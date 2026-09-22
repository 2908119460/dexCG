from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexcg.common.typing import ContactPlan
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    MODEL_COORDINATE_CONTRACT,
    object_aabb_center_numpy,
    partfield_grid_coordinates,
    require_model_coordinates,
    robot_base_point_cloud,
    robot_state_in_base,
)
from dexcg.models.contact.planner import ContactPlanner
from dexcg.models.contact.tokenizer import VISION_TOKENS, AllegroContactTokenizer
from dexcg.robots.allegro import ALLEGRO_CONTACT_TOKENS


def test_object_selection_preserves_base_coordinates():
    points = torch.tensor([[[0.2, -0.1, 0.0], [0.6, 0.3, 0.4], [9.0, 9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])
    selected = robot_base_point_cloud(points, mask)
    torch.testing.assert_close(selected, points[:, [0, 1, 0]])
    with pytest.raises(ValueError, match="at least one object point"):
        robot_base_point_cloud(points, torch.zeros_like(mask))


def test_state_transform_rotates_velocities_and_translates_positions():
    from scipy.spatial.transform import Rotation

    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", 0.7).as_matrix()
    base[:3, 3] = [-0.5, 0.2, 0.3]
    palm = np.eye(4)
    palm[:3, :3] = Rotation.from_euler("y", 0.4).as_matrix()
    state = np.arange(33, dtype=np.float32) / 10
    converted = robot_state_in_base(state, base, palm)
    np.testing.assert_allclose(
        base[:3, :3] @ converted[28:31] + base[:3, 3], state[28:31], atol=1e-6
    )
    for start in [22, 25]:
        np.testing.assert_allclose(
            base[:3, :3] @ converted[start : start + 3], state[start : start + 3], atol=1e-6
        )
    np.testing.assert_array_equal(converted[:22], state[:22])
    assert converted[-1] == state[-1]
    assert converted[31] == pytest.approx((base[:3, :3].T @ palm[:3, :3])[2, 0])


def test_grid_encoding_preserves_origin_and_absolute_translation():
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.2, -0.8, 0.5]]])
    grid = partfield_grid_coordinates(xyz)
    torch.testing.assert_close(grid * 4, xyz)
    torch.testing.assert_close(
        partfield_grid_coordinates(xyz + 0.1) - grid, torch.full_like(grid, 0.025)
    )
    with pytest.raises(ValueError, match="refusing clipping"):
        partfield_grid_coordinates(torch.tensor([[[2.0, 0.0, 0.0]]]))


def test_checkpoint_rejects_old_model_semantics_even_with_base_contacts():
    with pytest.raises(ValueError, match="coordinate contract"):
        require_model_coordinates({"contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT})
    require_model_coordinates(
        {
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        }
    )


def test_numpy_object_center_supports_batches() -> None:
    points = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [99.0, 99.0, 99.0]],
            [[-2.0, -4.0, -6.0], [0.0, 0.0, 0.0], [99.0, 99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, True, False], [True, True, False]])

    centers = object_aabb_center_numpy(points, mask)

    np.testing.assert_array_equal(centers, [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])


def test_decode_contacts_are_already_in_robot_base() -> None:
    vocabulary = {token: index for index, token in enumerate(ALLEGRO_CONTACT_TOKENS)}
    start = len(vocabulary)
    vocabulary["<|joint_start|>"] = start
    vocabulary["<|joint_end|>"] = start + 1
    vocabulary[VISION_TOKENS[0]] = start + 2
    for index in range(256):
        vocabulary[f"<pos_bin_{index}>"] = start + 3 + index
    base_tokenizer = SimpleNamespace(get_vocab=lambda: vocabulary)
    tokenizer = AllegroContactTokenizer(base_tokenizer)
    local = np.array([0.05, -0.02, 0.1], dtype=np.float32)
    ids = tokenizer.encode({ALLEGRO_CONTACT_TOKENS[0][1:-1]: [local]})
    plan = ContactPlan(
        torch.tensor([ids]),
        torch.ones(1, len(ids), dtype=torch.bool),
    )
    planner = SimpleNamespace(contact_tokenizer=tokenizer)

    decoded = ContactPlanner.decode_contacts(planner, plan)[0]

    actual = np.asarray(decoded[ALLEGRO_CONTACT_TOKENS[0][1:-1]][0])
    np.testing.assert_allclose(actual, local, atol=0.0044)


def test_position_quantization_matches_dexter_boundary_convention() -> None:
    vocabulary = {token: index for index, token in enumerate(ALLEGRO_CONTACT_TOKENS)}
    start = len(vocabulary)
    vocabulary["<|joint_start|>"] = start
    vocabulary["<|joint_end|>"] = start + 1
    vocabulary[VISION_TOKENS[0]] = start + 2
    for index in range(256):
        vocabulary[f"<pos_bin_{index}>"] = start + 3 + index
    tokenizer = AllegroContactTokenizer(
        SimpleNamespace(get_vocab=lambda: vocabulary), min_position=-0.4, max_position=0.4
    )
    link = ALLEGRO_CONTACT_TOKENS[0][1:-1]

    encoded = tokenizer.encode({link: [[0.0, 0.4, -0.4]]})
    decoded = tokenizer.decode(encoded)[link][0]

    expected_centers = (
        np.linspace(-0.4, 0.4, 256, dtype=np.float32)[:-1]
        + np.linspace(-0.4, 0.4, 256, dtype=np.float32)[1:]
    ) / 2.0
    np.testing.assert_allclose(
        decoded,
        [expected_centers[127], expected_centers[254], expected_centers[0]],
    )
