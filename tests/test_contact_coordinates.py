from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexcg.common.typing import ContactPlan
from dexcg.models.contact.coordinates import (
    center_point_cloud,
    object_aabb_center_numpy,
    restore_contact_positions,
)
from dexcg.models.contact.planner import ContactPlanner
from dexcg.models.contact.tokenizer import VISION_TOKENS, AllegroContactTokenizer
from dexcg.robots.allegro import ALLEGRO_CONTACT_TOKENS


def test_center_point_cloud_uses_only_masked_object_bounds() -> None:
    points = torch.tensor([[[0.2, -0.1, 0.0], [0.6, 0.3, 0.4], [9.0, 9.0, 9.0]]])
    mask = torch.tensor([[True, True, False]])

    centered, center = center_point_cloud(points, mask)

    torch.testing.assert_close(center, torch.tensor([[0.4, 0.1, 0.2]]))
    torch.testing.assert_close(centered[0, :2].amin(0), -centered[0, :2].amax(0))
    torch.testing.assert_close(restore_contact_positions(centered[:, :2], center), points[:, :2])


def test_center_point_cloud_rejects_empty_object_mask() -> None:
    with pytest.raises(ValueError, match="at least one object point"):
        center_point_cloud(torch.zeros(1, 4, 3), torch.zeros(1, 4, dtype=torch.bool))


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


def test_decode_contacts_restores_robot_base_coordinates() -> None:
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
        torch.tensor([[0.6, 0.2, -0.1]]),
    )
    planner = SimpleNamespace(contact_tokenizer=tokenizer)

    decoded = ContactPlanner.decode_contacts(planner, plan)[0]

    actual = np.asarray(decoded[ALLEGRO_CONTACT_TOKENS[0][1:-1]][0])
    np.testing.assert_allclose(actual, local + [0.6, 0.2, -0.1], atol=0.002)


def test_position_quantization_matches_dexter_boundary_convention() -> None:
    vocabulary = {token: index for index, token in enumerate(ALLEGRO_CONTACT_TOKENS)}
    start = len(vocabulary)
    vocabulary["<|joint_start|>"] = start
    vocabulary["<|joint_end|>"] = start + 1
    vocabulary[VISION_TOKENS[0]] = start + 2
    for index in range(256):
        vocabulary[f"<pos_bin_{index}>"] = start + 3 + index
    tokenizer = AllegroContactTokenizer(SimpleNamespace(get_vocab=lambda: vocabulary))
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
