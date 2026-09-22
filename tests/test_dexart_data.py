from types import SimpleNamespace

import numpy as np
import zarr

from dexcg.data.dexart import (
    DexArtEpisode,
    episode_coordinate_arrays,
    write_dexart_dataset,
)
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    OBJECT_CENTER_DEFINITION,
)
from dexcg.models.contact.tokenizer import VISION_TOKENS, AllegroContactTokenizer
from dexcg.robots.allegro import ALLEGRO_CONTACT_TOKENS


def contact_tokenizer() -> AllegroContactTokenizer:
    vocabulary = {token: index for index, token in enumerate(ALLEGRO_CONTACT_TOKENS)}
    start = len(vocabulary)
    vocabulary["<|joint_start|>"] = start
    vocabulary["<|joint_end|>"] = start + 1
    vocabulary[VISION_TOKENS[0]] = start + 2
    for index in range(256):
        vocabulary[f"<pos_bin_{index}>"] = start + 3 + index
    return AllegroContactTokenizer(SimpleNamespace(get_vocab=lambda: vocabulary))


def test_stable_contact_fills_targets_through_grasp_transition() -> None:
    raw_points = [np.zeros((16, 3), dtype=np.float32) for _ in range(4)]
    raw_masks = [np.zeros(16, dtype=np.bool_) for _ in range(4)]
    raw_points[3][4] = [0.3, 0.2, 0.1]
    raw_masks[3][4] = True
    stable_points = np.zeros((16, 3), dtype=np.float32)
    stable_mask = np.zeros(16, dtype=np.bool_)
    stable_points[4] = [0.1, 0.2, 0.3]
    stable_mask[4] = True

    episode = DexArtEpisode(
        observations={},
        actions=[np.zeros(22, dtype=np.float32) for _ in range(4)],
        raw_contact_points=raw_points,
        raw_contact_masks=raw_masks,
        stable_contact_step=2,
        stable_contact_points=stable_points,
        stable_contact_mask=stable_mask,
        object_id="object",
        task_id="task",
        annotation_views=np.empty((5, 1, 1, 3), dtype=np.uint8),
        camera_extrinsics=np.empty((5, 4, 4), dtype=np.float32),
        language={},
        annotation_raw="{}",
    )

    target_points, target_masks = episode.contact_targets()

    np.testing.assert_array_equal(target_masks[:3], np.broadcast_to(stable_mask, (3, 16)))
    np.testing.assert_array_equal(target_points[:3], np.broadcast_to(stable_points, (3, 16, 3)))
    np.testing.assert_array_equal(target_masks[3], raw_masks[3])
    np.testing.assert_array_equal(target_points[3], raw_points[3])


def test_contact_coordinates_do_not_depend_on_object_center() -> None:
    centers = [0.5, 0.7, 0.9]
    point_clouds = [
        np.array(
            [[center - 0.2, 0.0, 0.0], [center + 0.2, 0.0, 0.0], [9.0, 9.0, 9.0]],
            dtype=np.float32,
        )
        for center in centers
    ]
    episode = DexArtEpisode(
        observations={
            "point_cloud": point_clouds,
            "object_point_mask": [np.array([True, True, False])] * 3,
            "object_center": [np.array([center, 0.0, 0.0]) for center in centers],
        },
        actions=[np.zeros(22, dtype=np.float32) for _ in centers],
        raw_contact_points=[np.zeros((16, 3), dtype=np.float32) for _ in centers],
        raw_contact_masks=[np.zeros(16, dtype=np.bool_) for _ in centers],
        stable_contact_step=1,
        stable_contact_points=np.zeros((16, 3), dtype=np.float32),
        stable_contact_mask=np.zeros(16, dtype=np.bool_),
        object_id="object",
        task_id="task",
        annotation_views=np.empty((5, 1, 1, 3), dtype=np.uint8),
        camera_extrinsics=np.empty((5, 4, 4), dtype=np.float32),
        language={},
        annotation_raw="{}",
    )

    object_centers, valid, target_centers = episode_coordinate_arrays(episode)

    np.testing.assert_allclose(object_centers[:, 0], centers)
    np.testing.assert_array_equal(valid, [True, True, True])
    np.testing.assert_allclose(target_centers[:, 0], [0.0, 0.0, 0.0])


def test_zarr_round_trip_preserves_full_resolution_centers_and_contract(tmp_path) -> None:
    centers = np.array([[0.5, -0.1, 0.2], [0.7, -0.1, 0.2]], dtype=np.float32)
    stable_points = np.zeros((16, 3), dtype=np.float32)
    stable_mask = np.zeros(16, dtype=np.bool_)
    stable_points[0] = centers[1] + np.array([0.1, 0.0, 0.0], dtype=np.float32)
    stable_mask[0] = True
    episode = DexArtEpisode(
        observations={
            "img": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in centers],
            "depth": [np.zeros((2, 2), dtype=np.float32) for _ in centers],
            "point_cloud": [np.zeros((4, 3), dtype=np.float32) for _ in centers],
            "object_point_mask": [np.ones(4, dtype=np.bool_) for _ in centers],
            "object_center": list(centers),
            "imagin_robot": [np.zeros((1, 7), dtype=np.float32) for _ in centers],
            "state": [np.zeros(32, dtype=np.float32) for _ in centers],
            "agent_pos": [np.zeros(33, dtype=np.float32) for _ in centers],
        },
        actions=[np.zeros(22, dtype=np.float32) for _ in centers],
        raw_contact_points=[np.zeros((16, 3), dtype=np.float32) for _ in centers],
        raw_contact_masks=[np.zeros(16, dtype=np.bool_) for _ in centers],
        stable_contact_step=1,
        stable_contact_points=stable_points,
        stable_contact_mask=stable_mask,
        object_id="148",
        task_id="faucet-148",
        annotation_views=np.zeros((5, 2, 2, 3), dtype=np.uint8),
        camera_extrinsics=np.broadcast_to(np.eye(4, dtype=np.float32), (5, 4, 4)).copy(),
        language={
            "class_name": "faucet",
            "grasped_object_part": "handle",
            "low_level_grasp_instruction": "grasp the handle",
            "high_level_grasp_instruction": "turn on the faucet",
        },
        annotation_raw="{}",
    )
    tokenizer = contact_tokenizer()
    output_path = tmp_path / "faucet.zarr"

    write_dexart_dataset(
        output_path,
        [episode],
        tokenizer,
        max_token_length=66,
        attributes={"split": "seen", "task": "faucet"},
    )

    root = zarr.open_group(str(output_path), mode="r")
    assert root.attrs["contact_coordinate_contract"] == CONTACT_COORDINATE_CONTRACT
    assert root.attrs["object_center_definition"] == OBJECT_CENTER_DEFINITION
    assert root.attrs["contact_token_frame"] == "robot_base"
    np.testing.assert_array_equal(root["data/object_center"][:], centers)
    np.testing.assert_array_equal(root["data/object_center_valid"][:], [True, True])
    decoded = tokenizer.decode(root["data/contact_target_token_ids"][0])
    np.testing.assert_allclose(decoded["allegro_palm"][0], stable_points[0], atol=0.0044)
