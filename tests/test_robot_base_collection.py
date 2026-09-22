from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexcg.data.dexart import DexArtEpisode
from dexcg.data.robot_base import (
    InvalidObjectObservation,
    append_episode,
    balanced_quotas,
    base_state,
    recover_store,
    sample_object_points,
)
from dexcg.models.contact.tokenizer import AllegroContactTokenizer
from dexcg.robots.allegro import ALLEGRO_CONTACT_TOKENS


def test_world_state_roundtrip_with_rotated_translated_base():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.4, 0.8]).as_matrix()
    transform[:3, 3] = [0.8, -0.3, 0.4]
    original = np.arange(32, dtype=np.float32) / 10
    converted = base_state(original, transform)
    for section in (slice(22, 25), slice(25, 28)):
        np.testing.assert_allclose(
            converted[section] @ transform[:3, :3].T, original[section], atol=1e-6
        )
    np.testing.assert_allclose(
        converted[28:31] @ transform[:3, :3].T + transform[:3, 3], original[28:31], atol=1e-6
    )
    np.testing.assert_array_equal(converted[:22], original[:22])
    assert converted[-1] == original[-1]


def test_object_sampling_retains_body_outside_old_crop_and_pixel_provenance():
    source = {
        "camera_xyz": np.array([[0, 0, -1], [3, 0, -1], [1, 1, -1], [0, 0, 0]], dtype=np.float32),
        "actor_ids": np.array([10, 11, 99, 10]),
        "base_from_camera_gl": np.eye(4),
        "intrinsics": np.eye(3),
    }
    sample = sample_object_points(source, [10, 11], np.random.RandomState(0), 32)
    assert set(sample["point_actor_id"]) == {10, 11}
    assert int(sample["object_visible_pixel_count"]) == 2
    np.testing.assert_array_equal(
        sample["point_cloud"], source["camera_xyz"][sample["point_pixel_index"]]
    )
    np.testing.assert_allclose(sample["object_center"], [1.5, 0, -1])
    with pytest.raises(InvalidObjectObservation):
        sample_object_points(source, [888], np.random.RandomState(0))


@pytest.mark.parametrize("count,low,high", [(11, 22, 23), (17, 14, 15)])
def test_exact_250_balanced_quotas(count, low, high):
    quotas = balanced_quotas(list(range(count)), 250, 0)
    assert sum(quotas.values()) == 250
    assert min(quotas.values()) == low and max(quotas.values()) == high


def make_tokenizer():
    tokens = [*ALLEGRO_CONTACT_TOKENS, "<|joint_start|>", "<|joint_end|>", "<|vision_pad|>"]
    tokens += [f"<pos_bin_{index}>" for index in range(256)]
    vocabulary = dict(zip(tokens, range(len(tokens)), strict=True))
    return AllegroContactTokenizer(
        SimpleNamespace(get_vocab=lambda: vocabulary), min_position=-1.0, max_position=1.2
    )


def make_episode():
    points = np.zeros((16, 3), dtype=np.float32)
    points[0] = [0.8, 0.2, 0.3]
    masks = np.zeros(16, dtype=bool)
    masks[0] = True
    return DexArtEpisode(
        observations={
            "point_cloud": [np.ones((4, 3), dtype=np.float32)] * 2,
            "object_point_mask": [np.ones(4, dtype=bool)] * 2,
            "object_center": [np.array([0.7, 0.2, 0.3])] * 2,
            "state": [np.zeros(32, dtype=np.float32)] * 2,
            "agent_pos": [np.zeros(33, dtype=np.float32)] * 2,
            "palm_pose_robot_base": [np.eye(4)] * 2,
        },
        actions=[np.zeros(22)] * 2,
        raw_contact_points=[points] * 2,
        raw_contact_masks=[masks] * 2,
        stable_contact_step=0,
        stable_contact_points=points,
        stable_contact_mask=masks,
        object_id="148",
        task_id="faucet",
        annotation_views=np.zeros((5, 2, 2, 3), dtype=np.uint8),
        camera_extrinsics=np.broadcast_to(np.eye(4), (5, 4, 4)).copy(),
        language={"class_name": "faucet"},
        annotation_raw="{}",
    )


def test_incremental_storage_robot_base_tokens_and_interrupted_append(tmp_path):
    path = tmp_path / "episodes.zarr"
    tokenizer, episode = make_tokenizer(), make_episode()
    append_episode(path, episode, tokenizer, {"contact_token_frame": "robot_base"}, {"seed": 42})
    root = recover_store(path)
    decoded = tokenizer.decode(root["data/contact_target_token_ids"][0])
    np.testing.assert_allclose(decoded["allegro_palm"][0], [0.8, 0.2, 0.3], atol=0.0044)
    root["data/point_cloud"].append(np.ones((1, 4, 3)))
    root["meta/episode_ends"].append(np.array([99]))
    root = recover_store(path)
    assert root["data/point_cloud"].shape[0] == 2
    assert root["meta/episode_ends"][:].tolist() == [2]
    append_episode(path, episode, tokenizer, {"contact_token_frame": "robot_base"}, {"seed": 43})
    root = recover_store(path)
    assert root["meta/episode_ends"][:].tolist() == [2, 4]
    assert root.attrs["committed_episodes"] == 2
    assert root["meta/seed"][:].tolist() == [42, 43]
