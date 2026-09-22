from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from dexcg.data.training import DexArtTrainingDataset
from dexcg.models.contact.coordinates import CONTACT_COORDINATE_CONTRACT
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


def test_training_reads_precomputed_base_frame_targets(tmp_path) -> None:
    path = tmp_path / "task.zarr"
    root = zarr.group(str(path))
    root.attrs.update(
        split="seen",
        task="task",
        contact_coordinate_contract=CONTACT_COORDINATE_CONTRACT,
        point_cloud_frame="robot_base",
        contact_token_frame="robot_base",
        state_spatial_frame="robot_base",
        length_unit="metre",
        contact_tokenizer={"min_position": -1.0, "max_position": 1.2, "position_bins": 256},
    )
    data = root.create_group("data")
    meta = root.create_group("meta")
    centers = np.array([0.5, 0.7, 0.9], dtype=np.float32)
    point_cloud = np.stack(
        [
            np.array([[center - 0.2, 0.0, 0.0], [center + 0.2, 0.0, 0.0], [9, 9, 9]])
            for center in centers
        ]
    ).astype(np.float32)
    object_mask = np.broadcast_to([True, True, False], (3, 3)).copy()
    contact_points = np.zeros((3, 16, 3), dtype=np.float32)
    contact_points[:, 0, 0] = [0.8, 0.8, 1.0]
    contact_mask = np.zeros((3, 16), dtype=np.bool_)
    contact_mask[:, 0] = True
    data.create_dataset("point_cloud", data=point_cloud)
    data.create_dataset("object_point_mask", data=object_mask)
    data.create_dataset("imagin_robot", data=np.zeros((3, 1, 7), dtype=np.float32))
    data.create_dataset("agent_pos", data=np.zeros((3, 33), dtype=np.float32))
    data.create_dataset("action", data=np.zeros((3, 1), dtype=np.float32))
    data.create_dataset("contact_target_points", data=contact_points)
    data.create_dataset("contact_target_mask", data=contact_mask)
    tokenizer = contact_tokenizer()
    ids = np.full((3, 66), tokenizer.joint_end_id, dtype=np.int64)
    token_mask = np.zeros((3, 66), dtype=np.bool_)
    encoded = tokenizer.encode({"allegro_palm": [[0.1, 0.0, 0.0]]})
    ids[:, : len(encoded)] = encoded
    token_mask[:, : len(encoded)] = True
    data.create_dataset("contact_target_token_ids", data=ids)
    data.create_dataset("contact_target_token_mask", data=token_mask)
    data.create_dataset("object_center", data=np.stack([centers, np.zeros(3), np.zeros(3)], axis=1))
    data.create_dataset("object_center_valid", data=np.array([True, False, True]))
    meta.create_dataset("episode_ends", data=np.array([3], dtype=np.int64))
    meta.create_dataset("stable_contact_steps", data=np.array([1], dtype=np.int64))
    meta.create_dataset("low_level_grasp_instruction", data=np.array(["grasp"]))

    dataset = DexArtTrainingDataset([path], obs_horizon=1, action_horizon=1)

    assert len(dataset) == 2
    for index in range(2):
        sample = dataset[index]
        decoded = tokenizer.decode(sample["contact_token_ids"])["allegro_palm"][0]
        np.testing.assert_allclose(decoded, [0.1, 0.0, 0.0], atol=0.0044)


def test_training_rejects_old_contact_coordinates(tmp_path) -> None:
    path = tmp_path / "old.zarr"
    root = zarr.group(str(path))
    root.attrs.update(split="seen", task="task")

    with pytest.raises(ValueError, match="recollect with robot-base"):
        DexArtTrainingDataset([path])


def test_future_contacts_and_terminal_action_masks_in_memory(monkeypatch):
    # Two episodes: an interior gap, a discarded tail, then a new episode.
    # A memory store avoids creating a validation dataset on disk.
    root = zarr.group(store=zarr.storage.MemoryStore())
    root.attrs.update(
        split="seen", task="task", contact_coordinate_contract=CONTACT_COORDINATE_CONTRACT,
        point_cloud_frame="robot_base", contact_token_frame="robot_base",
        state_spatial_frame="robot_base", length_unit="metre",
        contact_tokenizer={"min_position": -1., "max_position": 1.2, "position_bins": 256},
    )
    data, meta = root.create_group("data"), root.create_group("meta")
    tokenizer = contact_tokenizer()
    contacts = np.zeros((8, 16), dtype=bool)
    contacts[[0, 3, 6, 7], 0] = True
    ids = np.full((8, 66), tokenizer.joint_end_id, dtype=np.int64)
    masks = np.zeros_like(ids, dtype=bool)
    for step in range(8):
        tokens = tokenizer.encode({"allegro_palm": [[step / 10., 0, 0]]} if contacts[step].any() else {})
        ids[step, :len(tokens)] = tokens
        masks[step, :len(tokens)] = True
    states = np.zeros((8, 33), dtype=np.float32)
    states[4:6, 32] = 999  # Discarded tails must not affect normalization.
    arrays = {
        "point_cloud": np.zeros((8, 2, 3), dtype=np.float32),
        "object_point_mask": np.ones((8, 2), dtype=bool),
        "object_center": np.zeros((8, 3), dtype=np.float32),
        "object_center_valid": np.ones(8, dtype=bool), "agent_pos": states,
        "action": np.arange(8, dtype=np.float32)[:, None],
        "contact_target_mask": contacts, "contact_target_token_ids": ids,
        "contact_target_token_mask": masks,
    }
    for name, array in arrays.items():
        data.create_dataset(name, data=array)
    meta.create_dataset("episode_ends", data=np.array([6, 8]))
    meta.create_dataset("low_level_grasp_instruction", data=np.array(["first", "second"]))
    monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: root)
    dataset = DexArtTrainingDataset(["in-memory"], action_horizon=4)
    assert len(dataset) == 6
    assert [s[3] for s in dataset.samples[0]] == [0, 1, 2, 3, 6, 7]
    assert dataset.effective_episode_ends[0].tolist() == [4, 8]
    assert dataset.state_max[32] == 0
    sample = dataset[1]
    np.testing.assert_array_equal(sample["contact_token_ids"].numpy(), ids[3])
    assert sample["action"].squeeze(1).tolist() == [1., 2., 3., 0.]
    assert sample["action_valid_mask"].tolist() == [True, True, True, False]
    assert dataset[3]["action_valid_mask"].tolist() == [True, False, False, False]
    assert dataset[4]["action"].squeeze(1).tolist() == [6., 7., 0., 0.]
    # Exercise split validation and train-only state ranges without any files.
    import json
    from pathlib import Path
    meta.create_dataset("object_id", data=np.array(["100", "200"]))
    data["agent_pos"][6:, 0] = 9
    data["point_cloud"][:, 1, 0] = 1
    manifest = {"train": {"episodes": {"task": {"100": [0]}}},
                "validation": {"episodes": {"task": {"200": [1]}}}}
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: json.dumps(manifest))
    training = DexArtTrainingDataset(["in-memory"], split_manifest="manifest", point_count=1)
    validation = DexArtTrainingDataset(["in-memory"], split_manifest="manifest", split="validation",
        point_count=1, state_statistics=(training.state_min, training.state_max))
    assert [s[1] for s in training.samples[0]] == [0, 0, 0, 0]
    assert [s[1] for s in validation.samples[0]] == [1, 1]
    assert training.state_max[0] == validation.state_max[0] == 0
    assert validation[0]["observation"]["agent_pos"][-1, 0] == 9
    assert training[0]["observation"]["point_cloud"].shape == (2, 1, 3)
    np.testing.assert_array_equal(validation[0]["observation"]["point_cloud"],
                                  validation[0]["observation"]["point_cloud"])
    manifest["validation"]["episodes"]["task"]["100"] = [0]
    with pytest.raises(ValueError, match="exactly once"):
        DexArtTrainingDataset(["in-memory"], split_manifest="manifest")
