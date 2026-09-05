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


def test_training_reads_precomputed_object_centered_targets(tmp_path) -> None:
    path = tmp_path / "task.zarr"
    root = zarr.group(str(path))
    root.attrs.update(
        split="seen",
        task="task",
        contact_coordinate_contract=CONTACT_COORDINATE_CONTRACT,
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
    data.create_dataset("agent_pos", data=np.zeros((3, 2), dtype=np.float32))
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
    data.create_dataset(
        "object_center", data=np.stack([centers, np.zeros(3), np.zeros(3)], axis=1)
    )
    data.create_dataset("object_center_valid", data=np.array([True, False, True]))
    meta.create_dataset("episode_ends", data=np.array([3], dtype=np.int64))
    meta.create_dataset("stable_contact_steps", data=np.array([1], dtype=np.int64))
    meta.create_dataset("low_level_grasp_instruction", data=np.array(["grasp"]))

    dataset = DexArtTrainingDataset([path], obs_horizon=1, action_horizon=1)

    assert len(dataset) == 2
    for index in range(2):
        sample = dataset[index]
        decoded = tokenizer.decode(sample["contact_token_ids"])["allegro_palm"][0]
        np.testing.assert_allclose(decoded, [0.1, 0.0, 0.0], atol=0.002)


def test_training_rejects_unmigrated_contact_coordinates(tmp_path) -> None:
    path = tmp_path / "old.zarr"
    root = zarr.group(str(path))
    root.attrs.update(split="seen", task="task")

    with pytest.raises(ValueError, match="Migrate it before training"):
        DexArtTrainingDataset([path])
