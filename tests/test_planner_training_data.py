from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from dexcg.data.planner_training import DexArtPlannerDataset, corrupt_previous_contacts
from dexcg.models.contact.coordinates import CONTACT_COORDINATE_CONTRACT


def _write_task(path: Path, trajectories_per_object: int = 14) -> None:
    root = zarr.group(str(path))
    root.attrs.update(
        split="seen",
        task="faucet",
        contact_coordinate_contract=CONTACT_COORDINATE_CONTRACT,
        point_cloud_frame="robot_base",
        contact_token_frame="robot_base",
        state_spatial_frame="robot_base",
        length_unit="metre",
        contact_tokenizer={"min_position": -1.0, "max_position": 1.2, "position_bins": 256},
    )
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_count = 2 * trajectories_per_object
    length = 3
    total = episode_count * length
    point_cloud = np.zeros((total, 4, 3), dtype=np.float32)
    point_cloud[:, :, 0] = np.arange(total)[:, None]
    contact_mask = np.ones((total, 16), dtype=np.bool_)
    contact_mask[:, 1:] = False
    contact_mask[1::length] = False
    token_ids = np.full((total, 6), 3, dtype=np.int64)
    token_ids[:, :6] = np.array([2, 10, 20, 21, 22, 3])
    token_mask = np.ones((total, 6), dtype=np.bool_)
    token_mask[1::length, 2:] = False

    data.create_dataset("point_cloud", data=point_cloud)
    data.create_dataset("object_point_mask", data=np.ones((total, 4), dtype=np.bool_))
    data.create_dataset("object_center", data=np.zeros((total, 3), dtype=np.float32))
    data.create_dataset("object_center_valid", data=np.ones(total, dtype=np.bool_))
    data.create_dataset("contact_target_mask", data=contact_mask)
    data.create_dataset("contact_target_token_ids", data=token_ids)
    data.create_dataset("contact_target_token_mask", data=token_mask)
    meta.create_dataset("episode_ends", data=np.arange(length, total + 1, length, dtype=np.int64))
    meta.create_dataset(
        "object_id",
        data=np.asarray(
            ["object_a"] * trajectories_per_object + ["object_b"] * trajectories_per_object,
            dtype=object,
        ),
        object_codec=zarr.codecs.VLenUTF8(),
    )
    for name, prefix in (
        ("low_level_grasp_instruction", "low"),
        ("high_level_grasp_instruction", "high"),
    ):
        meta.create_dataset(
            name,
            data=np.asarray([f"{prefix} {index}" for index in range(episode_count)], dtype=object),
            object_codec=zarr.codecs.VLenUTF8(),
        )


def test_planner_split_excludes_empty_targets_and_preserves_trajectory_history(
    tmp_path,
) -> None:
    path = tmp_path / "faucet.zarr"
    _write_task(path)
    kwargs = {
        "paths": [path],
        "deployment_instructions": {"faucet": "open the faucet"},
        "split_seed": 7,
        "history_interval": 1,
    }
    train = DexArtPlannerDataset(split="train", **kwargs)
    validation = DexArtPlannerDataset(split="validation", **kwargs)

    assert len(train.episodes) == 22
    assert len(validation.episodes) == 6
    assert len(train) == 44
    assert len(validation) == 12
    assert train.audit()["excluded_empty_targets"] == 22
    train_episodes = {(episode.task_index, episode.episode_index) for episode in train.episodes}
    validation_episodes = {
        (episode.task_index, episode.episode_index) for episode in validation.episodes
    }
    assert train_episodes.isdisjoint(validation_episodes)

    first_episode_samples = [
        train[index]
        for index, sample in enumerate(train.samples)
        if sample.episode == train.samples[0].episode
    ]
    assert first_episode_samples[0]["previous_mask"].sum() == 0
    assert first_episode_samples[1]["previous_mask"].sum() == 6
    assert all(sample["target_mask"].sum() == 6 for sample in first_episode_samples)


def test_single_frame_samples_do_not_read_previous_contacts(tmp_path) -> None:
    path = tmp_path / "faucet.zarr"
    _write_task(path)
    dataset = DexArtPlannerDataset(
        [path],
        split="all",
        deployment_instructions={"faucet": "open the faucet"},
        history_interval=1,
        use_previous_contact=False,
    )
    assert all(sample.previous_step is None for sample in dataset.samples)
    sample = dataset[1]
    assert sample["step"] == 2
    assert not sample["previous_mask"].any()
    assert sample["previous_step"] == -1
    root = zarr.open_group(str(path), mode="a")
    root["data/contact_target_token_ids"][0] = np.full(6, 999)
    after = dataset[1]
    for key in ("point_cloud", "target_ids", "target_mask", "previous_ids", "previous_mask"):
        assert torch.equal(sample[key], after[key])
    assert dataset.audit()["use_previous_contact"] is False


def test_language_family_is_constant_within_trajectory_for_an_epoch(tmp_path) -> None:
    path = tmp_path / "faucet.zarr"
    _write_task(path)
    dataset = DexArtPlannerDataset(
        [path],
        split="all",
        deployment_instructions={"faucet": "deploy"},
        history_interval=1,
    )
    first = dataset[0]
    batch = {
        "episode_key": [first["episode_key"], first["episode_key"]],
        "low_level_language": ["low", "low"],
        "high_level_language": ["high", "high"],
        "deployment_language": ["deploy", "deploy"],
    }

    assert len(set(dataset.languages_for_epoch(batch, epoch=3))) == 1
    assert dataset.languages_for_epoch(batch, epoch=3) != dataset.languages_for_epoch(
        batch, epoch=4
    )
    assert dataset.languages_for_family(batch, "low_level") == ["low", "low"]
    assert dataset.languages_for_family(batch, "high_level") == ["high", "high"]
    assert dataset.languages_for_family(batch, "deployment") == ["deploy", "deploy"]


def _history_batch() -> dict[str, object]:
    return {
        "episode_key": ["0:0", "0:1", "0:2"],
        "step": torch.tensor([8, 16, 24]),
        "previous_ids": torch.tensor(
            [[1, 10, 20, 21, 22, 2], [1, 10, 20, 21, 22, 2], [1, 10, 20, 21, 22, 2]]
        ),
        "previous_mask": torch.ones(3, 6, dtype=torch.bool),
        "target_ids": torch.tensor([[1, 30, 40, 41, 42, 2]]).repeat(3, 1),
        "target_mask": torch.ones(3, 6, dtype=torch.bool),
    }


def test_history_corruption_is_deterministic_and_preserves_targets() -> None:
    batch = _history_batch()
    kwargs = {
        "position_id_to_bin": {20: 0, 21: 1, 22: 2},
        "position_token_ids": [20, 21, 22],
        "seed": 42,
        "epoch": 3,
        "drop_probability": 0.0,
        "perturb_probability": 1.0,
        "max_bin_offset": 2,
    }
    first, first_stats = corrupt_previous_contacts(batch, **kwargs)
    second, second_stats = corrupt_previous_contacts(batch, **kwargs)

    assert torch.equal(first["previous_ids"], second["previous_ids"])
    assert torch.equal(first["previous_mask"], second["previous_mask"])
    assert first_stats == second_stats
    assert first_stats["perturbed"] == 3
    assert first_stats["changed_tokens"] > 0
    assert torch.equal(first["target_ids"], batch["target_ids"])
    assert torch.equal(first["target_mask"], batch["target_mask"])
    assert first["target_ids"].data_ptr() == batch["target_ids"].data_ptr()
    assert torch.equal(batch["previous_ids"], _history_batch()["previous_ids"])


def test_history_corruption_drop_and_clean_paths() -> None:
    batch = _history_batch()
    common = {
        "position_id_to_bin": {20: 0, 21: 1, 22: 2},
        "position_token_ids": [20, 21, 22],
        "seed": 1,
        "epoch": 1,
        "max_bin_offset": 1,
    }
    dropped, drop_stats = corrupt_previous_contacts(
        batch, drop_probability=1.0, perturb_probability=0.0, **common
    )
    clean, clean_stats = corrupt_previous_contacts(
        batch, drop_probability=0.0, perturb_probability=0.0, **common
    )

    assert not dropped["previous_mask"].any()
    assert drop_stats["dropped"] == 3
    assert torch.equal(clean["previous_ids"], batch["previous_ids"])
    assert torch.equal(clean["previous_mask"], batch["previous_mask"])
    assert clean_stats["clean"] == 3


def test_planner_data_rejects_unbalanced_object_trajectory_counts(tmp_path) -> None:
    path = tmp_path / "faucet.zarr"
    _write_task(path, trajectories_per_object=3)

    with pytest.raises(ValueError, match="14 or 9"):
        DexArtPlannerDataset(
            [path],
            split="train",
            deployment_instructions={"faucet": "open"},
        )
