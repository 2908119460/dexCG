"""Trajectory-aware contact-planner samples from DexArt demonstrations."""

from __future__ import annotations

import bisect
import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from dexcg.data.dexart import TARGET_TOKEN_IDS, TARGET_TOKEN_MASK
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    require_dataset_coordinates,
)

LANGUAGE_FAMILIES = ("low_level", "high_level", "deployment")


@dataclass(frozen=True)
class PlannerEpisode:
    task_index: int
    episode_index: int
    start: int
    end: int
    object_id: str
    valid_steps: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.task_index}:{self.episode_index}"


@dataclass(frozen=True)
class PlannerSample:
    episode: PlannerEpisode
    step: int
    previous_step: int | None
    weight: float


def _validation_count(episode_count: int) -> int:
    if episode_count == 14:
        return 3
    if episode_count == 9:
        return 2
    if episode_count in (15, 22, 23):
        return max(1, round(episode_count * 0.2))
    if episode_count < 2:
        raise ValueError("At least two trajectories per object are required for validation")
    raise ValueError(
        "Balanced planner data must contain 14 or 9 legacy, or 15, 22, or 23 "
        "new trajectories per object; "
        f"received {episode_count}"
    )


def _split_episode_indices(object_ids: Sequence[str], seed: int) -> tuple[set[int], set[int]]:
    by_object: dict[str, list[int]] = defaultdict(list)
    for episode_index, object_id in enumerate(object_ids):
        by_object[str(object_id)].append(episode_index)

    train: set[int] = set()
    validation: set[int] = set()
    for object_id in sorted(by_object):
        indices = np.asarray(by_object[object_id], dtype=np.int64)
        object_seed = int.from_bytes(object_id.encode("utf-8"), "little") % (2**32)
        rng = np.random.default_rng(np.random.SeedSequence([seed, object_seed]))
        shuffled = indices[rng.permutation(len(indices))]
        validation_count = _validation_count(len(indices))
        validation.update(map(int, shuffled[:validation_count]))
        train.update(map(int, shuffled[validation_count:]))
    return train, validation


class DexArtPlannerDataset(Dataset):
    """Use every nonempty target while preserving explicit trajectory history."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        split: str,
        deployment_instructions: Mapping[str, str],
        split_seed: int = 42,
        history_interval: int = 8,
        use_previous_contact: bool = True,
        use_robot_state: bool = False,
    ) -> None:
        if split not in {"train", "validation", "all"}:
            raise ValueError("split must be train, validation, or all")
        if not paths:
            raise ValueError("At least one dataset path is required")
        if history_interval < 1:
            raise ValueError("history_interval must be at least one")

        self.paths = tuple(Path(path).resolve() for path in paths)
        self.split = split
        self.split_seed = int(split_seed)
        self.history_interval = int(history_interval)
        self.use_previous_contact = bool(use_previous_contact)
        self.use_robot_state = bool(use_robot_state)
        self.deployment_instructions = dict(deployment_instructions)
        self._roots: dict[tuple[int, int], Any] = {}
        self.tasks: list[str] = []
        self.episodes: list[PlannerEpisode] = []
        self.samples: list[PlannerSample] = []
        self.excluded_empty_targets = 0
        self.excluded_invalid_centers = 0

        for task_index, path in enumerate(self.paths):
            root = zarr.open_group(str(path), mode="r")
            if root.attrs.get("split") != "seen":
                raise ValueError(f"Planner dataset must be seen split: {path}")
            require_dataset_coordinates(root.attrs)
            if self.use_robot_state:
                for key in ("agent_pos", "palm_pose_robot_base"):
                    if f"data/{key}" not in root:
                        raise ValueError(f"Robot-conditioned VLM requires data/{key}: {path}")
            task = str(root.attrs["task"])
            if task not in self.deployment_instructions:
                raise ValueError(f"Missing deployment instruction for task {task!r}")
            self.tasks.append(task)

            episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
            starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
            object_ids = [str(value) for value in root["meta/object_id"][:]]
            train_indices, validation_indices = (
                _split_episode_indices(object_ids, self.split_seed)
                if split != "all"
                else (set(range(len(episode_ends))), set())
            )
            selected = (
                set(range(len(episode_ends)))
                if split == "all"
                else train_indices
                if split == "train"
                else validation_indices
            )
            target_contacts = root["data/contact_target_mask"]
            center_valid = root["data/object_center_valid"]
            nonempty = np.asarray(target_contacts[:], dtype=bool).any(axis=1)
            valid_centers = np.asarray(center_valid[:], dtype=bool)

            for episode_index, (start, end, object_id) in enumerate(
                zip(starts, episode_ends, object_ids, strict=True)
            ):
                if episode_index not in selected:
                    continue
                valid = valid_centers[start:end]
                contacts = nonempty[start:end]
                self.excluded_invalid_centers += int((~valid).sum())
                self.excluded_empty_targets += int((valid & ~contacts).sum())
                valid_steps = (np.flatnonzero(valid & contacts) + start).tolist()
                if not valid_steps:
                    raise ValueError(f"Episode {task}:{episode_index} has no valid planner targets")
                episode = PlannerEpisode(
                    task_index=task_index,
                    episode_index=episode_index,
                    start=int(start),
                    end=int(end),
                    object_id=object_id,
                    valid_steps=tuple(valid_steps),
                )
                self.episodes.append(episode)

        total_samples = sum(len(episode.valid_steps) for episode in self.episodes)
        episode_count = len(self.episodes)
        for episode in self.episodes:
            episode_weight = total_samples / (episode_count * len(episode.valid_steps))
            valid_steps = episode.valid_steps
            for step in valid_steps:
                previous_limit = step - self.history_interval
                previous_index = bisect.bisect_right(valid_steps, previous_limit) - 1
                previous_step = (
                    valid_steps[previous_index]
                    if self.use_previous_contact and previous_index >= 0
                    else None
                )
                self.samples.append(PlannerSample(episode, step, previous_step, episode_weight))

    def __len__(self) -> int:
        return len(self.samples)

    def _root(self, task_index: int):
        key = (os.getpid(), task_index)
        if key not in self._roots:
            self._roots[key] = zarr.open_group(str(self.paths[task_index]), mode="r")
        return self._roots[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        episode = sample.episode
        root = self._root(episode.task_index)
        data = root["data"]
        target_ids = np.asarray(data[TARGET_TOKEN_IDS][sample.step], dtype=np.int64)
        target_mask = np.asarray(data[TARGET_TOKEN_MASK][sample.step], dtype=np.bool_)
        previous_ids = np.full_like(target_ids, int(target_ids[-1]))
        previous_mask = np.zeros_like(target_mask)
        if sample.previous_step is not None:
            previous_ids = np.asarray(data[TARGET_TOKEN_IDS][sample.previous_step], dtype=np.int64)
            previous_mask = np.asarray(
                data[TARGET_TOKEN_MASK][sample.previous_step], dtype=np.bool_
            )

        episode_index = episode.episode_index
        task = self.tasks[episode.task_index]
        result = {
            "point_cloud": torch.from_numpy(
                np.asarray(data["point_cloud"][sample.step], dtype=np.float32)
            ),
            "object_point_mask": torch.from_numpy(
                np.asarray(data["object_point_mask"][sample.step], dtype=np.bool_)
            ),
            "object_center": torch.from_numpy(
                np.asarray(data["object_center"][sample.step], dtype=np.float32)
            ),
            "target_ids": torch.from_numpy(target_ids),
            "target_mask": torch.from_numpy(target_mask),
            "previous_ids": torch.from_numpy(previous_ids),
            "previous_mask": torch.from_numpy(previous_mask),
            "sample_weight": torch.tensor(sample.weight, dtype=torch.float32),
            "task": task,
            "low_level_language": str(root["meta/low_level_grasp_instruction"][episode_index]),
            "high_level_language": str(root["meta/high_level_grasp_instruction"][episode_index]),
            "deployment_language": self.deployment_instructions[task],
            "episode_key": episode.key,
            "episode_index": episode_index,
            "step": sample.step - episode.start,
            "absolute_step": sample.step,
            "previous_step": -1 if sample.previous_step is None else sample.previous_step,
        }
        if self.use_robot_state:
            result["robot_qpos"] = torch.from_numpy(
                np.asarray(data["agent_pos"][sample.step, :22], dtype=np.float32)
            )
            result["palm_pose_robot_base"] = torch.from_numpy(
                np.asarray(data["palm_pose_robot_base"][sample.step], dtype=np.float32)
            )
        return result

    def languages_for_epoch(self, batch: Mapping[str, Any], epoch: int) -> list[str]:
        result = []
        for row, episode_key in enumerate(batch["episode_key"]):
            task_index, episode_index = map(int, episode_key.split(":"))
            family_index = (self.split_seed + epoch + task_index + episode_index) % len(
                LANGUAGE_FAMILIES
            )
            family = LANGUAGE_FAMILIES[family_index]
            result.append(str(batch[f"{family}_language"][row]))
        return result

    @staticmethod
    def languages_for_family(batch: Mapping[str, Any], family: str) -> list[str]:
        if family not in LANGUAGE_FAMILIES:
            raise ValueError(f"unknown language family: {family}")
        return [str(value) for value in batch[f"{family}_language"]]

    def split_manifest(self) -> dict[str, Any]:
        tasks: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for episode in self.episodes:
            task = self.tasks[episode.task_index]
            tasks[task][episode.object_id].append(episode.episode_index)
        return {
            "split": self.split,
            "split_seed": self.split_seed,
            "history_interval": self.history_interval,
            "use_previous_contact": self.use_previous_contact,
            "episodes": {
                task: {object_id: sorted(indices) for object_id, indices in objects.items()}
                for task, objects in tasks.items()
            },
        }

    def audit(self) -> dict[str, Any]:
        return {
            "coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "split": self.split,
            "trajectory_count": len(self.episodes),
            "training_sample_count": len(self.samples),
            "excluded_empty_targets": self.excluded_empty_targets,
            "excluded_invalid_centers": self.excluded_invalid_centers,
            "history_interval": self.history_interval,
            "use_previous_contact": self.use_previous_contact,
            "empty_contact_policy": "excluded",
            "use_robot_state": self.use_robot_state,
        }


def diagnostic_samples(dataset, seed=42):
    """Equal object/trajectory quotas, independent of labels and predictions."""
    groups = defaultdict(list)
    lookup = {(s.episode.key, s.step): i for i, s in enumerate(dataset.samples)}
    for episode in dataset.episodes:
        groups[(episode.task_index, episode.object_id)].append(episode)
    indices, manifest = [], []
    for (task_index, object_id), episodes in sorted(groups.items()):
        key = f"{seed}:{dataset.split}:{task_index}:{object_id}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        episodes = sorted(episodes, key=lambda episode: episode.episode_index)
        if len(episodes) < 2:
            raise ValueError("Diagnostic needs two trajectories per object in each split")
        for position in sorted(rng.choice(len(episodes), 2, replace=False).tolist()):
            episode = episodes[position]
            if len(episode.valid_steps) < 3:
                raise ValueError("Diagnostic needs three valid frames per selected trajectory")
            for offset in np.linspace(0, len(episode.valid_steps) - 1, 3, dtype=int):
                step = episode.valid_steps[offset]
                indices.append(lookup[(episode.key, step)])
                manifest.append({"task": dataset.tasks[task_index], "object_id": object_id,
                                 "episode_key": episode.key, "absolute_step": step,
                                 "step": step - episode.start})
    assert len(set(indices)) == len(indices)
    return indices, manifest


def corrupt_previous_contacts(
    batch: Mapping[str, Any],
    position_id_to_bin: Mapping[int, int],
    position_token_ids: Sequence[int],
    *,
    seed: int,
    epoch: int,
    drop_probability: float,
    perturb_probability: float,
    max_bin_offset: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Apply deterministic trajectory-history corruption without changing stored data."""
    if drop_probability < 0.0 or perturb_probability < 0.0:
        raise ValueError("history corruption probabilities must be non-negative")
    if drop_probability + perturb_probability > 1.0:
        raise ValueError("history corruption probabilities must sum to at most one")
    if max_bin_offset < 1 and perturb_probability > 0.0:
        raise ValueError("max_bin_offset must be positive when perturbation is enabled")

    previous_ids = batch["previous_ids"].clone()
    previous_mask = batch["previous_mask"].clone()
    position_tokens = np.asarray(position_token_ids, dtype=np.int64)
    stats = {"absent": 0, "clean": 0, "dropped": 0, "perturbed": 0, "changed_tokens": 0}

    for row, (episode_key, step) in enumerate(
        zip(batch["episode_key"], batch["step"], strict=True)
    ):
        if not bool(previous_mask[row].any()):
            stats["absent"] += 1
            continue
        key = f"{seed}:{epoch}:{episode_key}:{int(step)}".encode("utf-8")
        deterministic_seed = int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "little")
        rng = np.random.default_rng(deterministic_seed)
        draw = float(rng.random())
        if draw < drop_probability:
            previous_mask[row].zero_()
            stats["dropped"] += 1
            continue
        if draw >= drop_probability + perturb_probability:
            stats["clean"] += 1
            continue

        changed = 0
        valid_columns = previous_mask[row].nonzero(as_tuple=False).flatten().tolist()
        for column in valid_columns:
            token_id = int(previous_ids[row, column])
            bin_index = position_id_to_bin.get(token_id)
            if bin_index is None:
                continue
            magnitude = int(rng.integers(1, max_bin_offset + 1))
            direction = -1 if int(rng.integers(0, 2)) == 0 else 1
            candidate = bin_index + direction * magnitude
            if candidate < 0 or candidate >= len(position_tokens):
                candidate = bin_index - direction * magnitude
            new_bin = int(np.clip(candidate, 0, len(position_tokens) - 1))
            if new_bin != bin_index:
                previous_ids[row, column] = int(position_tokens[new_bin])
                changed += 1
        stats["perturbed"] += 1
        stats["changed_tokens"] += changed

    result = dict(batch)
    result["previous_ids"] = previous_ids
    result["previous_mask"] = previous_mask
    return result, stats
