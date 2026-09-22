"""Task-balanced sequence sampling from collected DexArt trajectories."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from dexcg.data.dexart import TARGET_TOKEN_IDS, TARGET_TOKEN_MASK
from dexcg.models.contact.coordinates import require_dataset_coordinates


class DexArtTrainingDataset(Dataset):
    """Sample observation/action windows without crossing episode boundaries."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        obs_horizon: int = 2,
        action_horizon: int = 16,
        split_manifest: str | Path | None = None,
        split: str = "train",
        state_statistics: tuple[torch.Tensor, torch.Tensor] | None = None,
        point_count: int | None = None,
        preload: bool = False,
    ) -> None:
        if not paths:
            raise ValueError("At least one dataset path is required")
        self.paths = tuple(Path(path).resolve() for path in paths)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.point_count = point_count
        self.preload = preload
        self._memory = {}
        self.split = split
        manifest = json.loads(Path(split_manifest).read_text()) if split_manifest else None
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if split == "validation" and (manifest is None or state_statistics is None):
            raise ValueError("Validation requires a fixed split and training-only statistics")
        if self.obs_horizon < 1 or self.action_horizon < 1:
            raise ValueError("observation and action horizons must be positive")
        self._roots: dict[tuple[int, int], Any] = {}
        self.tasks: list[str] = []
        self.samples: list[list[tuple[int, int, int, int, int]]] = []
        self.effective_episode_ends: list[np.ndarray] = []
        state_min = None
        state_max = None

        for task_index, path in enumerate(self.paths):
            root = zarr.open_group(str(path), mode="r")
            if root.attrs.get("split") != "seen":
                raise ValueError(f"Training dataset must be seen split: {path}")
            require_dataset_coordinates(root.attrs)
            task = str(root.attrs["task"])
            episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
            starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
            selected = set(range(len(episode_ends)))
            if manifest is not None:
                object_ids = np.asarray(root["meta/object_id"][:]).astype(str)
                partitions = {}
                for name in ("train", "validation"):
                    indices = []
                    for object_id, episodes in manifest[name]["episodes"][task].items():
                        for ep in episodes:
                            if ep < 0 or ep >= len(episode_ends) or object_ids[ep] != str(object_id):
                                raise ValueError(f"Split object/episode mismatch: {task}:{ep}")
                        indices.extend(episodes)
                    if len(indices) != len(set(indices)):
                        raise ValueError("Duplicate trajectory in split manifest")
                    partitions[name] = set(indices)
                if (partitions["train"] & partitions["validation"]
                        or partitions["train"] | partitions["validation"] != selected):
                    raise ValueError("Split must cover every trajectory exactly once")
                selected = partitions[split]
            center_valid = np.asarray(root["data/object_center_valid"][:], dtype=np.bool_)
            # The token mask includes the structural START/END tokens. Use the
            # link mask to identify an actually nonempty contact plan.
            contact_valid = np.asarray(root["data/contact_target_mask"][:], dtype=np.bool_).any(axis=1)
            task_samples: list[tuple[int, int, int, int, int]] = []
            effective_ends = episode_ends.copy()
            for episode_index, (start, end) in enumerate(zip(starts, episode_ends, strict=True)):
                if episode_index not in selected:
                    continue
                start, end = int(start), int(end)
                nonempty = np.flatnonzero(contact_valid[start:end]) + start
                if not len(nonempty):
                    raise ValueError(f"Episode {task}:{episode_index} has no contact target")
                effective_ends[episode_index] = int(nonempty[-1]) + 1
                for step in range(start, int(effective_ends[episode_index])):
                    if not center_valid[step]:
                        continue
                    contact_step = int(nonempty[np.searchsorted(nonempty, step)])
                    task_samples.append((task_index, episode_index, start, step, contact_step))
            self.tasks.append(task)
            self.samples.append(task_samples)
            self.effective_episode_ends.append(effective_ends)
            if not task_samples:
                raise ValueError(f"No usable training samples: {path}")

            state = np.asarray(root["data/agent_pos"][:], dtype=np.float32)
            state = state[[sample[3] for sample in task_samples]]
            current_min = state.min(axis=0)
            current_max = state.max(axis=0)
            state_min = current_min if state_min is None else np.minimum(state_min, current_min)
            state_max = current_max if state_max is None else np.maximum(state_max, current_max)

        self.samples_per_task = max(len(samples) for samples in self.samples)
        self.state_min = torch.from_numpy(state_min)
        self.state_max = torch.from_numpy(state_max)
        if state_statistics is not None:
            self.state_min, self.state_max = (value.clone() for value in state_statistics)
        self._flat_samples = [(task, index) for task, samples in enumerate(self.samples)
                              for index in range(len(samples))]
        if preload:
            # RAM only; keep all 10000 points so each training access can sample
            # a fresh subset. No derived datasets or disk caches are written.
            from dexcg.robots.geometry import robot_geometry
            for task_index in range(len(self.paths)):
                root = self._root(task_index)
                keys = ("point_cloud", "object_point_mask", "object_center", "agent_pos",
                        "palm_pose_robot_base", "action", TARGET_TOKEN_IDS, TARGET_TOKEN_MASK)
                memory = {f"data/{key}": np.asarray(root[f"data/{key}"][:])
                          for key in keys if f"data/{key}" in root}
                memory["meta/low_level_grasp_instruction"] = np.asarray(root["meta/low_level_grasp_instruction"][:])
                states = memory["data/agent_pos"]
                memory["data/imagin_robot"] = np.concatenate([
                    robot_geometry(states[start:start+512, :22])
                    for start in range(0, len(states), 512)])
                self._memory[task_index] = memory

    def __len__(self) -> int:
        if self.split == "validation":
            return len(self._flat_samples)
        return len(self.paths) * self.samples_per_task

    def _root(self, task_index: int):
        if task_index in self._memory:
            return self._memory[task_index]
        key = (os.getpid(), task_index)
        if key not in self._roots:
            self._roots[key] = zarr.open_group(str(self.paths[task_index]), mode="r")
        return self._roots[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = index % len(self.paths)
        sample_index = (index // len(self.paths)) % len(self.samples[task_index])
        if self.split == "validation":
            task_index, sample_index = self._flat_samples[index]
        _, episode_index, episode_start, step, contact_step = self.samples[task_index][sample_index]
        root = self._root(task_index)
        # Excluded terminal frames must not re-enter supervision through an
        # earlier sample's future action window.
        episode_end = int(self.effective_episode_ends[task_index][episode_index])

        observation_indices = np.clip(
            np.arange(step - self.obs_horizon + 1, step + 1),
            episode_start,
            episode_end - 1,
        )
        requested_actions = np.arange(step, step + self.action_horizon)
        action_valid_mask = requested_actions < episode_end
        real_actions = np.asarray(
            root["data/action"][step : min(step + self.action_horizon, episode_end)],
            dtype=np.float32,
        )
        actions = np.zeros((self.action_horizon, *real_actions.shape[1:]), dtype=np.float32)
        actions[:len(real_actions)] = real_actions
        observation = {
            name: torch.from_numpy(np.asarray(root[f"data/{name}"][observation_indices]))
            for name in (
                "point_cloud",
                "object_point_mask",
                "object_center",
                "agent_pos",
            )
        }
        from dexcg.robots.geometry import robot_geometry
        if self.point_count is not None:
            points, mask = observation["point_cloud"], observation["object_point_mask"]
            if points.shape[-2] < self.point_count or not mask.bool().all():
                raise ValueError("SMP requires enough pure object points for sampling without replacement")
            rng = np.random if self.split == "train" else np.random.RandomState(index)
            indices = rng.choice(points.shape[-2], self.point_count, replace=False)
            observation["point_cloud"] = points[:, indices]
            observation["object_point_mask"] = mask[:, indices]

        if "data/palm_pose_robot_base" in root:
            observation["palm_pose_robot_base"] = torch.from_numpy(
                np.asarray(root["data/palm_pose_robot_base"][observation_indices])
            )

        geometry = (root["data/imagin_robot"][observation_indices] if self.preload else
                    robot_geometry(observation["agent_pos"].numpy()[..., :22]))
        observation["imagin_robot"] = torch.from_numpy(geometry)
        return {
            "observation": observation,
            "action": torch.from_numpy(actions),
            "action_valid_mask": torch.from_numpy(action_valid_mask.astype(np.bool_)),
            "contact_token_ids": torch.from_numpy(
                np.asarray(root[f"data/{TARGET_TOKEN_IDS}"][contact_step], dtype=np.int64)
            ),
            "contact_token_mask": torch.from_numpy(
                np.asarray(root[f"data/{TARGET_TOKEN_MASK}"][contact_step], dtype=np.bool_)
            ),
            "language": str(root["meta/low_level_grasp_instruction"][episode_index]),
        }
