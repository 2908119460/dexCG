"""Task-balanced sequence sampling from collected DexArt trajectories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from dexcg.data.dexart import TARGET_TOKEN_IDS, TARGET_TOKEN_MASK
from dexcg.models.contact.coordinates import CONTACT_COORDINATE_CONTRACT


class DexArtTrainingDataset(Dataset):
    """Sample observation/action windows without crossing episode boundaries."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        obs_horizon: int = 2,
        action_horizon: int = 16,
    ) -> None:
        if not paths:
            raise ValueError("At least one dataset path is required")
        self.paths = tuple(Path(path).resolve() for path in paths)
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self._roots: dict[tuple[int, int], Any] = {}
        self.tasks: list[str] = []
        self.samples: list[list[tuple[int, int, int, int]]] = []
        state_min = None
        state_max = None

        for task_index, path in enumerate(self.paths):
            root = zarr.open_group(str(path), mode="r")
            if root.attrs.get("split") != "seen":
                raise ValueError(f"Training dataset must be seen split: {path}")
            contract = root.attrs.get("contact_coordinate_contract")
            if contract != CONTACT_COORDINATE_CONTRACT:
                raise ValueError(
                    f"Training dataset {path} uses contact coordinate contract {contract!r}; "
                    f"expected {CONTACT_COORDINATE_CONTRACT!r}. Migrate it before training."
                )
            task = str(root.attrs["task"])
            episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
            starts = np.concatenate((np.zeros(1, dtype=np.int64), episode_ends[:-1]))
            center_valid = np.asarray(root["data/object_center_valid"][:], dtype=np.bool_)
            task_samples: list[tuple[int, int, int, int]] = []
            for episode_index, (start, end) in enumerate(zip(starts, episode_ends, strict=True)):
                task_samples.extend(
                    (task_index, episode_index, int(start), step)
                    for step in range(int(start), int(end))
                    if center_valid[step]
                )
            self.tasks.append(task)
            self.samples.append(task_samples)

            state = np.asarray(root["data/agent_pos"][:], dtype=np.float32)
            current_min = state.min(axis=0)
            current_max = state.max(axis=0)
            state_min = current_min if state_min is None else np.minimum(state_min, current_min)
            state_max = current_max if state_max is None else np.maximum(state_max, current_max)

        self.samples_per_task = max(len(samples) for samples in self.samples)
        self.state_min = torch.from_numpy(state_min)
        self.state_max = torch.from_numpy(state_max)

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_task

    def _root(self, task_index: int):
        key = (os.getpid(), task_index)
        if key not in self._roots:
            self._roots[key] = zarr.open_group(str(self.paths[task_index]), mode="r")
        return self._roots[key]

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index = index % len(self.paths)
        sample_index = (index // len(self.paths)) % len(self.samples[task_index])
        _, episode_index, episode_start, step = self.samples[task_index][sample_index]
        root = self._root(task_index)
        episode_end = int(root["meta/episode_ends"][episode_index])

        observation_indices = np.clip(
            np.arange(step - self.obs_horizon + 1, step + 1),
            episode_start,
            episode_end - 1,
        )
        action_indices = np.clip(
            np.arange(step, step + self.action_horizon),
            episode_start,
            episode_end - 1,
        )
        observation = {
            name: torch.from_numpy(np.asarray(root[f"data/{name}"][observation_indices]))
            for name in ("point_cloud", "object_point_mask", "imagin_robot", "agent_pos")
        }
        return {
            "observation": observation,
            "action": torch.from_numpy(
                np.asarray(root["data/action"][action_indices], dtype=np.float32)
            ),
            "contact_token_ids": torch.from_numpy(
                np.asarray(root[f"data/{TARGET_TOKEN_IDS}"][step], dtype=np.int64)
            ),
            "contact_token_mask": torch.from_numpy(
                np.asarray(root[f"data/{TARGET_TOKEN_MASK}"][step], dtype=np.bool_)
            ),
            "language": str(root["meta/low_level_grasp_instruction"][episode_index]),
        }
