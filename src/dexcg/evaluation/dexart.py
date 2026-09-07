"""Multi-seed closed-loop DexArt evaluation."""

from __future__ import annotations

import json
import math
import random
import subprocess
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dexcg.envs import DexArtAdapter
from dexcg.training.objective import DexCGTrainingObjective


class _VideoWriter:
    def __init__(self, path: Path, shape: tuple[int, int], fps: int = 10) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = shape
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is closed")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode evaluation video")


def _tensor_observation(
    history: deque[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    items = list(history)
    while len(items) < 2:
        items.insert(0, items[0])
    return {
        key: torch.from_numpy(np.stack([item[key] for item in items[-2:]]))
        .unsqueeze(0)
        .to(device=device)
        for key in ("point_cloud", "object_point_mask", "imagin_robot", "agent_pos")
    }


def _frame(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    image = np.asarray(observation["instance_1-rgb"])
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _has_object_points(observation: Mapping[str, np.ndarray]) -> bool:
    return bool(np.asarray(observation["object_point_mask"]).any())


def _wilson(successes: int, episodes: int) -> tuple[float, float]:
    rate = successes / episodes
    z = 1.959963984540054
    denominator = 1.0 + z * z / episodes
    center = (rate + z * z / (2.0 * episodes)) / denominator
    margin = z * math.sqrt(
        (rate * (1.0 - rate) + z * z / (4.0 * episodes)) / episodes
    ) / denominator
    return center - margin, center + margin


@torch.no_grad()
def _evaluate_task(
    objective: DexCGTrainingObjective,
    task: str,
    task_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    epoch: int,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    instruction = str(task_config["instruction"])
    successes_by_seed: dict[str, int] = {}
    episodes_per_seed = int(evaluation_config["episodes_per_seed"])
    video_path = output_dir / "evaluation" / "videos" / f"{task}_epoch_{epoch:04d}.mp4"
    writer = None
    total_successes = 0
    invalid_observation_episodes = 0
    invalid_observation_frames = 0
    initial_observation_retries = int(
        evaluation_config.get("initial_observation_retries", 8)
    )
    if initial_observation_retries < 0:
        raise ValueError("initial_observation_retries must be non-negative")
    try:
        for seed_index, seed_value in enumerate(evaluation_config["seeds"]):
            seed = int(seed_value)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            adapter = DexArtAdapter.create(task, str(evaluation_config["split"]), 1.0e-2)
            adapter.environment.seed(seed)
            seed_successes = 0
            try:
                for episode_index in range(episodes_per_seed):
                    raw_observation = adapter.reset()
                    current_observation = DexArtAdapter.observation(raw_observation)
                    invalid_observation = False
                    for retry in range(initial_observation_retries + 1):
                        if _has_object_points(current_observation):
                            break
                        invalid_observation = True
                        invalid_observation_frames += 1
                        if retry == initial_observation_retries:
                            current_observation = None
                            break
                        raw_observation = adapter.observe()
                        current_observation = DexArtAdapter.observation(raw_observation)

                    if current_observation is None:
                        invalid_observation_episodes += 1
                        continue

                    history = deque([current_observation], maxlen=2)
                    record_video = seed_index == 0 and episode_index == 0
                    if record_video:
                        first_frame = _frame(raw_observation)
                        writer = _VideoWriter(video_path, first_frame.shape[:2])
                        writer.append(first_frame)
                    success = adapter.is_success
                    steps = 0
                    while steps < int(task_config["max_steps"]) and not success:
                        observation = _tensor_observation(history, device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            actions = objective.predict_action(
                                observation,
                                [instruction],
                                int(evaluation_config["num_inference_steps"]),
                                int(evaluation_config["action_steps"]),
                            )[0]
                        for action in actions.float().cpu().numpy():
                            raw_observation, _, done, _ = adapter.step(action)
                            next_observation = DexArtAdapter.observation(raw_observation)
                            if _has_object_points(next_observation):
                                history.append(next_observation)
                            else:
                                # Advance the environment, but keep the last valid model history.
                                invalid_observation = True
                                invalid_observation_frames += 1
                            if record_video:
                                writer.append(_frame(raw_observation))
                            steps += 1
                            success = success or adapter.is_success
                            if done or steps >= int(task_config["max_steps"]):
                                break
                        if done:
                            break
                    invalid_observation_episodes += int(invalid_observation)
                    seed_successes += int(success)
                    total_successes += int(success)
                    if record_video:
                        writer.close()
                        writer = None
            finally:
                adapter.close()
            successes_by_seed[str(seed)] = seed_successes
    finally:
        if writer is not None:
            writer.close()

    episodes = episodes_per_seed * len(evaluation_config["seeds"])
    low, high = _wilson(total_successes, episodes)
    seed_rates = [value / episodes_per_seed for value in successes_by_seed.values()]
    return {
        "successes": total_successes,
        "episodes": episodes,
        "success_rate": total_successes / episodes,
        "invalid_observation_episodes": invalid_observation_episodes,
        "invalid_observation_frames": invalid_observation_frames,
        "seed_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "ci95": [low, high],
        "successes_by_seed": successes_by_seed,
    }


def evaluate_seen_tasks(
    objective: DexCGTrainingObjective,
    config: Mapping[str, Any],
    epoch: int,
    output_dir: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    gather_group: torch.distributed.ProcessGroup | None = None,
) -> dict[str, Any] | None:
    import torch.distributed as dist

    evaluation_config = dict(config["evaluation"])
    evaluation_config["num_inference_steps"] = config["diffusion"]["num_inference_steps"]
    local_results = {}
    for task_index, (task, task_config) in enumerate(evaluation_config["tasks"].items()):
        if task_index % world_size == rank:
            local_results[task] = _evaluate_task(
                objective,
                task,
                task_config,
                evaluation_config,
                epoch,
                output_dir,
                device,
            )
    # SAPIEN may change the process-wide current CUDA device while evaluating.
    # Restore the rank-local device before returning to distributed training.
    torch.cuda.set_device(device)
    if torch.cuda.current_device() != device.index:
        raise RuntimeError(
            f"rank {rank}: current CUDA device is {torch.cuda.current_device()}, "
            f"expected {device.index}"
        )

    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_results, group=gather_group)
    if rank != 0:
        return None
    tasks = {task: result for shard in gathered for task, result in shard.items()}
    result = {
        "epoch": epoch,
        "split": evaluation_config["split"],
        "tasks": tasks,
        "mean_success_rate": float(np.mean([item["success_rate"] for item in tasks.values()])),
    }
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result_path = evaluation_dir / f"success_rates_epoch_{epoch:04d}.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "evaluation.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    return result
