"""Multi-seed closed-loop DexArt evaluation."""

from __future__ import annotations

import json
import math
import random
import subprocess
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from dexcg.data.robot_base import InvalidObjectObservation, RobotBaseCollectionAdapter
from dexcg.models.contact.coordinates import object_aabb_center_numpy
from dexcg.robots.geometry import robot_geometry
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
        for key in (
            "point_cloud",
            "object_point_mask",
            "object_center",
            "imagin_robot",
            "agent_pos",
            "palm_pose_robot_base",
            "planner_point_cloud",
            "planner_object_point_mask",
        )
        if key in items[-1]
    }


def _frame(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    image = np.asarray(observation["instance_1-rgb"])
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _split_object_observation(sample, planner_point_count, rng):
    """Split one validated metric capture; never rescale XYZ or synthesize points."""
    if planner_point_count not in (1024, 10000):
        raise ValueError("planner_point_count must explicitly be 1024 or 10000")
    points = np.asarray(sample["point_cloud"])
    mask = np.asarray(sample["object_point_mask"])
    if (points.shape != (10000, 3) or mask.shape != (10000,)
            or not (mask == 1).all() or not np.isfinite(points).all()
            or len(np.unique(points, axis=0)) != 10000):
        raise InvalidObjectObservation("Expected 10000 distinct finite object XYZ points")
    if not np.isfinite(sample["object_center"]).all():
        raise InvalidObjectObservation("Nonfinite object center")
    indices = rng.choice(10000, 1024, replace=False)
    policy_points = points[indices].copy()
    result = {key: sample[key] for key in (
        "object_center", "agent_pos", "palm_pose_robot_base"
    )}
    result.update(
        point_cloud=policy_points,
        object_point_mask=np.ones(1024, dtype=bool),
        planner_point_cloud=(points if planner_point_count == 10000 else policy_points).copy(),
        planner_object_point_mask=np.ones(planner_point_count, dtype=bool),
        imagin_robot=robot_geometry(np.asarray(sample["agent_pos"])[..., :22]),
    )
    return result


@contextmanager
def _point_input_guards(objective, planner_point_count, counts):
    """Check and count actual calls at PartField and DP3, not just the config."""
    def planner_guard(module, args):
        points = args[0]
        if (points.ndim != 3 or points.shape[1:] != (planner_point_count, 3)
                or not torch.isfinite(points).all()):
            raise ValueError("PartField received incorrect point count or invalid XYZ")
        counts["partfield_calls"] += 1
        counts["partfield_actual_point_count"] = int(points.shape[1])

    def policy_guard(module, args):
        observation = args[0]
        points, robot = observation["point_cloud"], observation["imagin_robot"]
        if (points.ndim != 3 or points.shape[1:] != (1024, 3)
                or robot.ndim != 3 or robot.shape[1:] != (96, 7)
                or not torch.isfinite(points).all() or not torch.isfinite(robot).all()):
            raise ValueError("DP3 requires 1024 object points and 96 robot geometry points")
        counts["dp3_calls"] += 1
        counts["dp3_actual_object_point_count"] = int(points.shape[1])
        counts["dp3_actual_robot_point_count"] = int(robot.shape[1])

    handles = []
    try:
        handles.append(objective.model.contact_planner.point_encoder.register_forward_pre_hook(planner_guard))
        handles.append(objective.model.observation_encoder.encoder.register_forward_pre_hook(policy_guard))
        yield
    finally:
        for handle in handles:
            handle.remove()


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
    # Required explicitly: an old config must not silently feed a 10000-point
    # planner the legacy 1024-point observation.
    planner_point_count = int(evaluation_config["planner_point_count"])
    if planner_point_count not in (1024, 10000):
        raise ValueError("planner_point_count must be 1024 or 10000")
    if int(evaluation_config.get("smp_point_count", 1024)) != 1024:
        raise ValueError("This SMP evaluation requires 1024 object points")
    resolution = list(evaluation_config.get("point_capture_resolution", [1024, 1024]))
    if resolution != [1024, 1024]:
        raise ValueError("Evaluation camera must match the 1024 x 1024 collection sensor")
    episodes_per_seed = int(evaluation_config["episodes_per_seed"])
    seeds = [int(seed) for seed in evaluation_config["seeds"]]
    if episodes_per_seed < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Positive episode count and nonempty unique seeds required")
    initial_observation_retries = int(evaluation_config.get("initial_observation_retries", 8))
    if initial_observation_retries < 0:
        raise ValueError("initial_observation_retries must be non-negative")
    if int(evaluation_config["action_steps"]) < 1 or int(task_config["max_steps"]) < 1:
        raise ValueError("action_steps and max_steps must be positive")

    instruction = str(task_config["instruction"])
    video_path = output_dir / "evaluation" / "videos" / f"{task}_vlm{planner_point_count}_epoch_{epoch:04d}.mp4"
    writer = None
    successes_by_seed = {}
    total_successes = 0
    invalid_observation_episodes = 0
    invalid_observation_frames = 0
    invalid_reasons = {}
    center_deltas_mm = []
    input_counts = {"partfield_calls": 0, "dp3_calls": 0,
                    "partfield_actual_point_count": None,
                    "dp3_actual_object_point_count": None,
                    "dp3_actual_robot_point_count": None}
    episode_results = []

    def read_observation(adapter, raw, subset_rng):
        nonlocal invalid_observation_frames
        try:
            # The collection adapter checks actor IDs, distinct pixels/XYZ,
            # finite depth, and depth-to-XYZ round trips on every capture.
            sample = adapter.observation(raw)
            return _split_object_observation(sample, planner_point_count, subset_rng)
        except InvalidObjectObservation as error:
            invalid_observation_frames += 1
            reason = str(error).split(":", 1)[0]
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            return None

    try:
        with _point_input_guards(objective, planner_point_count, input_counts):
            for seed_index, seed in enumerate(seeds):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                adapter = RobotBaseCollectionAdapter.create(task, str(evaluation_config["split"]), 1.0e-2)
                seed_successes = 0
                try:
                    adapter.point_count = 10000
                    adapter.enable_high_resolution_object_cloud(resolution)
                    for episode_index in range(episodes_per_seed):
                        # Reset streams per episode: different rollout lengths must
                        # not change the next episode's initial state in paired runs.
                        episode_seed = int(np.random.SeedSequence([seed, episode_index]).generate_state(1)[0])
                        random.seed(episode_seed)
                        np.random.seed(episode_seed)
                        torch.manual_seed(episode_seed)
                        adapter.environment.seed(episode_seed)
                        adapter.point_rng = np.random.RandomState(episode_seed)
                        subset_rng = np.random.default_rng(episode_seed)
                        invalid_before = invalid_observation_frames
                        raw_observation = adapter.reset()
                        current_observation = read_observation(adapter, raw_observation, subset_rng)
                        for _ in range(initial_observation_retries):
                            if current_observation is not None:
                                break
                            raw_observation = adapter.observe()
                            current_observation = read_observation(adapter, raw_observation, subset_rng)

                        success = False
                        steps = 0
                        failure_reason = None
                        if current_observation is None:
                            failure_reason = "invalid_initial_observation"
                        else:
                            history = deque([current_observation], maxlen=2)
                            record_video = bool(evaluation_config.get("record_video", True)) and seed_index == 0 and episode_index == 0
                            if record_video:
                                first_frame = _frame(raw_observation)
                                writer = _VideoWriter(video_path, first_frame.shape[:2])
                                writer.append(first_frame)
                            success = adapter.is_success
                            previous_contact_plan = None
                            stop = False
                            while steps < int(task_config["max_steps"]) and not success and not stop:
                                current = history[-1]
                                sampled_center = object_aabb_center_numpy(current["point_cloud"], current["object_point_mask"])
                                center_deltas_mm.append(1000.0 * float(np.linalg.norm(sampled_center - current["object_center"])))
                                observation = _tensor_observation(history, device)
                                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                                    prediction = objective.predict_action_with_diagnostics(
                                        observation, [instruction],
                                        int(evaluation_config["num_inference_steps"]),
                                        int(evaluation_config["action_steps"]),
                                        previous_contact_plan=previous_contact_plan,
                                    )
                                actions = prediction.actions[0]
                                previous_contact_plan = prediction.contact_plan
                                if len(actions) == 0 or not torch.isfinite(actions).all():
                                    raise ValueError("Policy produced empty or nonfinite actions")
                                for action in actions.float().cpu().numpy():
                                    raw_observation, _, done, _ = adapter.step(action)
                                    steps += 1
                                    if record_video:
                                        writer.append(_frame(raw_observation))
                                    success = bool(adapter.is_success)
                                    stop = bool(done) or steps >= int(task_config["max_steps"])
                                    # No next policy input is needed after terminal success/done.
                                    if success or stop:
                                        break
                                    next_observation = read_observation(adapter, raw_observation, subset_rng)
                                    if next_observation is None:
                                        # Never feed stale history or duplicate points. Preserve
                                        # this episode in the denominator and report sensor failure.
                                        failure_reason = "invalid_rollout_observation"
                                        stop = True
                                        break
                                    history.append(next_observation)
                        invalid = invalid_observation_frames > invalid_before
                        invalid_observation_episodes += int(invalid)
                        seed_successes += int(success)
                        total_successes += int(success)
                        episode_results.append({"seed": seed, "episode": episode_index,
                            "episode_seed": episode_seed, "success": bool(success),
                            "steps": steps, "failure_reason": failure_reason,
                            "invalid_observation_frames": invalid_observation_frames - invalid_before})
                        if writer is not None:
                            writer.close()
                            writer = None
                finally:
                    adapter.close()
                successes_by_seed[str(seed)] = seed_successes
    finally:
        if writer is not None:
            writer.close()

    episodes = episodes_per_seed * len(seeds)
    low, high = _wilson(total_successes, episodes)
    seed_rates = [value / episodes_per_seed for value in successes_by_seed.values()]
    center_deltas = np.asarray(center_deltas_mm, dtype=np.float64)
    return {
        "successes": total_successes, "episodes": episodes,
        "success_rate": total_successes / episodes,
        "invalid_observation_episodes": invalid_observation_episodes,
        "invalid_observation_frames": invalid_observation_frames,
        "invalid_observation_reasons": invalid_reasons,
        "episode_results": episode_results,
        "point_input_contract": {
            "sensor_contract": "aligned_rgbd_object_10000_v2",
            "coordinate_frame": "robot_base", "length_unit": "metre",
            "capture_resolution": resolution, "captured_object_points": 10000,
            "planner_point_count": planner_point_count, "smp_object_point_count": 1024,
            "sampling": "without_replacement", "invalid_rollout_policy": "count_as_failure",
            **input_counts,
        },
        "seed_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "ci95": [low, high], "successes_by_seed": successes_by_seed,
        "full_resolution_vs_sampled_center_delta_mm": {
            "samples": int(center_deltas.size),
            "mean": float(center_deltas.mean()) if center_deltas.size else None,
            "median": float(np.median(center_deltas)) if center_deltas.size else None,
            "p95": float(np.percentile(center_deltas, 95)) if center_deltas.size else None,
            "max": float(center_deltas.max()) if center_deltas.size else None,
        },
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
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type == "cuda" and torch.cuda.current_device() != device.index:
        raise RuntimeError(
            f"rank {rank}: current CUDA device is {torch.cuda.current_device()}, "
            f"expected {device.index}"
        )

    gathered: list[dict[str, Any] | None] = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(gathered, local_results, group=gather_group)
    else:
        gathered[0] = local_results
    if rank != 0:
        return None
    tasks = {task: result for shard in gathered for task, result in shard.items()}
    result = {
        "epoch": epoch,
        "split": evaluation_config["split"],
        "planner_point_count": evaluation_config.get("planner_point_count"),
        "planner_checkpoint": evaluation_config.get("planner_checkpoint"),
        "planner_checkpoint_sha256": evaluation_config.get("planner_checkpoint_sha256"),
        "tasks": tasks,
        "mean_success_rate": float(np.mean([item["success_rate"] for item in tasks.values()])),
    }
    if not evaluation_config.get("write_results", True):
        return result
    evaluation_dir = output_dir / "evaluation" / "success_rates"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    point_label = evaluation_config.get("planner_point_count", "unspecified")
    result_path = evaluation_dir / f"success_rates_epoch_{epoch:04d}_vlm{point_label}.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "evaluation.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True) + "\n")
    return result
