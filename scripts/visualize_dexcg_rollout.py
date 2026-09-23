#!/usr/bin/env python3
"""Render offline contact predictions or online DexCG policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
import zarr

from dexcg.common.config import load_config
from dexcg.envs import DexArtAdapter
from dexcg.models import DexCG
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    OBJECT_CENTER_DEFINITION,
    require_dataset_coordinates,
    require_model_coordinates,
)
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS
from dexcg.robots.geometry import robot_geometry
from dexcg.training import DexCGTrainingObjective
from dexcg.visualization import (
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    compute_projection_bounds,
    render_planner_frame,
    render_rollout_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "outputs/qwen-frozen/config.yaml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs/qwen-frozen/checkpoints/epoch=2600-seen_score=0.343.ckpt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data_visualize/dexcg/bucket_epoch2600"
OUTPUT_NAMES = ("rollout.npz", "metadata.json", "visualization.mp4")
OFFLINE_VARIANTS = {
    "qwen-1024": (
        "outputs/qwen-finetuned-balanced-20ep/checkpoints/"
        "epoch=0007-generation_score=0.498799217807.ckpt", 1024,
    ),
    "qwen-10000": (
        "outputs/qwen-partfield-frozen-10000/checkpoints/"
        "epoch=0012-generation_score=0.489804077136.ckpt", 10000,
    ),
}
TASKS = ("faucet", "bucket", "laptop", "toilet")
ONLINE_CHECKPOINTS = {
    "qwen-1024": "epoch=0900-new_10000_seen_score=0.265-old_1024_seen_score=0.350.ckpt",
    "qwen-10000": "epoch=1600-new_10000_seen_score=0.302-old_1024_seen_score=0.320.ckpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", default="bucket", choices=TASKS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-vlm", choices=tuple(OFFLINE_VARIANTS))
    mode.add_argument("--online-vlm", choices=tuple(ONLINE_CHECKPOINTS))
    parser.add_argument("--split", default="seen")
    parser.add_argument("--seed", type=int, default=1003)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_state_names(module: torch.nn.Module) -> set[str]:
    parameter_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    buffer_names = {
        name for name, _ in module.named_buffers() if not name.startswith("model.contact_planner.")
    }
    return parameter_names | buffer_names


def load_objective(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[DexCGTrainingObjective, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require_model_coordinates(checkpoint)
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    state = checkpoint["model"]
    for name in ("state_min", "state_max"):
        if name not in state:
            raise ValueError(f"Checkpoint model state has no {name}")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = DexCG.from_config(
        load_config(resolve(Path(config["model_config"]))),
        PROJECT_ROOT,
        torch_dtype=torch.bfloat16,
    )
    objective = DexCGTrainingObjective(
        model,
        state["state_min"],
        state["state_max"],
        config["diffusion"],
        config["loss"],
        train_contact_planner=bool(config["contact_training"]["enabled"]),
    )
    expected = _compact_state_names(objective)
    allowed = {
        name for name in objective.state_dict() if not name.startswith("model.contact_planner.")
    }
    missing = expected.difference(state)
    unexpected = set(state).difference(allowed)
    if missing or unexpected:
        raise RuntimeError(
            f"invalid compact checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    objective.load_state_dict(state, strict=False)
    return objective.to(device).eval(), checkpoint


def tensor_observation(
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
        )
    }


def has_object_points(observation: dict[str, np.ndarray]) -> bool:
    return bool(
        np.asarray(observation["object_point_mask"]).any()
        and np.isfinite(np.asarray(observation["object_center"])).all()
    )


def empty_trace() -> dict[str, list[np.ndarray] | list[int]]:
    return {
        "img": [],
        "point_cloud": [],
        "imagin_robot": [],
        "raw_contact_points": [],
        "raw_contact_mask": [],
        "actions": [],
        "inference_index": [],
        "basis": [],
        "contact_token_ids": [],
        "contact_token_mask": [],
        "object_center": [],
        "model_point_cloud": [],
        "model_object_point_mask": [],
        "model_imagin_robot": [],
        "model_agent_pos": [],
        "predicted_contact_points": [],
        "predicted_contact_mask": [],
        "predicted_contact_link_indices": [],
    }


def decode_predicted_contacts(
    planner,
    contact_plan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decoded = planner.decode_contacts(contact_plan)[0]
    points = np.zeros((len(ALLEGRO_CONTACT_LINKS), 3), dtype=np.float32)
    mask = np.zeros(len(ALLEGRO_CONTACT_LINKS), dtype=np.bool_)
    link_indices = np.full(len(ALLEGRO_CONTACT_LINKS), -1, dtype=np.int16)
    cursor = 0
    name_to_index = {item.token_name: index for index, item in enumerate(ALLEGRO_CONTACT_LINKS)}
    for name, positions in decoded.items():
        if name not in name_to_index:
            raise ValueError(f"Unknown predicted contact link: {name}")
        for position in positions:
            if cursor >= len(ALLEGRO_CONTACT_LINKS):
                raise ValueError("Predicted plan exceeds the contact grammar limit")
            points[cursor] = np.asarray(position, dtype=np.float32)
            mask[cursor] = True
            link_indices[cursor] = name_to_index[name]
            cursor += 1
    return points, mask, link_indices


def append_frame(
    trace: dict[str, list],
    adapter: DexArtAdapter,
    raw_observation,
    inference_index: int,
) -> None:
    observation = adapter.observation(raw_observation)
    contact = adapter.contact_graph()
    trace["img"].append(observation["img"])
    trace["point_cloud"].append(observation["point_cloud"])
    trace["imagin_robot"].append(observation["imagin_robot"])
    trace["raw_contact_points"].append(contact.points)
    trace["raw_contact_mask"].append(contact.mask)
    trace["inference_index"].append(inference_index)


def stack_trace(trace: dict[str, list], joint_end_id: int) -> dict[str, np.ndarray]:
    token_rows = trace.pop("contact_token_ids")
    token_masks = trace.pop("contact_token_mask")
    max_tokens = max(len(row) for row in token_rows)
    token_ids = np.full((len(token_rows), max_tokens), joint_end_id, dtype=np.int64)
    token_mask = np.zeros((len(token_rows), max_tokens), dtype=np.bool_)
    for index, (row, mask) in enumerate(zip(token_rows, token_masks, strict=True)):
        token_ids[index, : len(row)] = row
        token_mask[index, : len(mask)] = mask

    arrays = {name: np.stack(values) for name, values in trace.items()}
    arrays["contact_token_ids"] = token_ids
    arrays["contact_token_mask"] = token_mask
    arrays["inference_index"] = arrays["inference_index"].astype(np.int16)
    return arrays


@torch.no_grad()
def collect_successful_rollout(
    objective: DexCGTrainingObjective,
    task: str,
    split: str,
    instruction: str,
    max_steps: int,
    action_steps: int,
    num_inference_steps: int,
    seed: int,
    max_attempts: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    adapter = DexArtAdapter.create(task, split, 1.0e-2)
    adapter.environment.seed(seed)
    try:
        for attempt in range(max_attempts):
            raw_observation = adapter.reset()
            current = adapter.observation(raw_observation)
            for _ in range(9):
                if has_object_points(current):
                    break
                raw_observation = adapter.observe()
                current = adapter.observation(raw_observation)
            if not has_object_points(current):
                print(f"attempt {attempt + 1}: no object points", flush=True)
                continue

            trace = empty_trace()
            history = deque([current], maxlen=2)
            stable_frame = 0 if adapter.is_stable_contact else -1
            success = adapter.is_success
            steps = 0
            done = False
            while steps < max_steps and not success and not done:
                observation = tensor_observation(history, device)
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                else:
                    autocast = nullcontext()
                with autocast:
                    prediction = objective.predict_action_with_diagnostics(
                        observation,
                        [instruction],
                        num_inference_steps,
                        action_steps,
                    )

                inference_index = len(trace["basis"])
                trace["model_point_cloud"].append(
                    observation["point_cloud"][0].float().cpu().numpy()
                )
                trace["model_object_point_mask"].append(
                    observation["object_point_mask"][0].bool().cpu().numpy()
                )
                trace["model_imagin_robot"].append(
                    observation["imagin_robot"][0].float().cpu().numpy()
                )
                trace["model_agent_pos"].append(observation["agent_pos"][0].float().cpu().numpy())
                basis = prediction.basis[0].float().cpu().numpy()
                gram = basis.T @ basis
                if basis.shape != (22, 4) or not np.isfinite(basis).all():
                    raise RuntimeError(
                        f"Invalid basis at inference {inference_index}: {basis.shape}"
                    )
                if not np.allclose(gram, np.eye(4), atol=1.0e-2):
                    raise RuntimeError(
                        f"Non-orthogonal basis at inference {inference_index}: "
                        f"error={np.linalg.norm(gram - np.eye(4)):.6g}"
                    )
                predicted_points, predicted_mask, link_indices = decode_predicted_contacts(
                    objective.model.contact_planner, prediction.contact_plan
                )
                trace["basis"].append(basis)
                trace["contact_token_ids"].append(
                    prediction.contact_plan.token_ids[0].long().cpu().numpy()
                )
                trace["contact_token_mask"].append(
                    prediction.contact_plan.attention_mask[0].bool().cpu().numpy()
                )
                trace["object_center"].append(
                    observation["object_center"][0, -1].float().cpu().numpy()
                )
                trace["predicted_contact_points"].append(predicted_points)
                trace["predicted_contact_mask"].append(predicted_mask)
                trace["predicted_contact_link_indices"].append(link_indices)

                if trace["img"]:
                    trace["inference_index"][-1] = inference_index
                else:
                    append_frame(trace, adapter, raw_observation, inference_index)

                for action in prediction.actions[0].float().cpu().numpy():
                    trace["actions"].append(np.asarray(action, dtype=np.float32))
                    raw_observation, _, done, _ = adapter.step(action)
                    current = adapter.observation(raw_observation)
                    if has_object_points(current):
                        history.append(current)
                    append_frame(trace, adapter, raw_observation, inference_index)
                    steps += 1
                    if stable_frame < 0 and adapter.is_stable_contact:
                        stable_frame = len(trace["img"]) - 1
                    success = success or adapter.is_success
                    if done or success or steps >= max_steps:
                        break

            print(
                f"attempt {attempt + 1}: object={adapter.object_id} steps={steps} "
                f"success={success} stable_frame={stable_frame}",
                flush=True,
            )
            if success and stable_frame >= 0 and trace["basis"]:
                tokenizer = objective.model.contact_planner.contact_tokenizer
                arrays = stack_trace(trace, tokenizer.joint_end_id)
                return arrays, {
                    "attempt": attempt + 1,
                    "object_id": adapter.object_id,
                    "stable_frame": stable_frame,
                    "steps": steps,
                }
    finally:
        adapter.close()
        if device.type == "cuda":
            torch.cuda.set_device(device)
    raise RuntimeError(
        f"No successful {task} rollout with stable contact in {max_attempts} attempts"
    )


def open_encoder(path: Path, crf: int, metadata: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            *(["-metadata", "comment=" + json.dumps(metadata, sort_keys=True)]
              if metadata is not None else []),
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def write_video(
    path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    crf: int,
) -> None:
    bounds = compute_projection_bounds(arrays["point_cloud"], arrays["imagin_robot"])
    encoder = open_encoder(path, crf)
    try:
        assert encoder.stdin is not None
        for frame_index in range(len(arrays["img"])):
            frame = render_rollout_frame(arrays, metadata, frame_index, bounds)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}")


def write_outputs(
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    crf: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.tmp"
    temporary_npz = output_dir / f"rollout{suffix}.npz"
    temporary_json = output_dir / f"metadata{suffix}.json"
    temporary_video = output_dir / f"visualization{suffix}.mp4"
    temporary_paths = (temporary_npz, temporary_json, temporary_video)
    try:
        with temporary_npz.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        write_video(temporary_video, arrays, metadata, crf)
        temporary_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_npz.replace(output_dir / "rollout.npz")
        temporary_json.replace(output_dir / "metadata.json")
        temporary_video.replace(output_dir / "visualization.mp4")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def offline_episodes(variant: str) -> dict[str, tuple[zarr.Group, int, int, int]]:
    episodes = {}
    for task in TASKS:
        root = zarr.open_group(
            str(PROJECT_ROOT / f"data/dexart-object-balanced-10000-point/{task}_expert.zarr"),
            mode="r",
        )
        require_dataset_coordinates(root.attrs)
        if root.attrs.get("complete") is not True or root.attrs.get("task") != task:
            raise ValueError(f"Incomplete or mismatched {task} demonstration data")
        successes = np.asarray(root["meta/simulator_success"][:], dtype=bool)
        successful = np.flatnonzero(successes)
        if not len(successful):
            raise ValueError(f"No successful episode for {task}")
        ends = root["meta/episode_ends"]
        lengths = np.diff(np.r_[0, np.asarray(ends[:], dtype=np.int64)])
        episode_index = int(successful[np.argmax(lengths[successful])])
        start = int(ends[episode_index - 1]) if episode_index else 0
        end = int(ends[episode_index])
        if start >= end or not np.asarray(root["data/object_center_valid"][start:end]).all():
            raise ValueError(f"Invalid object observations in longest {task} episode {episode_index}")
        episodes[task] = root, episode_index, start, end
    return episodes


def load_offline_planner(variant: str, device: torch.device):
    try:
        from train_contact_planner import load_compact_planner_state
    except ModuleNotFoundError:
        from scripts.train_contact_planner import load_compact_planner_state

    checkpoint_name, point_count = OFFLINE_VARIANTS[variant]
    checkpoint_path = resolve(Path(checkpoint_name))
    config = load_yaml(checkpoint_path.parent.parent / "config.yaml")
    model = DexCG.from_config(
        load_config(resolve(Path(config["model_config"]))),
        PROJECT_ROOT,
        torch_dtype=torch.bfloat16,
    )
    with torch.serialization.safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=True
        )
    require_model_coordinates(checkpoint)
    wrapper = type("PlannerWrapper", (), {"planner": model.contact_planner})()
    load_compact_planner_state(wrapper, checkpoint["planner"])
    del checkpoint
    return model.contact_planner.to(device).eval(), config, point_count


@torch.no_grad()
def write_offline_video(
    output_path: Path,
    task: str,
    variant: str,
    episode: tuple[zarr.Group, int, int, int],
    planner,
    instruction: str,
    point_count: int,
    device: torch.device,
    crf: int,
) -> None:
    root, episode_index, start, end = episode
    data = root["data"]
    clouds = np.asarray(data["point_cloud"][start:end], dtype=np.float32)
    qpos = np.asarray(data["agent_pos"][start:end, :22], dtype=np.float32)
    palms = np.asarray(data["palm_pose_robot_base"][start:end], dtype=np.float32)
    if clouds.shape[1:] != (10000, 3) or not np.isfinite(clouds).all():
        raise ValueError("Demo requires 10000 finite metric object points per frame")
    if not np.isfinite(qpos).all() or palms.shape[1:] != (4, 4) or not np.isfinite(palms).all():
        raise ValueError("Demo has invalid robot conditioning")
    if np.max(np.abs(clouds)) >= 1.96:
        raise ValueError("Object XYZ is outside the PartField grid extent")
    robots = robot_geometry(qpos)
    bounds = compute_projection_bounds(clouds[:, ::8], robots)
    object_id = str(root["meta/object_id"][episode_index])
    stable_step = int(root["meta/stable_contact_steps"][episode_index])
    if not 0 <= stable_step < end - start:
        raise ValueError("Stable contact step is outside the selected episode")
    encoder = open_encoder(output_path, crf)
    prediction = None
    try:
        assert encoder.stdin is not None
        for frame_index in range(end - start):
            prediction_step = frame_index
            if point_count == 1024:
                rng = np.random.default_rng(
                    np.random.SeedSequence([42, episode_index, frame_index])
                )
                points = clouds[frame_index, rng.choice(10000, 1024, replace=False)]
            else:
                points = clouds[frame_index]
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                plan = planner.plan(
                    torch.from_numpy(np.ascontiguousarray(points))[None].to(device),
                    [instruction],
                    robot_qpos=torch.from_numpy(qpos[frame_index][None]).to(device),
                    palm_pose_robot_base=torch.from_numpy(palms[frame_index][None]).to(device),
                )
            prediction = decode_predicted_contacts(planner, plan)
            raw_points = np.asarray(data["contact_raw_points"][start + frame_index])
            raw_mask = np.asarray(data["contact_raw_mask"][start + frame_index])
            frame = render_planner_frame(
                np.asarray(data["img"][start + frame_index]),
                clouds[frame_index, ::8], robots[frame_index], raw_points, raw_mask,
                *prediction, bounds, task=task, variant=variant, object_id=object_id,
                instruction=instruction, episode_index=episode_index,
                frame_index=frame_index, frame_count=end - start,
                prediction_step=prediction_step, stable_step=stable_step,
            )
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to encode {output_path}")
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        output_path.unlink(missing_ok=True)
        raise
    print(
        f"{variant}/{task}: episode={episode_index} object={object_id} "
        f"frames={end - start} predictions={end - start} -> {output_path}",
        flush=True,
    )


def render_offline_vlm(args: argparse.Namespace) -> None:
    if args.output_dir == DEFAULT_OUTPUT:
        raise ValueError("--output-dir must be specified for offline VLM videos")
    output_dir = resolve(args.output_dir)
    if output_dir.resolve() != (PROJECT_ROOT / "data_visualize/dexcg" / args.offline_vlm).resolve():
        raise ValueError("Offline output directory must match the approved variant directory")
    output_paths = {task: output_dir / f"{task}_framewise.mp4" for task in TASKS}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing videos: {existing}")
    episodes = offline_episodes(args.offline_vlm)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    planner, config, point_count = load_offline_planner(args.offline_vlm, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        write_offline_video(
            output_paths[task], task, args.offline_vlm, episodes[task],
            planner, str(config["evaluation"]["tasks"][task]["instruction"]),
            point_count, device, args.crf,
        )


@torch.no_grad()
def collect_policy_episode(objective, adapter, episode_seed, task_config, config, device):
    """Use the evaluation control loop, recording each state without extra policy calls."""
    from dexcg.data.robot_base import InvalidObjectObservation
    from dexcg.evaluation.dexart import _split_object_observation, _tensor_observation

    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)
    adapter.environment.seed(episode_seed)
    adapter.point_rng = np.random.RandomState(episode_seed)
    subset_rng = np.random.default_rng(episode_seed)
    raw = adapter.reset()
    sample = None
    for retry in range(int(config.get("initial_observation_retries", 8)) + 1):
        try:
            sample = adapter.observation(raw)
            current = _split_object_observation(sample, config["planner_point_count"], subset_rng)
            break
        except InvalidObjectObservation:
            sample = None
            if retry < int(config.get("initial_observation_retries", 8)):
                raw = adapter.observe()
    if sample is None:
        return [], {"object_id": adapter.object_id, "failure_reason": "invalid_initial_observation"}

    frames = []
    history = deque([current], maxlen=2)
    stable_step = 0 if adapter.is_stable_contact else -1
    success, done, steps = bool(adapter.is_success), False, 0
    previous_plan, prediction_step, prediction_count = None, 0, 0
    decoded = None
    clipped_actions, action_values = 0, 0
    sensor_error = pixel_error = 0.0

    def record():
        nonlocal sensor_error, pixel_error
        contact = adapter.contact_graph()
        sensor_error = max(sensor_error, float(sample["depth_to_point_max_error_m"]))
        pixel_error = max(pixel_error, float(sample["pixel_reprojection_max_error_pixels"]))
        frames.append({
            "image": sample["img"].copy(),
            "point_cloud": sample["point_cloud"][::8].copy(),
            "robot": robot_geometry(sample["agent_pos"][:22]),
            "raw_points": contact.points, "raw_mask": contact.mask,
            "prediction": decoded, "prediction_step": prediction_step,
        })

    while steps < int(task_config["max_steps"]) and not success and not done:
        observation = _tensor_observation(history, device)
        torch.cuda.set_device(device) if device.type == "cuda" else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction = objective.predict_action_with_diagnostics(
                observation, [task_config["instruction"]],
                int(config["num_inference_steps"]), int(config["action_steps"]),
                previous_contact_plan=previous_plan,
            )
        previous_plan = prediction.contact_plan
        decoded = decode_predicted_contacts(objective.model.contact_planner, previous_plan)
        prediction_step = steps
        prediction_count += 1
        if frames:
            frames[-1].update(prediction=decoded, prediction_step=prediction_step)
        else:
            record()
        actions = prediction.actions[0].float().cpu().numpy()
        if not len(actions) or not np.isfinite(actions).all():
            raise ValueError("Policy produced empty or nonfinite actions")
        for action in actions:
            clipped_actions += int((np.abs(action) > 1).sum())
            action_values += action.size
            raw, _, done, _ = adapter.step(action)
            steps += 1
            success = bool(adapter.is_success)
            done = bool(done) or steps >= int(task_config["max_steps"])
            # This terminal capture is for display only; never reuse a stale state.
            try:
                sample = adapter.observation(raw)
                if not success and not done:
                    current = _split_object_observation(sample, config["planner_point_count"], subset_rng)
                    history.append(current)
            except InvalidObjectObservation as error:
                return [], {"object_id": adapter.object_id, "steps": steps,
                            "failure_reason": str(error)}
            if stable_step < 0 and adapter.is_stable_contact:
                stable_step = steps
            record()
            if success or done:
                break
    return frames, {
        "object_id": adapter.object_id, "episode_seed": episode_seed,
        "success": success, "steps": steps, "stable_step": stable_step,
        "prediction_count": prediction_count, "failure_reason": None,
        "action_outside_unit_fraction": clipped_actions / max(action_values, 1),
        "depth_to_point_max_error_m": sensor_error,
        "pixel_reprojection_max_error_pixels": pixel_error,
    }


def write_policy_video(path, frames, metadata, crf):
    bounds = compute_projection_bounds(
        np.stack([frame["point_cloud"] for frame in frames]),
        np.stack([frame["robot"] for frame in frames]),
    )
    encoder = open_encoder(path, crf, metadata)
    try:
        for index, sample in enumerate(frames):
            frame = render_planner_frame(
                sample["image"], sample["point_cloud"], sample["robot"],
                sample["raw_points"], sample["raw_mask"], *sample["prediction"], bounds,
                task=metadata["task"], variant=metadata["variant"],
                object_id=metadata["object_id"], instruction=metadata["instruction"],
                episode_index=metadata["episode"], frame_index=index, frame_count=len(frames),
                prediction_step=sample["prediction_step"], stable_step=metadata["stable_step"],
                success=metadata["success"],
            )
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to encode {path}")
    except BaseException:
        if encoder.poll() is None:
            encoder.kill()
        encoder.wait()
        path.unlink(missing_ok=True)
        raise


def render_online_vlm(args):
    from dexcg.data.robot_base import RobotBaseCollectionAdapter
    from dexcg.evaluation.dexart import _point_input_guards
    from train import load_variant

    run_dir = PROJECT_ROOT / "outputs/smp-robot-base-dual-vlm"
    output_dir = resolve(args.output_dir)
    expected_dir = PROJECT_ROOT / "data_visualize/dexcg" / args.online_vlm / "SMP"
    if output_dir.resolve() != expected_dir.resolve():
        raise ValueError(f"Online output directory must be {expected_dir}")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    config = load_yaml(run_dir / "config.yaml")
    evaluation = dict(config["evaluation"])
    variant_key = "old_1024" if args.online_vlm == "qwen-1024" else "new_10000"
    variant = evaluation["variants"][variant_key]
    evaluation.update(planner_point_count=int(variant["point_count"]),
                      num_inference_steps=int(config["diffusion"]["num_inference_steps"]))
    device = torch.device(args.device)
    checkpoint_path = run_dir / "checkpoints" / ONLINE_CHECKPOINTS[args.online_vlm]
    objective, checkpoint = load_objective(config, checkpoint_path, device)
    expected_hash = checkpoint["run_identity"]["variants"][variant_key]["checkpoint_sha256"]
    if sha256(resolve(Path(variant["checkpoint"]))) != expected_hash:
        raise ValueError("VLM checkpoint differs from the one used in SMP evaluation")
    objective.model.contact_planner = load_variant(objective.model, variant).to(device)
    objective.eval()
    report = json.loads((run_dir / "evaluation/success_rates" /
        f"success_rates_epoch_{checkpoint['epoch']:04d}_vlm{variant['point_count']}.json").read_text())
    if report["planner_checkpoint_sha256"] != expected_hash:
        raise ValueError("Evaluation report and VLM checkpoint disagree")
    driver = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if driver.is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", str(driver))
    output_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        task_config = evaluation["tasks"][task]
        accepted = {True: 0, False: 0}
        used_objects, used_seeds = set(), set()
        for path in output_dir.glob(f"{task}_*.mp4"):
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
                "-of", "json", str(path)], text=True))
            saved = json.loads(probe["format"]["tags"]["comment"])
            if saved["variant"] != args.online_vlm or saved["smp_checkpoint"] != str(checkpoint_path):
                raise ValueError(f"Existing video uses another model: {path}")
            accepted[saved["success"]] += 1
            used_objects.add(saved["object_id"])
            used_seeds.add(saved["episode_seed"])
        targets = {True: 2, False: 3}
        if accepted == targets:
            continue
        candidates = [item for item in report["tasks"][task]["episode_results"]
                      if not item["failure_reason"] and item["episode_seed"] not in used_seeds]
        candidates.sort(key=lambda item: not item["success"])
        counts = {"partfield_calls": 0, "dp3_calls": 0}
        adapter = RobotBaseCollectionAdapter.create(task, evaluation["split"], 1.0e-2)
        adapter.point_count = 10000
        adapter.enable_high_resolution_object_cloud(evaluation["point_capture_resolution"])
        initial_object_cursor = getattr(adapter.environment, "i", None)
        backups = {True: [], False: []}
        attempts = 0

        def save(frames, result, candidate):
            outcome = bool(result["success"])
            label = "success" if outcome else "failure"
            path = output_dir / f"{task}_{label}_{accepted[outcome] + 1:02d}.mp4"
            if path.exists():
                raise FileExistsError(path)
            metadata = {**result, "task": task, "variant": args.online_vlm,
                "episode": candidate["episode"], "evaluation_seed": candidate["seed"],
                "instruction": task_config["instruction"], "split": evaluation["split"],
                "smp_checkpoint": str(checkpoint_path), "smp_epoch": checkpoint["epoch"],
                "planner_checkpoint": variant["checkpoint"], "planner_sha256": expected_hash,
                "planner_point_count": variant["point_count"], "smp_point_count": 1024,
                "action_steps": evaluation["action_steps"], "max_steps": task_config["max_steps"],
                "num_inference_steps": evaluation["num_inference_steps"], "frames": len(frames),
                "fps": FPS, "source": "online_vlm_smp_rollout"}
            write_policy_video(path, frames, metadata, args.crf)
            accepted[outcome] += 1
            used_objects.add(result["object_id"])
            print(f"SAVED {path.name}: " + json.dumps(metadata, sort_keys=True), flush=True)

        try:
            with _point_input_guards(objective, variant["point_count"], counts):
                for candidate in candidates:
                    if accepted == targets or attempts >= args.max_attempts:
                        break
                    attempts += 1
                    # Faucet/bucket/toilet rotate objects on reset independently
                    # of the seed. Restore the evaluated episode's cursor too.
                    if initial_object_cursor is not None:
                        adapter.environment.i = (
                            initial_object_cursor + candidate["episode"]
                        ) % len(adapter.environment.instance_list)
                    frames, result = collect_policy_episode(
                        objective, adapter, candidate["episode_seed"], task_config, evaluation, device)
                    print(f"{args.online_vlm}/{task} attempt={attempts} "
                          f"seed={candidate['episode_seed']} " + json.dumps(result), flush=True)
                    if not frames or result["failure_reason"]:
                        continue
                    outcome = result["success"]
                    if accepted[outcome] >= targets[outcome]:
                        continue
                    if result["object_id"] in used_objects:
                        if len(backups[outcome]) < targets[outcome]:
                            backups[outcome].append((frames, result, candidate))
                        continue
                    save(frames, result, candidate)
                # Object diversity is preferred, but outcome quotas take priority.
                for outcome in (True, False):
                    for frames, result, candidate in backups[outcome]:
                        if accepted[outcome] < targets[outcome]:
                            save(frames, result, candidate)
            print(f"{args.online_vlm}/{task}: accepted={accepted} objects={sorted(used_objects)} "
                  f"input_audit={counts}", flush=True)
            if accepted != targets:
                raise RuntimeError(f"{task}: quota incomplete after {attempts} attempts: {accepted}")
        finally:
            adapter.close()
            if device.type == "cuda":
                torch.cuda.set_device(device)


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    if args.online_vlm is not None:
        render_online_vlm(args)
        return
    if args.offline_vlm is not None:
        render_offline_vlm(args)
        return
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")

    config_path = resolve(args.config)
    checkpoint_path = resolve(args.checkpoint)
    output_dir = resolve(args.output_dir)
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {[str(path) for path in existing]}")
    config = load_yaml(config_path)
    task_config = config["evaluation"]["tasks"][args.task]
    device = torch.device(args.device)

    print(f"loading {checkpoint_path} on {device}", flush=True)
    objective, checkpoint = load_objective(config, checkpoint_path, device)
    action_steps = int(config["evaluation"]["action_steps"])
    num_inference_steps = int(config["diffusion"]["num_inference_steps"])
    arrays, rollout = collect_successful_rollout(
        objective=objective,
        task=args.task,
        split=args.split,
        instruction=str(task_config["instruction"]),
        max_steps=int(task_config["max_steps"]),
        action_steps=action_steps,
        num_inference_steps=num_inference_steps,
        seed=args.seed,
        max_attempts=args.max_attempts,
        device=device,
    )
    metadata = {
        "format": "dexcg.online_rollout_visualization.v1",
        "task": args.task,
        "split": args.split,
        "object_id": rollout["object_id"],
        "instruction": str(task_config["instruction"]),
        "seed": args.seed,
        "attempt": rollout["attempt"],
        "success": True,
        "stable_frame": rollout["stable_frame"],
        "steps": rollout["steps"],
        "frames": len(arrays["img"]),
        "inference_calls": len(arrays["basis"]),
        "action_steps_per_inference": action_steps,
        "num_inference_steps": num_inference_steps,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "basis_shape": list(arrays["basis"].shape),
        "basis_semantics": "final QR basis including PhysGraph bias",
        "basis_value_limit": float(np.max(np.abs(arrays["basis"]))),
        "frame_action_alignment": "actions[i] transitions frame i to frame i+1",
        "inference_alignment": (
            "inference_index maps each frame to basis, contact tokens, object center, "
            "predicted contacts, and model observation history"
        ),
        "heatmap_layout": "rows=22_action_channels, columns=[B1,B2,B3,B4]_current_inference",
        "heatmap_colormap": "ColorBrewer_RdBu_r",
        "point_cloud_frame": "robot_base",
        "raw_contact_frame": "robot_base",
        "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
        "contact_token_frame": "robot_base",
        "object_center_definition": OBJECT_CENTER_DEFINITION,
        "predicted_contact_frame": "robot_base",
        "fps": FPS,
        "video_resolution": [FRAME_WIDTH, FRAME_HEIGHT],
    }
    write_outputs(output_dir, arrays, metadata, args.crf)
    print(json.dumps({"output_dir": str(output_dir), **rollout}, indent=2), flush=True)


if __name__ == "__main__":
    main()
