#!/usr/bin/env python3
"""Evaluate a planner and frozen policy in closed-loop DexArt tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
import zarr

from dexcg.common.config import load_config
from dexcg.evaluation import evaluate_seen_tasks
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    OBJECT_CENTER_DEFINITION,
    require_model_coordinates,
)
from dexcg.models.dexcg import DexCG
from dexcg.training import DexCGTrainingObjective

try:
    from train_contact_planner import (
        load_compact_planner_state,
        load_policy_checkpoint,
        load_training_config,
        resolve,
    )
except ModuleNotFoundError:
    from scripts.train_contact_planner import (
        load_compact_planner_state,
        load_policy_checkpoint,
        load_training_config,
        resolve,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--planner-point-count", required=True, type=int, choices=(1024, 10000))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint-coordinate-contract",
        default=CONTACT_COORDINATE_CONTRACT,
        help="Required robot-base source contract for both checkpoints.",
    )
    parser.add_argument("--episodes-per-seed", type=int)
    parser.add_argument("--seeds", type=parse_int_csv)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_planner_point_count(checkpoint_path: Path, expected_count: int) -> dict:
    """Bind the requested point count to the checkpoint's recorded training data."""
    run_config_path = checkpoint_path.parent.parent / "config.yaml"
    run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
    paths = run_config.get("data", {}).get("paths", [])
    if not paths:
        raise ValueError(f"No training data recorded for planner: {run_config_path}")
    sources = []
    for path in paths:
        dataset_path = resolve(path)
        root = zarr.open_group(str(dataset_path), mode="r")
        shape = root["data/point_cloud"].shape
        if len(shape) != 3 or shape[1:] != (expected_count, 3):
            raise ValueError(
                f"Planner training data {dataset_path} has shape {shape}; "
                f"--planner-point-count={expected_count} does not match"
            )
        if root.attrs.get("point_cloud_frame") != "robot_base" or root.attrs.get("length_unit") != "metre":
            raise ValueError(f"Planner training data is not robot-base metres: {dataset_path}")
        sources.append(str(dataset_path))
    return {"run_config": str(run_config_path), "run_config_sha256": sha256(run_config_path),
            "verified_training_point_count": expected_count, "training_data": sources}


def load_planner_checkpoint(
    model: DexCG, checkpoint_path: Path, expected_coordinate_contract: str
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require_model_coordinates(checkpoint)
    contract = checkpoint.get("contact_coordinate_contract")
    if contract != expected_coordinate_contract:
        raise RuntimeError(
            f"Planner checkpoint {checkpoint_path} uses coordinate contract {contract!r}; "
            f"expected {expected_coordinate_contract!r}"
        )
    wrapper = type("PlannerWrapper", (), {"planner": model.contact_planner})()
    load_compact_planner_state(wrapper, checkpoint["planner"])
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.episodes_per_seed is not None and args.episodes_per_seed < 1:
        raise ValueError("--episodes-per-seed must be positive")

    config = load_training_config(args.config)
    config["evaluation"].update(
        planner_point_count=args.planner_point_count,
        smp_point_count=1024,
        point_capture_resolution=[1024, 1024],
        record_video=not args.no_video,
    )
    if args.episodes_per_seed is not None:
        config["evaluation"]["episodes_per_seed"] = args.episodes_per_seed
    if args.seeds is not None:
        config["evaluation"]["seeds"] = list(args.seeds)
    planner_path = resolve(args.planner_checkpoint)
    point_count_evidence = verify_planner_point_count(planner_path, args.planner_point_count)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    evaluation_group = None
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=device)
        evaluation_group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))

    output_dir = resolve(args.output_dir)
    policy_path = resolve(config["policy_checkpoint"])
    config["evaluation"]["planner_checkpoint"] = str(planner_path)
    config["evaluation"]["planner_checkpoint_sha256"] = sha256(planner_path)
    try:
        model = DexCG.from_config(
            load_config(resolve(config["model_config"])),
            PROJECT_ROOT,
            torch_dtype=torch.bfloat16,
        )
        policy_checkpoint, state_min, state_max = load_policy_checkpoint(
            model,
            policy_path,
            expected_coordinate_contract=args.checkpoint_coordinate_contract,
        )
        planner_checkpoint = load_planner_checkpoint(
            model, planner_path, args.checkpoint_coordinate_contract
        )
        objective = DexCGTrainingObjective(
            model,
            state_min,
            state_max,
            config["diffusion"],
            {
                "action_likelihood_std": 1.0,
                "gate_alpha": 2.0,
                "gate_alpha0": 0.5,
                "gate_kappa": 20.0,
                "coefficient": 1.0,
                "reconstruction": 1.0,
                "gate": 1.0,
                "alignment": 1.0,
                "contact": 0.0,
            },
            train_contact_planner=True,
        ).to(device)
        objective.eval()

        if rank == 0:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")
            (output_dir / "evaluation" / "videos").mkdir(parents=True, exist_ok=True)
            saved_config = copy.deepcopy(config)
            saved_config["output_dir"] = str(output_dir)
            saved_config["planner_checkpoint"] = str(planner_path)
            saved_config["checkpoint_coordinate_contract"] = args.checkpoint_coordinate_contract
            with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
                yaml.safe_dump(saved_config, stream, sort_keys=False)
            metadata = {
                "format": "dexcg.closed_loop_interface_ablation.v1",
                "purpose": (
                    "robot-base planner and policy evaluated with matching coordinate contracts"
                ),
                "runtime_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
                "runtime_object_center_definition": OBJECT_CENTER_DEFINITION,
                "checkpoint_coordinate_contract": args.checkpoint_coordinate_contract,
                "planner_checkpoint": str(planner_path),
                "planner_checkpoint_epoch": int(planner_checkpoint.get("epoch", 0)),
                "planner_checkpoint_sha256": sha256(planner_path),
                "policy_checkpoint": str(policy_path),
                "policy_checkpoint_epoch": int(policy_checkpoint.get("epoch", 0)),
                "policy_checkpoint_sha256": sha256(policy_path),
                "planner_point_count": args.planner_point_count,
                "smp_object_point_count": 1024,
                "point_capture_resolution": [1024, 1024],
                "point_count_verification": point_count_evidence,
                "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
                stream.write("\n")
        if world_size > 1:
            dist.barrier()

        result = evaluate_seen_tasks(
            objective,
            config,
            int(planner_checkpoint.get("epoch", 0)),
            output_dir,
            device,
            rank,
            world_size,
            evaluation_group,
        )
        if rank == 0:
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
