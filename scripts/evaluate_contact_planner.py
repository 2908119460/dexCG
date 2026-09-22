#!/usr/bin/env python3
"""Evaluate free-running planner contacts on the trajectory validation split."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from dexcg.common.config import load_config
from dexcg.data.planner_training import diagnostic_samples
from dexcg.common.typing import ContactPlan
from dexcg.evaluation.planner import ContactMetricAccumulator
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    require_model_coordinates,
    robot_base_point_cloud,
)
from dexcg.models.dexcg import DexCG

try:
    from train_contact_planner import (
        load_compact_planner_state,
        load_training_config,
        move_batch,
        planner_dataset,
        resolve,
    )
except ModuleNotFoundError:
    from scripts.train_contact_planner import (
        load_compact_planner_state,
        load_training_config,
        move_batch,
        planner_dataset,
        resolve,
    )


def parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_planner_balanced_dev_v2_lr.yaml")
    parser.add_argument("--planner-checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--diagnose-generalization", action="store_true")
    parser.add_argument("--diagnostic-plan-only", action="store_true")
    parser.add_argument(
        "--modes",
        type=parse_csv,
        default=parse_csv("no_history,ground_truth_history,predicted_history"),
    )
    parser.add_argument(
        "--language-families",
        type=parse_csv,
        default=parse_csv("low_level,high_level,deployment"),
    )
    return parser.parse_args()


@torch.inference_mode()
def diagnose_generalization(args, config):
    if not args.planner_checkpoint:
        raise ValueError("Diagnostic requires a fixed planner checkpoint")
    datasets = {split: planner_dataset(config, split) for split in ("train", "validation")}
    selections = {split: diagnostic_samples(dataset) for split, dataset in datasets.items()}
    memberships = [{e.key for e in dataset.episodes} for dataset in datasets.values()]
    if memberships[0] & memberships[1]:
        raise ValueError("Train/validation trajectory overlap")
    for dataset in datasets.values():
        if dataset.use_previous_contact or not dataset.use_robot_state:
            raise ValueError("Diagnostic requires audited single-frame robot-conditioned inputs")
    output = {
        "format": "dexcg.planner_generalization_diagnostic.v1",
        "config": str(resolve(args.config)),
        "checkpoint": str(resolve(args.planner_checkpoint)),
        "seed": 42, "coordinate_contract": CONTACT_COORDINATE_CONTRACT,
        "sampling": "two trajectories per object; first/middle/last valid frame per trajectory",
        "loss_aggregation": "mean per-frame CE; equal frame quotas per object; no dataset weights",
        "token_accuracy_aggregation": "correct tokens / all target tokens",
        "precision": "FP32 weights with BF16 autocast",
        "history": False, "model_mode": "eval", "batch_size": args.batch_size,
        "sample_manifest": {split: selection[1] for split, selection in selections.items()},
        "results": {},
    }
    if args.diagnostic_plan_only:
        print(json.dumps({"counts": {s: len(v[0]) for s, v in selections.items()},
                          "trajectory_overlap": 0, "checkpoint": output["checkpoint"]}), flush=True)
        return
    if not args.output or not resolve(args.output).parent.is_dir() or resolve(args.output).exists():
        raise ValueError("Diagnostic requires a new result path in an existing approved directory")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    model = DexCG.from_config(load_config(resolve(config["model_config"])),
                             Path(__file__).resolve().parents[1],
                             torch_dtype=torch.float32).to(device)
    output["checkpoint_epoch"] = _load_planner_checkpoint(model, resolve(args.planner_checkpoint))
    model.requires_grad_(False).eval()
    assert not any(module.training for module in model.modules())
    tokenizer = model.contact_planner.contact_tokenizer
    # Audit all chosen inputs/targets before any model evaluation.
    audit = {}
    for split, dataset in datasets.items():
        for index in selections[split][0]:
            row = dataset[index]
            if not row["object_point_mask"].all():
                raise ValueError("Diagnostic point cloud contains non-object points")
            robot_base_point_cloud(row["point_cloud"][None], row["object_point_mask"][None])
            model.contact_planner.robot_state_projector.features(
                row["robot_qpos"][None], row["palm_pose_robot_base"][None])
            ids = row["target_ids"][row["target_mask"]].tolist()
            if tokenizer.encode(tokenizer.decode(ids)) != ids:
                raise ValueError("Target coordinate token round trip failed")
        audit[split] = {"samples": len(selections[split][0]), "object_only": True,
                        "robot_pose_valid": True, "target_round_trip": True}
    output["input_audit"] = audit
    print(json.dumps({"event": "diagnostic_preflight", **audit}), flush=True)
    for split, dataset in datasets.items():
        output["results"][split] = {}
        indices = selections[split][0]
        for family in args.language_families:
            started = time.perf_counter()
            accumulator = ContactMetricAccumulator(tokenizer)
            totals = {"loss_sum": 0.0, "samples": 0, "correct": 0, "tokens": 0}
            task_results = {}
            for task_index, task in enumerate(dataset.tasks):
                task_indices = [i for i in indices if dataset.samples[i].episode.task_index == task_index]
                loader = DataLoader(Subset(dataset, task_indices), batch_size=args.batch_size,
                                    num_workers=args.num_workers, shuffle=False)
                local = dict.fromkeys(totals, 0)
                for cpu_batch in loader:
                    batch = move_batch(cpu_batch, device)
                    languages = dataset.languages_for_family(cpu_batch, family)
                    robot = {key: batch[key] for key in ("robot_qpos", "palm_pose_robot_base")}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, metrics = model.contact_planner.training_loss(
                            batch["point_cloud"], languages, batch["target_ids"],
                            batch["target_mask"], sample_weight=None, **robot)
                        plan = model.contact_planner.plan(batch["point_cloud"], languages, **robot)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite diagnostic loss")
                    rows = len(languages)
                    local["loss_sum"] += loss.item() * rows
                    local["samples"] += rows
                    local["correct"] += metrics["correct"].item()
                    local["tokens"] += metrics["count"].item()
                    accumulator.update(plan.token_ids, plan.attention_mask, batch["target_ids"],
                                       batch["target_mask"], {"task": cpu_batch["task"]})
                task_results[task] = {"loss": local["loss_sum"] / local["samples"],
                                      "token_accuracy": local["correct"] / local["tokens"], **local}
                for key in totals:
                    totals[key] += local[key]
                print(json.dumps({"event": "diagnostic_task_done", "split": split,
                                  "family": family, "task": task, **task_results[task]}), flush=True)
            result = {"teacher_forcing": {"loss": totals["loss_sum"] / totals["samples"],
                       "token_accuracy": totals["correct"] / totals["tokens"], **totals},
                      "teacher_forcing_by_task": task_results, "generation": accumulator.result(),
                      "seconds": time.perf_counter() - started}
            output["results"][split][family] = result
            print(json.dumps({"event": "diagnostic_condition_done", "split": split,
                              "family": family, **result}), flush=True)
    with resolve(args.output).open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_planner_checkpoint(model: DexCG, checkpoint_path: Path) -> int:
    # Resumable checkpoints also contain NumPy's uint32 RNG state.
    with torch.serialization.safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    require_model_coordinates(checkpoint)
    if checkpoint.get("contact_coordinate_contract") != CONTACT_COORDINATE_CONTRACT:
        raise RuntimeError("Planner checkpoint coordinate contract is incompatible")
    wrapper = type("PlannerWrapper", (), {"planner": model.contact_planner})()
    load_compact_planner_state(wrapper, checkpoint["planner"])
    return int(checkpoint.get("epoch", 0))


def _depth_groups(dataset, limit: int) -> list[list[int]]:
    index_by_step: dict[tuple[str, int], int] = {}
    depths: list[int] = []
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples[:limit]):
        previous_index = (
            None
            if sample.previous_step is None
            else index_by_step[(sample.episode.key, sample.previous_step)]
        )
        depth = 0 if previous_index is None else depths[previous_index] + 1
        depths.append(depth)
        groups[depth].append(index)
        index_by_step[(sample.episode.key, sample.step)] = index
    return [groups[depth] for depth in sorted(groups)]


def _ground_truth_history(batch: dict[str, Any]) -> ContactPlan:
    return ContactPlan(batch["previous_ids"], batch["previous_mask"])


@torch.no_grad()
def evaluate_mode(
    model: DexCG,
    dataset,
    language_family: str,
    mode: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    max_samples: int | None,
) -> tuple[dict[str, Any], float]:
    if mode not in {"no_history", "ground_truth_history", "predicted_history"}:
        raise ValueError(f"unknown history mode: {mode}")
    limit = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if mode == "predicted_history":
        index_groups = _depth_groups(dataset, limit)
    else:
        index_groups = [list(range(limit))]

    accumulator = ContactMetricAccumulator(model.contact_planner.contact_tokenizer)
    predictions: dict[tuple[str, int], ContactPlan] = {}
    start_time = time.perf_counter()
    for indices in index_groups:
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
        for cpu_batch in loader:
            batch = move_batch(cpu_batch, device)
            point_cloud = robot_base_point_cloud(batch["point_cloud"], batch["object_point_mask"])
            languages = dataset.languages_for_family(cpu_batch, language_family)
            previous_plan = None
            if mode == "ground_truth_history":
                previous_plan = _ground_truth_history(batch)
            elif mode == "predicted_history":
                prior_rows = []
                for episode_key, absolute_step in zip(
                    cpu_batch["episode_key"], cpu_batch["previous_step"].tolist(), strict=True
                ):
                    prior_rows.append(
                        None
                        if absolute_step < 0
                        else predictions[(str(episode_key), int(absolute_step))]
                    )
                if any(plan is not None for plan in prior_rows):
                    end_id = model.contact_planner.contact_tokenizer.joint_end_id
                    width = max(
                        int(plan.token_ids.shape[1]) for plan in prior_rows if plan is not None
                    )
                    ids = torch.full(
                        (len(prior_rows), width), end_id, dtype=torch.long, device=device
                    )
                    mask = torch.zeros((len(prior_rows), width), dtype=torch.bool, device=device)
                    for row, plan in enumerate(prior_rows):
                        if plan is None:
                            continue
                        length = int(plan.token_ids.shape[1])
                        ids[row, :length] = plan.token_ids[0]
                        mask[row, :length] = plan.attention_mask[0]
                    previous_plan = ContactPlan(ids, mask)
            robot_inputs = (
                {key: batch[key] for key in ("robot_qpos", "palm_pose_robot_base")}
                if model.contact_planner.robot_state_projector is not None
                else {}
            )
            plan = model.contact_planner.plan(
                point_cloud, languages, previous_plan=previous_plan, **robot_inputs
            )
            if mode == "predicted_history":
                for row, (episode_key, absolute_step) in enumerate(
                    zip(
                        cpu_batch["episode_key"],
                        cpu_batch["absolute_step"].tolist(),
                        strict=True,
                    )
                ):
                    predictions[(str(episode_key), int(absolute_step))] = ContactPlan(
                        plan.token_ids[row : row + 1].detach(),
                        plan.attention_mask[row : row + 1].detach(),
                    )
            accumulator.update(
                plan.token_ids,
                plan.attention_mask,
                batch["target_ids"],
                batch["target_mask"],
                {
                    "task": cpu_batch["task"],
                    "history_presence": [
                        "present" if bool(value) else "absent"
                        for value in cpu_batch["previous_mask"].any(dim=1).tolist()
                    ],
                },
            )
    return accumulator.result(), time.perf_counter() - start_time


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers must be non-negative")
    config = load_training_config(args.config)
    unknown_families = set(args.language_families) - {"low_level", "high_level", "deployment"}
    if unknown_families:
        raise ValueError(f"unknown language families: {sorted(unknown_families)}")

    if args.diagnose_generalization or args.diagnostic_plan_only:
        diagnose_generalization(args, config)
        return

    device = torch.device("cuda", 0)
    model = DexCG.from_config(
        load_config(resolve(config["model_config"])),
        Path(__file__).resolve().parents[1],
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    checkpoint_epoch = 0
    if args.planner_checkpoint:
        checkpoint_epoch = _load_planner_checkpoint(model, resolve(args.planner_checkpoint))
    dataset = planner_dataset(config, "validation")

    results = {}
    for language_family in args.language_families:
        results[language_family] = {}
        for mode in args.modes:
            metrics, seconds = evaluate_mode(
                model,
                dataset,
                language_family,
                mode,
                device,
                args.batch_size,
                args.num_workers,
                args.max_samples,
            )
            results[language_family][mode] = {"seconds": seconds, **metrics}
            print(
                json.dumps(
                    {
                        "event": "planner_generation_metrics",
                        "language_family": language_family,
                        "history_mode": mode,
                        **results[language_family][mode],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    output = {
        "format": "dexcg.contact_planner_generation.v1",
        "config": str(resolve(args.config)),
        "checkpoint": None
        if args.planner_checkpoint is None
        else str(resolve(args.planner_checkpoint)),
        "checkpoint_epoch": checkpoint_epoch,
        "split": "validation",
        "sample_limit": args.max_samples,
        "results": results,
    }
    if args.output:
        output_path = resolve(args.output)
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"Output parent directory does not exist: {output_path.parent}")
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2, sort_keys=True)
            stream.write("\n")


if __name__ == "__main__":
    main()
