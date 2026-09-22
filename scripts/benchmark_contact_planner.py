#!/usr/bin/env python3
"""Measure real planner fine-tuning throughput without retaining benchmark artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from dexcg.common.config import load_config
from dexcg.models.dexcg import DexCG
from dexcg.training import ContactPlannerTrainingObjective

try:
    from train_contact_planner import (
        loader_for,
        move_batch,
        optimizer_for,
        planner_dataset,
        resolve,
    )
except ModuleNotFoundError:
    from scripts.train_contact_planner import (
        loader_for,
        move_batch,
        optimizer_for,
        planner_dataset,
        resolve,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_planner_balanced_dev.yaml")
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument(
        "--batch-sizes",
        type=parse_ints,
        default=parse_ints("1,2,4,8,9,10,11,12,16,24,32"),
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--io-trials", type=int, default=100)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    return parser.parse_args()


def gpu_status(index: int) -> dict[str, int | str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    name, total, used, utilization = [item.strip() for item in output.split(",")]
    return {
        "index": index,
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_before_mib": int(used),
        "utilization_before_percent": int(utilization),
    }


def preflight_gpu(index: int, allow_shared: bool) -> dict[str, int | str]:
    status = gpu_status(index)
    if not allow_shared and (
        int(status["memory_used_before_mib"]) > 1024
        or int(status["utilization_before_percent"]) > 10
    ):
        raise RuntimeError(
            f"GPU {index} is not idle: {status['memory_used_before_mib']} MiB used, "
            f"{status['utilization_before_percent']}% utilization"
        )
    return status


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)
    return ordered[index]


def benchmark_io(
    dataset, batch_size: int, trials: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        (0, 1, False),
        (0, 1, True),
        (2, 2, True),
        (4, 2, True),
        (4, 4, True),
        (8, 2, True),
        (8, 4, True),
    ]
    results = []
    for workers, prefetch, pin_memory in candidates:
        options: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": workers,
            "pin_memory": pin_memory,
            "persistent_workers": workers > 0,
            "drop_last": False,
        }
        if workers > 0:
            options["prefetch_factor"] = prefetch
        loader = torch.utils.data.DataLoader(**options)
        iterator = iter(loader)
        waits = []
        rows = 0
        for trial in range(trials + 5):
            start = time.perf_counter()
            batch, iterator = next_batch(iterator, loader)
            elapsed = time.perf_counter() - start
            if trial >= 5:
                waits.append(elapsed)
                rows += int(batch["target_ids"].shape[0])
        result = {
            "num_workers": workers,
            "prefetch_factor": prefetch if workers > 0 else None,
            "pin_memory": pin_memory,
            "samples_per_second": rows / sum(waits),
            "median_batch_wait_ms": 1000.0 * statistics.median(waits),
            "p95_batch_wait_ms": 1000.0 * percentile(waits, 0.95),
        }
        results.append(result)
        del iterator, loader
        gc.collect()
    return max(results, key=lambda item: item["samples_per_second"]), results


def benchmark_batch(
    objective,
    optimizer,
    dataset,
    device,
    batch_size: int,
    loader_settings: dict[str, Any],
    warmup_steps: int,
    trials: int,
) -> dict[str, Any]:
    loader = loader_for(
        dataset,
        batch_size,
        int(loader_settings["num_workers"]),
        int(loader_settings["prefetch_factor"] or 1),
        shuffle=True,
        pin_memory=bool(loader_settings["pin_memory"]),
    )
    iterator = iter(loader)
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    waits: list[float] = []
    computes: list[float] = []
    rows = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for trial in range(warmup_steps + trials):
        wait_start = time.perf_counter()
        batch, iterator = next_batch(iterator, loader)
        batch = move_batch(batch, device)
        languages = dataset.languages_for_epoch(batch, 0)
        wait_elapsed = time.perf_counter() - wait_start

        torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = objective(batch, languages)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        compute_elapsed = time.perf_counter() - compute_start
        if trial >= warmup_steps:
            waits.append(wait_elapsed)
            computes.append(compute_elapsed)
            rows += int(batch["target_ids"].shape[0])
    total = sum(waits) + sum(computes)
    result = {
        "batch_size_per_gpu": batch_size,
        "trials": trials,
        "samples_per_second": rows / total,
        "median_step_ms": 1000.0 * statistics.median(computes),
        "p95_step_ms": 1000.0 * percentile(computes, 0.95),
        "median_data_wait_ms": 1000.0 * statistics.median(waits),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    del iterator, loader, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def gradient_audit(objective, optimizer, dataset, device, loader_settings):
    loader = loader_for(
        dataset,
        1,
        int(loader_settings["num_workers"]),
        int(loader_settings["prefetch_factor"] or 1),
        shuffle=True,
        pin_memory=bool(loader_settings["pin_memory"]),
    )
    batch = move_batch(next(iter(loader)), device)
    languages = dataset.languages_for_epoch(batch, 0)
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, _ = objective(batch, languages)
    loss.backward()

    modules = {
        "partfield": objective.planner.point_encoder,
        "point_projector": objective.planner.point_projector,
        "qwen": objective.planner.language_model,
        "embedding": objective.planner.language_model.get_input_embeddings(),
    }
    result = {}
    for name, module in modules.items():
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        gradients = [parameter.grad for parameter in parameters]
        result[name] = {
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "parameter_tensors": len(parameters),
            "tensors_with_gradient": sum(gradient is not None for gradient in gradients),
            "tensors_with_nonzero_gradient": sum(
                gradient is not None and bool(torch.count_nonzero(gradient).item())
                for gradient in gradients
            ),
            "finite": all(
                gradient is not None and bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ),
        }
        if not parameters or not result[name]["finite"]:
            raise RuntimeError(f"Gradient audit failed for {name}: {result[name]}")
        if result[name]["tensors_with_nonzero_gradient"] == 0:
            raise RuntimeError(f"Gradient audit found no nonzero gradient for {name}")

    downstream = (
        objective.model.observation_encoder,
        objective.model.contact_encoder,
        objective.model.smp,
        objective.model.physgraph,
    )
    result["downstream"] = {
        "trainable_parameters": sum(
            parameter.numel()
            for module in downstream
            if module is not None
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        "gradient_tensors": sum(
            parameter.grad is not None
            for module in downstream
            if module is not None
            for parameter in module.parameters()
        ),
    }
    if result["downstream"] != {"trainable_parameters": 0, "gradient_tensors": 0}:
        raise RuntimeError(f"Downstream freeze audit failed: {result['downstream']}")
    optimizer.zero_grad(set_to_none=True)
    del loader, batch, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def benchmark_validation(
    objective,
    dataset,
    device,
    batch_size,
    loader_settings,
    warmup_steps,
    trials,
):
    loader = loader_for(
        dataset,
        batch_size,
        int(loader_settings["num_workers"]),
        int(loader_settings["prefetch_factor"] or 1),
        shuffle=True,
        pin_memory=bool(loader_settings["pin_memory"]),
    )
    iterator = iter(loader)
    objective.eval()
    waits = []
    computes = []
    rows = 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for trial in range(warmup_steps + trials):
        wait_start = time.perf_counter()
        batch, iterator = next_batch(iterator, loader)
        batch = move_batch(batch, device)
        languages = dataset.languages_for_epoch(batch, 0)
        wait_elapsed = time.perf_counter() - wait_start
        torch.cuda.synchronize(device)
        compute_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            objective(batch, languages)
        torch.cuda.synchronize(device)
        compute_elapsed = time.perf_counter() - compute_start
        if trial >= warmup_steps:
            waits.append(wait_elapsed)
            computes.append(compute_elapsed)
            rows += int(batch["target_ids"].shape[0])
    result = {
        "batch_size_per_gpu": batch_size,
        "samples_per_second": rows / (sum(waits) + sum(computes)),
        "median_forward_ms": 1000.0 * statistics.median(computes),
        "median_data_wait_ms": 1000.0 * statistics.median(waits),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    del loader, iterator, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.warmup_steps < 1 or args.io_trials < 1:
        raise ValueError("trial and warmup counts must be positive")
    status = preflight_gpu(args.gpu_index, args.allow_shared_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    with resolve(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = planner_dataset(config, str(config["data"]["split"]))

    io_probe_batch = min(8, max(args.batch_sizes))
    best_io, io_results = benchmark_io(dataset, io_probe_batch, args.io_trials)

    model = DexCG.from_config(
        load_config(resolve(config["model_config"])), PROJECT_ROOT, torch_dtype=torch.float32
    )
    objective = ContactPlannerTrainingObjective(model).to(device)
    optimizer = optimizer_for(objective, config["optimizer"])
    gradients = gradient_audit(objective, optimizer, dataset, device, best_io)
    batch_results = []
    for batch_size in args.batch_sizes:
        try:
            result = benchmark_batch(
                objective,
                optimizer,
                dataset,
                device,
                batch_size,
                best_io,
                args.warmup_steps,
                args.trials,
            )
            batch_results.append(result)
            print(json.dumps({"event": "batch_result", **result}, sort_keys=True), flush=True)
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"event": "batch_oom", "batch_size_per_gpu": batch_size},
                    sort_keys=True,
                ),
                flush=True,
            )
            break
    if not batch_results:
        raise RuntimeError("Even batch size one ran out of GPU memory")
    best_batch = max(batch_results, key=lambda item: item["samples_per_second"])
    samples_per_second = float(best_batch["samples_per_second"])
    validation_dataset = planner_dataset(config, "validation")
    validation_result = benchmark_validation(
        objective,
        validation_dataset,
        device,
        int(best_batch["batch_size_per_gpu"]),
        best_io,
        args.warmup_steps,
        args.trials,
    )
    result = {
        "gpu": status,
        "dataset_samples": len(dataset),
        "trajectory_count": len(dataset.episodes),
        "selected": {
            "batch_size_per_gpu": int(best_batch["batch_size_per_gpu"]),
            "num_workers": int(best_io["num_workers"]),
            "prefetch_factor": best_io["prefetch_factor"],
            "pin_memory": bool(best_io["pin_memory"]),
            "samples_per_second_per_gpu": samples_per_second,
            "seconds_per_training_epoch_one_gpu": len(dataset) / samples_per_second,
            "seconds_per_100_training_epochs_one_gpu": 100 * len(dataset) / samples_per_second,
        },
        "io_results": io_results,
        "batch_results": batch_results,
        "gradient_audit": gradients,
        "validation_result": validation_result,
        "trainable_parameters": objective.trainable_parameter_summary(),
        "note": "No benchmark checkpoint or result file was retained.",
    }
    print(json.dumps({"event": "benchmark_complete", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
