#!/usr/bin/env python3
"""Trajectory-aware contact-planner fine-tuning entry point."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import hashlib
import subprocess
import sys
import time
import shutil
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dexcg.common.config import load_config
from dexcg.data import DexArtPlannerDataset
from dexcg.data.planner_training import LANGUAGE_FAMILIES, corrupt_previous_contacts, diagnostic_samples
from dexcg.evaluation import evaluate_seen_tasks
from dexcg.evaluation.planner import ContactMetricAccumulator, evaluate_contact_condition, composite_contact_score
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    MODEL_COORDINATE_CONTRACT,
    require_model_coordinates,
)
from dexcg.models.dexcg import DexCG
from dexcg.training import ContactPlannerTrainingObjective, DexCGTrainingObjective

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_BENCHMARK_SETTINGS = (
    "epochs",
    "batch_size_per_gpu",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_planner_balanced_dev.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size-per-gpu", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, help="Explicit smoke-test limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-four-gpus", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_training_config(path: str | Path, overrides: Mapping[str, Any] | None = None):
    with resolve(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for name, value in (overrides or {}).items():
        if value is not None:
            config[name] = value
    validate_training_config(config)
    return config


def validate_training_config(config: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_BENCHMARK_SETTINGS if config.get(name) is None]
    if missing:
        raise ValueError(
            "Planner training settings are still awaiting the GPU benchmark: " + ", ".join(missing)
        )
    for name in ("epochs", "batch_size_per_gpu"):
        if int(config[name]) < 1:
            raise ValueError(f"{name} must be at least one")
    if int(config.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(config["num_workers"]) > 0 and int(config["prefetch_factor"]) < 1:
        raise ValueError("prefetch_factor must be at least one when workers are enabled")
    if str(config["mixed_precision"]) != "bf16":
        raise ValueError("The audited planner training path currently supports bf16 only")
    if str(config["optimizer"]["scheduler"]) != "cosine":
        raise ValueError("The planner optimizer must use the audited cosine scheduler")
    optimizer = config["optimizer"]
    learning_rates = optimizer.get("learning_rates")
    if learning_rates is not None:
        expected = {"qwen", "point_projector"}
        if not config.get("freeze_partfield", False):
            expected.add("partfield")
        if set(learning_rates) not in (expected, expected | {"robot_state_projector"}):
            raise ValueError(f"optimizer.learning_rates must contain exactly {sorted(expected)}")
        if any(float(value) <= 0.0 for value in learning_rates.values()):
            raise ValueError("all planner learning rates must be positive")
    else:
        learning_rate = optimizer.get("learning_rate")
        if learning_rate is None or float(learning_rate) <= 0.0:
            raise ValueError("optimizer.learning_rate must be positive")

    validation = config["validation"]
    if bool(validation.get("enabled", False)) and config.get("data", {}).get("split") == "all":
        raise ValueError("Validation requires data.split=train to prevent trajectory leakage")
    if config.get("data", {}).get("use_previous_contact", False):
        raise ValueError("Stage-1 single-frame training forbids previous contact inputs")
    if bool(validation.get("enabled", False)) and int(validation["interval_epochs"]) < 1:
        raise ValueError("validation.interval_epochs must be positive when validation is enabled")
    families = tuple(validation.get("language_families", LANGUAGE_FAMILIES))
    if not families or any(family not in LANGUAGE_FAMILIES for family in families):
        raise ValueError(f"validation language families must be drawn from {LANGUAGE_FAMILIES}")
    if validation.get("selection_family", "deployment") not in families:
        raise ValueError("validation selection_family must be evaluated")
    early_stopping = validation.get("early_stopping", {})
    if bool(early_stopping.get("enabled", False)):
        if int(early_stopping.get("patience", 0)) < 1:
            raise ValueError("early-stopping patience must be positive")
        if float(early_stopping.get("min_delta", 0.0)) < 0.0:
            raise ValueError("early-stopping min_delta must be non-negative")

    history = config.get("history_corruption", {})
    drop = float(history.get("drop_probability", 0.0))
    perturb = float(history.get("perturb_probability", 0.0))
    if drop < 0.0 or perturb < 0.0 or drop + perturb > 1.0:
        raise ValueError(
            "history corruption probabilities must be non-negative and sum to at most one"
        )
    if perturb > 0.0 and int(history.get("max_bin_offset", 0)) < 1:
        raise ValueError("history max_bin_offset must be positive when perturbation is enabled")
    if int(config["checkpoint"].get("keep_validation_best", 1)) < 1:
        raise ValueError("checkpoint.keep_validation_best must be at least one")
    if validation.get("selection_metric") == "composite_contact":
        if not validation.get("enabled") or int(validation["interval_epochs"]) != 1:
            raise ValueError("Composite selection requires validation every epoch")
        if early_stopping.get("enabled", False):
            raise ValueError("Composite selection does not use CE early stopping")
        if int(config["checkpoint"].get("keep_generation_best", 0)) != 2:
            raise ValueError("Composite selection must retain the best two checkpoints")
        if float(validation.get("position_score_scale_m", 0)) <= 0:
            raise ValueError("Composite position score needs a positive scale")
        if int(validation.get("diagnostic_batch_size", 0)) != 3:
            raise ValueError("Use diagnostic batch 3 to match the audited six-condition protocol")


def eligible_gpus(histories, minimum_free_mib=25600):
    eligible = []
    for index, history in histories.items():
        if len(history) < 5:
            continue
        minimum_free = min(value[0] for value in history)
        if minimum_free > minimum_free_mib:
            eligible.append((-minimum_free, index))
    return [index for _, index in sorted(eligible)[:4]]


def wait_and_launch(config_path):
    """Monitor without CUDA allocation; launch one four-rank run after 60 stable seconds."""
    config = load_training_config(config_path)
    if not config.get("training_authorized"):
        raise PermissionError("Training not authorized")
    import socket

    monitor_lock = socket.socket(socket.AF_UNIX)
    monitor_lock.bind("\0dexcg-monitor-" + hashlib.sha256(str(resolve(config_path)).encode()).hexdigest())
    output = resolve(config["output_dir"])
    if output.exists() and any(p.name != "benchmark.json" for p in output.iterdir()) and not config.get("resume"):
        raise FileExistsError(output)
    benchmark_path = output / "benchmark.json"
    if config.get("freeze_partfield"):
        benchmark = json.loads(benchmark_path.read_text())
        if not benchmark.get("passed") or benchmark.get("config_sha256") != sha256_path(resolve(config_path)):
            raise RuntimeError("Frozen training requires a successful benchmark for this config")
        if max(benchmark.get("peak_allocated_gib", 100),
               benchmark.get("peak_reserved_gib", 100)) >= 23:
            raise RuntimeError("Benchmark leaves insufficient memory margin below 25 GiB")
    watched = [resolve(config_path), resolve(config["model_config"]), Path(__file__),
               PROJECT_ROOT / "src/dexcg/evaluation/planner.py",
               PROJECT_ROOT / "src/dexcg/data/planner_training.py"]
    watched.extend((PROJECT_ROOT / "src/dexcg").rglob("*.py"))
    if config["validation"].get("reference_result"):
        watched.append(resolve(config["validation"]["reference_result"]))
    digests = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
    if config.get("freeze_partfield"):
        watched.append(benchmark_path)
        source = resolve(load_config(resolve(config["model_config"])).contact_planner["checkpoint"]) / "model.safetensors"
        if benchmark["source_checkpoint_sha256"] != sha256_path(source):
            raise RuntimeError("Original checkpoint changed since preflight")
        for name, digest in benchmark["code_sha256"].items():
            if sha256_path(resolve(name)) != digest:
                raise RuntimeError(f"Code changed since preflight: {name}")
        digests[benchmark_path] = sha256_path(benchmark_path)
        digests[source] = sha256_path(source)
    histories = {}
    print(json.dumps({"event": "waiting_for_gpus", "pid": os.getpid(),
                      "minimum_free_mib_per_gpu": 25600, "utilization_filter": False,
                      "stable_seconds": 60, "poll_seconds": 15, "config": config}), flush=True)
    polls = 0
    while True:
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
               for path, digest in digests.items()):
            raise RuntimeError("Prepared code/config changed while waiting; recheck before launching")
        rows = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).splitlines()
        snapshot = {}
        for row in rows:
            index, free, utilization = map(int, row.split(","))
            snapshot[index] = {"free_mib": free, "utilization": utilization}
            histories.setdefault(index, deque(maxlen=5)).append((free, utilization))
        for index in set(histories) - set(snapshot):
            del histories[index]
        selected = eligible_gpus(histories)
        if polls % 4 == 0 or len(selected) == 4:
            print(json.dumps({"event": "gpu_poll", "time": time.time(),
                              "gpus": snapshot, "eligible": selected}), flush=True)
        if len(selected) == 4:
            environment = os.environ.copy()
            runtime_dir = Path(f"/tmp/dexcg-partfield-frozen-{os.getpid()}")
            runtime_dir.mkdir(mode=0o700)
            environment.update(CUDA_VISIBLE_DEVICES=",".join(map(str, selected)),
                               NCCL_IB_DISABLE="1", NCCL_P2P_DISABLE="1", NCCL_RAS_ENABLE="0",
                               TMPDIR=str(runtime_dir), PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                       "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                       str(Path(__file__).resolve()), "--config", str(resolve(config_path))]
            print(json.dumps({"event": "launching_training", "gpus": selected,
                              "time": time.time(), "command": command}), flush=True)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
            print(json.dumps({"event": "training_exited", "returncode": completed.returncode,
                              "time": time.time()}), flush=True)
            active = []
            marker = f"TMPDIR={runtime_dir}".encode()
            for process in Path("/proc").glob("[0-9]*"):
                try:
                    if marker in (process / "environ").read_bytes().split(b"\0"):
                        active.append(process.name)
                except (OSError, PermissionError):
                    continue
            if not active:
                shutil.rmtree(runtime_dir)
                print(json.dumps({"event": "runtime_cleanup", "path": str(runtime_dir)}), flush=True)
            else:
                print(json.dumps({"event": "runtime_cleanup_deferred", "active_pids": active}), flush=True)
            raise SystemExit(completed.returncode)
        polls += 1
        time.sleep(15)


def move_batch(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {name: move_batch(item, device) for name, item in value.items()}
    return value


def loader_for(
    dataset: DexArtPlannerDataset,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    sampler=None,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "drop_last": False,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    return DataLoader(**options)


def scheduler_for(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def optimizer_for(objective: ContactPlannerTrainingObjective, config: Mapping[str, Any]):
    if any(p.dtype != torch.float32 for p in objective.parameters() if p.requires_grad):
        raise ValueError("AdamW requires FP32 trainable weights; use bf16 autocast for computation")
    learning_rates = config.get("learning_rates")
    if learning_rates is None:
        parameters = [parameter for parameter in objective.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("Planner objective has no trainable parameters")
        groups = [{"name": "planner", "params": parameters}]
        default_lr = float(config["learning_rate"])
    else:
        components = {
            "partfield": objective.planner.point_encoder,
            "point_projector": objective.planner.point_projector,
            "qwen": objective.planner.language_model,
        }
        if getattr(objective.planner, "robot_state_projector", None) is not None:
            components["robot_state_projector"] = objective.planner.robot_state_projector
        embedding_ids = {
            id(parameter)
            for parameter in objective.planner.language_model.get_input_embeddings().parameters()
        }
        groups = []
        assigned: set[int] = set()
        for component_name, module in components.items():
            decay_parameters = []
            no_decay_parameters = []
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in assigned:
                    raise RuntimeError(
                        f"planner optimizer parameter assigned twice: {component_name}.{name}"
                    )
                assigned.add(parameter_id)
                no_decay = parameter.ndim < 2 or parameter_id in embedding_ids
                target = no_decay_parameters if no_decay else decay_parameters
                target.append(parameter)
            for decay_name, parameters, weight_decay in (
                ("decay", decay_parameters, float(config["weight_decay"])),
                ("no_decay", no_decay_parameters, 0.0),
            ):
                if parameters:
                    groups.append(
                        {
                            "name": f"{component_name}.{decay_name}",
                            "params": parameters,
                            "lr": float(learning_rates[component_name]),
                            "weight_decay": weight_decay,
                        }
                    )
        expected = {
            id(parameter) for parameter in objective.parameters() if parameter.requires_grad
        }
        if assigned != expected:
            raise RuntimeError(
                f"planner optimizer coverage mismatch: missing={len(expected - assigned)}, "
                f"unexpected={len(assigned - expected)}"
            )
        default_lr = 1.0
    return torch.optim.AdamW(
        groups,
        lr=default_lr,
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )


def optimizer_learning_rates(optimizer) -> dict[str, float]:
    return {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}


def update_early_stopping(
    loss: float,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    bad_epochs: int,
    min_delta: float,
) -> tuple[float, int, int]:
    if loss < best_loss - min_delta:
        return loss, epoch, 0
    return best_loss, best_epoch, bad_epochs + 1


def planner_checkpoint_metadata(objective):
    if not getattr(objective, "freeze_partfield", False):
        return {}
    return {"freeze_partfield": True, "planner_state_scope": "full",
            "frozen_partfield_sha256": module_digest(objective.planner.point_encoder),
            "initialization": getattr(objective, "initialization_metadata", {})}


def compact_planner_state(objective: ContactPlannerTrainingObjective):
    if getattr(objective, "freeze_partfield", False):
        return objective.planner.state_dict()
    names = {
        name for name, parameter in objective.planner.named_parameters() if parameter.requires_grad
    }
    return {name: value for name, value in objective.planner.state_dict().items() if name in names}


def load_compact_planner_state(
    objective: ContactPlannerTrainingObjective, state: Mapping[str, torch.Tensor]
) -> None:
    full = objective.planner.state_dict()
    expected = set(full) if set(state) == set(full) else set(compact_planner_state(objective))
    missing = expected.difference(state)
    unexpected = set(state).difference(expected)
    if missing or unexpected:
        raise RuntimeError(
            f"invalid planner checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    objective.planner.load_state_dict(state, strict=False)
    objective.planner.language_model.tie_weights()


def load_policy_checkpoint(
    model: DexCG,
    checkpoint_path: Path,
    expected_coordinate_contract: str = CONTACT_COORDINATE_CONTRACT,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require_model_coordinates(checkpoint)
    if checkpoint.get("contact_coordinate_contract") != expected_coordinate_contract:
        raise RuntimeError(
            f"Policy checkpoint {checkpoint_path} uses coordinate contract "
            f"{checkpoint.get('contact_coordinate_contract')!r}; expected "
            f"{expected_coordinate_contract!r}"
        )
    state = checkpoint.get("model")
    if not isinstance(state, dict) or "state_min" not in state or "state_max" not in state:
        raise RuntimeError(f"Policy checkpoint {checkpoint_path} has no normalization state")
    source = {
        name.removeprefix("model."): value
        for name, value in state.items()
        if name.startswith("model.") and not name.startswith("model.contact_planner.")
    }
    expected = {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("contact_planner.")
    }
    missing = set(expected).difference(source)
    unexpected = set(source).difference(expected)
    shape_mismatch = {
        name: (tuple(source[name].shape), tuple(expected[name].shape))
        for name in set(source).intersection(expected)
        if source[name].shape != expected[name].shape
    }
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Policy checkpoint does not exactly match the frozen downstream model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"shape_mismatch={shape_mismatch}"
        )
    model.load_state_dict(source, strict=False)
    return checkpoint, state["state_min"], state["state_max"]


def save_atomic(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def save_latest(
    path: Path,
    objective: ContactPlannerTrainingObjective,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    save_atomic(
        {
            "format": "dexcg.contact_planner.v1",
            **planner_checkpoint_metadata(objective),
            "epoch": epoch,
            "global_step": global_step,
            "planner": compact_planner_state(objective),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
            "training_state": dict(training_state or {}),
        },
        path,
    )


def save_validation_checkpoint(
    checkpoint_dir: Path,
    objective: ContactPlannerTrainingObjective,
    epoch: int,
    loss: float,
    keep: int,
) -> None:
    pattern = re.compile(r"epoch=(\d+)-deployment_loss=([0-9.]+)\.ckpt")
    existing = []
    for path in checkpoint_dir.glob("epoch=*-deployment_loss=*.ckpt"):
        match = pattern.fullmatch(path.name)
        if match:
            existing.append((float(match.group(2)), path))
    if len(existing) >= keep and loss >= max(item[0] for item in existing):
        return
    path = checkpoint_dir / f"epoch={epoch:04d}-deployment_loss={loss:.6f}.ckpt"
    save_atomic(
        {
            "format": "dexcg.contact_planner.v1",
            **planner_checkpoint_metadata(objective),
            "epoch": epoch,
            "deployment_loss": loss,
            "planner": compact_planner_state(objective),
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        },
        path,
    )
    existing.append((loss, path))
    for _, stale in sorted(existing, key=lambda item: item[0])[keep:]:
        stale.unlink(missing_ok=True)


def save_top_checkpoint(
    checkpoint_dir: Path,
    objective: ContactPlannerTrainingObjective,
    epoch: int,
    score: float,
    keep: int,
) -> None:
    pattern = re.compile(r"epoch=(\d+)-seen_score=([0-9.]+)\.ckpt")
    existing = []
    for path in checkpoint_dir.glob("epoch=*-seen_score=*.ckpt"):
        match = pattern.fullmatch(path.name)
        if match:
            existing.append((float(match.group(2)), path))
    if len(existing) >= keep and score <= min(item[0] for item in existing):
        return
    path = checkpoint_dir / f"epoch={epoch:04d}-seen_score={score:.3f}.ckpt"
    save_atomic(
        {
            "format": "dexcg.contact_planner.v1",
            **planner_checkpoint_metadata(objective),
            "epoch": epoch,
            "seen_score": score,
            "planner": compact_planner_state(objective),
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        },
        path,
    )
    existing.append((score, path))
    for _, stale in sorted(existing, key=lambda item: item[0], reverse=True)[keep:]:
        stale.unlink(missing_ok=True)


def save_generation_checkpoint(checkpoint_dir, objective, epoch, selection, keep=2):
    score = round(float(selection["score"]), 12)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid generation checkpoint score")
    pattern = re.compile(r"epoch=(\d+)-generation_score=([0-9.]+)\.ckpt")
    existing = []
    for path in checkpoint_dir.glob("epoch=*-generation_score=*.ckpt"):
        match = pattern.fullmatch(path.name)
        if match:
            existing.append((float(match[2]), int(match[1]), path))
    path = checkpoint_dir / f"epoch={epoch:04d}-generation_score={score:.12f}.ckpt"
    ranked = sorted(existing + [(score, epoch, path)], key=lambda item: (-item[0], item[1]))
    if path not in [item[2] for item in ranked[:keep]]:
        return
    save_atomic({"format": "dexcg.contact_planner.v1",
            **planner_checkpoint_metadata(objective), "epoch": epoch,
                 "generation_selection": selection, "planner": compact_planner_state(objective),
                 "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
                 "model_coordinate_contract": MODEL_COORDINATE_CONTRACT}, path)
    for _, _, stale in ranked[keep:]:
        stale.unlink(missing_ok=True)


def append_json(path: Path, value: Mapping[str, Any]) -> None:
    line = json.dumps(dict(value), sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def reduce_sums(values: Mapping[str, float], device: torch.device):
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


@torch.no_grad()
def validate(
    objective: ContactPlannerTrainingObjective,
    dataset: DexArtPlannerDataset,
    loader: DataLoader,
    language_family: str,
    device: torch.device,
) -> dict[str, float]:
    objective.eval()
    sums = {"weighted_loss": 0.0, "samples": 0.0, "correct": 0.0, "tokens": 0.0}
    for batch in loader:
        batch = move_batch(batch, device)
        languages = dataset.languages_for_family(batch, language_family)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, metrics = objective(batch, languages)
        rows = int(batch["target_ids"].shape[0])
        sums["weighted_loss"] += float(loss.item()) * rows
        sums["samples"] += rows
        sums["correct"] += float(metrics["correct"].item())
        sums["tokens"] += float(metrics["count"].item())
    sums = reduce_sums(sums, device)
    return {
        "loss": sums["weighted_loss"] / max(sums["samples"], 1.0),
        "token_accuracy": sums["correct"] / max(sums["tokens"], 1.0),
        "samples": sums["samples"],
        "tokens": sums["tokens"],
    }


def planner_dataset(config: Mapping[str, Any], split: str) -> DexArtPlannerDataset:
    data = config["data"]
    return DexArtPlannerDataset(
        [resolve(path) for path in data["paths"]],
        split=split,
        deployment_instructions=data["deployment_instructions"],
        split_seed=int(data["split_seed"]),
        history_interval=int(data["history_interval"]),
        use_previous_contact=bool(data.get("use_previous_contact", True)),
        use_robot_state=bool(
            load_config(resolve(config["model_config"])).contact_planner.get(
                "use_robot_state", False
            )
        ),
    )


def generation_sample_indices(dataset, samples_per_task=0, samples_per_trajectory=0):
    """Select fixed frames before inference, without looking at predictions."""
    if samples_per_trajectory:
        lookup = {(sample.episode.key, sample.step): index
                  for index, sample in enumerate(dataset.samples)}
        return [lookup[(episode.key, episode.valid_steps[offset])]
                for episode in dataset.episodes
                for offset in np.linspace(0, len(episode.valid_steps) - 1,
                                          min(samples_per_trajectory, len(episode.valid_steps)),
                                          dtype=int)]
    selected = []
    for task_index in range(len(dataset.tasks)):
        indices = [
            i for i, sample in enumerate(dataset.samples) if sample.episode.task_index == task_index
        ]
        selected.extend([
            indices[i]
            for i in np.linspace(
                0, len(indices) - 1, min(samples_per_task, len(indices)), dtype=int
            )
        ])
    return selected


@torch.no_grad()
def validate_generation(objective, dataset, device, samples_per_task=0,
                        samples_per_trajectory=0, batch_size=1):
    """Fixed held-out frames, generated with deployment text and no target prefix."""
    objective.eval()
    accumulator = ContactMetricAccumulator(objective.planner.contact_tokenizer)
    selected = generation_sample_indices(dataset, samples_per_task, samples_per_trajectory)
    for start in range(0, len(selected), batch_size):
        batch = move_batch(torch.utils.data.default_collate(
            [dataset[index] for index in selected[start:start + batch_size]]), device)
        robot = (
            {key: batch[key] for key in ("robot_qpos", "palm_pose_robot_base")}
            if objective.planner.robot_state_projector is not None
            else {}
        )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            plan = objective.planner.plan(
                batch["point_cloud"], batch["deployment_language"], **robot
            )
        accumulator.update(
            plan.token_ids, plan.attention_mask, batch["target_ids"], batch["target_mask"],
            {"task": batch["task"]},
        )
    return accumulator.result()


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def planner_diagnostic(objective, train_dataset, validation_dataset, diagnostics, device, config):
    objective.eval()
    results = {}
    for split, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        results[split] = {}
        for family in LANGUAGE_FAMILIES:
            results[split][family] = evaluate_contact_condition(
                objective.planner, dataset, diagnostics[split][0], family, device,
                int(config["diagnostic_batch_size"]))
            print(json.dumps({"event": "fixed_diagnostic", "split": split, "family": family}), flush=True)
    return {"sample_manifest": {split: values[1] for split, values in diagnostics.items()},
            "results": results, "selection": composite_contact_score(
                results["validation"], float(config["position_score_scale_m"]))}


def main() -> None:
    args = parse_args()
    if args.wait_for_four_gpus:
        wait_and_launch(args.config)
        return
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be at least one")
    overrides = {
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "output_dir": args.output_dir,
    }
    config = load_training_config(args.config, overrides)
    if config.get("training_authorized") is False:
        raise PermissionError(
            "This configuration is pending discussion; formal training is not authorized."
        )
    if args.resume:
        config["resume"] = True

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=device)
        evaluation_group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))
    else:
        evaluation_group = None

    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    train_dataset = planner_dataset(config, str(config["data"]["split"]))
    validation_dataset = (
        planner_dataset(config, "validation") if bool(config["validation"]["enabled"]) else None
    )
    composite_selection = config["validation"].get("selection_metric") == "composite_contact"
    diagnostics = ({split: diagnostic_samples(dataset) for split, dataset in
                    (("train", train_dataset), ("validation", validation_dataset))}
                   if composite_selection else {})
    reference_score = None
    if composite_selection and config["validation"].get("reference_result"):
        reference = json.loads(resolve(config["validation"]["reference_result"]).read_text())
        if reference["sample_manifest"] != {split: values[1] for split, values in diagnostics.items()}:
            raise ValueError("Reference and current diagnostic samples differ")
        reference_score = composite_contact_score(
            reference["results"]["validation"], config["validation"]["position_score_scale_m"])["score"]
    sampler = (
        DistributedSampler(train_dataset, world_size, rank, shuffle=True, seed=seed)
        if world_size > 1
        else None
    )
    validation_sampler = (
        list(range(rank, len(validation_dataset), world_size))
        if world_size > 1 and validation_dataset is not None
        else None
    )
    loader = loader_for(
        train_dataset,
        int(config["batch_size_per_gpu"]),
        int(config["num_workers"]),
        int(config["prefetch_factor"]),
        sampler=sampler,
        shuffle=sampler is None,
        pin_memory=bool(config["pin_memory"]),
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = loader_for(
            validation_dataset,
            int(config["batch_size_per_gpu"]),
            int(config["num_workers"]),
            int(config["prefetch_factor"]),
            sampler=validation_sampler,
            pin_memory=bool(config["pin_memory"]),
        )

    model = DexCG.from_config(
        load_config(resolve(config["model_config"])), PROJECT_ROOT, torch_dtype=torch.float32
    )
    policy_checkpoint = None
    evaluation_objective = None
    if config.get("policy_checkpoint"):
        policy_checkpoint, state_min, state_max = load_policy_checkpoint(
            model, resolve(config["policy_checkpoint"])
        )
        evaluation_objective = DexCGTrainingObjective(
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
        )
    elif config["evaluation"].get("enabled", True):
        raise ValueError("Closed-loop evaluation requires a robot-base policy checkpoint")
    objective = ContactPlannerTrainingObjective(
        model, freeze_partfield=bool(config.get("freeze_partfield", False))
    ).to(device)
    if evaluation_objective is not None:
        evaluation_objective.to(device)
    objective.initialization_metadata = {
        "source_checkpoint": load_config(resolve(config["model_config"])).contact_planner["checkpoint"],
        "source_checkpoint_sha256": sha256_path(resolve(load_config(resolve(config["model_config"])).contact_planner["checkpoint"]) / "model.safetensors"),
        "seed": int(config["seed"]),
    }
    optimizer = optimizer_for(objective, config["optimizer"])
    accumulation = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = scheduler_for(optimizer, int(config["optimizer"]["warmup_steps"]), total_updates)

    output_dir = resolve(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    frozen_digest = module_digest(objective.planner.point_encoder) if objective.freeze_partfield else None
    latest_path = checkpoint_dir / "latest.ckpt"
    if config["resume"] and not latest_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {latest_path}")
    model_configuration = load_config(resolve(config["model_config"]))
    with resolve(config["model_config"]).open(encoding="utf-8") as stream:
        model_snapshot = yaml.safe_load(stream)
    if config["resume"]:
        with (output_dir / "config.yaml").open(encoding="utf-8") as stream:
            previous_config = yaml.safe_load(stream)
        for key in (
            "freeze_partfield",
            "seed",
            "epochs",
            "data",
            "optimizer",
            "batch_size_per_gpu",
            "gradient_accumulation_steps",
            "validation",
        ):
            if previous_config.get(key) != config.get(key):
                raise ValueError(f"Resume would change {key}; start a separate run")
        if previous_config.get("model_snapshot") != model_snapshot:
            raise ValueError("Resume model architecture/input contract changed")
        if previous_config.get("runtime", {}).get("world_size") != world_size:
            raise ValueError("Resume world size changed")
    if bool(model_configuration.contact_planner.get("use_previous_contact", False)) != bool(
        config["data"].get("use_previous_contact", False)
    ):
        raise ValueError("Model and dataset previous-contact settings disagree")
    if rank == 0:
        if output_dir.exists() and any(p.name != "benchmark.json" for p in output_dir.iterdir()) and not bool(config["resume"]):
            raise FileExistsError(
                f"Refusing to overwrite nonempty planner output directory: {output_dir}"
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if config["evaluation"].get("enabled", False):
            (output_dir / "evaluation" / "videos").mkdir(parents=True, exist_ok=True)
        saved_config = copy.deepcopy(config)
        saved_config["model_snapshot"] = model_snapshot
        saved_config["source_checkpoint_sha256"] = sha256_path(
            resolve(model_configuration.contact_planner["checkpoint"]) / "model.safetensors"
        )
        saved_config["frozen_partfield_sha256"] = frozen_digest
        saved_config["runtime"] = {
            "world_size": world_size,
            "global_batch_size": int(config["batch_size_per_gpu"]) * world_size * accumulation,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "trainable_parameters": objective.trainable_parameter_summary(),
            "policy_checkpoint_epoch": int(policy_checkpoint["epoch"])
            if policy_checkpoint
            else None,
        }
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, sort_keys=False)
        with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as stream:
            manifests = {"train": train_dataset.split_manifest()}
            if validation_dataset is not None:
                manifests["validation"] = validation_dataset.split_manifest()
            json.dump(manifests, stream, indent=2, sort_keys=True)
            stream.write("\n")
        audits = {"training": train_dataset.audit()}
        if validation_dataset is not None:
            audits["validation"] = validation_dataset.audit()
        if composite_selection:
            audits["diagnostic_samples"] = {split: values[1] for split, values in diagnostics.items()}
        with (output_dir / "data_audit.json").open("w", encoding="utf-8") as stream:
            json.dump(audits, stream, indent=2, sort_keys=True)
            stream.write("\n")
    if world_size > 1:
        dist.barrier()

    start_epoch = 1
    global_step = 0
    best_validation_loss = math.inf
    best_validation_epoch = 0
    bad_validation_epochs = 0
    if bool(config["resume"]) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        require_model_coordinates(checkpoint)
        if checkpoint.get("contact_coordinate_contract") != CONTACT_COORDINATE_CONTRACT:
            raise RuntimeError("Planner checkpoint coordinate contract is incompatible")
        load_compact_planner_state(objective, checkpoint["planner"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        training_state = checkpoint.get("training_state", {})
        best_validation_loss = float(training_state.get("best_validation_loss", math.inf))
        best_validation_epoch = int(training_state.get("best_validation_epoch", 0))
        bad_validation_epochs = int(training_state.get("bad_validation_epochs", 0))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])

    distributed = (
        DistributedDataParallel(
            objective,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        if world_size > 1
        else objective
    )
    if objective.freeze_partfield and not config["resume"]:
        # Exercise all four DDP reducers before baseline evaluation or formal updates.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        probe = move_batch(next(iter(loader)), device)
        distributed.train()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            probe_loss, _ = distributed(probe, train_dataset.languages_for_epoch(probe, 0))
        if not torch.isfinite(probe_loss):
            raise FloatingPointError("DDP preflight produced nonfinite loss")
        probe_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("DDP preflight produced nonfinite gradients")
        if any(p.grad is not None for p in objective.planner.point_encoder.parameters()):
            raise RuntimeError("Frozen PartField received gradients")
        peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
        if world_size > 1:
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        reserved = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
        if world_size > 1:
            dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
        if max(peak.item(), reserved.item()) >= 23:
            raise RuntimeError("DDP preflight exceeds the 23 GiB allocation/reservation limit")
        optimizer.zero_grad(set_to_none=True)
        del probe, probe_loss
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        if rank == 0:
            benchmark_path = output_dir / "benchmark.json"
            report = json.loads(benchmark_path.read_text())
            report["ddp_preflight"] = {"world_size": world_size, "passed": True,
                                       "peak_allocated_gib": peak.item(),
                                       "peak_reserved_gib": reserved.item(), "optimizer_updates": 0}
            benchmark_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"event": "ddp_preflight", **report["ddp_preflight"]}), flush=True)
    optimizer.zero_grad(set_to_none=True)
    history_config = config.get("history_corruption", {})
    validation_config = config["validation"]
    validation_families = tuple(validation_config.get("language_families", LANGUAGE_FAMILIES))
    selection_family = str(validation_config.get("selection_family", "deployment"))
    early_stopping = validation_config.get("early_stopping", {})
    stop = bool(early_stopping.get("enabled", False)) and bad_validation_epochs >= int(
        early_stopping["patience"]
    )
    if stop and rank == 0:
        append_json(
            output_dir / "train.log",
            {
                "event": "resume_already_early_stopped",
                "best_validation_epoch": best_validation_epoch,
                "best_validation_loss": best_validation_loss,
                "bad_validation_epochs": bad_validation_epochs,
            },
        )
    if objective.freeze_partfield and not config["resume"]:
        if rank == 0:
            baseline = planner_diagnostic(objective, train_dataset, validation_dataset, diagnostics, device, validation_config)
            baseline["epoch"] = 0
            baseline["frozen_partfield_sha256"] = frozen_digest
            (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        if world_size > 1:
            dist.barrier()

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        distributed.train()
        sums = {
            "weighted_loss": 0.0,
            "samples": 0.0,
            "correct": 0.0,
            "tokens": 0.0,
            "history_absent": 0.0,
            "history_clean": 0.0,
            "history_dropped": 0.0,
            "history_perturbed": 0.0,
            "history_changed_tokens": 0.0,
        }
        for batch_index, batch in enumerate(loader, start=1):
            languages = train_dataset.languages_for_epoch(batch, epoch)
            batch, history_stats = corrupt_previous_contacts(
                batch,
                objective.planner.contact_tokenizer.position_id_to_bin,
                objective.planner.contact_tokenizer.position_token_ids,
                seed=int(history_config.get("seed", seed)),
                epoch=epoch,
                drop_probability=float(history_config.get("drop_probability", 0.0)),
                perturb_probability=float(history_config.get("perturb_probability", 0.0)),
                max_bin_offset=int(history_config.get("max_bin_offset", 1)),
            )
            for name, value in history_stats.items():
                sums[f"history_{name}"] += float(value)
            batch = move_batch(batch, device)
            sync = batch_index % accumulation == 0 or batch_index == len(loader)
            context = distributed.no_sync() if world_size > 1 and not sync else torch.enable_grad()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metrics = distributed(batch, languages)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Nonfinite loss at epoch {epoch}, batch {batch_index}"
                        )
                    window_start = ((batch_index - 1) // accumulation) * accumulation
                    window_size = min(accumulation, len(loader) - window_start)
                    scaled_loss = loss / window_size
                scaled_loss.backward()
            rows = int(batch["target_ids"].shape[0])
            sums["weighted_loss"] += float(loss.detach().item()) * rows
            sums["samples"] += rows
            sums["correct"] += float(metrics["correct"].item())
            sums["tokens"] += float(metrics["count"].item())
            if sync:
                torch.nn.utils.clip_grad_norm_(
                    objective.parameters(),
                    float(config["gradient_clip_norm"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if rank == 0 and global_step % 100 == 0:
                    append_json(
                        output_dir / "train.log",
                        {
                            "event": "progress",
                            "epoch": epoch,
                            "global_step": global_step,
                            "batch_loss": float(loss.detach()),
                        },
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break

        reduced = reduce_sums(sums, device)
        if rank == 0:
            summary = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rates": optimizer_learning_rates(optimizer),
                "loss": reduced["weighted_loss"] / max(reduced["samples"], 1.0),
                "token_accuracy": reduced["correct"] / max(reduced["tokens"], 1.0),
                "samples": reduced["samples"],
                "tokens": reduced["tokens"],
                "history": {
                    name.removeprefix("history_"): int(value)
                    for name, value in reduced.items()
                    if name.startswith("history_")
                },
            }
            append_json(output_dir / "train.log", summary)

        validation_due = (
            not stop
            and validation_loader is not None
            and epoch % int(validation_config["interval_epochs"]) == 0
        )
        if validation_due:
            family_results = {
                family: validate(
                    objective,
                    validation_dataset,
                    validation_loader,
                    family,
                    device,
                )
                for family in validation_families
            }
            result = {"epoch": epoch, "families": family_results}
            selection_loss = float(family_results[selection_family]["loss"])
            if not math.isfinite(selection_loss):
                raise FloatingPointError("Nonfinite validation loss")
            min_delta = float(early_stopping.get("min_delta", 0.0))
            (
                best_validation_loss,
                best_validation_epoch,
                bad_validation_epochs,
            ) = update_early_stopping(
                selection_loss,
                epoch,
                best_validation_loss,
                best_validation_epoch,
                bad_validation_epochs,
                min_delta,
            )
            result["selection"] = {
                "family": selection_family,
                "loss": selection_loss,
                "best_loss": best_validation_loss,
                "best_epoch": best_validation_epoch,
                "bad_epochs": bad_validation_epochs,
            }
        if rank == 0 and validation_due:
            if composite_selection:
                condition_results = {}
                for split, dataset in (("train", train_dataset), ("validation", validation_dataset)):
                    condition_results[split] = {}
                    for family in LANGUAGE_FAMILIES:
                        condition_results[split][family] = evaluate_contact_condition(
                            objective.planner, dataset, diagnostics[split][0], family, device,
                            int(validation_config["diagnostic_batch_size"]))
                        print(json.dumps({"event": "epoch_diagnostic_condition", "epoch": epoch,
                                          "split": split, "family": family,
                                          "teacher_forcing": condition_results[split][family]["teacher_forcing"],
                                          "generation": condition_results[split][family]["generation"]["overall"]}),
                              flush=True)
                result["diagnostics"] = condition_results
                result["loss_monitor"] = result.pop("selection")
                result["selection"] = {
                    "metric": "composite_contact", "split": "validation",
                    "formula": "mean_languages((token_accuracy + link_f1 + 1/(1+xyz_error/scale))/3)",
                    **composite_contact_score(condition_results["validation"],
                                               float(validation_config["position_score_scale_m"]))}
                if reference_score is not None:
                    result["selection"].update(reference_epoch3_score=reference_score,
                                               reaches_reference=result["selection"]["score"] >= reference_score)
                append_json(output_dir / "validation.log", result)
                save_generation_checkpoint(checkpoint_dir, objective, epoch, result["selection"],
                                           int(config["checkpoint"]["keep_generation_best"]))
            else:
                generation_count = int(validation_config.get("generation_samples_per_task", 0))
                if generation_count:
                    result["generation"] = validate_generation(
                        objective, validation_dataset, device, generation_count
                    )
                append_json(output_dir / "validation.log", result)
                save_validation_checkpoint(
                    checkpoint_dir, objective, epoch, selection_loss,
                    int(config["checkpoint"].get("keep_validation_best", 2)),
                )
        if validation_due and bool(early_stopping.get("enabled", False)):
            stop = bad_validation_epochs >= int(early_stopping["patience"])

        checkpoint_due = (
            epoch % int(config["checkpoint"]["interval_epochs"]) == 0
            or epoch == int(config["epochs"])
            or stop
        )
        if rank == 0 and checkpoint_due:
            save_latest(
                latest_path,
                objective,
                optimizer,
                scheduler,
                epoch,
                global_step,
                {
                    "best_validation_loss": best_validation_loss,
                    "best_validation_epoch": best_validation_epoch,
                    "bad_validation_epochs": bad_validation_epochs,
                },
            )
        if world_size > 1:
            dist.barrier()

        if frozen_digest is not None and module_digest(objective.planner.point_encoder) != frozen_digest:
            raise RuntimeError("Frozen PartField state changed during training")
        evaluation = config["evaluation"]
        evaluation_due = (
            not stop
            and bool(evaluation.get("enabled", True))
            and epoch >= int(evaluation["first_epoch"])
            and epoch % int(evaluation["interval_epochs"]) == 0
        )
        if evaluation_due:
            evaluation_objective.eval()
            result = evaluate_seen_tasks(
                evaluation_objective,
                config,
                epoch,
                output_dir,
                device,
                rank,
                world_size,
                evaluation_group,
            )
            if rank == 0:
                score = float(result["mean_success_rate"])
                save_top_checkpoint(
                    checkpoint_dir,
                    objective,
                    epoch,
                    score,
                    int(config["checkpoint"]["keep_top"]),
                )
                append_json(
                    output_dir / "train.log",
                    {"epoch": epoch, "seen_score": score, "event": "closed_loop_evaluation"},
                )
            if world_size > 1:
                dist.barrier()
        if stop:
            break

    if objective.freeze_partfield and not args.max_steps:
        if rank == 0:
            candidates = list(checkpoint_dir.glob("epoch=*-generation_score=*.ckpt"))
            if not candidates:
                raise RuntimeError("No generation checkpoint was saved")
            best = max(candidates, key=lambda p: float(p.stem.split("generation_score=")[1]))
            selected = torch.load(best, map_location=device, weights_only=True)
            load_compact_planner_state(objective, selected["planner"])
            final = planner_diagnostic(objective, train_dataset, validation_dataset, diagnostics, device, validation_config)
            final.update(checkpoint=str(best), epoch=selected["epoch"],
                         frozen_partfield_sha256=module_digest(objective.planner.point_encoder))
            if final["frozen_partfield_sha256"] != frozen_digest:
                raise RuntimeError("Best checkpoint changed frozen PartField")
            (output_dir / "final_evaluation.json").write_text(json.dumps(final, indent=2) + "\n")
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
