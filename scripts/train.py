#!/usr/bin/env python3
"""Distributed dexCG training entry point."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dexcg.common.config import load_config
from dexcg.data import DexArtTrainingDataset
from dexcg.evaluation import evaluate_seen_tasks
from dexcg.models.dexcg import DexCG
from dexcg.training import DexCGTrainingObjective

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def move_batch(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    return batch


def teacher_forcing_probability(epoch: int, config: dict[str, Any]) -> float:
    full = int(config["full_until_epoch"])
    end = int(config["decay_until_epoch"])
    final = float(config["final_probability"])
    if epoch <= full:
        return 1.0
    if epoch >= end:
        return final
    progress = (epoch - full) / (end - full)
    return 1.0 + progress * (final - 1.0)


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.parameter_names = tuple(
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        )
        self.buffer_names = tuple(
            name
            for name, _ in module.named_buffers()
            if not name.startswith("model.contact_planner.")
        )
        self.state_names = frozenset((*self.parameter_names, *self.buffer_names))
        self.module = copy.deepcopy(module).requires_grad_(False).eval()
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        source_parameters = dict(source.named_parameters())
        target_parameters = dict(self.module.named_parameters())
        for name in self.parameter_names:
            target_parameters[name].lerp_(
                source_parameters[name].detach(), 1.0 - self.decay
            )
        source_buffers = dict(source.named_buffers())
        target_buffers = dict(self.module.named_buffers())
        for name in self.buffer_names:
            target_buffers[name].copy_(source_buffers[name])

    def state_dict(self):
        return {
            name: value
            for name, value in self.module.state_dict().items()
            if name in self.state_names
        }

    def load_state_dict(self, state_dict) -> None:
        _load_compact_state_dict(self.module, state_dict, self.state_names)


def compact_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    parameter_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    buffer_names = {
        name
        for name, _ in module.named_buffers()
        if not name.startswith("model.contact_planner.")
    }
    names = parameter_names | buffer_names
    return {name: value for name, value in module.state_dict().items() if name in names}


def _load_compact_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    expected_names: set[str] | frozenset[str],
) -> None:
    missing = set(expected_names) - set(state_dict)
    unexpected = set(state_dict) - set(expected_names)
    if missing or unexpected:
        raise RuntimeError(
            f"invalid compact checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    module.load_state_dict(state_dict, strict=False)


def optimizer_for(model: DexCGTrainingObjective, config: dict[str, Any]):
    planner_ids = {
        id(parameter)
        for parameter in model.model.contact_planner.parameters()
        if parameter.requires_grad
    }
    planner = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in planner_ids
    ]
    policy = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in planner_ids
    ]
    groups = [{"params": policy, "lr": float(config["learning_rate"])}]
    if planner:
        groups.append(
            {"params": planner, "lr": float(config["planner_learning_rate"])}
        )
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in config["betas"]),
        weight_decay=float(config["weight_decay"]),
    )


def scheduler_for(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def save_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_latest(
    path: Path,
    objective: DexCGTrainingObjective,
    ema: EMA,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
) -> None:
    save_atomic(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": compact_state_dict(objective),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        path,
    )


def save_top_checkpoint(
    checkpoint_dir: Path,
    objective: DexCGTrainingObjective,
    ema: EMA,
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
        {"epoch": epoch, "seen_score": score, "model": ema.state_dict()},
        path,
    )
    existing.append((score, path))
    for _, stale in sorted(existing, key=lambda item: item[0], reverse=True)[keep:]:
        stale.unlink(missing_ok=True)


def reduce_epoch_metrics(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


def main() -> None:
    args = parse_args()
    with resolve(args.config).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.batch_size_per_gpu is not None:
        config["batch_size_per_gpu"] = args.batch_size_per_gpu
    if args.no_resume:
        config["resume"] = False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        raise RuntimeError("dexCG training requires torchrun with at least two GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=device)

    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    output_dir = resolve(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation" / "videos").mkdir(parents=True, exist_ok=True)
        saved_config = copy.deepcopy(config)
        saved_config["runtime"] = {
            "world_size": world_size,
            "global_batch_size": int(config["batch_size_per_gpu"]) * world_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(saved_config, stream, sort_keys=False)
    dist.barrier()

    dataset = DexArtTrainingDataset(
        [resolve(path) for path in config["data"]["paths"]],
        obs_horizon=int(config["data"]["obs_horizon"]),
        action_horizon=int(config["data"]["action_horizon"]),
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size_per_gpu"]),
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=True,
    )

    model_config = load_config(resolve(config["model_config"]))
    model = DexCG.from_config(model_config, PROJECT_ROOT, torch_dtype=torch.bfloat16)
    objective = DexCGTrainingObjective(
        model,
        dataset.state_min,
        dataset.state_max,
        config["diffusion"],
        config["loss"],
        train_contact_planner=bool(config["contact_training"]["enabled"]),
    ).to(device)
    ema = EMA(objective, float(config["checkpoint"]["ema_decay"]))
    optimizer = optimizer_for(objective, config["optimizer"])
    accumulation = int(config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * int(config["epochs"])
    scheduler = scheduler_for(
        optimizer, int(config["optimizer"]["warmup_steps"]), total_updates
    )

    start_epoch = 1
    global_step = 0
    latest_path = checkpoint_dir / "latest.ckpt"
    if bool(config["resume"]) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        _load_compact_state_dict(
            objective,
            checkpoint["model"],
            frozenset(compact_state_dict(objective)),
        )
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if rank == 0:
            print(f"resumed epoch={start_epoch - 1} global_step={global_step}", flush=True)

    distributed = DistributedDataParallel(
        objective,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
    )
    optimizer.zero_grad(set_to_none=True)
    stop = False
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        sampler.set_epoch(epoch)
        distributed.train()
        probability = (
            teacher_forcing_probability(epoch, config["contact_teacher_forcing"])
            if objective.train_contact_planner
            else 1.0
        )
        sums = {
            "batches": 0.0,
            "loss": 0.0,
            "loss_coefficient": 0.0,
            "loss_reconstruction": 0.0,
            "loss_gate": 0.0,
            "loss_alignment": 0.0,
            "loss_contact": 0.0,
            "contact_correct": 0.0,
            "contact_count": 0.0,
            "predicted_contact_rows": 0.0,
            "batch_rows": 0.0,
        }
        for batch_index, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            sync = batch_index % accumulation == 0 or batch_index == len(loader)
            context = distributed.no_sync() if not sync else torch.enable_grad()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metrics = distributed(batch, probability)
                    scaled_loss = loss / accumulation
                scaled_loss.backward()
            for key in sums:
                if key == "batches":
                    continue
                sums[key] += float(metrics[key].item())
            sums["batches"] += 1.0
            if sync:
                torch.nn.utils.clip_grad_norm_(
                    objective.parameters(), float(config["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(objective)
                global_step += 1
                if rank == 0 and global_step % 50 == 0:
                    print(
                        f"epoch={epoch:04d} batch={batch_index}/{len(loader)} "
                        f"step={global_step} loss={metrics['loss'].item():.4f}",
                        flush=True,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop = True
                    break

        reduced = reduce_epoch_metrics(sums, device)
        if rank == 0:
            batches = reduced["batches"]
            summary = {
                "epoch": epoch,
                "global_step": global_step,
                "contact_planner_trainable": objective.train_contact_planner,
                "learning_rate": scheduler.get_last_lr()[0],
                **{
                    key: reduced[key] / batches
                    for key in (
                        "loss",
                        "loss_coefficient",
                        "loss_reconstruction",
                        "loss_gate",
                        "loss_alignment",
                    )
                },
            }
            if objective.train_contact_planner:
                summary["teacher_forcing"] = probability
                summary["loss_contact"] = reduced["loss_contact"] / batches
                summary["contact_accuracy"] = reduced["contact_correct"] / max(
                    reduced["contact_count"], 1.0
                )
                summary["predicted_contact_fraction"] = reduced[
                    "predicted_contact_rows"
                ] / max(reduced["batch_rows"], 1.0)
            print(json.dumps(summary, sort_keys=True), flush=True)

        checkpoint_due = epoch % int(config["checkpoint"]["interval_epochs"]) == 0 or stop
        if rank == 0 and checkpoint_due:
            save_latest(latest_path, objective, ema, optimizer, scheduler, epoch, global_step)
        dist.barrier()

        evaluation = config["evaluation"]
        evaluation_due = (
            not stop
            and epoch >= int(evaluation["first_epoch"])
            and epoch % int(evaluation["interval_epochs"]) == 0
        )
        if evaluation_due:
            ema.module.eval()
            result = evaluate_seen_tasks(
                ema.module, config, epoch, output_dir, device, rank, world_size
            )
            if rank == 0:
                score = float(result["mean_success_rate"])
                save_top_checkpoint(
                    checkpoint_dir,
                    objective,
                    ema,
                    epoch,
                    score,
                    int(config["checkpoint"]["keep_top"]),
                )
                print(f"evaluation epoch={epoch:04d} seen_score={score:.4f}", flush=True)
            dist.barrier()
        if stop:
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
