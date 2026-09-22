#!/usr/bin/env python3
"""Distributed dexCG training entry point."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time
import subprocess
import sys
import shutil
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dexcg.common.config import ProjectConfig, load_config
from dexcg.data import DexArtTrainingDataset
from dexcg.evaluation import evaluate_seen_tasks
from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    MODEL_COORDINATE_CONTRACT,
    require_model_coordinates,
)
from dexcg.models.dexcg import DexCG
from dexcg.training import DexCGTrainingObjective

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_balanced.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-per-gpu", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test limit")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Restore latest policy, EMA, optimizer and scheduler")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-suite", action="store_true")
    parser.add_argument("--continue-auto", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Benchmark, preflight, then train")
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--workers", type=int)
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
        planner = getattr(getattr(module, "model", None), "contact_planner", None)
        memo = {id(planner): planner} if planner is not None and not any(
            p.requires_grad for p in planner.parameters()) else None
        self.module = copy.deepcopy(module, memo).requires_grad_(False).eval()
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, source: torch.nn.Module) -> None:
        source_parameters = dict(source.named_parameters())
        target_parameters = dict(self.module.named_parameters())
        for name in self.parameter_names:
            target_parameters[name].lerp_(source_parameters[name].detach(), 1.0 - self.decay)
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

    def assert_synchronized(self, group=None) -> None:
        """Fail before evaluation if ranks would evaluate different policies."""
        if not dist.is_initialized() or dist.get_world_size(group) == 1:
            return
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        hashes = [None] * dist.get_world_size(group)
        dist.all_gather_object(hashes, digest.hexdigest(), group=group)
        if len(set(hashes)) != 1:
            raise RuntimeError("EMA differs across ranks; refusing to combine evaluation results")


def compact_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    parameter_names = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    buffer_names = {
        name for name, _ in module.named_buffers() if not name.startswith("model.contact_planner.")
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
        groups.append({"params": planner, "lr": float(config["planner_learning_rate"])})
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


def reduce_epoch_metrics(values: dict[str, float], device: torch.device) -> dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor([values[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return dict(zip(keys, tensor.cpu().tolist(), strict=True))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_identity(config):
    from evaluate_dexart_closed_loop import verify_planner_point_count
    variants = {}
    for name, item in config["evaluation"]["variants"].items():
        path = resolve(item["checkpoint"])
        evidence = verify_planner_point_count(path, int(item["point_count"]))
        variants[name] = {**item, "checkpoint_sha256": file_sha256(path), **evidence}
    return {"variants": variants,
            "split_manifest_sha256": file_sha256(resolve(config["data"]["split_manifest"])),
            "model_config_sha256": file_sha256(resolve(config["model_config"])),
            "contact_embedding_source": "original_dexter_frozen_lookup_saved_in_SMP",
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT}


def load_variant(model, item):
    from train_contact_planner import load_compact_planner_state
    from types import SimpleNamespace
    path = resolve(item["checkpoint"])
    saved_config = yaml.safe_load((path.parent.parent / "config.yaml").read_text())
    # Recreate the original initialization before applying a compact checkpoint:
    # otherwise the other variant's unsaved weights could leak into this one.
    fresh = DexCG.from_config(ProjectConfig(**saved_config["model_snapshot"]),
                              PROJECT_ROOT, torch_dtype=torch.bfloat16)
    planner = fresh.contact_planner
    reference = model.contact_planner.contact_tokenizer
    candidate = planner.contact_tokenizer
    for attr in ("link_token_ids", "position_token_ids", "joint_start_id", "joint_end_id"):
        left, right = getattr(reference, attr), getattr(candidate, attr)
        same = bool(np.array_equal(left, right)) if isinstance(left, (list, tuple, np.ndarray)) \
            or isinstance(right, (list, tuple, np.ndarray)) else left == right
        if not same:
            raise ValueError(f"Planner vocabulary mismatch: {attr}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require_model_coordinates(checkpoint)
    load_compact_planner_state(SimpleNamespace(planner=planner), checkpoint["planner"])
    planner.requires_grad_(False).eval()
    return planner


def paired_evaluation(ema, config, identity, epoch, output_dir, device, rank, world_size, group,
                      preflight=False):
    driver = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
    if driver.is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", str(driver))
    results = {}
    original = ema.module.model.contact_planner
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            ema.assert_synchronized(group)
            for name, item in config["evaluation"]["variants"].items():
                planner = load_variant(ema.module.model, item).to(device)
                ema.module.model.contact_planner = planner
                evaluation_config = copy.deepcopy(config)
                evaluation_config["evaluation"].update(
                    planner_point_count=int(item["point_count"]),
                    planner_checkpoint=str(resolve(item["checkpoint"])),
                    planner_checkpoint_sha256=identity["variants"][name]["checkpoint_sha256"],
                )
                if preflight:
                    evaluation_config["evaluation"].update(episodes_per_seed=1, seeds=[1000],
                        initial_observation_retries=1, action_steps=1, record_video=False,
                        write_results=False)
                    for task in evaluation_config["evaluation"]["tasks"].values():
                        task["max_steps"] = 2
                try:
                    result = evaluate_seen_tasks(ema.module.eval(), evaluation_config, epoch,
                        output_dir, device, rank, world_size, group)
                finally:
                    ema.module.model.contact_planner = original
                    del planner
                    torch.cuda.set_device(device)
                    torch.cuda.empty_cache()
                if rank == 0:
                    results[name] = result
                if preflight:
                    passed = [all(t["point_input_contract"]["partfield_calls"] > 0
                                  and t["point_input_contract"]["dp3_calls"] > 0
                                  for t in result["tasks"].values()) if rank == 0 else None]
                    dist.broadcast_object_list(passed, src=0, group=group)
                    if not passed[0]:
                        raise RuntimeError(f"{name}: not every task exercised the real VLM and SMP interfaces")
                dist.barrier()
    finally:
        ema.module.model.contact_planner = original
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    return results


def save_ranked(checkpoint_dir, ema, epoch, results, identity, keep):
    manifest_path = checkpoint_dir / "rankings.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    rankings = {}
    score_suffix = "".join(
        f"-{name}_seen_score={float(result['mean_success_rate']):.3f}"
        for name, result in sorted(results.items())
    )
    checkpoint_name = f"epoch={epoch:04d}{score_suffix}.ckpt"
    for name, result in results.items():
        rows = previous.get(name, []) + [{"epoch": epoch, "score": float(result["mean_success_rate"]),
                                         "checkpoint": checkpoint_name}]
        rows = {row["epoch"]: row for row in rows}
        rankings[name] = sorted(rows.values(), key=lambda row: (-row["score"], row["epoch"]))[:keep]
    retained = {row["epoch"] for rows in rankings.values() for row in rows}
    if epoch in retained:
        save_atomic({"epoch": epoch, "model": ema.state_dict(), "run_identity": identity,
            "scores": {name: result["mean_success_rate"] for name, result in results.items()},
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT},
            checkpoint_dir / checkpoint_name)
    write_json_atomic(rankings, manifest_path)
    # Only files from this ranking scheme are eligible for pruning.
    for path in checkpoint_dir.glob("epoch=*.ckpt"):
        match = re.fullmatch(r"epoch=(\d+)(?:-[a-zA-Z0-9_]+_seen_score=[0-9.]+)*\.ckpt", path.name)
        if match and int(match.group(1)) not in retained:
            path.unlink()


def current_gpus():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                     "--format=csv,noheader,nounits"], text=True)
    return [{"index": int(a), "free_mib": int(b), "utilization": int(c)}
            for a, b, c in (line.split(",") for line in output.strip().splitlines())]


def preparation_dir(output_dir):
    directory = output_dir / "preparation"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def automatic_run(args, config):
    output_dir = resolve(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.continue_auto:
        raise FileExistsError(f"Automatic fresh run requires an empty output directory: {output_dir}")
    if any((output_dir / "checkpoints").glob("*.ckpt")) and not args.resume:
        raise FileExistsError("Automatic fresh launch refuses existing training checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(os.environ["TMPDIR"])
    if not runtime_dir.name.startswith("dexcg-smp-"):
        raise ValueError("Set TMPDIR to the approved dedicated dexcg-smp runtime directory")
    results = []
    benchmark = config["benchmark"]
    minimum = int(float(benchmark["minimum_free_gib"]) * 1024)

    def choose_cards():
        while True:
            candidates = sorted((g for g in current_gpus() if g["free_mib"] >= minimum),
                                key=lambda g: (-g["free_mib"], g["utilization"]))
            if len(candidates) >= 4:
                return candidates[:4]
            print(f"Waiting for four GPUs with at least {minimum / 1024:g} GiB free", flush=True)
            time.sleep(30)

    def run_child(extra, cards, log_name):
        (runtime_dir / "cache" / "torch" / "kernels").mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(str(g["index"]) for g in cards),
            PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false", WANDB_DISABLED="true",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            MPLCONFIGDIR=str(runtime_dir / "matplotlib"), XDG_CACHE_HOME=str(runtime_dir / "cache"),
            TORCH_EXTENSIONS_DIR=str(runtime_dir / "torch_extensions"),
            NUMBA_CACHE_DIR=str(runtime_dir / "numba"), TRITON_CACHE_DIR=str(runtime_dir / "triton"),
            CUDA_CACHE_PATH=str(runtime_dir / "cuda"))
        command = [sys.executable, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   str(Path(__file__).resolve()), "--config", str(resolve(args.config)),
                   "--output-dir", str(output_dir), "--epochs", str(config["epochs"]), *extra]
        log_dir = output_dir if log_name == "train.log" else preparation_dir(output_dir)
        with (log_dir / log_name).open("a") as stream:
            process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=runtime_dir)
            write_json_atomic({"pid": process.pid, "command": command, "gpus": cards,
                               "stage": log_name}, preparation_dir(output_dir) / "process.json")
            status = process.wait()
            write_json_atomic({"pid": process.pid, "command": command, "gpus": cards,
                               "stage": log_name, "exit_code": status}, preparation_dir(output_dir) / "process.json")
            return status

    if args.resume:
        if not (output_dir / "checkpoints" / "latest.ckpt").is_file():
            raise FileNotFoundError("Automatic resume requires latest.ckpt")
        cards = choose_cards()
        if run_child(["--resume", "--global-batch", str(config["global_batch_size"]),
                      "--batch-size-per-gpu", str(config["batch_size_per_gpu"])], cards, "train.log"):
            raise RuntimeError("Resumed training failed; see train.log")
        return

    cards = choose_cards()
    write_json_atomic({"initial_gpus": cards, "started": time.time()}, preparation_dir(output_dir) / "launch.json")
    # Keep decoded data in RAM across cases. Each case resets the complete
    # lower policy, EMA, optimizer and RNG, and creates a fresh data loader.
    cases = [(4 * micro, micro) for micro in benchmark["micro_batches"]]
    selection_path = preparation_dir(output_dir) / "benchmark_selection.json"
    if args.continue_auto and selection_path.exists():
        selection = json.loads(selection_path.read_text())
        results = json.loads((preparation_dir(output_dir) / "benchmark.json").read_text())
    else:
        if run_child(["--benchmark-suite"], cards, "benchmark.log"):
            raise RuntimeError("Benchmark suite failed; training was not started")
        results = json.loads((preparation_dir(output_dir) / "benchmark.json").read_text())
    usable = []
    for global_batch, micro in cases:
        rows = [row for row in results if row["global_batch"] == global_batch and row["micro_batch"] == micro]
        if len(rows) != int(benchmark["repetitions"]) or any(row.get("status") != "passed" for row in rows):
            continue
        usable.append({"global_batch": global_batch, "micro_batch": micro,
                       "median_samples_per_second": float(np.median([r["samples_per_second"] for r in rows])),
                       "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows)})
    if not usable:
        raise RuntimeError("No benchmark candidate passed; training was not started")
    # If throughput differs by less than 3%, favor the smaller effective batch
    # and then lower memory. Do not increase learning rate with batch size.
    if not (args.continue_auto and selection_path.exists()):
        fastest = max(row["median_samples_per_second"] for row in usable)
        candidates = [row for row in usable if row["median_samples_per_second"] >= fastest * .97]
        selected = min(candidates, key=lambda row: (row["global_batch"], row["peak_reserved_gib"]))
        write_json_atomic({"candidates": usable, "selected": selected,
                           "selection_rule": "within 3% of fastest: smaller global batch, then lower VRAM"},
                          selection_path)
    else:
        selected = selection["selected"]
    extra = ["--global-batch", str(selected["global_batch"]),
             "--batch-size-per-gpu", str(selected["micro_batch"])]
    cards = choose_cards()
    if run_child([*extra, "--preflight"], cards, "preflight.log"):
        raise RuntimeError("Real simulation/save-load preflight failed; training was not started")
    cards = choose_cards()
    if run_child(extra, cards, "train.log"):
        raise RuntimeError("Training process failed; see train.log")


def cleanup_runtime_directory():
    """Remove only the dedicated, completed run's temporary tree."""
    raw = os.environ.get("TMPDIR", "")
    runtime = Path(raw)
    if not raw or runtime.parent != Path("/tmp") or not runtime.name.startswith("dexcg-smp-"):
        return
    if runtime.is_symlink() or not runtime.is_dir():
        return
    for process in Path("/proc").glob("[0-9]*"):
        if process.name == str(os.getpid()):
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            cwd = (process / "cwd").resolve(strict=True)
        except FileNotFoundError:
            continue
        except PermissionError:
            return  # Cannot verify the tree is unused; preserve it.
        if cwd == runtime or runtime in cwd.parents:
            print(f"Preserving active runtime directory: {runtime}", flush=True)
            return
    shutil.rmtree(runtime)
    print(f"Cleaned completed run's temporary directory: {runtime}", flush=True)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text())
    for attr, key in (("epochs", "epochs"), ("output_dir", "output_dir"),
                      ("batch_size_per_gpu", "batch_size_per_gpu"), ("workers", "num_workers"),
                      ("global_batch", "global_batch_size")):
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    if args.no_resume:
        config["resume"] = False
    if args.resume:
        if args.no_resume:
            raise ValueError("--resume and --no-resume are mutually exclusive")
        config["resume"] = True
    if args.auto:
        try:
            automatic_run(args, config)
        finally:
            cleanup_runtime_directory()
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        raise RuntimeError("This SMP run requires exactly four GPUs")
    if config["contact_training"]["enabled"] or config["loss"]["contact"] != 0:
        raise ValueError("This SMP run freezes the VLM and uses only GT contacts")
    micro = int(config["batch_size_per_gpu"])
    global_batch = int(config["global_batch_size"])
    if global_batch != micro * world_size:
        raise ValueError("This run uses no gradient accumulation: global batch must be four times per-GPU batch")
    accumulation = global_batch // (micro * world_size)
    config["gradient_accumulation_steps"] = accumulation
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=24), device_id=device)
    group = dist.new_group(backend="gloo", timeout=timedelta(hours=24))
    try:
        run_distributed(args, config, device, rank, world_size, group, accumulation)
    finally:
        dist.destroy_process_group()


def benchmark_suite(distributed, objective, ema, dataset, config, device, rank, world_size, group, output_dir):
    import gc
    initial = {name: value.detach().cpu().clone() for name, value in compact_state_dict(objective).items()}
    initial_ema = {name: value.detach().cpu().clone() for name, value in ema.state_dict().items()}
    parameters = [p for p in objective.parameters() if p.requires_grad]
    settings = config["benchmark"]
    cases = [(world_size * micro, micro) for micro in settings["micro_batches"]]
    results = []
    for repeat in range(int(settings["repetitions"])):
        for global_batch, micro in cases[::1 if repeat % 2 == 0 else -1]:
            torch.manual_seed(int(config["seed"]) + rank)
            np.random.seed(int(config["seed"]) + rank)
            random.seed(int(config["seed"]) + rank)
            _load_compact_state_dict(objective, initial, frozenset(initial))
            ema.load_state_dict(initial_ema)
            distributed.train()
            accumulation = global_batch // (world_size * micro)
            optimizer = optimizer_for(objective, config["optimizer"])
            scheduler = scheduler_for(optimizer, int(config["optimizer"]["warmup_steps"]),
                (len(dataset) // global_batch) * int(config["epochs"]))
            sampler = DistributedSampler(dataset, shuffle=True, seed=int(config["seed"]), drop_last=True)
            sampler.set_epoch(1)
            loader = DataLoader(dataset, batch_size=micro, sampler=sampler,
                num_workers=int(config["num_workers"]), pin_memory=True, drop_last=True)
            iterator = iter(loader)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            warmup = int(settings["warmup_updates"])
            measured = int(settings["measured_updates"])
            if rank == 0:
                print(f"benchmark repetition={repeat} global_batch={global_batch} micro_batch={micro} accumulation={accumulation}", flush=True)
            for update in range(warmup + measured):
                if update == warmup:
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.monotonic()
                for part in range(accumulation):
                    batch = move_batch(next(iterator), device)
                    with nullcontext() if part == accumulation - 1 else distributed.no_sync():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss, metrics = distributed(batch, 1.0)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite benchmark loss")
                        (loss / accumulation).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip_norm"]), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(objective)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.monotonic() - started, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            memory = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                                  dtype=torch.float64, device=device)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            ema.assert_synchronized(group)
            if rank == 0:
                result = {"status": "passed", "global_batch": global_batch, "micro_batch": micro,
                    "accumulation": accumulation, "workers": config["num_workers"], "repetition": repeat,
                    "measured_updates": measured, "elapsed_seconds": elapsed.item(),
                    "samples_per_second": global_batch * measured / elapsed.item(),
                    "peak_allocated_gib": memory[0].item() / 2**30, "peak_reserved_gib": memory[1].item() / 2**30,
                    "loss": float(loss.detach()), "gradient_norm": float(norm),
                    "estimated_training_hours": len(dataset) * config["epochs"] / (global_batch * measured / elapsed.item()) / 3600,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
                results.append(result)
                write_json_atomic(results, preparation_dir(output_dir) / "benchmark.json")
                print(json.dumps(result), flush=True)
            del iterator, loader, optimizer, scheduler, batch, loss, metrics
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()


def run_distributed(args, config, device, rank, world_size, group, accumulation):
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.set_num_threads(2)
    output_dir = resolve(config["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    latest_path = checkpoint_dir / "latest.ckpt"
    if config["resume"] and not latest_path.exists():
        raise FileNotFoundError("Resume requires an existing latest.ckpt")
    if not config["resume"] and any(checkpoint_dir.glob("*.ckpt")) and not args.benchmark:
        raise FileExistsError("Fresh training refuses existing policy checkpoints")
    identity = run_identity(config)
    dataset = DexArtTrainingDataset(
        [resolve(path) for path in config["data"]["paths"]],
        obs_horizon=int(config["data"]["obs_horizon"]),
        action_horizon=int(config["data"]["action_horizon"]),
        split_manifest=resolve(config["data"]["split_manifest"]),
        point_count=int(config["data"]["point_count"]),
        preload=bool(config["data"]["preload"]) and not args.preflight,
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=int(config["batch_size_per_gpu"]), sampler=sampler,
        num_workers=int(config["num_workers"]), pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0, drop_last=True)
    model = DexCG.from_config(load_config(resolve(config["model_config"])), PROJECT_ROOT,
                             torch_dtype=torch.bfloat16)
    objective = DexCGTrainingObjective(model, dataset.state_min, dataset.state_max,
        config["diffusion"], config["loss"], train_contact_planner=False).to(device)
    distributed = DistributedDataParallel(objective, device_ids=[device.index],
        output_device=device.index, gradient_as_bucket_view=True)
    ema = EMA(objective, float(config["checkpoint"]["ema_decay"]))
    ema.assert_synchronized(group)
    optimizer = optimizer_for(objective, config["optimizer"])
    # Only complete effective batches are stepped; no underweighted partial
    # accumulation group and no silent global batch change at the epoch tail.
    usable_batches = len(loader) // accumulation * accumulation
    updates_per_epoch = usable_batches // accumulation
    if updates_per_epoch < 1:
        raise ValueError("Dataset is too small for one effective batch")
    scheduler = scheduler_for(optimizer, int(config["optimizer"]["warmup_steps"]),
                              updates_per_epoch * int(config["epochs"]))
    start_epoch, global_step = 1, 0
    if config["resume"] and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        require_model_coordinates(checkpoint)
        if checkpoint.get("run_identity") != identity:
            raise ValueError("Resume identity differs: VLM, coordinates, split or model config")
        _load_compact_state_dict(objective, checkpoint["model"], frozenset(compact_state_dict(objective)))
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step = int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])
        states = checkpoint["rank_rng_states"][rank]
        torch.set_rng_state(states["torch"].cpu())
        torch.cuda.set_rng_state(states["cuda"].cpu(), device)
        np.random.set_state(states["numpy"])
        random.setstate(states["python"])
        if rank == 0:
            print(json.dumps({"event": "resumed", "checkpoint_epoch": start_epoch - 1,
                "next_epoch": start_epoch, "global_step": global_step,
                "restored": ["model", "ema", "optimizer", "scheduler", "rank_rng_states"]}), flush=True)
    ema.assert_synchronized(group)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"train_retained_frames": [len(s) for s in dataset.samples],
            "balanced_samples_per_epoch": len(dataset), "updates_per_epoch": updates_per_epoch,
            "micro_batch": config["batch_size_per_gpu"], "global_batch": config["global_batch_size"],
            "trainable_parameters": sum(p.numel() for p in objective.parameters() if p.requires_grad),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    dist.barrier()
    if args.benchmark_suite:
        benchmark_suite(distributed, objective, ema, dataset, config, device, rank, world_size, group, output_dir)
        return
    if args.preflight:
        import io
        # Exercise actual serialization in memory without leaving disposable checkpoints.
        stream = io.BytesIO()
        state = compact_state_dict(objective)
        torch.save(state, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location=device, weights_only=True)
        _load_compact_state_dict(objective, restored, frozenset(state))
        for name in state:
            if not torch.equal(state[name], restored[name]):
                raise AssertionError(f"Checkpoint round trip changed {name}")
        results = paired_evaluation(ema, config, identity, 0, output_dir, device, rank, world_size, group, True)
        if rank == 0:
            write_json_atomic({"status": "passed", "run_identity": identity,
                "save_reload": "passed", "purpose": "untrained policy interface check only",
                "variants": results},
                preparation_dir(output_dir) / "preflight.json")
        return
    if not args.benchmark and rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config["runtime"] = {"world_size": world_size, "global_batch_size": config["global_batch_size"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "updates_per_epoch": updates_per_epoch}
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        write_json_atomic(identity, preparation_dir(output_dir) / "run_identity.json")
    dist.barrier()
    warmup = int(config["benchmark"]["warmup_updates"])
    measured = int(config["benchmark"]["measured_updates"])
    benchmark_start = None
    benchmark_steps = 0
    parameters = [p for p in objective.parameters() if p.requires_grad]
    started = time.monotonic()
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        sampler.set_epoch(epoch)
        distributed.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {}
        batches = 0
        epoch_start = time.monotonic()
        iterator = iter(loader)
        for batch_index in range(1, usable_batches + 1):
            batch = move_batch(next(iterator), device)
            sync = batch_index % accumulation == 0
            with nullcontext() if sync else distributed.no_sync():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metrics = distributed(batch, 1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite SMP loss")
                (loss / accumulation).backward()
            for key, value in metrics.items():
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value.detach()
            batches += 1
            if not sync:
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(objective)
            global_step += 1
            if args.benchmark:
                benchmark_steps += 1
                if benchmark_steps == warmup:
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    torch.cuda.reset_peak_memory_stats(device)
                    benchmark_start = time.monotonic()
                if benchmark_steps == warmup + measured:
                    torch.cuda.synchronize(device)
                    elapsed = torch.tensor(time.monotonic() - benchmark_start, device=device)
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    memory = torch.tensor([torch.cuda.max_memory_allocated(device),
                                           torch.cuda.max_memory_reserved(device)], dtype=torch.float64, device=device)
                    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                    ema.assert_synchronized(group)
                    if rank == 0:
                        result = {"status": "passed", "global_batch": config["global_batch_size"],
                            "micro_batch": config["batch_size_per_gpu"], "accumulation": accumulation,
                            "workers": config["num_workers"], "measured_updates": measured,
                            "elapsed_seconds": elapsed.item(),
                            "samples_per_second": config["global_batch_size"] * measured / elapsed.item(),
                            "peak_allocated_gib": memory[0].item() / 2**30,
                            "peak_reserved_gib": memory[1].item() / 2**30,
                            "loss": float(loss), "gradient_norm": float(norm),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "estimated_training_hours": len(dataset) * config["epochs"] / (config["global_batch_size"] * measured / elapsed.item()) / 3600}
                        write_json_atomic(result, preparation_dir(output_dir) / "benchmark_case.json")
                        print(json.dumps(result), flush=True)
                    return
            if rank == 0 and global_step % 50 == 0:
                print(f"epoch={epoch} step={global_step} loss={float(loss):.5f}", flush=True)
            if args.max_steps is not None and global_step >= args.max_steps:
                return
        # Keep input-loader RNG independent of validation and simulation below.
        reduced = reduce_epoch_metrics({**{k: float(v) for k, v in sums.items()}, "batches": batches}, device)
        elapsed_epoch = time.monotonic() - epoch_start
        if rank == 0:
            summary = {"epoch": epoch, "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0], "teacher_forcing": 1.0,
                "predicted_contact_fraction": reduced["predicted_contact_rows"] / max(reduced["batch_rows"], 1),
                "epoch_seconds": elapsed_epoch, "elapsed_hours": (time.monotonic() - started) / 3600,
                **{k: v / reduced["batches"] for k, v in reduced.items() if k.startswith("loss")}}
            print(json.dumps(summary), flush=True)
        evaluation = config["evaluation"]
        if (evaluation.get("enabled", True) and epoch >= int(evaluation["first_epoch"])
                and epoch % int(evaluation["interval_epochs"]) == 0):
            results = paired_evaluation(ema, config, identity, epoch, output_dir, device, rank, world_size, group)
            if rank == 0:
                save_ranked(checkpoint_dir, ema, epoch, results, identity, int(config["checkpoint"]["keep_top_per_variant"]))
            dist.barrier()
        if epoch % int(config["checkpoint"]["interval_epochs"]) == 0 or epoch == int(config["epochs"]):
            rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                   "numpy": np.random.get_state(), "python": random.getstate()}
            rng_states = [None] * world_size
            dist.all_gather_object(rng_states, rng, group=group)
            if rank == 0:
                save_atomic({"epoch": epoch, "global_step": global_step, "run_identity": identity,
                    "model": compact_state_dict(objective), "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rank_rng_states": rng_states,
                    "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
                    "model_coordinate_contract": MODEL_COORDINATE_CONTRACT}, latest_path)
            dist.barrier()


if __name__ == "__main__":
    main()
