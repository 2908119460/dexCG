#!/usr/bin/env python3
"""Restart distributed training after failures once local preflight checks pass."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "model",
    "ema",
    "optimizer",
    "scheduler",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--restart-delay", type=float, default=30.0)
    parser.add_argument("--min-free-memory-mib", type=int, default=8192)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return config


def _checkpoint_summary(config: dict[str, Any]) -> str:
    if not bool(config.get("resume")):
        return "resume disabled"
    output_dir = _resolve(config["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / "latest.ckpt"
    if not checkpoint_path.is_file():
        return "no latest checkpoint; training will start at epoch 1"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    missing = CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing keys {sorted(missing)}: {checkpoint_path}")
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    if epoch < 0 or global_step < 0:
        raise ValueError(
            f"checkpoint has negative epoch/global_step: {checkpoint_path}"
        )
    return f"checkpoint epoch={epoch} global_step={global_step}"


def _gpu_free_memory() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    free_memory = {}
    for line in result.stdout.splitlines():
        index, memory = (part.strip() for part in line.split(",", maxsplit=1))
        free_memory[int(index)] = int(memory)
    return free_memory


def preflight(
    config_path: Path, devices: tuple[int, ...], min_free_memory_mib: int
) -> str:
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("at least two distinct CUDA device indices are required")
    config = _load_config(config_path)
    checkpoint = _checkpoint_summary(config)
    free_memory = _gpu_free_memory()
    missing = [device for device in devices if device not in free_memory]
    if missing:
        raise RuntimeError(f"CUDA devices do not exist: {missing}")
    low_memory = {
        device: free_memory[device]
        for device in devices
        if free_memory[device] < min_free_memory_mib
    }
    if low_memory:
        raise RuntimeError(
            f"insufficient free GPU memory (need {min_free_memory_mib} MiB): {low_memory}"
        )
    memory = ", ".join(f"gpu{device}={free_memory[device]}MiB" for device in devices)
    return f"{checkpoint}; {memory}"


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.config)
    devices = tuple(int(item.strip()) for item in args.cuda_visible_devices.split(","))
    if args.check_only:
        summary = preflight(config_path, devices, args.min_free_memory_mib)
        print(f"[{_timestamp()}] preflight passed: {summary}", flush=True)
        return 0
    stop_event = threading.Event()
    child: subprocess.Popen | None = None

    def stop(signum, frame) -> None:
        del frame
        stop_event.set()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)

    torchrun = Path(sys.executable).with_name("torchrun")
    if not torchrun.is_file():
        raise FileNotFoundError(f"torchrun was not found next to {sys.executable}")
    command = [
        str(torchrun),
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        "scripts/train.py",
        "--config",
        str(config_path),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in devices)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    failures = 0
    while not stop_event.is_set():
        try:
            summary = preflight(config_path, devices, args.min_free_memory_mib)
        except Exception as error:
            print(f"[{_timestamp()}] preflight failed: {error}", flush=True)
            stop_event.wait(args.restart_delay)
            continue

        print(f"[{_timestamp()}] preflight passed: {summary}", flush=True)
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            start_new_session=True,
        )
        return_code = child.wait()
        child = None
        if stop_event.is_set():
            print(f"[{_timestamp()}] monitor stopped by signal", flush=True)
            return 128
        if return_code == 0:
            print(f"[{_timestamp()}] training completed normally", flush=True)
            return 0

        failures += 1
        print(
            f"[{_timestamp()}] training exited with code {return_code}; "
            f"restart attempt {failures} after {args.restart_delay:g}s",
            flush=True,
        )
        stop_event.wait(args.restart_delay)
    return 128


if __name__ == "__main__":
    raise SystemExit(main())
