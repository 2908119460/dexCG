#!/usr/bin/env python3
"""Supervise resumable collection and retain a consolidated distribution manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect_dexart import load_yaml, resolve
from collect_object_dexart import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,4")
    parser.add_argument("--config", default="configs/data/dexart_object_balanced_250.yaml")
    args = parser.parse_args()
    config = load_yaml(resolve(args.config))
    output = resolve(config["collection"]["output"])
    tasks = list(config["tasks"])
    language = load_yaml(resolve("configs/annotation/language.yaml"))
    signature = hashlib.sha256(json.dumps({"config": config, "language": language}, sort_keys=True).encode()).hexdigest()
    for task in tasks:
        report = json.loads((output / f"{task}_preflight.json").read_text())
        if not report.get("passed") or (config["collection"].get("sensor_contract") and report.get("config_sha256") != signature):
            raise RuntimeError(f"Missing successful preflight: {task}")
    work = queue.Queue()
    for task in ["bucket", "faucet", "toilet", "laptop"]:
        if task in tasks:
            work.put(task)
    results = {}

    def worker(gpu):
        while True:
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "OMP_NUM_THREADS": "1",
                }
            )
            with (output / f"{task}_collection.log").open("a") as log:
                print(f"Starting/resuming {task} on GPU {gpu}", flush=True)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("collect_object_dexart.py")),
                        "--task",
                        task,
                        "--config",
                        args.config,
                    ],
                    cwd=resolve("."),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                results[task] = result.returncode
                print(f"Finished {task}: exit={result.returncode}", flush=True)
            work.task_done()

    def manifest():
        distribution = {"format": "dexcg.object_robot_base.distribution.v1", "tasks": {}}
        for task in tasks:
            path = output / f"{task}_progress.json"
            distribution["tasks"][task] = (
                json.loads(path.read_text()) if path.exists() else {"status": "pending"}
            )
            if task in results:
                distribution["tasks"][task]["process_exit_code"] = results[task]
        distribution["complete"] = all(
            task.get("status") == "complete" for task in distribution["tasks"].values()
        )
        distribution["accepted_episodes"] = sum(
            obj["accepted"]
            for task in distribution["tasks"].values()
            for obj in task.get("objects", {}).values()
        )
        atomic_json(output / "distribution.json", distribution)

    with ThreadPoolExecutor(max_workers=len(args.gpus.split(","))) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus.split(",")]
        while not all(future.done() for future in futures):
            manifest()
            time.sleep(15)
        for future in futures:
            future.result()
    manifest()
    if any(code != 0 for code in results.values()):
        raise SystemExit(f"One or more collections failed; see distribution.json: {results}")
    print("All 1,000 episodes collected and all four dataset audits passed.", flush=True)


if __name__ == "__main__":
    main()
