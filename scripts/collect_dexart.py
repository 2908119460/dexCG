#!/usr/bin/env python3
"""Collect successful DexArt demonstrations with contact and language annotations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from dexcg.annotation import AnnotationFormatError, GemmaGraspAnnotator
from dexcg.data import DexArtEpisode, write_dexart_dataset
from dexcg.envs import DexArtAdapter
from dexcg.models.contact.tokenizer import AllegroContactTokenizer
from dexcg.policy.dexart_expert import load_dexart_expert
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("faucet", "bucket", "laptop", "toilet"))
    parser.add_argument("--device", required=True, help="Torch device, for example cuda:0")
    parser.add_argument("--config", default="configs/data/dexart.yaml")
    parser.add_argument("--language-config", default="configs/annotation/language.yaml")
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--episodes", type=int, default=None)
    count.add_argument("--quota-per-object", type=int, default=None)
    parser.add_argument("--max-attempts-per-object", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve(path: str) -> Path:
    return PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batched(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[None] for name, value in observation.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_object_ids(task: str, split: str) -> list[int]:
    from dexart.env.task_setting import TRAIN_CONFIG

    try:
        return [int(object_id) for object_id in TRAIN_CONFIG[task][split]]
    except KeyError as error:
        raise ValueError(f"Unknown DexArt task/split: {task}/{split}") from error


def balanced_collection_settings(
    task: str,
    task_config: dict[str, Any],
    collection_config: dict[str, Any],
    quota_override: int | None = None,
    attempt_override: int | None = None,
) -> tuple[list[int], int, int] | None:
    quota = quota_override if quota_override is not None else task_config.get("quota_per_object")
    if quota is None:
        return None
    max_attempts = (
        attempt_override
        if attempt_override is not None
        else collection_config.get("max_attempts_per_object")
    )
    if not isinstance(quota, int) or quota <= 0:
        raise ValueError(f"quota_per_object must be a positive integer, received {quota!r}")
    if not isinstance(max_attempts, int) or max_attempts < quota:
        raise ValueError(
            "max_attempts_per_object must be an integer greater than or equal to "
            f"quota_per_object ({quota}), received {max_attempts!r}"
        )
    return split_object_ids(task, collection_config["split"]), quota, max_attempts


def empty_object_statistics(object_ids: list[int], quota: int) -> dict[str, dict[str, Any]]:
    return {
        str(object_id): {
            "quota": quota,
            "attempts": 0,
            "simulator_successes": 0,
            "accepted_episodes": 0,
            "successful_without_state_3": 0,
            "annotation_failures": 0,
            "timesteps": 0,
            "episode_indices": [],
            "episode_lengths": [],
        }
        for object_id in object_ids
    }


def _distribution_template(data_config: dict[str, Any]) -> dict[str, Any]:
    collection = data_config["collection"]
    tasks = {}
    for task, task_config in data_config["tasks"].items():
        quota = task_config.get("quota_per_object")
        object_ids = split_object_ids(task, collection["split"]) if quota is not None else []
        tasks[task] = {
            "status": "pending",
            "output": task_config["output"],
            "expected_object_ids": object_ids,
            "quota_per_object": quota,
            "max_attempts_per_object": collection.get("max_attempts_per_object"),
            "objects": empty_object_statistics(object_ids, quota) if quota is not None else {},
        }
    return {
        "format": "dexcg.dexart.distribution.v1",
        "split": collection["split"],
        "seed": collection["seed"],
        "complete": False,
        "tasks": tasks,
    }


def update_distribution(
    data_config: dict[str, Any],
    task: str,
    status: str,
    object_statistics: dict[str, dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    distribution_value = data_config["collection"].get("distribution")
    if distribution_value is None:
        return
    distribution_path = resolve(distribution_value)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(distribution_path).encode()).hexdigest()[:16]
    lock_path = Path("/tmp") / f"dexcg-distribution-{lock_name}.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if distribution_path.exists():
            distribution = json.loads(distribution_path.read_text(encoding="utf-8"))
        else:
            distribution = _distribution_template(data_config)
        task_distribution = distribution["tasks"][task]
        task_distribution["status"] = status
        if object_statistics is not None:
            task_distribution["objects"] = object_statistics
            task_distribution["attempts"] = sum(
                item["attempts"] for item in object_statistics.values()
            )
            task_distribution["accepted_episodes"] = sum(
                item["accepted_episodes"] for item in object_statistics.values()
            )
            task_distribution["timesteps"] = sum(
                item["timesteps"] for item in object_statistics.values()
            )
        if error is None:
            task_distribution.pop("error", None)
        else:
            task_distribution["error"] = error
        distribution["complete"] = all(
            item["status"] == "complete" for item in distribution["tasks"].values()
        )
        temporary_path = distribution_path.with_name(
            f".{distribution_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.write_text(
            json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_path.replace(distribution_path)


def collect_episode(
    adapter: DexArtAdapter,
    expert,
    annotator: GemmaGraspAnnotator,
    task_id: str,
    device: str,
    camera_config: dict,
) -> tuple[DexArtEpisode | None, bool]:
    observation = adapter.reset()
    values = {
        name: []
        for name in (
            "img",
            "depth",
            "point_cloud",
            "object_point_mask",
            "imagin_robot",
            "state",
            "agent_pos",
        )
    }
    actions: list[np.ndarray] = []
    raw_points: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    stable_step = None
    stable_graph = None
    annotation_views = None
    camera_extrinsics = None
    success = False

    for _ in range(adapter.horizon):
        graph = adapter.contact_graph()
        if adapter.is_stable_contact and stable_step is None:
            stable_step = len(actions)
            stable_graph = graph
            annotation_views, camera_extrinsics = adapter.render_annotation_views(
                observation=observation,
                directions=camera_config["directions"],
                resolution=camera_config["resolution"],
                fov_degrees=camera_config["fov_degrees"],
                distance_margin=camera_config["distance_margin"],
            )

        sample = adapter.observation(observation)
        for name, value in sample.items():
            values[name].append(value)
        raw_points.append(graph.points)
        raw_masks.append(graph.mask)

        action = expert.predict(observation=batched(observation), deterministic=True)[0]
        actions.append(np.asarray(action, dtype=np.float32))
        observation, _, done, _ = adapter.step(action)
        success = success or adapter.is_success

        if adapter.is_stable_contact and stable_step is None:
            stable_step = len(actions)
            stable_graph = adapter.contact_graph()
            annotation_views, camera_extrinsics = adapter.render_annotation_views(
                observation=observation,
                directions=camera_config["directions"],
                resolution=camera_config["resolution"],
                fov_degrees=camera_config["fov_degrees"],
                distance_margin=camera_config["distance_margin"],
            )
            if done:
                stable_step -= 1
        if done:
            break

    if not success or stable_step is None or stable_graph is None:
        return None, success

    language, annotation_raw = annotator.annotate(
        annotation_views,
        stable_graph.link_names,
        adapter.object_id,
        task_id,
    )
    return (
        DexArtEpisode(
            observations=values,
            actions=actions,
            raw_contact_points=raw_points,
            raw_contact_masks=raw_masks,
            stable_contact_step=stable_step,
            stable_contact_points=stable_graph.points,
            stable_contact_mask=stable_graph.mask,
            object_id=adapter.object_id,
            task_id=task_id,
            annotation_views=annotation_views,
            camera_extrinsics=camera_extrinsics,
            language=language,
            annotation_raw=annotation_raw,
        ),
        success,
    )


def collect_legacy(
    task: str,
    episode_count: int,
    checkpoint: Path,
    device: str,
    collection_config: dict[str, Any],
    task_config: dict[str, Any],
    language_config: dict[str, Any],
    annotator: GemmaGraspAnnotator,
) -> tuple[list[DexArtEpisode], int, int, int]:
    adapter = DexArtAdapter.create(
        task_name=task,
        split=collection_config["split"],
        impulse_threshold=collection_config["contact_impulse_threshold"],
    )
    adapter.environment.seed(collection_config["seed"])
    expert = load_dexart_expert(checkpoint, adapter.environment, device)
    episodes: list[DexArtEpisode] = []
    successful_without_stable_contact = 0
    annotation_failures = 0
    consecutive_annotation_failures = 0
    attempts = 0
    progress = tqdm(total=episode_count, desc=task)
    try:
        while len(episodes) < episode_count:
            attempts += 1
            try:
                episode, simulator_success = collect_episode(
                    adapter,
                    expert,
                    annotator,
                    task_config["task_id"],
                    device,
                    language_config["camera"],
                )
            except AnnotationFormatError as error:
                annotation_failures += 1
                consecutive_annotation_failures += 1
                progress.write(
                    f"Discarding attempt {attempts} after invalid Gemma annotation: {error}"
                )
                if consecutive_annotation_failures >= 20:
                    raise RuntimeError(
                        "Gemma produced 20 consecutive invalid annotations"
                    ) from error
                continue
            if episode is None:
                successful_without_stable_contact += int(simulator_success)
                continue
            consecutive_annotation_failures = 0
            episodes.append(episode)
            progress.update(1)
            progress.set_postfix(attempts=attempts, object=episode.object_id)
    finally:
        progress.close()
        adapter.close()
    return episodes, attempts, successful_without_stable_contact, annotation_failures


def collect_balanced(
    task: str,
    object_ids: list[int],
    quota: int,
    max_attempts: int,
    checkpoint: Path,
    device: str,
    data_config: dict[str, Any],
    task_config: dict[str, Any],
    language_config: dict[str, Any],
    annotator: GemmaGraspAnnotator,
    statistics: dict[str, dict[str, Any]],
) -> list[DexArtEpisode]:
    collection_config = data_config["collection"]
    episodes: list[DexArtEpisode] = []
    expert = None
    consecutive_annotation_failures = 0
    progress = tqdm(total=len(object_ids) * quota, desc=f"{task} balanced")
    try:
        for object_id in object_ids:
            adapter = DexArtAdapter.create(
                task_name=task,
                split=collection_config["split"],
                impulse_threshold=collection_config["contact_impulse_threshold"],
                object_id=object_id,
            )
            adapter.environment.seed(collection_config["seed"])
            if expert is None:
                expert = load_dexart_expert(checkpoint, adapter.environment, device)
            else:
                expert.set_env(adapter.environment)
            item = statistics[str(object_id)]
            try:
                while item["accepted_episodes"] < quota and item["attempts"] < max_attempts:
                    item["attempts"] += 1
                    try:
                        episode, simulator_success = collect_episode(
                            adapter,
                            expert,
                            annotator,
                            task_config["task_id"],
                            device,
                            language_config["camera"],
                        )
                    except AnnotationFormatError as error:
                        item["simulator_successes"] += 1
                        item["annotation_failures"] += 1
                        consecutive_annotation_failures += 1
                        progress.write(
                            f"Discarding {task}/{object_id} attempt {item['attempts']} "
                            f"after invalid Gemma annotation: {error}"
                        )
                        if consecutive_annotation_failures >= 20:
                            raise RuntimeError(
                                "Gemma produced 20 consecutive invalid annotations"
                            ) from error
                        continue
                    item["simulator_successes"] += int(simulator_success)
                    if episode is None:
                        item["successful_without_state_3"] += int(simulator_success)
                        if item["attempts"] % 10 == 0:
                            update_distribution(data_config, task, "running", statistics)
                        continue
                    if episode.object_id != str(object_id):
                        raise RuntimeError(
                            f"Requested object {object_id}, but DexArt returned {episode.object_id}"
                        )
                    consecutive_annotation_failures = 0
                    episode_index = len(episodes)
                    episode_length = len(episode.actions)
                    episodes.append(episode)
                    item["accepted_episodes"] += 1
                    item["timesteps"] += episode_length
                    item["episode_indices"].append(episode_index)
                    item["episode_lengths"].append(episode_length)
                    progress.update(1)
                    progress.set_postfix(
                        object=object_id,
                        accepted=f"{item['accepted_episodes']}/{quota}",
                        attempts=item["attempts"],
                    )
                    update_distribution(data_config, task, "running", statistics)
            finally:
                adapter.close()
            if item["accepted_episodes"] < quota:
                update_distribution(data_config, task, "failed", statistics)
                raise RuntimeError(
                    f"{task} object {object_id} reached the limit of {max_attempts} attempts "
                    f"with only {item['accepted_episodes']}/{quota} accepted episodes"
                )
    finally:
        progress.close()
    return episodes


def main() -> None:
    args = parse_args()
    data_config = load_yaml(resolve(args.config))
    language_config = load_yaml(resolve(args.language_config))
    task_config = data_config["tasks"][args.task]
    collection_config = data_config["collection"]
    token_config = data_config["contact_tokens"]
    checkpoint = resolve(task_config["checkpoint"])
    output = resolve(task_config["output"])
    gemma_checkpoint = resolve(language_config["checkpoint"])
    balanced_settings = balanced_collection_settings(
        args.task,
        task_config,
        collection_config,
        args.quota_per_object,
        args.max_attempts_per_object,
    )

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if not gemma_checkpoint.exists():
        raise FileNotFoundError(f"Gemma checkpoint is missing: {gemma_checkpoint}")

    seed_everything(collection_config["seed"])
    object_statistics = None
    try:
        annotator = GemmaGraspAnnotator(
            checkpoint=gemma_checkpoint,
            prompt_path=resolve(language_config["prompt"]),
            device=args.device,
            max_new_tokens=language_config["max_new_tokens"],
            revision=language_config["revision"],
        )
        if balanced_settings is None:
            episode_count = (
                args.episodes
                if args.episodes is not None
                else collection_config["successful_episodes"]
            )
            if episode_count <= 0:
                raise ValueError("episodes must be positive")
            episodes, attempts, successful_without_stable_contact, annotation_failures = (
                collect_legacy(
                    args.task,
                    episode_count,
                    checkpoint,
                    args.device,
                    collection_config,
                    task_config,
                    language_config,
                    annotator,
                )
            )
        else:
            if args.episodes is not None:
                raise ValueError("--episodes cannot be used with a balanced collection config")
            object_ids, quota, max_attempts = balanced_settings
            object_statistics = empty_object_statistics(object_ids, quota)
            update_distribution(data_config, args.task, "running", object_statistics)
            episodes = collect_balanced(
                args.task,
                object_ids,
                quota,
                max_attempts,
                checkpoint,
                args.device,
                data_config,
                task_config,
                language_config,
                annotator,
                object_statistics,
            )
            attempts = sum(item["attempts"] for item in object_statistics.values())
            successful_without_stable_contact = sum(
                item["successful_without_state_3"] for item in object_statistics.values()
            )
            annotation_failures = sum(
                item["annotation_failures"] for item in object_statistics.values()
            )

        base_tokenizer = AutoTokenizer.from_pretrained(
            resolve(token_config["tokenizer"]), local_files_only=True
        )
        contact_tokenizer = AllegroContactTokenizer.build(
            base_tokenizer,
            position_bins=token_config["position_bins"],
            min_position=token_config["min_position"],
            max_position=token_config["max_position"],
        )
        metadata = {
            "task": args.task,
            "task_id": task_config["task_id"],
            "successful_episodes": len(episodes),
            "collection_attempts": attempts,
            "successful_without_state_3": successful_without_stable_contact,
            "annotation_failures": annotation_failures,
            "split": collection_config["split"],
            "seed": collection_config["seed"],
            "expert_checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
            "expert_sha256": sha256(checkpoint),
            "stable_contact": collection_config["stable_contact"],
            "contact_impulse_threshold": collection_config["contact_impulse_threshold"],
            "contact_link_order": [item.token_name for item in ALLEGRO_CONTACT_LINKS],
            "precontact_target_fill": "stable_contact_inclusive",
            "agent_pos": {
                "dimension": 33,
                "native_32d_conversion": "insert_zero_before_time_progress",
                "native_33d_conversion": "identity",
            },
            "language_annotation": dict(annotator.generation_metadata()),
            "camera": language_config["camera"],
            "contact_tokenizer": token_config,
        }
        if object_statistics is not None:
            metadata["balance"] = {
                "quota_per_object": balanced_settings[1],
                "max_attempts_per_object": balanced_settings[2],
                "expected_object_ids": balanced_settings[0],
                "objects": object_statistics,
            }
        write_dexart_dataset(
            output,
            episodes,
            contact_tokenizer,
            token_config["max_length"],
            metadata,
        )
        if object_statistics is not None:
            update_distribution(data_config, args.task, "complete", object_statistics)
        print(json.dumps({"output": str(output), **metadata}, indent=2))
    except Exception as error:
        if balanced_settings is not None:
            update_distribution(
                data_config,
                args.task,
                "failed",
                object_statistics,
                f"{type(error).__name__}: {error}",
            )
        raise


if __name__ == "__main__":
    main()
