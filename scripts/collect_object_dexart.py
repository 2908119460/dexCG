#!/usr/bin/env python3
"""Resume balanced, object-only DexArt collection with explicit robot-base coordinates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json

import numpy as np
import torch
from collect_dexart import collect_episode, load_yaml, resolve, seed_everything, sha256
from transformers import AutoTokenizer

from dexcg.annotation import AnnotationFormatError, GemmaGraspAnnotator
from dexcg.data.robot_base import (
    ROBOT_BASE_CONTRACT,
    InvalidObjectObservation,
    RobotBaseCollectionAdapter,
    append_episode,
    balanced_quotas,
    recover_store,
    validate_episode,
)
from dexcg.models.contact.tokenizer import AllegroContactTokenizer
from dexcg.policy.dexart_expert import load_dexart_expert


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def attempt_seed(seed, task, object_id, attempt):
    text = f"{seed}:{task}:{object_id}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


class AuditAnnotator:
    """No model loading and no dataset writes during geometry preflight."""

    def annotate(self, *args):
        return {}, ""


def audit_live(adapter, sample, original):
    env = adapter.environment
    transform = env.robot.get_pose().to_transformation_matrix()
    camera = adapter.highres_camera if adapter.highres_camera is not None else env.cameras["instance_1"]
    camera_world = camera.get_model_matrix()
    from_base = transform @ sample["camera_to_robot_base"]
    errors = {"camera_transform_max_error": float(abs(camera_world - from_base).max())}
    source = env.object_cloud_source
    pixels = sample["point_pixel_index"]
    camera_xyz = source["camera_xyz"][pixels]
    world_xyz = camera_xyz @ camera_world[:3, :3].T + camera_world[:3, 3]
    restored = sample["point_cloud"] @ transform[:3, :3].T + transform[:3, 3]
    errors["point_world_roundtrip_max_error_m"] = float(abs(world_xyz - restored).max())
    errors["palm_world_roundtrip_max_error_m"] = float(
        abs(
            sample["state"][28:31] @ transform[:3, :3].T
            + transform[:3, 3]
            - env.palm_link.get_pose().p
        ).max()
    )
    for key, section, actual in (
        ("linear_velocity_roundtrip", slice(22, 25), env.palm_link.get_velocity()),
        ("angular_velocity_roundtrip", slice(25, 28), env.palm_link.get_angular_velocity()),
    ):
        errors[key] = float(abs(sample["state"][section] @ transform[:3, :3].T - actual).max())
    source_depth = np.asarray(sample["depth"])
    depth = np.asarray(source_depth).reshape(-1)[pixels]
    errors["depth_max_error_m"] = float(abs(depth + camera_xyz[:, 2]).max())
    if max(errors.values()) > 2e-5:
        raise RuntimeError(f"Live coordinate audit failed: {errors}")
    optical_xyz = camera_xyz * np.array([1, -1, -1])
    homogeneous_uv = optical_xyz @ source["intrinsics"].T
    uv = homogeneous_uv[:, :2] / homogeneous_uv[:, 2:]
    width = np.asarray(source_depth).shape[1]
    expected_uv = np.stack([pixels % width, pixels // width], axis=1) + 0.5
    pixel_error = float(abs(uv - expected_uv).max())
    pixel_tolerance = 0.02
    if pixel_error > pixel_tolerance:
        raise InvalidObjectObservation(f"Camera pixel reprojection failed: {pixel_error} pixels")
    errors["pixel_reprojection_max_error_pixels"] = pixel_error
    object_ids = {link.get_id() for link in env.instance_links}
    if not set(map(int, sample["point_actor_id"])).issubset(object_ids):
        raise RuntimeError("Point cloud contains a non-object actor")
    np.testing.assert_array_equal(sample["point_actor_id"], source["actor_ids"][pixels])
    # The partial IK chain's fixed root is the articulation root in all four tasks.
    np.testing.assert_allclose(
        env.kinematic_model.start_link.get_pose().to_transformation_matrix(), transform, atol=1e-6
    )
    return errors


class AuditedAdapter(RobotBaseCollectionAdapter):
    def __init__(self, environment, impulse_threshold=1e-2):
        super().__init__(environment, impulse_threshold)
        self.audit_maxima = {}
        self.audited_frames = 0

    def observation(self, observation):
        sample = super().observation(observation)
        for key, value in audit_live(self, sample, observation).items():
            self.audit_maxima[key] = max(self.audit_maxima.get(key, 0.0), value)
        self.audited_frames += 1
        return sample

    def contact_graph(self):
        graph = super().contact_graph()
        world_points = [[] for _ in self.contact_links]
        for contact in self.environment.scene.get_contacts():
            actors = {contact.actor0, contact.actor1}
            hand = actors.intersection(self.contact_link_index)
            if len(hand) != 1 or not actors.intersection(self.object_links):
                continue
            if (
                sum(np.abs(point.impulse).sum() for point in contact.points)
                < self.impulse_threshold
            ):
                continue
            index = self.contact_link_index[hand.pop()]
            world_points[index].extend(point.position for point in contact.points)
        transform = self.environment.robot.get_pose().to_transformation_matrix()
        error = 0.0
        for index, points in enumerate(world_points):
            if bool(points) != bool(graph.mask[index]):
                raise RuntimeError("Contact mask disagrees with simulator contact actors")
            if points:
                restored = graph.points[index] @ transform[:3, :3].T + transform[:3, 3]
                error = max(error, float(abs(restored - np.mean(points, axis=0)).max()))
        if error > 1e-6:
            raise RuntimeError(f"Contact world round trip failed: {error}")
        self.audit_maxima["contact_world_roundtrip_max_error_m"] = max(
            self.audit_maxima.get("contact_world_roundtrip_max_error_m", 0.0), error
        )
        return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["faucet", "bucket", "laptop", "toilet"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default="configs/data/dexart_object_balanced_250.yaml")
    parser.add_argument("--language-config", default="configs/annotation/language.yaml")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--max-attempts-per-object",
        type=int,
        help="Override the retry limit for this resume without changing the dataset config hash",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = load_yaml(resolve(args.config))
    language = load_yaml(resolve(args.language_config))
    collection = cfg["collection"]
    from dexart.env.task_setting import TRAIN_CONFIG

    objects = TRAIN_CONFIG[args.task][collection["split"]]
    quotas = balanced_quotas(objects, collection["successful_episodes"], collection["seed"])
    token_cfg = cfg["contact_tokens"]
    tokenizer = AllegroContactTokenizer.build(
        AutoTokenizer.from_pretrained(resolve(token_cfg["tokenizer"]), local_files_only=True),
        position_bins=token_cfg["position_bins"],
        min_position=token_cfg["min_position"],
        max_position=token_cfg["max_position"],
    )
    output_dir = resolve(collection["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # A persistent lock inode is safe for concurrent invocations and resumptions.
    with (output_dir / f"{args.task}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, cfg, language, objects, quotas, tokenizer, output_dir)


def run(args, cfg, language, objects, quotas, tokenizer, output_dir):
    collection, task = cfg["collection"], cfg["tasks"][args.task]
    path = output_dir / f"{args.task}_expert.zarr"
    progress_path = output_dir / f"{args.task}_progress.json"
    signature = hashlib.sha256(
        json.dumps({"config": cfg, "language": language}, sort_keys=True).encode()
    ).hexdigest()
    progress = {
        "task": args.task,
        "status": "preflight" if args.preflight else "running",
        "config_sha256": signature,
        "objects": {
            str(i): {
                "quota": quotas[str(i)],
                "attempts": 0,
                "accepted": 0,
                "annotation_failures": 0,
                "invalid_observations": 0,
                "simulator_successes": 0,
            }
            for i in objects
        },
    }
    root = None
    if not args.preflight:
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["config_sha256"] != signature:
                raise ValueError("Resume configuration differs from the original run")
        if path.exists():
            root = recover_store(path)
            stored_signature = root.attrs.get("config_sha256")
            if stored_signature != signature and (
                root.attrs.get("committed_episodes", 0) or stored_signature is not None
            ):
                raise ValueError("Dataset configuration differs from the requested run")
            for item in progress["objects"].values():
                item["accepted"] = 0
            if root.attrs.get("committed_episodes", 0):
                for obj in root["meta/object_id"][:]:
                    progress["objects"][str(obj)]["accepted"] += 1
        progress.pop("error", None)
        progress["status"] = "running"
        atomic_json(progress_path, progress)
    checkpoint = resolve(task["checkpoint"])
    attributes = {
        "format": "dexcg.dexart.object_robot_base.v1",
        "task": args.task,
        "task_id": task["task_id"],
        "split": collection["split"],
        "seed": collection["seed"],
        "config_sha256": signature,
        "expert_checkpoint": task["checkpoint"],
        "expert_sha256": sha256(checkpoint),
        "expected_object_ids": list(objects),
        "object_quotas": quotas,
        "contact_tokenizer": cfg["contact_tokens"],
        "point_cloud_frame": "robot_base",
        "contact_point_frame": "robot_base",
        "contact_token_frame": "robot_base",
        "state_spatial_frame": "robot_base",
        "length_unit": "metre",
        "robot_base_definition": (
            "DexArt robot.get_pose(); URDF root named world, not simulation world"
        ),
        "contact_coordinate_contract": ROBOT_BASE_CONTRACT,
        "point_cloud_source": (
            "full-resolution visible instance_links actor IDs; no scene crop or noise"
        ),
        "object_center_definition": "full-resolution visible object AABB midpoint in robot_base",
        "point_count": collection["point_count"],
        "sensor_contract": collection.get("sensor_contract"),
        "minimum_distinct_object_pixels": collection.get("minimum_distinct_object_pixels", 1),
        "pixel_reprojection_tolerance": 0.02,
        "depth_reconstruction_tolerance_m": 1e-4,
        "point_capture_resolution": collection.get("point_resolution"),
        "expert_observations": (
            "unmodified native DexArt including world-frame state and noisy scene cloud"
        ),
        "state_layout": {
            "0:22": "joint qpos, radians",
            "22:25": "palm linear velocity, base m/s",
            "25:28": "palm angular velocity, base rad/s",
            "28:31": "palm XYZ, base metres",
            "31 (bucket only)": "palm local +X direction dot base +Z",
            "last": "current_step / horizon",
        },
        "agent_pos_layout": "33D state; non-bucket inserts zero at index 31",
        "action_semantics": {
            "0:3": "normalized base-frame end-link center-of-mass linear velocity",
            "3:6": "normalized base-frame end-link angular velocity",
            "6:22": "normalized hand joint target angles",
            "range": [-1, 1],
            "execution": "DexArt recover_action and IK",
        },
        "image_depth_semantics": (
            "RGB uint8; depth is positive camera optical-axis distance in metres"
        ),
        "camera_to_robot_base": "maps OpenGL camera XYZ to robot-base XYZ",
        "annotation_camera_extrinsics": "maps SAPIEN camera local (+X forward) to robot-base",
        "contact_target_rule": "stable contact through stable step inclusive, then current contact",
        "contact_impulse_threshold": collection["contact_impulse_threshold"],
    }
    annotator = (
        AuditAnnotator()
        if args.preflight
        else GemmaGraspAnnotator(
            checkpoint=resolve(language["checkpoint"]),
            prompt_path=resolve(language["prompt"]),
            device=args.device,
            max_new_tokens=language["max_new_tokens"],
            revision=language["revision"],
        )
    )
    if not args.preflight:
        attributes["language_annotation"] = dict(annotator.generation_metadata())
    expert = None
    accepted_this_run = 0
    preflight = {"task": args.task, "config_sha256": signature, "sensor_contract": collection.get("sensor_contract"), "objects": {}, "successful_rollout": False}
    try:
        for object_id in objects:
            item = progress["objects"][str(object_id)]
            if not args.preflight and item["accepted"] >= item["quota"]:
                continue
            adapter = AuditedAdapter.create(
                args.task, collection["split"], collection["contact_impulse_threshold"], object_id
            )
            if "point_resolution" in collection:
                adapter.enable_high_resolution_object_cloud(collection["point_resolution"])
            adapter.point_count = collection["point_count"]
            try:
                if args.preflight:
                    adapter.environment.seed(
                        attempt_seed(collection["seed"], args.task, object_id, 0)
                    )
                    obs = adapter.reset()
                    sample = adapter.observation(obs)
                    preflight["objects"][str(object_id)] = {
                        "world_from_base": adapter.environment.robot.get_pose()
                        .to_transformation_matrix()
                        .tolist(),
                        "root_link": adapter.environment.robot.get_links()[0].get_name(),
                        "object_actor_ids": [
                            link.get_id() for link in adapter.environment.instance_links
                        ],
                        "visible_pixels": int(sample["object_visible_pixel_count"]),
                        "point_xyz_min": sample["point_cloud"].min(0).tolist(),
                        "point_xyz_max": sample["point_cloud"].max(0).tolist(),
                        "errors": adapter.audit_maxima.copy(),
                    }
                    if preflight["successful_rollout"]:
                        continue
                if expert is None:
                    expert = load_dexart_expert(checkpoint, adapter.environment, args.device)
                else:
                    expert.set_env(adapter.environment)
                attempts = (
                    3
                    if args.preflight
                    else (
                        args.max_attempts_per_object
                        if args.max_attempts_per_object is not None
                        else collection["max_attempts_per_object"]
                    )
                )
                if attempts < 1:
                    raise ValueError("max attempts per object must be positive")
                while (
                    item["accepted"] < (1 if args.preflight else item["quota"])
                    and item["attempts"] < attempts
                ):
                    item["attempts"] += 1
                    seed = attempt_seed(collection["seed"], args.task, object_id, item["attempts"])
                    seed_everything(seed)
                    adapter.environment.seed(seed)
                    adapter.point_rng = np.random.RandomState(seed ^ 0xA5A5A5A5)
                    if not args.preflight:
                        atomic_json(progress_path, progress)
                    try:
                        episode, success = collect_episode(
                            adapter,
                            expert,
                            annotator,
                            task["task_id"],
                            args.device,
                            language["camera"],
                        )
                        item["simulator_successes"] += int(success)
                        if episode is None:
                            continue
                        if episode.object_id != str(object_id):
                            raise RuntimeError(
                                "Simulator returned a different object than requested"
                            )
                        error = validate_episode(episode, tokenizer)
                    except AnnotationFormatError as exc:
                        item["annotation_failures"] += 1
                        print(f"annotation failure {args.task}/{object_id}: {exc}", flush=True)
                        continue
                    except InvalidObjectObservation as exc:
                        item["invalid_observations"] += 1
                        print(f"invalid observation {args.task}/{object_id}: {exc}", flush=True)
                        continue
                    if args.preflight:
                        points, masks = episode.contact_targets()
                        preflight.update(
                            {
                                "successful_rollout": True,
                                "rollout_object": object_id,
                                "rollout_steps": len(episode.actions),
                                "contact_xyz_min": points[masks].min(0).tolist(),
                                "contact_xyz_max": points[masks].max(0).tolist(),
                                "token_roundtrip_max_error_m": error,
                                "rollout_errors": adapter.audit_maxima,
                            }
                        )
                    else:
                        env = adapter.environment
                        append_episode(
                            path,
                            episode,
                            tokenizer,
                            attributes,
                            {
                                "attempt": item["attempts"],
                                "seed": np.uint32(seed),
                                "world_from_robot_base": (
                                    env.robot.get_pose().to_transformation_matrix()
                                ),
                                "action_velocity_limits": env.velocity_limit,
                                "robot_joint_limits": env.robot.get_qlimits(),
                                "control_time_step": env.control_time_step,
                                "simulator_success": True,
                                "object_actor_ids_json": json.dumps(
                                    [link.get_id() for link in env.instance_links]
                                ),
                            },
                        )
                    item["accepted"] += 1
                    accepted_this_run += 1
                    progress["audited_frames"] = (
                        progress.get("audited_frames", 0) + adapter.audited_frames
                    )
                    adapter.audited_frames = 0
                    progress.setdefault("coordinate_audit_maxima", {})
                    for key, value in adapter.audit_maxima.items():
                        progress["coordinate_audit_maxima"][key] = max(
                            progress["coordinate_audit_maxima"].get(key, 0.0), value
                        )
                    total = sum(stat["accepted"] for stat in progress["objects"].values())
                    print(
                        f"ACCEPT {args.task} total={total}/250 object={object_id} "
                        f"count={item['accepted']}/{item['quota']} attempt={item['attempts']}",
                        flush=True,
                    )
                    if not args.preflight:
                        atomic_json(progress_path, progress)
                        if args.stop_after and accepted_this_run >= args.stop_after:
                            progress["status"] = "paused"
                            atomic_json(progress_path, progress)
                            return
                if not args.preflight and item["accepted"] < item["quota"]:
                    raise RuntimeError(f"{args.task}/{object_id} exhausted attempts: {item}")
            finally:
                adapter.close()
        if args.preflight:
            if not preflight["successful_rollout"]:
                raise RuntimeError("No successful preflight rollout")
            preflight["passed"] = True
            atomic_json(output_dir / f"{args.task}_preflight.json", preflight)
        else:
            root = recover_store(path)
            if root.attrs["committed_episodes"] != collection["successful_episodes"]:
                raise RuntimeError("Final episode count mismatch")
            from audit_object_dataset import audit_robot_base

            report = audit_robot_base(path, tokenizer)
            atomic_json(output_dir / f"{args.task}_audit.json", report)
            root.attrs.update(
                {"complete": True, "successful_episodes": root.attrs["committed_episodes"]}
            )
            progress["status"] = "complete"
            atomic_json(progress_path, progress)
    except Exception as exc:
        if not args.preflight:
            progress.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            atomic_json(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
