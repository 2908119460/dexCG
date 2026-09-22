#!/usr/bin/env python3
"""Read every stored frame to audit label purity and robot-frame contact ranges.

Stored masks establish label purity only; renderer actor-ID provenance and frame
transforms must also be checked at collection time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from dexcg.data.robot_base import ROBOT_BASE_CONTRACT, validate_sensor_frame


def audit(path: Path, expected_episodes: int, lower: float, upper: float) -> dict:
    root = zarr.open_group(str(path), mode="r")
    points = root["data/point_cloud"]
    masks = root["data/object_point_mask"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    objects = [str(value) for value in root["meta/object_id"][:]]
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] == 0:
        raise ValueError(f"{path}: expected nonempty XYZ point clouds")
    if masks.shape != points.shape[:-1]:
        raise ValueError(f"{path}: point/mask shapes do not match")
    if (
        not len(ends)
        or len(objects) != len(ends)
        or ends[-1] != len(points)
        or np.any(np.diff(np.r_[0, ends]) <= 0)
    ):
        raise ValueError(f"{path}: invalid episode boundaries or object metadata")

    counts = Counter(objects)
    object_frames = Counter()
    for object_id, length in zip(objects, np.diff(np.r_[0, ends]), strict=True):
        object_frames[object_id] += int(length)
    impure_frames = empty_frames = nonfinite_points = object_points = 0
    minimum_fraction = 1.0
    first_impure_frames = []
    chunk_size = 128
    for start in range(0, len(points), chunk_size):
        xyz = np.asarray(points[start : start + chunk_size])
        labels = np.asarray(masks[start : start + chunk_size])
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"{path}: nonbinary object mask near frame {start}")
        labels = labels.astype(bool)
        fractions = labels.mean(axis=1)
        bad = np.flatnonzero(~labels.all(axis=1)) + start
        first_impure_frames.extend(bad[: max(0, 10 - len(first_impure_frames))].tolist())
        impure_frames += len(bad)
        empty_frames += int((~labels.any(axis=1)).sum())
        nonfinite_points += int((~np.isfinite(xyz).all(axis=-1)).sum())
        object_points += int(labels.sum())
        minimum_fraction = min(minimum_fraction, float(fractions.min()))

    contact_ranges = {}
    for kind in ("raw", "target"):
        coordinates = root[f"data/contact_{kind}_points"]
        active = root[f"data/contact_{kind}_mask"]
        if coordinates.shape[:-1] != active.shape or coordinates.shape[-1] != 3:
            raise ValueError(f"{path}: invalid {kind} contact shapes")
        if len(coordinates) != len(points):
            raise ValueError(f"{path}: {kind} contacts do not align with frames")
        minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
        valid_count = outside_count = nonfinite_count = 0
        for start in range(0, len(points), chunk_size):
            labels = np.asarray(active[start : start + chunk_size])
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"{path}: nonbinary {kind} contact mask")
            xyz = np.asarray(coordinates[start : start + chunk_size])[labels.astype(bool)]
            finite = np.isfinite(xyz).all(axis=1)
            nonfinite_count += int((~finite).sum())
            xyz = xyz[finite]
            if not len(xyz):
                continue
            minimum = np.minimum(minimum, xyz.min(axis=0))
            maximum = np.maximum(maximum, xyz.max(axis=0))
            valid_count += len(xyz)
            outside_count += int(((xyz < lower) | (xyz > upper)).any(axis=1).sum())
        contact_ranges[kind] = {
            "finite_active_contacts": valid_count,
            "nonfinite_active_contacts": nonfinite_count,
            "xyz_min": minimum.tolist() if valid_count else None,
            "xyz_max": maximum.tolist() if valid_count else None,
            "outside_proposed_token_range": outside_count,
            "outside_fraction": outside_count / valid_count if valid_count else None,
        }

    return {
        "path": str(path),
        "episodes": len(ends),
        "expected_episode_count_met": len(ends) == expected_episodes,
        "frames": len(points),
        "episodes_per_observed_object": dict(sorted(counts.items())),
        "frames_per_observed_object": dict(sorted(object_frames.items())),
        "observed_object_episode_count_spread": max(counts.values()) - min(counts.values()),
        "object_point_fraction": object_points / int(np.prod(masks.shape)),
        "minimum_frame_object_fraction": minimum_fraction,
        "impure_frames": impure_frames,
        "empty_object_frames": empty_frames,
        "first_impure_frames": first_impure_frames,
        "nonfinite_points": nonfinite_points,
        "all_frames_label_pure_and_finite": impure_frames == 0 and nonfinite_points == 0,
        "proposed_token_range": [lower, upper],
        "stored_contact_ranges": contact_ranges,
        "limitations": [
            "Stored masks cannot independently prove renderer actor-ID provenance.",
            "Observed object counts do not establish coverage of an expected object list.",
            "Stored coordinates alone do not prove their reference frame or units.",
        ],
    }


def audit_robot_base(path: Path, tokenizer, require_complete: bool = True) -> dict:
    """Independently decode stored tokens and reconstruct all clouds from depth pixels."""
    from dexart.env.task_setting import TRAIN_CONFIG

    from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

    root = zarr.open_group(str(path), mode="r")
    if root.attrs.get("contact_coordinate_contract") != ROBOT_BASE_CONTRACT:
        raise ValueError("Expected a robot-base dataset")
    data, meta = root["data"], root["meta"]
    strict_sensor = root.attrs.get("sensor_contract") == "aligned_rgbd_object_10000_v2"
    if strict_sensor:
        resolution = tuple(root.attrs["point_capture_resolution"])[::-1]
        if data["point_cloud"].shape[1:] != (10000, 3) or data["depth"].shape[1:] != resolution:
            raise ValueError("Dataset dimensions violate sensor contract")
        if data["img"].shape[1:] != (*resolution, 3) or data["segmentation_actor_id"].shape[1:] != resolution:
            raise ValueError("RGB/depth/segmentation do not share a resolution")

    ends = np.asarray(meta["episode_ends"][:])
    objects = list(map(str, meta["object_id"][:]))
    counts = Counter(objects)
    expected_ids = set(map(str, TRAIN_CONFIG[root.attrs["task"]][root.attrs["split"]]))
    quotas = root.attrs["object_quotas"]
    if not set(counts).issubset(expected_ids):
        raise ValueError("Unexpected object ID")
    coverage = set(counts) == expected_ids
    quotas_met = all(counts[obj] == quotas[obj] for obj in expected_ids)
    if require_complete and (not coverage or not quotas_met or len(ends) != 250):
        raise ValueError(f"Incomplete object quotas: {dict(counts)}")
    if root.attrs["committed_episodes"] != len(ends):
        raise ValueError("Uncommitted episode data")
    for name, array in data.arrays():
        if len(array) != ends[-1]:
            raise ValueError(f"Step count mismatch: {name}")
    for group in (meta, root["annotation"]):
        for name, array in group.arrays():
            if len(array) != len(ends):
                raise ValueError(f"Episode count mismatch: {name}")
    if not np.asarray(meta["simulator_success"][:]).all():
        raise ValueError("Stored episode is not a simulator success")
    maximum = {
        "depth_to_point_max_error_m": 0.0,
        "palm_position_max_error_m": 0.0,
        "token_roundtrip_max_error_m": 0.0,
    }
    maximum["pixel_reprojection_max_error_pixels"] = 0.0
    minimum_visible_pixels = None
    repeated_frames = saturated_actions = action_values = 0
    contact_count = 0
    xyz_min, xyz_max = np.full(3, np.inf), np.full(3, -np.inf)
    actions_min, actions_max = np.full(22, np.inf), np.full(22, -np.inf)
    for episode, (start, end) in enumerate(zip(np.r_[0, ends[:-1]], ends, strict=True)):
        allowed_actors = json.loads(str(meta["object_actor_ids_json"][episode]))
        stable_step = int(meta["stable_contact_steps"][episode])
        if not 0 <= stable_step < end - start:
            raise ValueError("Stable step outside trajectory")
        frame_chunk = 4 if strict_sensor else 128
        for lo in range(int(start), int(end), frame_chunk):
            hi = min(lo + frame_chunk, int(end))
            if strict_sensor:
                for index in range(lo, hi):
                    keys = ("depth", "img", "segmentation_actor_id", "point_pixel_index", "point_cloud",
                            "camera_intrinsics", "camera_to_robot_base", "object_visible_pixel_count",
                            "object_point_mask", "point_actor_id")
                    sample = {key: np.asarray(data[key][index]) for key in keys}
                    quality = validate_sensor_frame(sample, allowed_actors, 10000)
                    for key, value in quality.items():
                        maximum[key] = max(maximum.get(key, 0.0), value)
                        recorded = float(data[key][index])
                        if not np.isfinite(recorded) or not np.isclose(recorded, value, rtol=1e-5, atol=1e-8):
                            raise ValueError(f"Stored quality metric disagrees at frame {index}: {key}")
                    visible_count = int(sample["object_visible_pixel_count"])
                    minimum_visible_pixels = visible_count if minimum_visible_pixels is None else min(minimum_visible_pixels, visible_count)
            xyz = np.asarray(data["point_cloud"][lo:hi])
            if not np.isfinite(xyz).all() or not np.asarray(data["object_point_mask"][lo:hi]).all():
                raise ValueError("Nonfinite or impure point cloud")
            xyz_min = np.minimum(xyz_min, xyz.min(axis=(0, 1)))
            xyz_max = np.maximum(xyz_max, xyz.max(axis=(0, 1)))
            if not np.isin(data["point_actor_id"][lo:hi], allowed_actors).all():
                raise ValueError("Non-object actor ID stored")
            pixels = np.asarray(data["point_pixel_index"][lo:hi])
            depth = np.asarray(data["depth"][lo:hi])
            width = depth.shape[-1]
            if (pixels < 0).any() or (pixels >= np.prod(depth.shape[1:])).any():
                raise ValueError("Invalid pixel index")
            for pixel_row in pixels:
                repeated_frames += int(len(np.unique(pixel_row)) < len(pixel_row))
            z = np.take_along_axis(depth.reshape(len(depth), -1), pixels, axis=1)
            intrinsics = np.asarray(data["camera_intrinsics"][lo:hi])
            x = (pixels % width + 0.5 - intrinsics[:, None, 0, 2]) * z / intrinsics[:, None, 0, 0]
            y = (pixels // width + 0.5 - intrinsics[:, None, 1, 2]) * z / intrinsics[:, None, 1, 1]
            camera_xyz = np.stack([x, -y, -z], axis=-1)
            transform = np.asarray(data["camera_to_robot_base"][lo:hi])
            restored = np.einsum("bnj,bij->bni", camera_xyz, transform[:, :3, :3])
            restored += transform[:, None, :3, 3]
            maximum["depth_to_point_max_error_m"] = max(
                maximum["depth_to_point_max_error_m"], float(abs(restored - xyz).max())
            )
            state = np.asarray(data["state"][lo:hi])
            palm = np.asarray(data["palm_pose_robot_base"][lo:hi])
            maximum["palm_position_max_error_m"] = max(
                maximum["palm_position_max_error_m"],
                float(abs(state[:, 28:31] - palm[:, :3, 3]).max()),
            )
            action = np.asarray(data["action"][lo:hi])
            if not np.isfinite(action).all() or (abs(action) > 1.000001).any():
                raise ValueError("Invalid normalized action")
            actions_min = np.minimum(actions_min, action.min(0))
            actions_max = np.maximum(actions_max, action.max(0))
            saturated_actions += int((abs(action) >= 1 - 1e-6).sum())
            action_values += action.size
            for kind in ("raw", "target"):
                points = np.asarray(data[f"contact_{kind}_points"][lo:hi])
                masks = np.asarray(data[f"contact_{kind}_mask"][lo:hi])
                active = points[masks]
                if (
                    not np.isfinite(active).all()
                    or ((active < tokenizer.min_position) | (active > tokenizer.max_position)).any()
                ):
                    raise ValueError("Contact would be clipped by token bounds")
                contact_count += len(active)
                ids = np.asarray(data[f"contact_{kind}_token_ids"][lo:hi])
                token_masks = np.asarray(data[f"contact_{kind}_token_mask"][lo:hi])
                for row, row_mask, row_ids, token_mask in zip(
                    points, masks, ids, token_masks, strict=True
                ):
                    decoded = tokenizer.decode(row_ids[token_mask])
                    expected_names = {
                        ALLEGRO_CONTACT_LINKS[i].token_name for i in np.flatnonzero(row_mask)
                    }
                    if set(decoded) != expected_names:
                        raise ValueError("Contact token links disagree with contact mask")
                    for index in np.flatnonzero(row_mask):
                        point = decoded[ALLEGRO_CONTACT_LINKS[index].token_name][0]
                        maximum["token_roundtrip_max_error_m"] = max(
                            maximum["token_roundtrip_max_error_m"],
                            float(abs(point - row[index]).max()),
                        )
    tolerance = (
        (tokenizer.max_position - tokenizer.min_position) / (tokenizer.position_bins - 1) / 2
    )
    if maximum["token_roundtrip_max_error_m"] > tolerance + 1e-6:
        raise ValueError(f"Contact token round trip failed: {maximum}")
    if maximum["depth_to_point_max_error_m"] > 1e-4 or maximum["palm_position_max_error_m"] > 1e-6:
        raise ValueError(f"Coordinate reconstruction failed: {maximum}")
    return {
        "passed": True,
        "episodes": len(ends),
        "frames": int(ends[-1]),
        "episodes_per_object": dict(sorted(counts.items())),
        "expected_object_coverage": coverage,
        "exact_quotas_met": quotas_met,
        "contact_token_frame": "robot_base",
        "active_raw_and_target_contacts": contact_count,
        "contact_clipping_count": 0,
        "all_stored_point_actor_ids_belong_to_target": True,
        "point_xyz_min": xyz_min.tolist(),
        "point_xyz_max": xyz_max.tolist(),
        "frames_with_repeated_object_pixels": repeated_frames,
        "sensor_contract": root.attrs.get("sensor_contract"),
        "minimum_visible_object_pixels": minimum_visible_pixels,
        "action_min": actions_min.tolist(),
        "action_max": actions_max.tolist(),
        "action_component_saturation_fraction": saturated_actions / action_values,
        "action_linear_velocity_reference_point": "DexArt end-link center of mass",
        **maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=250)
    parser.add_argument("--min-position", type=float, default=-0.4)
    parser.add_argument("--max-position", type=float, default=0.4)
    parser.add_argument("--robot-base", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.expected_episodes < 1:
        parser.error("expected episodes must be positive")
    if not np.isfinite([args.min_position, args.max_position]).all() or not (
        args.min_position < args.max_position
    ):
        parser.error("position bounds must be finite and increasing")
    if args.robot_base:
        from transformers import AutoTokenizer

        from dexcg.models.contact.tokenizer import AllegroContactTokenizer

        reports = []
        for path in args.paths:
            root = zarr.open_group(str(path), mode="r")
            config = root.attrs["contact_tokenizer"]
            tokenizer = AllegroContactTokenizer.build(
                AutoTokenizer.from_pretrained(
                    Path(__file__).resolve().parents[1] / config["tokenizer"],
                    local_files_only=True,
                ),
                position_bins=config["position_bins"],
                min_position=config["min_position"],
                max_position=config["max_position"],
            )
            reports.append(audit_robot_base(path, tokenizer, not args.allow_incomplete))
    else:
        reports = [
            audit(path, args.expected_episodes, args.min_position, args.max_position)
            for path in args.paths
        ]
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
