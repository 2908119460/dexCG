"""Auditable object-only demonstrations in the DexArt articulation-root frame."""

from __future__ import annotations

from pathlib import Path

import numcodecs
import numpy as np
import zarr

from dexcg.data.dexart import DexArtEpisode, encode_contact_graphs
from dexcg.envs.dexart import DexArtAdapter

ROBOT_BASE_CONTRACT = "dexart_robot_base_metric_v1"


class InvalidObjectObservation(ValueError):
    pass


def base_state(state: np.ndarray, world_from_base: np.ndarray) -> np.ndarray:
    """DexArt: 22 qpos, world linear/angular palm velocity, world palm XYZ, [z], time."""
    result = np.asarray(state, dtype=np.float32).copy()
    if result.shape not in ((32,), (33,)):
        raise ValueError(f"Unexpected DexArt state shape: {result.shape}")
    rotation = world_from_base[:3, :3]
    result[22:25] = result[22:25] @ rotation
    result[25:28] = result[25:28] @ rotation
    result[28:31] = (result[28:31] - world_from_base[:3, 3]) @ rotation
    return result


def sample_object_points(source, object_ids, rng, point_count=1024, *, require_unique=False):
    camera_xyz = source["camera_xyz"]
    labels = source["actor_ids"]
    valid = (
        np.isin(labels, object_ids)
        & np.isfinite(camera_xyz).all(axis=1)
        & (camera_xyz[:, 2] < -0.05)
    )
    pixels = np.flatnonzero(valid)
    if not len(pixels):
        raise InvalidObjectObservation("No finite visible target-object pixels")
    if require_unique and len(pixels) < point_count:
        raise InvalidObjectObservation(f"Insufficient distinct object pixels: {len(pixels)} < {point_count}")
    transform = source["base_from_camera_gl"]
    full_xyz = camera_xyz[pixels] @ transform[:3, :3].T + transform[:3, 3]
    center = (full_xyz.min(axis=0) + full_xyz.max(axis=0)) * 0.5
    if len(pixels) < point_count:
        selected = np.r_[
            np.arange(len(pixels)), rng.choice(len(pixels), point_count - len(pixels), replace=True)
        ]
        rng.shuffle(selected)
    else:
        selected = rng.choice(len(pixels), point_count, replace=False)
    return {
        "point_cloud": full_xyz[selected].astype(np.float32),
        "object_point_mask": np.ones(point_count, dtype=bool),
        "object_center": center.astype(np.float32),
        "object_visible_pixel_count": np.asarray(len(pixels), dtype=np.int32),
        "point_actor_id": labels[pixels[selected]].astype(np.int32),
        "point_pixel_index": pixels[selected].astype(np.int32),
        "camera_to_robot_base": transform.astype(np.float32),
        "camera_intrinsics": source["intrinsics"].astype(np.float32),
    }


class RobotBaseCollectionAdapter(DexArtAdapter):
    def __init__(self, environment, impulse_threshold=1e-2):
        super().__init__(environment, impulse_threshold)
        environment.capture_object_cloud_source = True
        self.point_rng = np.random.RandomState(0)
        self.point_count = 1024
        self.highres_camera = None
        self.highres_camera_name = None

    def enable_high_resolution_object_cloud(self, resolution):
        """Add a sensor-aligned camera for synchronized RGB, depth and segmentation."""

        width, height = (int(resolution[0]), int(resolution[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid high-resolution camera size: {resolution}")
        name = "dexcg_highres_instance_1"
        if name in self.environment.cameras:
            raise RuntimeError(f"camera already exists: {name}")
        source_camera = self.environment.cameras["instance_1"]
        source_intrinsics = source_camera.get_intrinsic_matrix()
        source_height = float(source_camera.height)
        fovy = 2.0 * np.arctan(source_height / (2.0 * float(source_intrinsics[1, 1])))
        camera = self.environment.scene.add_camera(
            name,
            width=width,
            height=height,
            fovy=float(fovy),
            near=0.1,
            far=10,
        )
        scaled = np.asarray(source_intrinsics, dtype=np.float64).copy()
        scaled[0] *= width / source_camera.width
        scaled[1] *= height / source_camera.height
        camera.set_perspective_parameters(
            source_camera.near, source_camera.far,
            scaled[0, 0], scaled[1, 1], scaled[0, 2], scaled[1, 2], scaled[0, 1],
        )
        np.testing.assert_allclose(camera.get_intrinsic_matrix(), scaled, atol=1e-4, rtol=0)
        camera.set_local_pose(source_camera.get_pose())
        self.environment.cameras[name] = camera
        self.highres_camera = camera
        self.highres_camera_name = name

    def _capture_highres_source(self):
        if self.highres_camera is None:
            return
        camera = self.highres_camera
        camera.set_local_pose(self.environment.cameras["instance_1"].get_pose())
        self.environment.scene.update_render()
        camera.take_picture()
        # Synchronous copies of attachments from one picture, without another render.
        position = camera.get_float_texture("Position")[..., :3].copy()
        actor_ids = camera.get_uint32_texture("Segmentation")[..., 1].astype(np.int32)
        rgb = np.rint(np.clip(camera.get_float_texture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
        depth = -position[..., 2].copy()
        valid = np.isfinite(position).all(axis=-1) & (depth >= camera.near) & (depth < camera.far)
        depth[~valid] = 0
        position[~valid] = 0
        self.environment.object_cloud_source = {
            "camera_xyz": position.reshape(-1, 3),
            "actor_ids": actor_ids.reshape(-1),
            "base_from_camera_gl": self.environment.get_camera_to_robot_pose(self.highres_camera_name),
            "intrinsics": camera.get_intrinsic_matrix().copy(),
            "depth": depth,
            "rgb": rgb,
            "segmentation_actor_id": actor_ids,
            "resolution": (int(camera.height), int(camera.width)),
        }

    def observation(self, observation):
        environment = self.environment
        self._capture_highres_source()
        world_from_base = environment.robot.get_pose().to_transformation_matrix()
        root_pose = environment.robot.get_links()[0].get_pose().to_transformation_matrix()
        np.testing.assert_allclose(world_from_base, root_pose, atol=1e-6)
        object_ids = [link.get_id() for link in environment.instance_links]
        robot_ids = [link.get_id() for link in environment.robot.get_links()]
        if set(object_ids) & set(robot_ids):
            raise RuntimeError("Object and robot actor IDs overlap")
        sample = sample_object_points(
            environment.object_cloud_source, object_ids, self.point_rng, self.point_count,
            require_unique=self.highres_camera is not None
        )
        state = base_state(observation["state"], world_from_base)
        if len(state) == 33:
            rotation = world_from_base[:3, :3]
            state[31] = (np.asarray(environment.palm_vector) @ rotation)[2]
        agent_pos = state if len(state) == 33 else np.r_[state[:31], np.float32(0), state[31:]]
        sample.update(
            {
                "img": np.rint(np.clip(observation["instance_1-rgb"], 0, 1) * 255).astype(np.uint8),
                "depth": np.asarray(observation["instance_1-depth"], dtype=np.float32),
                "state": state,
                "agent_pos": agent_pos.astype(np.float32),
                "palm_pose_robot_base": self.robot_proprioception()["palm_pose_robot_base"],
            }
        )
        if self.highres_camera is not None:
            source = environment.object_cloud_source
            sample["img"] = source["rgb"]
            sample["depth"] = source["depth"]
            sample["segmentation_actor_id"] = source["segmentation_actor_id"]
            quality = validate_sensor_frame(sample, object_ids, self.point_count)
            sample.update({key: np.asarray(value, dtype=np.float32) for key, value in quality.items()})
        return sample


def validate_sensor_frame(sample, object_ids, point_count=10000):
    """Independently reconstruct selected XYZ from the values that will be stored."""
    depth = np.asarray(sample["depth"])
    image = np.asarray(sample["img"])
    segmentation = np.asarray(sample["segmentation_actor_id"])
    pixels = np.asarray(sample["point_pixel_index"])
    points = np.asarray(sample["point_cloud"])
    intrinsic = np.asarray(sample["camera_intrinsics"], dtype=np.float64)
    transform = np.asarray(sample["camera_to_robot_base"], dtype=np.float64)
    if depth.ndim != 2 or image.shape != (*depth.shape, 3) or segmentation.shape != depth.shape:
        raise InvalidObjectObservation("RGB, depth and segmentation resolutions disagree")
    if depth.dtype != np.float32 or image.dtype != np.uint8 or not np.issubdtype(segmentation.dtype, np.integer):
        raise InvalidObjectObservation("Unexpected RGB/depth/segmentation data types")
    if not np.issubdtype(pixels.dtype, np.integer) or np.asarray(sample["object_point_mask"]).shape != (point_count,):
        raise InvalidObjectObservation("Invalid pixel-index type or mask shape")
    if points.shape != (point_count, 3) or pixels.shape != (point_count,):
        raise InvalidObjectObservation("Incorrect point count or pixel-index shape")
    if not np.isfinite(depth).all() or (depth < 0).any() or not np.isfinite(points).all():
        raise InvalidObjectObservation("Invalid depth or XYZ")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0:
        raise InvalidObjectObservation("Invalid camera intrinsics")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise InvalidObjectObservation("Invalid camera transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
        raise InvalidObjectObservation("Non-rigid camera transform")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5):
        raise InvalidObjectObservation("Camera rotation changes handedness")
    if (pixels < 0).any() or (pixels >= depth.size).any() or len(np.unique(pixels)) != point_count:
        raise InvalidObjectObservation("Out-of-bounds or repeated object pixels")
    if len(np.unique(points, axis=0)) != point_count:
        raise InvalidObjectObservation("Repeated XYZ points")
    visible = np.isin(segmentation, object_ids) & (depth > 0.05)
    if int(visible.sum()) < point_count or int(sample["object_visible_pixel_count"]) != int(visible.sum()):
        raise InvalidObjectObservation("Insufficient or inconsistent visible object count")
    if not visible.reshape(-1)[pixels].all() or not np.asarray(sample["object_point_mask"]).all():
        raise InvalidObjectObservation("Non-object or invalid selected pixels")
    if not np.array_equal(segmentation.reshape(-1)[pixels], sample["point_actor_id"]):
        raise InvalidObjectObservation("Stored actor IDs disagree with rendered segmentation")
    width = depth.shape[1]
    uv = np.stack([pixels % width + 0.5, pixels // width + 0.5, np.ones(point_count)], axis=1)
    optical = uv @ np.linalg.inv(intrinsic).T
    optical *= depth.reshape(-1)[pixels, None]
    camera_xyz = optical * [1, -1, -1]
    restored = camera_xyz @ transform[:3, :3].T + transform[:3, 3]
    metric_error = float(np.abs(restored - points).max())
    actual_camera = (points - transform[:3, 3]) @ transform[:3, :3]
    projected = (actual_camera * [1, -1, -1]) @ intrinsic.T
    if (projected[:, 2] <= 0).any():
        raise InvalidObjectObservation("Selected point behind camera")
    pixel_error = float(np.abs(projected[:, :2] / projected[:, 2:] - uv[:, :2]).max())
    if metric_error > 1e-4 or pixel_error > 0.02:
        raise InvalidObjectObservation(f"Sensor reconstruction failed: {metric_error:.8g} m, {pixel_error:.8g} px")
    return {"depth_to_point_max_error_m": metric_error, "pixel_reprojection_max_error_pixels": pixel_error}


def balanced_quotas(object_ids, total, seed):
    if total < len(object_ids) or len(set(object_ids)) != len(object_ids):
        raise ValueError("Need unique object IDs and at least one episode per object")
    base, remainder = divmod(total, len(object_ids))
    order = np.random.RandomState(seed).permutation(object_ids)
    extras = set(map(int, order[:remainder]))
    return {str(i): base + int(i in extras) for i in object_ids}


def validate_episode(episode, tokenizer):
    observations = episode.observations
    points = np.stack(observations["point_cloud"])
    if not np.isfinite(points).all() or not np.stack(observations["object_point_mask"]).all():
        raise InvalidObjectObservation("Impure or nonfinite object cloud")
    for key in ("state", "agent_pos", "object_center", "palm_pose_robot_base"):
        if not np.isfinite(np.stack(observations[key])).all():
            raise InvalidObjectObservation(f"Nonfinite {key}")
    if not episode.stable_contact_mask.any():
        raise InvalidObjectObservation("Stable contact has no active contact above threshold")
    error_max = 0.0
    target_points, target_masks = episode.contact_targets()
    raw_points = np.stack(episode.raw_contact_points)
    raw_masks = np.stack(episode.raw_contact_masks)
    for points_array, masks in ((raw_points, raw_masks), (target_points, target_masks)):
        active = points_array[masks]
        if not np.isfinite(active).all():
            raise ValueError("Nonfinite contact coordinates")
        if np.any(active < tokenizer.min_position) or np.any(active > tokenizer.max_position):
            raise ValueError(
                f"Base-frame contact exceeds token range: {active.min(0)}, {active.max(0)}"
            )
        ids, _ = encode_contact_graphs(
            tokenizer, points_array, masks, np.zeros((len(points_array), 3)), 66
        )
        from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

        for xyz, mask, tokens in zip(points_array, masks, ids, strict=True):
            decoded = tokenizer.decode(tokens)
            for index in np.flatnonzero(mask):
                restored = decoded[ALLEGRO_CONTACT_LINKS[index].token_name][0]
                error_max = max(error_max, float(np.max(np.abs(restored - xyz[index]))))
    tolerance = (
        (tokenizer.max_position - tokenizer.min_position) / (tokenizer.position_bins - 1) / 2
    )
    if error_max > tolerance + 1e-6:
        raise ValueError(f"Contact-token round trip error {error_max} exceeds {tolerance}")
    return error_max


def recover_store(path):
    """Truncate an interrupted append to the last atomically committed episode."""
    root = zarr.open_group(str(path), mode="a")
    committed = int(root.attrs.get("committed_episodes", 0))
    steps = int(root["meta/episode_ends"][committed - 1]) if committed else 0
    for group_name in ("data", "meta", "annotation"):
        if group_name not in root:
            continue
        for _, array in root[group_name].arrays():
            length = steps if group_name == "data" else committed
            if len(array) < length:
                raise RuntimeError(f"Committed data missing: {array.path}")
            if len(array) != length:
                array.resize((length, *array.shape[1:]))
    return root


def append_episode(path: Path, episode: DexArtEpisode, tokenizer, attributes, episode_metadata):
    import shutil
    if shutil.disk_usage(path.parent).free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free; stop before writing another episode")
    error = validate_episode(episode, tokenizer)
    if attributes.get("sensor_contract") == "aligned_rgbd_object_10000_v2":
        import json
        object_ids = json.loads(episode_metadata["object_actor_ids_json"])
        for i in range(len(episode.actions)):
            sample = {key: values[i] for key, values in episode.observations.items()}
            validate_sensor_frame(sample, object_ids, int(attributes["point_count"]))
            if list(sample["depth"].shape) != list(attributes["point_capture_resolution"])[::-1]:
                raise InvalidObjectObservation("Stored resolution differs from collection contract")
    root = recover_store(path)
    if root.attrs.get("contact_coordinate_contract", ROBOT_BASE_CONTRACT) != ROBOT_BASE_CONTRACT:
        raise ValueError("Cannot append to a dataset with a different coordinate contract")
    committed = int(root.attrs.get("committed_episodes", 0))
    offset = int(root["meta/episode_ends"][committed - 1]) if committed else 0
    target_points, target_masks = episode.contact_targets()
    raw_points, raw_masks = (
        np.stack(episode.raw_contact_points),
        np.stack(episode.raw_contact_masks),
    )
    steps = {key: np.stack(values) for key, values in episode.observations.items()}
    steps.update(
        {
            "action": np.stack(episode.actions),
            "object_center_valid": np.ones(len(episode.actions), dtype=bool),
            "contact_raw_points": raw_points,
            "contact_raw_mask": raw_masks,
            "contact_target_points": target_points,
            "contact_target_mask": target_masks,
        }
    )
    for kind, points, masks in (
        ("raw", raw_points, raw_masks),
        ("target", target_points, target_masks),
    ):
        ids, mask = encode_contact_graphs(tokenizer, points, masks, np.zeros((len(points), 3)), 66)
        steps[f"contact_{kind}_token_ids"] = ids
        steps[f"contact_{kind}_token_mask"] = mask
    text = {
        **dict(episode.language),
        "object_id": episode.object_id,
        "task_id": episode.task_id,
        "annotation_raw": episode.annotation_raw,
    }
    meta = {key: np.asarray([value], dtype=object) for key, value in text.items()}
    meta.update(
        {
            "episode_ends": np.array([offset + len(episode.actions)], dtype=np.int64),
            "stable_contact_steps": np.array([episode.stable_contact_step], dtype=np.int32),
            "token_roundtrip_max_error_m": np.array([error], dtype=np.float32),
            **{
                key: np.asarray([value], dtype=object)
                if isinstance(value, str)
                else np.asarray(value)[None]
                for key, value in episode_metadata.items()
            },
        }
    )
    annotation = {
        "multiview_rgb": episode.annotation_views[None],
        "camera_extrinsics": episode.camera_extrinsics[None],
    }
    compressor = numcodecs.Blosc(cname="zstd", clevel=7 if attributes.get("sensor_contract") else 3, shuffle=numcodecs.Blosc.SHUFFLE)
    for name, arrays in (("data", steps), ("meta", meta), ("annotation", annotation)):
        group = root.require_group(name)
        for key, value in arrays.items():
            if key not in group:
                # Episode-sized RGB chunks and modest time chunks keep writes bounded.
                kwargs = {"object_codec": numcodecs.VLenUTF8()} if value.dtype == object else {}
                group.create_dataset(
                    key,
                    shape=(0, *value.shape[1:]),
                    dtype=value.dtype,
                    chunks=(1 if name != "data" or key in ("img", "depth", "segmentation_actor_id") else 64, *value.shape[1:]),
                    compressor=compressor,
                    **kwargs,
                )
            group[key].append(value, axis=0)
    if attributes.get("sensor_contract") == "aligned_rgbd_object_10000_v2":
        for index in range(offset, offset + len(episode.actions)):
            keys = ("depth", "img", "segmentation_actor_id", "point_pixel_index", "point_cloud",
                    "camera_intrinsics", "camera_to_robot_base", "object_visible_pixel_count",
                    "object_point_mask", "point_actor_id")
            sample = {key: root["data"][key][index] for key in keys}
            validate_sensor_frame(sample, object_ids, 10000)
    root.attrs.update(
        {
            **attributes,
            "contact_coordinate_contract": ROBOT_BASE_CONTRACT,
            "committed_episodes": committed + 1,
            "complete": False,
        }
    )
    return committed + 1
