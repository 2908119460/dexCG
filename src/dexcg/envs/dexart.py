"""DexArt adapter for observations, physical contacts, and annotation views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

VIEW_DIRECTIONS = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class ContactGraph:
    points: np.ndarray
    mask: np.ndarray

    @property
    def link_names(self) -> list[str]:
        return [
            link.token_name
            for link, active in zip(ALLEGRO_CONTACT_LINKS, self.mask, strict=True)
            if active
        ]


class DexArtAdapter:
    """Own one DexArt task environment and expose dexCG's data contract."""

    def __init__(self, environment, impulse_threshold: float = 1.0e-2) -> None:
        self.environment = environment
        self.impulse_threshold = impulse_threshold
        robot_links = {link.get_name(): link for link in environment.robot.get_links()}
        self.contact_links = tuple(robot_links[item.dexart_link] for item in ALLEGRO_CONTACT_LINKS)
        self.contact_link_index = {link: index for index, link in enumerate(self.contact_links)}
        self.object_links = set(environment.instance_links)
        self.annotation_cameras: dict[str, object] = {}

    @classmethod
    def create(
        cls,
        task_name: str,
        split: str,
        impulse_threshold: float,
        object_id: int | str | None = None,
    ) -> "DexArtAdapter":
        from dexart.env.create_env import create_env
        from dexart.env.task_setting import RANDOM_CONFIG, TRAIN_CONFIG

        object_ids = TRAIN_CONFIG[task_name][split]
        if object_id is not None:
            object_id = int(object_id)
            if object_id not in object_ids:
                raise ValueError(
                    f"Object {object_id} is not in the {task_name} {split} split: {object_ids}"
                )
            # DexArt treats a scalar as a positional index. A one-element list
            # keeps resetting the requested object ID instead.
            object_ids = [object_id]

        environment = create_env(
            task_name=task_name,
            use_visual_obs=True,
            use_gui=False,
            is_eval=True,
            pc_noise=True,
            pc_seg=True,
            index=object_ids,
            img_type="robot",
            rand_pos=RANDOM_CONFIG[task_name]["rand_pos"],
            rand_degree=RANDOM_CONFIG[task_name]["rand_degree"],
        )
        return cls(environment, impulse_threshold)

    @property
    def horizon(self) -> int:
        return self.environment.horizon

    @property
    def object_id(self) -> str:
        return str(self.environment.index)

    @property
    def is_success(self) -> bool:
        return bool(self.environment.is_eval_done)

    @property
    def is_stable_contact(self) -> bool:
        return int(self.environment.state) == 3

    def reset(self) -> Mapping[str, np.ndarray]:
        observation = self.environment.reset()
        self.object_links = set(self.environment.instance_links)
        return observation

    def step(self, action: np.ndarray):
        return self.environment.step(action)

    @staticmethod
    def observation(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = np.asarray(observation["instance_1-rgb"])
        state = np.asarray(observation["state"], dtype=np.float32)
        segmentation = np.asarray(observation.get("instance_1-seg_gt", ()), dtype=np.float32)
        if segmentation.shape == (len(observation["instance_1-point_cloud"]), 4):
            object_point_mask = np.any(segmentation[:, :2] > 0.5, axis=-1)
        else:
            object_point_mask = np.ones(len(observation["instance_1-point_cloud"]), dtype=np.bool_)
        if state.shape[-1] == 32:
            agent_pos = np.concatenate(
                (state[..., :-1], np.zeros_like(state[..., :1]), state[..., -1:]), axis=-1
            )
        elif state.shape[-1] == 33:
            agent_pos = state.copy()
        else:
            raise ValueError(f"Expected a 32D or 33D DexArt state, received {state.shape}")
        return {
            "img": np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
            "depth": np.asarray(observation["instance_1-depth"], dtype=np.float32),
            "point_cloud": np.asarray(
                observation["instance_1-point_cloud"][..., :3], dtype=np.float32
            ),
            "object_point_mask": object_point_mask,
            "imagin_robot": np.asarray(observation["imagination_robot"], dtype=np.float32),
            "state": state,
            "agent_pos": agent_pos,
        }

    def contact_graph(self) -> ContactGraph:
        per_link: list[list[np.ndarray]] = [[] for _ in ALLEGRO_CONTACT_LINKS]
        for contact in self.environment.scene.get_contacts():
            actors = {contact.actor0, contact.actor1}
            hand_links = actors.intersection(self.contact_link_index)
            if len(hand_links) != 1 or not actors.intersection(self.object_links):
                continue
            total_impulse = sum(np.abs(point.impulse).sum() for point in contact.points)
            if total_impulse < self.impulse_threshold:
                continue
            index = self.contact_link_index[hand_links.pop()]
            per_link[index].extend(np.asarray(point.position) for point in contact.points)

        root = self.environment.robot.get_pose().to_transformation_matrix()
        rotation_inv = root[:3, :3].T
        translation = root[:3, 3]
        points = np.zeros((len(ALLEGRO_CONTACT_LINKS), 3), dtype=np.float32)
        mask = np.zeros(len(ALLEGRO_CONTACT_LINKS), dtype=np.bool_)
        for index, positions in enumerate(per_link):
            if positions:
                world_point = np.mean(np.stack(positions), axis=0)
                points[index] = rotation_inv @ (world_point - translation)
                mask[index] = True
        return ContactGraph(points=points, mask=mask)

    def render_annotation_views(
        self,
        observation: Mapping[str, np.ndarray],
        directions: Sequence[str],
        resolution: Sequence[int],
        fov_degrees: float,
        distance_margin: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        scene_points = np.asarray(observation["instance_1-point_cloud"])[..., :3]
        segmentation = np.asarray(observation["instance_1-seg_gt"])
        hand_object_mask = np.any(segmentation[..., :3] > 0.5, axis=-1)
        robot_points = np.concatenate(
            (
                scene_points[hand_object_mask],
                np.asarray(observation["imagination_robot"])[..., :3],
            ),
            axis=0,
        )
        bounds_min = robot_points.min(axis=0)
        bounds_max = robot_points.max(axis=0)
        center_robot = 0.5 * (bounds_min + bounds_max)
        radius = np.linalg.norm(robot_points - center_robot, axis=1).max()
        robot_pose = self.environment.robot.get_pose()
        center = (robot_pose * self._point_pose(center_robot)).p
        fov = np.deg2rad(fov_degrees)
        distance = radius / np.sin(0.5 * fov) * distance_margin

        images = []
        extrinsics = []
        robot_inverse = self.environment.robot.get_pose().inv()
        for direction_name in directions:
            axis = VIEW_DIRECTIONS[direction_name]
            position = center + axis * distance
            look_direction = center - position
            up_reference = (
                np.array([0.0, 1.0, 0.0]) if abs(axis[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
            )
            right = np.cross(look_direction, up_reference)
            camera_name = f"dexcg_annotation_{direction_name.replace('+', 'p').replace('-', 'n')}"
            if camera_name not in self.annotation_cameras:
                self.environment.create_camera(
                    position,
                    look_direction,
                    right,
                    camera_name,
                    resolution=resolution,
                    fov=fov,
                )
                self.annotation_cameras[camera_name] = self.environment.cameras[camera_name]
            else:
                camera = self.annotation_cameras[camera_name]
                look = look_direction / np.linalg.norm(look_direction)
                right_unit = right / np.linalg.norm(right)
                up = np.cross(look, -right_unit)
                pose_matrix = np.eye(4)
                pose_matrix[:3] = np.stack([look, -right_unit, up, position], axis=1)
                import sapien.core as sapien

                camera.set_local_pose(sapien.Pose.from_transformation_matrix(pose_matrix))

            camera = self.annotation_cameras[camera_name]
            self.environment.scene.update_render()
            camera.take_picture()
            rgba = camera.get_color_rgba()
            images.append(np.rint(np.clip(rgba[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8))
            extrinsics.append((robot_inverse * camera.get_pose()).to_transformation_matrix())
        return np.stack(images), np.stack(extrinsics).astype(np.float32)

    @staticmethod
    def _point_pose(position: np.ndarray):
        import sapien.core as sapien

        return sapien.Pose(p=position)

    def close(self) -> None:
        self.environment.close()
