"""Robot graph metadata derived from a URDF and a small embodiment config."""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import torch
import yaml

_ALLEGRO_LINK = re.compile(r"link_(\d+)\.0(?P<tip>_tip)?$")


def _vector(element: ET.Element | None, attribute: str, default: str) -> list[float]:
    value = default if element is None else element.attrib.get(attribute, default)
    return [float(item) for item in value.split()]


def _rpy_matrix(rpy: list[float]) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float32,
    )


@dataclass
class RobotGraphSpec:
    """Topologically ordered link graph and its action/contact mappings."""

    link_names: tuple[str, ...]
    parent_indices: torch.Tensor
    origin_xyz: torch.Tensor
    origin_rotation: torch.Tensor
    joint_axes: torch.Tensor
    joint_types: torch.Tensor
    q_indices: torch.Tensor
    adjacency: torch.Tensor
    action_node_indices: torch.Tensor
    contact_node_indices: torch.Tensor
    node_kinds: torch.Tensor
    finger_ids: torch.Tensor
    anatomical_levels: torch.Tensor
    serial_mask: torch.Tensor
    synergy_mask: torch.Tensor
    qpos_dim: int

    @property
    def num_nodes(self) -> int:
        return len(self.link_names)

    @property
    def action_dim(self) -> int:
        return int(self.action_node_indices.numel())


def _anatomy(link_name: str) -> tuple[int, int]:
    match = _ALLEGRO_LINK.fullmatch(link_name)
    if match is None:
        return -1, -1
    link_index = int(match.group(1))
    finger_id = link_index // 4
    level = link_index % 4 + int(match.group("tip") is not None)
    return finger_id, level


def _node_kind(link_name: str) -> int:
    if link_name in {"base_link", "palm", "palm_center"}:
        return 1
    if _ALLEGRO_LINK.fullmatch(link_name):
        return 2
    return 0


def load_robot_graph_spec(
    config_path: str | Path,
    project_root: str | Path,
    contact_link_names: tuple[str, ...],
) -> RobotGraphSpec:
    """Load one embodiment config and resolve its URDF under the project root."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config: Mapping[str, Any] = yaml.safe_load(stream)
    urdf_path = Path(project_root) / str(config["urdf"])
    return _parse_urdf(
        urdf_path=urdf_path,
        root_link=str(config["root_link"]),
        arm_dof=int(config["arm_dof"]),
        hand_dof=int(config["hand_dof"]),
        action_dim=int(config["action_dim"]),
        arm_action_link=str(config["arm_action_link"]),
        contact_link_names=contact_link_names,
    )


def _parse_urdf(
    urdf_path: Path,
    root_link: str,
    arm_dof: int,
    hand_dof: int,
    action_dim: int,
    arm_action_link: str,
    contact_link_names: tuple[str, ...],
) -> RobotGraphSpec:
    root = ET.parse(urdf_path).getroot()
    joints: list[dict[str, Any]] = []
    active_joint_names: list[str] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib["type"]
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"Joint {joint.attrib.get('name')} has no parent or child")
        origin = joint.find("origin")
        record = {
            "name": joint.attrib["name"],
            "type": joint_type,
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "xyz": _vector(origin, "xyz", "0 0 0"),
            "rotation": _rpy_matrix(_vector(origin, "rpy", "0 0 0")),
            "axis": _vector(joint.find("axis"), "xyz", "0 0 0"),
        }
        joints.append(record)
        if joint_type in {"revolute", "continuous", "prismatic"}:
            active_joint_names.append(record["name"])

    expected_qpos = arm_dof + hand_dof
    if len(active_joint_names) != expected_qpos:
        raise ValueError(
            f"URDF has {len(active_joint_names)} active joints, expected {expected_qpos}"
        )
    if action_dim != expected_qpos:
        raise ValueError(f"action_dim={action_dim} does not match arm_dof + hand_dof")

    children: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        children.setdefault(joint["parent"], []).append(joint)

    ordered_links = [root_link]
    incoming: dict[str, dict[str, Any]] = {}
    queue = deque([root_link])
    while queue:
        parent = queue.popleft()
        for joint in children.get(parent, []):
            child = joint["child"]
            incoming[child] = joint
            ordered_links.append(child)
            queue.append(child)

    link_to_index = {name: index for index, name in enumerate(ordered_links)}
    if len(link_to_index) != len(ordered_links):
        raise ValueError("URDF graph contains a repeated child link")
    missing_contacts = sorted(set(contact_link_names).difference(link_to_index))
    if missing_contacts:
        raise ValueError(f"Contact links are missing from the URDF: {missing_contacts}")
    if arm_action_link not in link_to_index:
        raise ValueError(f"Arm action link {arm_action_link!r} is missing from the URDF")

    q_name_to_index = {name: index for index, name in enumerate(active_joint_names)}
    parent_indices = [-1]
    origin_xyz = [[0.0, 0.0, 0.0]]
    origin_rotation = [torch.eye(3)]
    axes = [[0.0, 0.0, 0.0]]
    joint_types = [0]
    q_indices = [-1]
    type_ids = {"fixed": 1, "revolute": 2, "continuous": 2, "prismatic": 3}
    for link_name in ordered_links[1:]:
        joint = incoming[link_name]
        parent_indices.append(link_to_index[joint["parent"]])
        origin_xyz.append(joint["xyz"])
        origin_rotation.append(joint["rotation"])
        axes.append(joint["axis"])
        joint_types.append(type_ids[joint["type"]])
        q_indices.append(q_name_to_index.get(joint["name"], -1))

    adjacency = torch.zeros((len(ordered_links), len(ordered_links)), dtype=torch.bool)
    for child_index, parent_index in enumerate(parent_indices):
        if parent_index >= 0:
            adjacency[child_index, parent_index] = True
            adjacency[parent_index, child_index] = True

    active_child_links = {
        joint["name"]: joint["child"] for joint in joints if joint["name"] in q_name_to_index
    }
    hand_joint_names = active_joint_names[arm_dof:]
    action_nodes = [link_to_index[arm_action_link]] * arm_dof
    action_nodes.extend(link_to_index[active_child_links[name]] for name in hand_joint_names)

    anatomy = [_anatomy(name) for name in ordered_links]
    finger_ids = torch.tensor([item[0] for item in anatomy], dtype=torch.long)
    levels = torch.tensor([item[1] for item in anatomy], dtype=torch.long)
    same_finger = finger_ids[:, None].eq(finger_ids[None, :]) & finger_ids[:, None].ge(0)
    serial_mask = same_finger & adjacency
    synergy_mask = (
        finger_ids[:, None].ne(finger_ids[None, :])
        & finger_ids[:, None].ge(0)
        & finger_ids[None, :].ge(0)
        & levels[:, None].eq(levels[None, :])
    )

    return RobotGraphSpec(
        link_names=tuple(ordered_links),
        parent_indices=torch.tensor(parent_indices, dtype=torch.long),
        origin_xyz=torch.tensor(origin_xyz, dtype=torch.float32),
        origin_rotation=torch.stack(origin_rotation),
        joint_axes=torch.tensor(axes, dtype=torch.float32),
        joint_types=torch.tensor(joint_types, dtype=torch.long),
        q_indices=torch.tensor(q_indices, dtype=torch.long),
        adjacency=adjacency,
        action_node_indices=torch.tensor(action_nodes, dtype=torch.long),
        contact_node_indices=torch.tensor(
            [link_to_index[name] for name in contact_link_names], dtype=torch.long
        ),
        node_kinds=torch.tensor([_node_kind(name) for name in ordered_links], dtype=torch.long),
        finger_ids=finger_ids,
        anatomical_levels=levels,
        serial_mask=serial_mask,
        synergy_mask=synergy_mask,
        qpos_dim=expected_qpos,
    )
