"""Reconstruct DexArt's 96 hand collision-surface points in robot-base metres."""

from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
import yaml


def _surface_points(geometry, count, rng):
    box, sphere = geometry.find("box"), geometry.find("sphere")
    if box is not None:
        size = np.fromstring(box.attrib["size"], sep=" ")
        areas = np.array([size[1] * size[2], size[0] * size[2], size[0] * size[1]])
        faces = rng.choice(6, count, p=np.repeat(areas, 2) / (2 * areas.sum()))
        points = rng.uniform(-0.5, 0.5, (count, 3)) * size
        points[np.arange(count), faces // 2] = (faces % 2 - 0.5) * size[faces // 2]
        return points
    if sphere is not None:
        points = rng.normal(size=(count, 3))
        return (
            points / np.linalg.norm(points, axis=1, keepdims=True) * float(sphere.attrib["radius"])
        )
    raise ValueError("Expected box or sphere hand collision geometry; refusing synthetic origins")


@lru_cache(maxsize=1)
def _geometry_template():
    from dexart.env.task_setting import IMG_CONFIG

    from dexcg.models.physgraph.graph_spec import _rpy_matrix, load_robot_graph_spec
    from dexcg.models.physgraph.kinematics import BatchedForwardKinematics
    from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

    project = Path(__file__).resolve().parents[3]
    config_path = project / "configs/robot/allegro_xarm6.yaml"
    spec = load_robot_graph_spec(
        config_path, project, tuple(link.dexart_link for link in ALLEGRO_CONTACT_LINKS)
    )
    config = yaml.safe_load(config_path.read_text())
    root = ET.parse(project / config["urdf"]).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    points, indices = [], []
    rng = np.random.default_rng(0)
    for name, count in IMG_CONFIG["robot"]["robot"].items():
        collisions = links[name].findall("collision")
        if len(collisions) != 1:
            raise ValueError(f"Expected one collision surface for {name}")
        collision = collisions[0]
        local = _surface_points(collision.find("geometry"), count, rng)
        origin = collision.find("origin")
        xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        rpy = (
            [0.0, 0.0, 0.0]
            if origin is None
            else list(map(float, origin.get("rpy", "0 0 0").split()))
        )
        points.append(local @ _rpy_matrix(rpy).numpy().T + xyz)
        indices.extend([spec.link_names.index(name)] * count)
    return (
        BatchedForwardKinematics(spec),
        torch.tensor(np.concatenate(points), dtype=torch.float32),
        torch.tensor(indices, dtype=torch.long),
    )


def robot_geometry(qpos: np.ndarray) -> np.ndarray:
    """Return [...,96,7] XYZ plus semantic labels; no renderer or simulator needed."""
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape[-1] != 22 or not np.isfinite(qpos).all():
        raise ValueError("robot geometry requires 22 finite joint angles in simulator order")
    kinematics, local_points, link_indices = _geometry_template()
    with torch.no_grad():
        transforms = kinematics.transforms(torch.from_numpy(qpos.reshape(-1, 22)))[:, link_indices]
        xyz = torch.einsum("bnij,nj->bni", transforms[..., :3, :3], local_points)
        xyz += transforms[..., :3, 3]
        result = torch.zeros((*xyz.shape[:-1], 7), dtype=torch.float32)
        result[..., :3] = xyz
        result[..., 5] = 1
    return result.numpy().reshape(*qpos.shape[:-1], len(local_points), 7)
