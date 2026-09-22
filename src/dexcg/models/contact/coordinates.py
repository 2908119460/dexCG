"""DexArt robot-base coordinates and explicit numerical encoding contracts."""

from __future__ import annotations

import numpy as np
import torch

CONTACT_COORDINATE_CONTRACT = "dexart_robot_base_metric_v1"
MODEL_COORDINATE_CONTRACT = "dexart_robot_base_model_v1"
CONTACT_MIN_POSITION = -1.0
CONTACT_MAX_POSITION = 1.2
CONTACT_POSITION_BINS = 256
POINT_ENCODER_EXTENT_M = 2.0
OBJECT_CENTER_DEFINITION = "masked_full_resolution_visible_object_xyz_aabb_midpoint"


def require_model_coordinates(checkpoint) -> None:
    if (
        checkpoint.get("contact_coordinate_contract") != CONTACT_COORDINATE_CONTRACT
        or checkpoint.get("model_coordinate_contract") != MODEL_COORDINATE_CONTRACT
    ):
        raise ValueError(
            "Checkpoint coordinate contract does not match the robot-base model; "
            "object-centered weights and world-frame state statistics cannot be resumed."
        )


def require_dataset_coordinates(attributes) -> None:
    expected = {
        "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
        "point_cloud_frame": "robot_base",
        "contact_token_frame": "robot_base",
        "state_spatial_frame": "robot_base",
        "length_unit": "metre",
    }
    for key, value in expected.items():
        if attributes.get(key) != value:
            raise ValueError(
                f"Dataset requires {key}={value!r}; recollect with robot-base coordinates"
            )
    tokenizer = attributes.get("contact_tokenizer", {})
    for key, value in (
        ("min_position", CONTACT_MIN_POSITION),
        ("max_position", CONTACT_MAX_POSITION),
        ("position_bins", CONTACT_POSITION_BINS),
    ):
        if tokenizer.get(key) != value:
            raise ValueError(f"Dataset contact tokenizer requires {key}={value}")
    if attributes.get("complete") is False:
        raise ValueError(
            "Dataset collection is incomplete; wait for collection and audit to finish"
        )


def partfield_grid_coordinates(point_cloud: torch.Tensor) -> torch.Tensor:
    """Fixed zero-preserving metric-to-grid scale; reserve margin before discretization."""
    xyz = point_cloud.float()
    if not torch.isfinite(xyz).all():
        raise ValueError("PartField XYZ must be finite")
    if (xyz.abs() >= POINT_ENCODER_EXTENT_M * 0.98).any():
        raise ValueError(
            "PartField robot-base points exceed the fixed grid range; refusing clipping"
        )
    return xyz / (2 * POINT_ENCODER_EXTENT_M)


def robot_base_point_cloud(point_cloud: torch.Tensor, object_mask: torch.Tensor) -> torch.Tensor:
    """Select labeled object XYZ without translating, rotating, or scaling them.

    Preserve the point count by repeating valid object points. Pure clouds pass
    through unchanged, so training and deployment use the same metric interface.
    """
    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3 or point_cloud.shape[1] == 0:
        raise ValueError("expected point_cloud [batch, nonempty points, 3]")
    if object_mask.shape != point_cloud.shape[:-1]:
        raise ValueError("object mask must match point_cloud batch and point dimensions")
    if not torch.all((object_mask == 0) | (object_mask == 1)):
        raise ValueError("object mask must be binary")
    valid = object_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("each point cloud must contain at least one object point")
    if not torch.isfinite(point_cloud[valid]).all():
        raise ValueError("object point coordinates must be finite")
    if valid.all():
        return point_cloud
    rows = []
    selection = torch.arange(point_cloud.shape[1], device=point_cloud.device)
    for xyz, mask in zip(point_cloud, valid, strict=True):
        indices = mask.nonzero(as_tuple=False).flatten()
        rows.append(xyz[indices[selection % len(indices)]])
    return torch.stack(rows)


def robot_state_in_base(
    state: np.ndarray, world_from_base: np.ndarray, world_from_palm: np.ndarray
) -> np.ndarray:
    """Transform native DexArt state; keep joint radians and progress unchanged."""
    result = np.asarray(state, dtype=np.float32).copy()
    if result.shape not in ((32,), (33,)) or not np.isfinite(result).all():
        raise ValueError("expected finite native DexArt state with 32 or 33 entries")
    if world_from_base.shape != (4, 4) or world_from_palm.shape != (4, 4):
        raise ValueError("base and palm poses must be 4x4 transforms")
    if not np.isfinite(world_from_base).all() or not np.isfinite(world_from_palm).all():
        raise ValueError("base and palm transforms must be finite")
    rotation = world_from_base[:3, :3]
    result[22:25] = result[22:25] @ rotation
    result[25:28] = result[25:28] @ rotation
    result[28:31] = (result[28:31] - world_from_base[:3, 3]) @ rotation
    if len(result) == 33:
        result[31] = (rotation.T @ world_from_palm[:3, :3])[2, 0]
    return result


def object_aabb_center(
    point_cloud: torch.Tensor, object_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Return the object AABB center, falling back to all points when no mask is given."""
    if point_cloud.shape[-1] < 3:
        raise ValueError("point_cloud must have at least three channels")
    xyz = point_cloud[..., :3]
    if object_mask is None:
        return 0.5 * (xyz.amin(dim=-2) + xyz.amax(dim=-2))
    if object_mask.shape != xyz.shape[:-1]:
        raise ValueError(
            f"object_mask shape {tuple(object_mask.shape)} does not match "
            f"point cloud shape {tuple(xyz.shape[:-1])}"
        )
    valid = object_mask.bool()
    if not torch.all(valid.any(dim=-1)):
        raise ValueError("each point cloud must contain at least one object point")
    lower = xyz.masked_fill(~valid.unsqueeze(-1), torch.inf).amin(dim=-2)
    upper = xyz.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=-2)
    return 0.5 * (lower + upper)


def object_aabb_center_numpy(
    point_cloud: np.ndarray, object_mask: np.ndarray | None = None
) -> np.ndarray:
    """NumPy equivalent supporting either one point cloud or a leading batch."""
    xyz = np.asarray(point_cloud)[..., :3]
    if xyz.shape[-1] != 3:
        raise ValueError("point_cloud must have at least three channels")
    if object_mask is None:
        lower = xyz.min(axis=-2)
        upper = xyz.max(axis=-2)
    else:
        mask = np.asarray(object_mask, dtype=np.bool_)
        if mask.shape != xyz.shape[:-1]:
            raise ValueError(
                f"object_mask shape {mask.shape} does not match point cloud shape {xyz.shape[:-1]}"
            )
        if not np.all(mask.any(axis=-1)):
            raise ValueError("each point cloud must contain at least one object point")
        lower = np.where(mask[..., None], xyz, np.inf).min(axis=-2)
        upper = np.where(mask[..., None], xyz, -np.inf).max(axis=-2)
    return np.asarray(0.5 * (lower + upper), dtype=np.float32)
