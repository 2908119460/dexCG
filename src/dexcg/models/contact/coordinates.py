"""Coordinate transforms for DextER-compatible contact planning."""

from __future__ import annotations

import numpy as np
import torch

CONTACT_COORDINATE_CONTRACT = "object_aabb_center_v1"
OBJECT_CENTER_DEFINITION = "masked_observed_object_xyz_aabb_midpoint"


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


def center_point_cloud(
    point_cloud: torch.Tensor, object_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate XYZ into DextER's object-centered frame and return its origin."""
    center = object_aabb_center(point_cloud, object_mask)
    centered = point_cloud.clone()
    centered[..., :3] -= center.unsqueeze(-2)
    return centered, center


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


def restore_contact_positions(
    local_positions: torch.Tensor, object_center: torch.Tensor
) -> torch.Tensor:
    """Translate object-centered contact positions back to the robot-base frame."""
    return local_positions + object_center.unsqueeze(-2)
