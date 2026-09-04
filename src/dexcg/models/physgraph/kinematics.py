"""Batched forward kinematics implemented with PyTorch operations."""

import torch
from torch import nn

from dexcg.models.physgraph.graph_spec import RobotGraphSpec


def _axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    outer = axis[:, None] * axis[None, :]
    identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
    cosine = angle.cos()[:, None, None]
    sine = angle.sin()[:, None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


class BatchedForwardKinematics(nn.Module):
    def __init__(self, spec: RobotGraphSpec) -> None:
        super().__init__()
        self.register_buffer("parent_indices", spec.parent_indices)
        self.register_buffer("origin_xyz", spec.origin_xyz)
        self.register_buffer("origin_rotation", spec.origin_rotation)
        self.register_buffer("joint_axes", spec.joint_axes)
        self.register_buffer("joint_types", spec.joint_types)
        self.register_buffer("q_indices", spec.q_indices)
        self.qpos_dim = spec.qpos_dim

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[-1] != self.qpos_dim:
            raise ValueError(f"Expected qpos [B, {self.qpos_dim}], received {tuple(qpos.shape)}")
        batch_size = qpos.shape[0]
        identity = torch.eye(4, device=qpos.device, dtype=qpos.dtype)
        transforms = [identity.expand(batch_size, -1, -1)]
        for node in range(1, self.parent_indices.numel()):
            parent = int(self.parent_indices[node])
            origin = identity.expand(batch_size, -1, -1).clone()
            origin[:, :3, :3] = self.origin_rotation[node].to(dtype=qpos.dtype)
            origin[:, :3, 3] = self.origin_xyz[node].to(dtype=qpos.dtype)

            motion = identity.expand(batch_size, -1, -1).clone()
            q_index = int(self.q_indices[node])
            joint_type = int(self.joint_types[node])
            if q_index >= 0 and joint_type == 2:
                motion[:, :3, :3] = _axis_angle_rotation(
                    self.joint_axes[node].to(dtype=qpos.dtype), qpos[:, q_index]
                )
            elif q_index >= 0 and joint_type == 3:
                motion[:, :3, 3] = (
                    self.joint_axes[node].to(dtype=qpos.dtype)[None]
                    * qpos[:, q_index : q_index + 1]
                )
            transforms.append(transforms[parent] @ origin @ motion)
        return torch.stack([transform[:, :3, 3] for transform in transforms], dim=1)
