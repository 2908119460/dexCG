"""Projection layers between PartField/Qwen and the contact planner."""

import torch
from torch import nn


class PointCloudProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 896, dropout: float = 0.0) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.proj(self.layernorm(point_tokens))))


class ContactProjector(nn.Module):
    def __init__(self, llm_dim: int, contact_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, contact_dim),
            nn.GELU(),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(token_embeddings)


class RobotStateProjector(nn.Module):
    """Current joint radians and base-frame palm pose encoded as two prefix tokens.

    Fixed scales preserve absolute position: qpos/pi, translation/2 metres,
    and the first two rotation-matrix columns (continuous 6D orientation).
    No dataset-dependent normalization, clipping, or temporal inputs.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(31, hidden_size), nn.GELU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.hidden_size = hidden_size

    @staticmethod
    def features(qpos: torch.Tensor, palm_pose: torch.Tensor) -> torch.Tensor:
        if qpos.ndim != 2 or qpos.shape[1] != 22 or palm_pose.shape != (len(qpos), 4, 4):
            raise ValueError("Robot inputs require qpos [B,22] and palm pose [B,4,4]")
        qpos, palm_pose = qpos.float(), palm_pose.float()
        if not torch.isfinite(qpos).all() or not torch.isfinite(palm_pose).all():
            raise ValueError("Robot inputs must be finite")
        rotation = palm_pose[:, :3, :3]
        identity = torch.eye(3, device=rotation.device)
        if not torch.allclose(
            rotation.transpose(1, 2) @ rotation, identity.expand_as(rotation), atol=1e-4
        ):
            raise ValueError("Palm rotation must be orthonormal")
        if not torch.allclose(
            torch.linalg.det(rotation), torch.ones(len(qpos), device=qpos.device), atol=1e-4
        ):
            raise ValueError("Palm rotation must be right-handed")
        bottom = palm_pose.new_tensor([0, 0, 0, 1]).expand(len(qpos), -1)
        if not torch.allclose(palm_pose[:, 3], bottom, atol=1e-6):
            raise ValueError("Palm pose must be a homogeneous transform")
        return torch.cat(
            (
                qpos / torch.pi,
                palm_pose[:, :3, 3] / 2,
                rotation[:, :, :2].transpose(1, 2).reshape(-1, 6),
            ),
            dim=1,
        )

    def forward(self, qpos: torch.Tensor, palm_pose: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=qpos.device.type, enabled=False):
            features = self.features(qpos, palm_pose).to(self.network[0].weight.dtype)
        return self.network(features).reshape(-1, 2, self.hidden_size)
