"""PointNet encoder matching DP3's point-cloud branch."""

from collections.abc import Sequence

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Sequence[int] = (64, 128, 256),
        out_channels: int = 64,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.point_mlp = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.Linear(current_dim, out_channels),
            nn.LayerNorm(out_channels) if layer_norm else nn.Identity(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        point_features = self.point_mlp(points)
        return self.projection(point_features.amax(dim=1))
