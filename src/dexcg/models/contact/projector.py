"""Projection layers between PartField/Qwen and the contact planner."""

import torch
from torch import nn


class PointCloudProjector(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 896) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.GELU()

    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        return self.activation(self.proj(self.layernorm(point_tokens)))


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
