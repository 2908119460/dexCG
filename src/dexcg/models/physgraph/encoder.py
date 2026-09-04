"""Transformer encoder whose attention accepts a per-head physical bias."""

import math

import torch
from torch import nn


class BiasedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, hidden_dim = inputs.shape
        qkv = self.qkv(inputs).reshape(batch_size, node_count, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if bias.shape != logits.shape:
            raise ValueError(f"Expected bias {tuple(logits.shape)}, received {tuple(bias.shape)}")
        attention = self.dropout(torch.softmax(logits + bias, dim=-1))
        encoded = attention @ value
        encoded = encoded.transpose(1, 2).reshape(batch_size, node_count, hidden_dim)
        return self.output(encoded)


class PhysicalGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = BiasedSelfAttention(hidden_dim, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs), bias)
        return inputs + self.feedforward(self.feedforward_norm(inputs))


class PhysicalGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            PhysicalGraphEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs, bias)
        return self.norm(inputs)
