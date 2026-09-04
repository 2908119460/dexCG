"""Tensor operations shared by model components."""

from collections.abc import Mapping

import torch


def take_observation_horizon(
    observation: Mapping[str, torch.Tensor], horizon: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[:, :horizon].reshape(-1, *value.shape[2:])
        for name, value in observation.items()
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)
