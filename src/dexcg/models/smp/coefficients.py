"""Projection into and reconstruction from SMP coefficient space."""

import torch


def coefficient_targets(
    basis: torch.Tensor,
    action: torch.Tensor,
    gate: torch.Tensor,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    stopped_projection = torch.einsum("bdk,btd->btk", basis.detach(), action)
    reconstruction_projection = torch.einsum("bdk,btd->btk", basis, action)
    return stopped_projection / (gate + eps), reconstruction_projection / (gate + eps)


def decode_action(
    basis: torch.Tensor, gate: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("bdk,btk->btd", basis, gate * coefficients)
