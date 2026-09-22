"""Loss terms used by the SMP objective."""

import torch
from torch.nn import functional as F


def dirichlet_kl(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    q, p = torch.broadcast_tensors(q.float(), p.float())
    q_sum = q.sum(dim=-1)
    p_sum = p.sum(dim=-1)
    normalizer = (
        torch.lgamma(q_sum)
        - torch.lgamma(p_sum)
        - torch.lgamma(q).sum(dim=-1)
        + torch.lgamma(p).sum(dim=-1)
    )
    expectation = ((q - p) * (torch.digamma(q) - torch.digamma(q_sum).unsqueeze(-1))).sum(dim=-1)
    return normalizer + expectation


def router_alignment_loss(
    posterior: torch.Tensor,
    prior: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values = dirichlet_kl(posterior, prior)
    if valid_mask is None:
        return values.sum(dim=1).mean()
    mask = valid_mask.to(dtype=values.dtype)
    return (
        (values * mask).sum(dim=1).div(mask.sum(dim=1).clamp_min(1.0))
        * values.shape[1]
    ).mean()


def sticky_gate_loss(
    global_concentration: torch.Tensor,
    posterior_concentration: torch.Tensor,
    alpha: float,
    alpha0: float,
    kappa: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SMP's global, initial, and temporally sticky Dirichlet KL terms."""
    gates = posterior_concentration.float()
    global_q = global_concentration.float()
    if global_q.ndim == 1:
        global_q = global_q.unsqueeze(0).expand(gates.shape[0], -1)
    theta_mean = global_q / global_q.sum(dim=-1, keepdim=True)
    gate_mean = gates / gates.sum(dim=-1, keepdim=True)
    global_prior = torch.full_like(global_q, alpha)
    global_term = dirichlet_kl(global_q, global_prior)
    initial_term = dirichlet_kl(gates[:, 0], alpha0 * theta_mean)
    if gates.shape[1] > 1:
        sticky_prior = kappa * gate_mean[:, :-1] + alpha0 * theta_mean.unsqueeze(1)
        temporal = dirichlet_kl(gates[:, 1:], sticky_prior)
        if valid_mask is None:
            temporal_term = temporal.sum(dim=1)
        else:
            transition_mask = valid_mask[:, 1:].to(dtype=temporal.dtype) * valid_mask[:, :-1].to(dtype=temporal.dtype)
            temporal_term = (temporal * transition_mask).sum(dim=1).div(
                transition_mask.sum(dim=1).clamp_min(1.0)
            ) * temporal.shape[1]
    else:
        temporal_term = torch.zeros_like(initial_term)
    return (global_term + initial_term + temporal_term).mean()


def reconstruction_loss(reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstructed, target)


def coefficient_denoising_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prediction, target)
