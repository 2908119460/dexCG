"""Typed model outputs."""

from dataclasses import dataclass

import torch


@dataclass
class ContactPlan:
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    object_center: torch.Tensor | None = None


@dataclass
class DexCGOutput:
    observation_feature: torch.Tensor
    contact_feature: torch.Tensor
    basis: torch.Tensor
    prior_concentration: torch.Tensor
    prior_gate: torch.Tensor
    coefficient_prediction: torch.Tensor | None = None
    posterior_concentration: torch.Tensor | None = None
    posterior_gate: torch.Tensor | None = None
    coefficient_target: torch.Tensor | None = None
    reconstructed_action: torch.Tensor | None = None
