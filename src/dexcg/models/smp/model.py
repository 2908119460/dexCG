"""Contact-conditioned SMP network head."""

from collections.abc import Sequence

import torch
from torch import nn

from dexcg.models.smp.coefficients import coefficient_targets, decode_action
from dexcg.models.smp.experts import CoefficientExperts
from dexcg.models.smp.gating import (
    GlobalUsagePosterior,
    PosteriorGate,
    PriorGate,
    dirichlet_mean,
)
from dexcg.models.smp.skill_basis import SkillBasis


def _conditioner(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.Mish(),
        nn.Linear(output_dim, output_dim),
    )


class ContactConditionedSMP(nn.Module):
    """Compute B(s), z(s, contact), and g(s, contact[, action])."""

    def __init__(
        self,
        observation_dim: int,
        contact_dim: int,
        action_dim: int = 22,
        action_horizon: int = 16,
        num_experts: int = 4,
        condition_dim: int = 256,
        basis_hidden_dim: int = 256,
        gate_hidden_dim: int = 256,
        min_concentration: float = 1e-4,
        global_initial_concentration: float = 2.0,
        coefficient_eps: float = 1e-3,
        expert_down_dims: Sequence[int] = (64, 128, 256),
        expert_timestep_dim: int = 128,
        expert_kernel_size: int = 3,
        expert_groups: int = 8,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_experts = num_experts
        self.coefficient_eps = coefficient_eps
        contact_condition_dim = observation_dim + contact_dim
        self.expert_conditioner = _conditioner(contact_condition_dim, condition_dim)
        self.gate_conditioner = _conditioner(contact_condition_dim, condition_dim)
        self.basis = SkillBasis(observation_dim, action_dim, num_experts, basis_hidden_dim)
        self.prior = PriorGate(
            condition_dim,
            action_horizon,
            num_experts,
            gate_hidden_dim,
            min_concentration,
        )
        self.posterior = PosteriorGate(
            condition_dim,
            action_dim,
            num_experts,
            gate_hidden_dim,
            min_concentration,
        )
        self.global_usage = GlobalUsagePosterior(
            num_experts,
            global_initial_concentration,
            min_concentration,
        )
        self.experts = CoefficientExperts(
            num_experts,
            condition_dim,
            expert_timestep_dim,
            expert_down_dims,
            expert_kernel_size,
            expert_groups,
        )

    def conditions(
        self, observation: torch.Tensor, contact: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((observation, contact), dim=-1)
        return self.expert_conditioner(inputs), self.gate_conditioner(inputs)

    def route(
        self,
        observation: torch.Tensor,
        contact: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, gate_condition = self.conditions(observation, contact)
        prior_concentration = self.prior(gate_condition)
        outputs = {
            "prior_concentration": prior_concentration,
            "prior_gate": dirichlet_mean(prior_concentration),
            "global_concentration": self.global_usage(),
        }
        if action is not None:
            posterior_concentration = self.posterior(gate_condition, action)
            outputs["posterior_concentration"] = posterior_concentration
            outputs["posterior_gate"] = dirichlet_mean(posterior_concentration)
        return outputs

    def build_training_targets(
        self,
        observation: torch.Tensor,
        contact: torch.Tensor,
        action: torch.Tensor,
        basis_bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        basis = self.basis(observation, basis_bias)
        routing = self.route(observation, contact, action)
        target, reconstruction_coefficients = coefficient_targets(
            basis,
            action,
            routing["posterior_gate"],
            self.coefficient_eps,
        )
        return {
            "basis": basis,
            **routing,
            "coefficient_target": target,
            "reconstructed_action": decode_action(
                basis, routing["posterior_gate"], reconstruction_coefficients
            ),
        }

    def denoise(
        self,
        noisy_coefficients: torch.Tensor,
        timestep: torch.Tensor | int,
        observation: torch.Tensor,
        contact: torch.Tensor,
    ) -> torch.Tensor:
        expert_condition, _ = self.conditions(observation, contact)
        return self.experts(noisy_coefficients, timestep, expert_condition)

    @staticmethod
    def decode(basis: torch.Tensor, gate: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return decode_action(basis, gate, coefficients)
