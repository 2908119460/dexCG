"""Contact-conditioned prior and posterior expert gates."""

import torch
from torch import nn
from torch.nn import functional as F


class PositiveHead(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, min_concentration: float
    ) -> None:
        super().__init__()
        self.min_concentration = min_concentration
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(inputs)) + self.min_concentration


class PriorGate(nn.Module):
    """Deployment router p(g | s, contact)."""

    def __init__(
        self,
        condition_dim: int,
        action_horizon: int,
        num_experts: int,
        hidden_dim: int = 256,
        min_concentration: float = 1e-4,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.num_experts = num_experts
        self.head = PositiveHead(
            condition_dim,
            action_horizon * num_experts,
            hidden_dim,
            min_concentration,
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.head(condition).reshape(-1, self.action_horizon, self.num_experts)


class PosteriorGate(nn.Module):
    """Training router q(g | s, action, contact)."""

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        min_concentration: float = 1e-4,
    ) -> None:
        super().__init__()
        self.head = PositiveHead(
            condition_dim + action_dim,
            num_experts,
            hidden_dim,
            min_concentration,
        )

    def forward(self, condition: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        expanded = condition[:, None].expand(-1, action.shape[1], -1)
        return self.head(torch.cat((expanded, action), dim=-1))


class GlobalUsagePosterior(nn.Module):
    def __init__(
        self,
        num_experts: int,
        initial_concentration: float = 2.0,
        min_concentration: float = 1e-4,
    ) -> None:
        super().__init__()
        initial = torch.log(torch.expm1(torch.tensor(initial_concentration - min_concentration)))
        self.raw_concentration = nn.Parameter(initial.repeat(num_experts))
        self.min_concentration = min_concentration

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw_concentration) + self.min_concentration


def dirichlet_mean(concentration: torch.Tensor) -> torch.Tensor:
    return concentration / concentration.sum(dim=-1, keepdim=True)
