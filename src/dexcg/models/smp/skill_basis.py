"""State-conditioned orthogonal skill basis."""

import torch
from torch import nn


class SkillBasis(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if num_experts > action_dim:
            raise ValueError("num_experts must not exceed action_dim")
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim * num_experts),
        )

    def forward(self, state: torch.Tensor, basis_bias: torch.Tensor | None = None) -> torch.Tensor:
        unconstrained = self.network(state).reshape(-1, self.action_dim, self.num_experts)
        if basis_bias is not None:
            if basis_bias.shape != unconstrained.shape:
                raise ValueError(
                    f"Expected basis_bias {tuple(unconstrained.shape)}, "
                    f"received {tuple(basis_bias.shape)}"
                )
            unconstrained = unconstrained + basis_bias.to(dtype=unconstrained.dtype)
        basis, upper = torch.linalg.qr(unconstrained.float(), mode="reduced")
        diagonal = torch.diagonal(upper, dim1=-2, dim2=-1)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        return (basis * signs.unsqueeze(-2)).to(dtype=unconstrained.dtype)

    @staticmethod
    def orthogonality_error(basis: torch.Tensor) -> torch.Tensor:
        gram = basis.transpose(-2, -1) @ basis
        identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        return torch.linalg.matrix_norm(gram - identity, ord="fro")
