import torch
from torch import nn

from dexcg.common.typing import ContactPlan
from dexcg.models.contact.token_encoder import ContactTokenEncoder
from dexcg.models.dexcg import DexCG
from dexcg.models.observation.dp3_encoder import DP3ObservationEncoder
from dexcg.models.smp.model import ContactConditionedSMP


class TinyContactPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)

    def plan(self, point_cloud: torch.Tensor, languages: list[str]) -> ContactPlan:
        self.last_point_cloud = point_cloud
        batch_size = point_cloud.shape[0]
        ids = torch.tensor([2, 10, 20, 21, 22, 3], device=point_cloud.device)
        return ContactPlan(ids.expand(batch_size, -1), torch.ones(batch_size, 6, dtype=torch.bool))

    def embed_contact_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def test_full_model_accepts_only_observation_and_language_task_information() -> None:
    observation_encoder = DP3ObservationEncoder(
        obs_horizon=2,
        feature_dim=12,
        point_feature_dim=8,
        point_mlp_dims=(8, 16),
        state_dim=5,
        state_mlp_dims=(8,),
    )
    contact_encoder = ContactTokenEncoder(
        llm_dim=16,
        feature_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
    )
    smp = ContactConditionedSMP(
        observation_dim=12,
        contact_dim=8,
        action_dim=6,
        action_horizon=8,
        num_experts=4,
        condition_dim=16,
        basis_hidden_dim=16,
        gate_hidden_dim=16,
        expert_down_dims=(8, 16),
        expert_timestep_dim=8,
        expert_groups=4,
    )
    model = DexCG(observation_encoder, TinyContactPlanner(), contact_encoder, smp)
    observation = {
        "point_cloud": torch.randn(2, 2, 32, 3),
        "imagin_robot": torch.randn(2, 2, 8, 7),
        "agent_pos": torch.randn(2, 2, 5),
    }
    output = model(
        observation,
        ["grasp the cup handle", "lift the bottle"],
        noisy_coefficients=torch.randn(2, 8, 4),
        timestep=torch.tensor([2, 4]),
        action=torch.randn(2, 8, 6),
    )
    assert output.basis.shape == (2, 6, 4)
    assert output.coefficient_prediction.shape == (2, 8, 4)
    assert output.prior_gate.shape == (2, 8, 4)
    assert output.posterior_gate.shape == (2, 8, 4)
    planner_center = 0.5 * (
        model.contact_planner.last_point_cloud.amin(dim=1)
        + model.contact_planner.last_point_cloud.amax(dim=1)
    )
    torch.testing.assert_close(planner_center, torch.zeros_like(planner_center), atol=1e-6, rtol=0)
