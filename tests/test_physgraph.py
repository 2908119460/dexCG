from pathlib import Path

import torch
from torch import nn

from dexcg.common.typing import ContactPlan
from dexcg.models.contact.token_encoder import ContactTokenEncoder
from dexcg.models.dexcg import DexCG
from dexcg.models.observation.dp3_encoder import DP3ObservationEncoder
from dexcg.models.physgraph.basis_bias import PhysGraphBasisBias
from dexcg.models.physgraph.bias import PhysicalGraphBias
from dexcg.models.physgraph.graph_spec import load_robot_graph_spec
from dexcg.models.physgraph.kinematics import BatchedForwardKinematics
from dexcg.models.smp.model import ContactConditionedSMP
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

ROOT = Path(__file__).resolve().parents[1]


class EmbeddingOnlyPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, 16)

    def embed_contact_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def graph_spec():
    return load_robot_graph_spec(
        ROOT / "configs/robot/allegro_xarm6.yaml",
        ROOT,
        tuple(link.dexart_link for link in ALLEGRO_CONTACT_LINKS),
    )


def test_urdf_graph_maps_qpos_actions_and_contact_links() -> None:
    spec = graph_spec()

    assert spec.num_nodes == 30
    assert spec.qpos_dim == 22
    assert spec.action_dim == 22
    assert spec.contact_node_indices.shape == (16,)
    assert spec.link_names[spec.action_node_indices[0]] == "palm_center"
    assert spec.link_names[spec.action_node_indices[6]] == "link_0.0"


def test_forward_kinematics_changes_with_current_qpos() -> None:
    spec = graph_spec()
    kinematics = BatchedForwardKinematics(spec)
    qpos = torch.zeros(2, spec.qpos_dim)
    qpos[1, 1] = 0.5

    positions = kinematics(qpos)

    assert positions.shape == (2, spec.num_nodes, 3)
    assert torch.isfinite(positions).all()
    assert not torch.allclose(positions[0], positions[1])


def test_all_four_biases_have_head_specific_dynamic_structure() -> None:
    spec = graph_spec()
    generator = PhysicalGraphBias(spec, num_heads=8, serial_heads=2, synergy_heads=2)
    robot_positions = BatchedForwardKinematics(spec)(torch.zeros(2, spec.qpos_dim))
    object_positions = torch.tensor([[[0.2, 0.1, 0.3]], [[1.0, 1.0, 1.0]]])
    positions = torch.cat((robot_positions, object_positions), dim=1)
    contacts = torch.zeros(2, spec.num_nodes, dtype=torch.bool)
    contacts[1, spec.contact_node_indices[1]] = True

    parts = generator.components(positions, contacts)

    expected_shape = (2, 8, spec.num_nodes + 1, spec.num_nodes + 1)
    assert set(parts) == {"spatial", "edge", "geometric", "anatomical"}
    assert all(part.shape == expected_shape for part in parts.values())
    assert not torch.allclose(parts["spatial"][0], parts["spatial"][1])
    assert not torch.allclose(parts["edge"][0], parts["edge"][1])
    assert not torch.allclose(parts["geometric"][0], parts["geometric"][1])
    assert parts["anatomical"][:, :2].abs().sum() > 0
    assert parts["anatomical"][:, 2:4].abs().sum() > 0
    assert torch.count_nonzero(parts["anatomical"][:, 4:]) == 0


def test_basis_bias_builds_object_node_and_planned_contact_edges() -> None:
    spec = graph_spec()
    contact_ids = tuple(range(100, 116))
    module = PhysGraphBasisBias(
        spec=spec,
        contact_token_ids=contact_ids,
        num_experts=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        serial_heads=1,
        synergy_heads=1,
    )
    observation = {
        "agent_pos": torch.zeros(2, 2, 33),
        "point_cloud": torch.randn(2, 2, 32, 3),
        "object_point_mask": torch.tensor([[[True] * 16 + [False] * 16] * 2, [[False] * 32] * 2]),
    }
    token_ids = torch.tensor([[1, 100, 2], [1, 103, 2]])

    nodes, positions, contacts = module.graph_inputs(observation, token_ids)
    raw_bias = module.raw_basis_bias(observation, token_ids)
    scaled_bias = module(observation, token_ids)

    assert nodes.shape == (2, spec.num_nodes + 1, 32)
    assert positions.shape == (2, spec.num_nodes + 1, 3)
    assert contacts[0, spec.contact_node_indices[0]]
    assert contacts[1, spec.contact_node_indices[3]]
    assert raw_bias.shape == (2, 22, 4)
    assert torch.count_nonzero(scaled_bias) == 0

    with torch.no_grad():
        module.output_scale.fill_(1.0)
    assert torch.allclose(module(observation, token_ids), raw_bias)


def test_zero_initialized_output_scale_receives_gradient() -> None:
    spec = graph_spec()
    module = PhysGraphBasisBias(
        spec=spec,
        contact_token_ids=tuple(range(100, 116)),
        num_experts=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        serial_heads=1,
        synergy_heads=1,
    )
    observation = {
        "agent_pos": torch.zeros(2, 2, 33),
        "point_cloud": torch.randn(2, 2, 32, 3),
        "object_point_mask": torch.ones(2, 2, 32, dtype=torch.bool),
    }
    token_ids = torch.tensor([[1, 100, 2], [1, 103, 2]])

    raw_residual = module.raw_basis_bias(observation, token_ids).detach()
    residual = module(observation, token_ids)
    (residual * raw_residual).sum().backward()

    assert module.output_scale.grad is not None
    assert torch.count_nonzero(module.output_scale.grad) == module.num_experts


def test_full_model_routes_physgraph_output_only_to_basis() -> None:
    torch.manual_seed(11)
    spec = graph_spec()
    physgraph = PhysGraphBasisBias(
        spec=spec,
        contact_token_ids=tuple(range(100, 116)),
        num_experts=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        serial_heads=1,
        synergy_heads=1,
    )
    observation_encoder = DP3ObservationEncoder(
        obs_horizon=2,
        feature_dim=12,
        point_feature_dim=8,
        point_mlp_dims=(8, 16),
        state_dim=33,
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
        action_dim=22,
        action_horizon=8,
        num_experts=4,
        condition_dim=16,
        basis_hidden_dim=16,
        gate_hidden_dim=16,
        expert_down_dims=(8, 16),
        expert_timestep_dim=8,
        expert_groups=4,
    )
    model = DexCG(
        observation_encoder,
        EmbeddingOnlyPlanner(),
        contact_encoder,
        smp,
        physgraph,
    ).eval()
    observation = {
        "agent_pos": torch.randn(2, 2, 33),
        "point_cloud": torch.randn(2, 2, 32, 3),
        "object_point_mask": torch.ones(2, 2, 32, dtype=torch.bool),
        "imagin_robot": torch.randn(2, 2, 8, 7),
    }
    contact_plan = ContactPlan(
        token_ids=torch.tensor([[1, 100, 20, 21, 22, 2], [1, 103, 20, 21, 22, 2]]),
        attention_mask=torch.ones(2, 6, dtype=torch.bool),
    )
    noisy = torch.randn(2, 8, 4)
    action = torch.randn(2, 8, 22)

    baseline = model.forward_with_contact(
        observation,
        contact_plan,
        noisy_coefficients=noisy,
        timestep=torch.tensor([2, 4]),
        action=action,
    )
    with torch.no_grad():
        physgraph.output_scale.fill_(0.2)
    biased = model.forward_with_contact(
        observation,
        contact_plan,
        noisy_coefficients=noisy,
        timestep=torch.tensor([2, 4]),
        action=action,
    )

    assert not torch.allclose(baseline.basis, biased.basis)
    assert torch.equal(baseline.prior_gate, biased.prior_gate)
    assert torch.equal(baseline.posterior_gate, biased.posterior_gate)
    assert torch.equal(baseline.coefficient_prediction, biased.coefficient_prediction)
