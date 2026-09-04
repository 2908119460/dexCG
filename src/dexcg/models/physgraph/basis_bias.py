"""Map a dynamically biased robot-object graph to an SMP basis residual."""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from dexcg.models.physgraph.bias import PhysicalGraphBias
from dexcg.models.physgraph.encoder import PhysicalGraphEncoder
from dexcg.models.physgraph.graph_spec import RobotGraphSpec
from dexcg.models.physgraph.kinematics import BatchedForwardKinematics


class ObjectNodeEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_projection = nn.Linear(3, hidden_dim)
        self.type_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self, points: torch.Tensor, point_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_mask is None:
            point_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        else:
            point_mask = point_mask.bool()
            if point_mask.shape != points.shape[:2]:
                raise ValueError("object point mask must match the point-cloud batch and points")
            has_object = point_mask.any(dim=1, keepdim=True)
            point_mask = torch.where(has_object, point_mask, torch.ones_like(point_mask))
        weights = point_mask.to(points.dtype)
        center = (points * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        point_features = self.point_mlp(points - center[:, None])
        blocked = torch.finfo(point_features.dtype).min
        shape = point_features.masked_fill(~point_mask[..., None], blocked).amax(dim=1)
        return shape + self.position_projection(center) + self.type_embedding, center


class PhysGraphBasisBias(nn.Module):
    """Produce a residual for the unconstrained skill basis, and nowhere else."""

    def __init__(
        self,
        spec: RobotGraphSpec,
        contact_token_ids: Sequence[int],
        num_experts: int,
        observation_horizon: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.0,
        max_path_distance: int = 8,
        geometric_min_hops: int = 1,
        initial_bandwidth: float = 0.1,
        serial_heads: int = 2,
        synergy_heads: int = 2,
        initial_output_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if len(contact_token_ids) != spec.contact_node_indices.numel():
            raise ValueError("contact_token_ids must align with the configured contact links")
        self.observation_horizon = observation_horizon
        self.qpos_dim = spec.qpos_dim
        self.action_dim = spec.action_dim
        self.num_experts = num_experts
        self.register_buffer("q_indices", spec.q_indices)
        self.register_buffer("joint_types", spec.joint_types)
        self.register_buffer("node_kinds", spec.node_kinds)
        self.register_buffer("finger_ids", spec.finger_ids)
        self.register_buffer("anatomical_levels", spec.anatomical_levels)
        self.register_buffer("action_node_indices", spec.action_node_indices)
        self.register_buffer("contact_node_indices", spec.contact_node_indices)
        self.register_buffer("contact_token_ids", torch.tensor(contact_token_ids, dtype=torch.long))

        self.kinematics = BatchedForwardKinematics(spec)
        self.robot_numeric_projection = nn.Linear(observation_horizon + 3, hidden_dim)
        self.node_kind_embedding = nn.Embedding(3, hidden_dim)
        self.joint_type_embedding = nn.Embedding(4, hidden_dim)
        self.finger_embedding = nn.Embedding(5, hidden_dim)
        self.level_embedding = nn.Embedding(6, hidden_dim)
        self.object_encoder = ObjectNodeEncoder(hidden_dim)
        self.bias_generator = PhysicalGraphBias(
            spec=spec,
            num_heads=num_heads,
            max_path_distance=max_path_distance,
            geometric_min_hops=geometric_min_hops,
            initial_bandwidth=initial_bandwidth,
            serial_heads=serial_heads,
            synergy_heads=synergy_heads,
        )
        self.encoder = PhysicalGraphEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.action_embedding = nn.Parameter(torch.zeros(self.action_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, num_experts)
        self.output_scale = nn.Parameter(torch.full((num_experts,), float(initial_output_scale)))
        nn.init.normal_(self.action_embedding, std=0.02)

    def _planned_contacts(self, token_ids: torch.Tensor) -> torch.Tensor:
        active = token_ids[:, :, None].eq(self.contact_token_ids[None, None]).any(dim=1)
        contacts = torch.zeros(
            (token_ids.shape[0], self.q_indices.numel()),
            dtype=torch.bool,
            device=token_ids.device,
        )
        contacts[:, self.contact_node_indices] = active
        return contacts

    def _robot_nodes(self, qpos_history: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        safe_indices = self.q_indices.clamp_min(0)
        per_node_qpos = qpos_history[:, :, safe_indices].transpose(1, 2)
        per_node_qpos = per_node_qpos * self.q_indices.ge(0)[None, :, None]
        numeric = torch.cat((per_node_qpos, positions), dim=-1)
        finger_ids = self.finger_ids.add(1).clamp_min(0)
        levels = self.anatomical_levels.add(1).clamp_min(0)
        return (
            self.robot_numeric_projection(numeric)
            + self.node_kind_embedding(self.node_kinds)
            + self.joint_type_embedding(self.joint_types)
            + self.finger_embedding(finger_ids)
            + self.level_embedding(levels)
        )

    def graph_inputs(
        self,
        observation: Mapping[str, torch.Tensor],
        contact_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agent_pos = observation["agent_pos"]
        if agent_pos.shape[1] < self.observation_horizon:
            raise ValueError("agent_pos history is shorter than observation_horizon")
        qpos_history = agent_pos[:, : self.observation_horizon, : self.qpos_dim]
        robot_positions = self.kinematics(qpos_history[:, -1])
        robot_nodes = self._robot_nodes(qpos_history, robot_positions)

        object_points = observation["point_cloud"][:, -1, :, :3]
        object_mask = observation.get("object_point_mask")
        if object_mask is not None:
            object_mask = object_mask[:, -1]
        object_node, object_position = self.object_encoder(object_points, object_mask)
        nodes = torch.cat((robot_nodes, object_node[:, None]), dim=1)
        positions = torch.cat((robot_positions, object_position[:, None]), dim=1)
        contacts = self._planned_contacts(contact_token_ids)
        return nodes, positions, contacts

    def raw_basis_bias(
        self,
        observation: Mapping[str, torch.Tensor],
        contact_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        nodes, positions, contacts = self.graph_inputs(observation, contact_token_ids)
        physical_bias = self.bias_generator(positions, contacts)
        encoded = self.encoder(nodes, physical_bias)
        action_features = encoded[:, self.action_node_indices] + self.action_embedding
        return self.output(action_features)

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        contact_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.raw_basis_bias(observation, contact_token_ids)
        return residual * self.output_scale
