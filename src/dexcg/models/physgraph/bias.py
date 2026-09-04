"""The four head-specific physically grounded attention biases from PhysGraph."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from dexcg.models.physgraph.graph_spec import RobotGraphSpec


class PhysicalGraphBias(nn.Module):
    """Compose spatial, edge, geometric, and anatomical graph biases."""

    def __init__(
        self,
        spec: RobotGraphSpec,
        num_heads: int = 8,
        max_path_distance: int = 8,
        geometric_min_hops: int = 1,
        initial_bandwidth: float = 0.1,
        serial_heads: int = 2,
        synergy_heads: int = 2,
    ) -> None:
        super().__init__()
        if serial_heads + synergy_heads > num_heads:
            raise ValueError("Anatomical head groups exceed num_heads")
        if initial_bandwidth <= 0:
            raise ValueError("initial_bandwidth must be positive")
        self.num_heads = num_heads
        self.max_path_distance = max_path_distance
        self.geometric_min_hops = geometric_min_hops
        self.serial_heads = serial_heads
        self.synergy_heads = synergy_heads
        self.register_buffer("robot_adjacency", spec.adjacency)
        self.register_buffer("serial_mask", spec.serial_mask)
        self.register_buffer("synergy_mask", spec.synergy_mask)

        self.spatial_embedding = nn.Embedding(max_path_distance + 1, num_heads)
        self.edge_embedding = nn.Embedding(4, num_heads)
        raw_bandwidth = math.log(math.expm1(initial_bandwidth))
        self.raw_bandwidth = nn.Parameter(torch.tensor(raw_bandwidth))
        self.geometric_head_weight = nn.Parameter(torch.ones(num_heads))
        self.serial_bonus = nn.Parameter(torch.tensor(1.0))
        self.synergy_bonus = nn.Parameter(torch.tensor(1.0))
        self.component_weights = nn.Parameter(torch.ones(4))
        nn.init.normal_(self.spatial_embedding.weight, std=0.02)
        nn.init.normal_(self.edge_embedding.weight, std=0.02)

    def _relations(self, contact_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, robot_nodes = contact_nodes.shape
        if robot_nodes != self.robot_adjacency.shape[0]:
            raise ValueError("contact_nodes does not match the configured robot graph")
        total_nodes = robot_nodes + 1
        adjacency = torch.zeros(
            (batch_size, total_nodes, total_nodes),
            dtype=torch.bool,
            device=contact_nodes.device,
        )
        adjacency[:, :robot_nodes, :robot_nodes] = self.robot_adjacency
        adjacency[:, :robot_nodes, robot_nodes] = contact_nodes
        adjacency[:, robot_nodes, :robot_nodes] = contact_nodes

        distance = torch.full(
            (batch_size, total_nodes, total_nodes),
            self.max_path_distance,
            dtype=torch.long,
            device=contact_nodes.device,
        )
        distance.masked_fill_(adjacency, 1)
        diagonal = torch.arange(total_nodes, device=contact_nodes.device)
        distance[:, diagonal, diagonal] = 0
        for intermediate in range(total_nodes):
            through = (
                distance[:, :, intermediate, None] + distance[:, None, intermediate, :]
            ).clamp_max(self.max_path_distance)
            distance = torch.minimum(distance, through)

        edge_types = torch.zeros_like(distance)
        edge_types[:, diagonal, diagonal] = 1
        edge_types[:, :robot_nodes, :robot_nodes].masked_fill_(self.robot_adjacency, 2)
        edge_types[:, :robot_nodes, robot_nodes].masked_fill_(contact_nodes, 3)
        edge_types[:, robot_nodes, :robot_nodes].masked_fill_(contact_nodes, 3)
        return distance, edge_types

    def components(
        self, node_positions: torch.Tensor, contact_nodes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if node_positions.ndim != 3 or node_positions.shape[-1] != 3:
            raise ValueError("node_positions must have shape [B, N, 3]")
        distance, edge_types = self._relations(contact_nodes.bool())
        spatial = self.spatial_embedding(distance).permute(0, 3, 1, 2)
        edge = self.edge_embedding(edge_types).permute(0, 3, 1, 2)

        squared_distance = torch.cdist(node_positions.float(), node_positions.float()).square()
        bandwidth = F.softplus(self.raw_bandwidth) + 1e-6
        proximity = torch.exp(-squared_distance / (2.0 * bandwidth.square()))
        distant = distance.gt(self.geometric_min_hops).to(proximity.dtype)
        geometric = (
            proximity[:, None] * distant[:, None] * self.geometric_head_weight[None, :, None, None]
        )

        batch_size, total_nodes, _ = node_positions.shape
        robot_nodes = total_nodes - 1
        anatomical = node_positions.new_zeros(
            (batch_size, self.num_heads, total_nodes, total_nodes)
        )
        if self.serial_heads:
            anatomical[:, : self.serial_heads, :robot_nodes, :robot_nodes] = (
                self.serial_bonus * self.serial_mask
            )
        synergy_start = self.serial_heads
        synergy_end = synergy_start + self.synergy_heads
        if self.synergy_heads:
            anatomical[:, synergy_start:synergy_end, :robot_nodes, :robot_nodes] = (
                self.synergy_bonus * self.synergy_mask
            )
        return {
            "spatial": spatial.to(dtype=node_positions.dtype),
            "edge": edge.to(dtype=node_positions.dtype),
            "geometric": geometric.to(dtype=node_positions.dtype),
            "anatomical": anatomical,
        }

    def forward(self, node_positions: torch.Tensor, contact_nodes: torch.Tensor) -> torch.Tensor:
        parts = self.components(node_positions, contact_nodes)
        return sum(
            self.component_weights[index] * parts[name]
            for index, name in enumerate(("spatial", "edge", "geometric", "anatomical"))
        )
