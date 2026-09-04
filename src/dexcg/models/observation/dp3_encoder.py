"""DP3 observation encoder for scene, robot geometry, and proprioception."""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from dexcg.common.tensors import take_observation_horizon
from dexcg.models.observation.pointnet import PointNetEncoder


def _mlp(input_dim: int, dims: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for output_dim in dims:
        layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        input_dim = output_dim
    return nn.Sequential(*layers)


class DP3Encoder(nn.Module):
    """Per-timestep DP3 encoder.

    `imagin_robot[..., :3]` is concatenated with the scene cloud before
    PointNet, while `agent_pos` is encoded by its own MLP.
    """

    def __init__(
        self,
        point_cloud_channels: int = 3,
        point_feature_dim: int = 64,
        point_mlp_dims: Sequence[int] = (64, 128, 256),
        state_dim: int = 33,
        state_mlp_dims: Sequence[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.point_cloud_channels = point_cloud_channels
        self.point_encoder = PointNetEncoder(
            in_channels=point_cloud_channels,
            hidden_dims=point_mlp_dims,
            out_channels=point_feature_dim,
        )
        self.state_encoder = _mlp(state_dim, state_mlp_dims)
        self.output_dim = point_feature_dim + state_mlp_dims[-1]

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        scene_points = observation["point_cloud"][..., : self.point_cloud_channels]
        robot_points = observation["imagin_robot"][..., : self.point_cloud_channels]
        point_feature = self.point_encoder(torch.cat((scene_points, robot_points), dim=1))
        state_feature = self.state_encoder(observation["agent_pos"])
        return torch.cat((point_feature, state_feature), dim=-1)


class DP3ObservationEncoder(nn.Module):
    """Encode a DP3 observation history into the state feature `s`."""

    def __init__(
        self,
        obs_horizon: int = 2,
        feature_dim: int = 256,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.encoder = DP3Encoder(**encoder_kwargs)
        self.temporal_fusion = nn.Sequential(
            nn.Linear(obs_horizon * self.encoder.output_dim, feature_dim),
            nn.Mish(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.output_dim = feature_dim

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch_size = observation["point_cloud"].shape[0]
        flattened = take_observation_horizon(observation, self.obs_horizon)
        encoded = self.encoder(flattened).reshape(batch_size, -1)
        return self.temporal_fusion(encoded)
