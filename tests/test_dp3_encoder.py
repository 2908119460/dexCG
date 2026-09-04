import torch

from dexcg.models.observation.dp3_encoder import DP3Encoder, DP3ObservationEncoder


def test_dp3_encoder_matches_dp3_feature_concatenation() -> None:
    encoder = DP3Encoder(
        point_feature_dim=16,
        point_mlp_dims=(8, 16),
        state_dim=5,
        state_mlp_dims=(8, 8),
    )
    observation = {
        "point_cloud": torch.randn(3, 32, 3),
        "imagin_robot": torch.randn(3, 8, 7),
        "agent_pos": torch.randn(3, 5),
    }
    assert encoder(observation).shape == (3, 24)


def test_observation_history_is_fused_without_task_id() -> None:
    encoder = DP3ObservationEncoder(
        obs_horizon=2,
        feature_dim=32,
        point_feature_dim=16,
        point_mlp_dims=(8, 16),
        state_dim=5,
        state_mlp_dims=(8, 8),
    )
    observation = {
        "point_cloud": torch.randn(3, 2, 32, 3),
        "imagin_robot": torch.randn(3, 2, 8, 7),
        "agent_pos": torch.randn(3, 2, 5),
    }
    assert encoder(observation).shape == (3, 32)
