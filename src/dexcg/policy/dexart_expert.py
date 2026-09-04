"""Loader for the PPO experts released with DexArt and used by DP3."""

from pathlib import Path

from torch import nn


def dexart_policy_kwargs() -> dict:
    from dexart.env.task_setting import IMG_CONFIG
    from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP

    return {
        "features_extractor_class": PointNetImaginationExtractorGP,
        "features_extractor_kwargs": {
            "pc_key": "instance_1-point_cloud",
            "gt_key": "instance_1-seg_gt",
            "extractor_name": "smallpn",
            "imagination_keys": [f"imagination_{key}" for key in IMG_CONFIG["robot"]],
            "state_key": "state",
        },
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }


def load_dexart_expert(checkpoint: str | Path, environment, device: str):
    from stable_baselines3 import PPO

    return PPO.load(
        str(checkpoint),
        environment,
        device,
        policy_kwargs=dexart_policy_kwargs(),
        check_obs_space=False,
        force_load=True,
    )
