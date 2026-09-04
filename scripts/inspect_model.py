#!/usr/bin/env python3
"""Load and summarize the complete dexCG architecture."""

from pathlib import Path

import torch

from dexcg.common.config import load_config
from dexcg.models.dexcg import DexCG
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS


def parameter_count(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/model/dexcg.yaml")
    model = DexCG.from_config(config, project_root=root)
    total, trainable = parameter_count(model)
    print(model)
    print(f"contact links: {len(ALLEGRO_CONTACT_LINKS)}")
    print(f"parameters: {total:,} total, {trainable:,} trainable")


if __name__ == "__main__":
    main()
