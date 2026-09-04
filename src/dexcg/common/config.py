"""Project configuration loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    observation: dict[str, Any]
    contact_planner: dict[str, Any]
    contact_encoder: dict[str, Any]
    physgraph: dict[str, Any]
    smp: dict[str, Any]
    policy: dict[str, Any]


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    return ProjectConfig(**values)
