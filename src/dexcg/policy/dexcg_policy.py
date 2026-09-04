"""Network-level fast/slow invocation wrapper."""

from collections.abc import Mapping

import torch
from torch import nn

from dexcg.common.typing import DexCGOutput
from dexcg.models.dexcg import DexCG
from dexcg.policy.contact_cache import ContactCache


class DexCGPolicy(nn.Module):
    def __init__(self, model: DexCG, contact_update_interval: int = 8) -> None:
        super().__init__()
        self.model = model
        self.contact_cache = ContactCache(contact_update_interval)
        self.action_step = 0

    def reset(self) -> None:
        self.action_step = 0
        self.contact_cache.reset()

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        languages: list[str],
        noisy_coefficients: torch.Tensor | None = None,
        timestep: torch.Tensor | int | None = None,
        action: torch.Tensor | None = None,
    ) -> DexCGOutput:
        plan = self.contact_cache.get(
            self.action_step,
            languages,
            lambda: self.model.plan_contact(observation, languages),
        )
        self.action_step += 1
        return self.model.forward_with_contact(
            observation,
            plan,
            noisy_coefficients=noisy_coefficients,
            timestep=timestep,
            action=action,
        )
