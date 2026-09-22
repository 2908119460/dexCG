"""Planner-only objective that leaves every downstream policy module frozen."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from dexcg.models.contact.coordinates import robot_base_point_cloud
from dexcg.models.dexcg import DexCG


class ContactPlannerTrainingObjective(nn.Module):
    def __init__(self, model: DexCG, freeze_partfield: bool = False) -> None:
        super().__init__()
        originally_trainable = {
            name
            for name, parameter in model.contact_planner.named_parameters()
            if parameter.requires_grad
        }
        model.requires_grad_(False)
        for name, parameter in model.contact_planner.named_parameters():
            parameter.requires_grad_(name in originally_trainable)
        self.freeze_partfield = freeze_partfield
        if freeze_partfield:
            model.contact_planner.point_encoder.requires_grad_(False)
        self.model = model
        self._set_downstream_eval()

    @property
    def planner(self):
        return self.model.contact_planner

    def _set_downstream_eval(self) -> None:
        if self.freeze_partfield:
            self.planner.point_encoder.eval()
        self.model.observation_encoder.eval()
        self.model.contact_encoder.eval()
        self.model.smp.eval()
        if self.model.physgraph is not None:
            self.model.physgraph.eval()

    def train(self, mode: bool = True) -> "ContactPlannerTrainingObjective":
        super().train(mode)
        self._set_downstream_eval()
        return self

    def forward(
        self, batch: Mapping[str, Any], languages: list[str]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        point_cloud = robot_base_point_cloud(batch["point_cloud"], batch["object_point_mask"])
        loss, contact_metrics = self.planner.training_loss(
            point_cloud,
            languages,
            batch["target_ids"],
            batch["target_mask"],
            previous_ids=batch["previous_ids"],
            previous_mask=batch["previous_mask"],
            sample_weight=batch["sample_weight"],
            **(
                {
                    "robot_qpos": batch["robot_qpos"],
                    "palm_pose_robot_base": batch["palm_pose_robot_base"],
                }
                if getattr(self.planner, "robot_state_projector", None) is not None
                else {}
            ),
        )
        return loss, {
            "loss": loss.detach(),
            "correct": contact_metrics["correct"].detach(),
            "count": contact_metrics["count"].detach(),
            "samples": torch.tensor(
                batch["target_ids"].shape[0], device=batch["target_ids"].device
            ),
        }

    def trainable_parameter_summary(self) -> dict[str, int]:
        groups = {
            "partfield": self.planner.point_encoder,
            "point_projector": self.planner.point_projector,
            "qwen": self.planner.language_model,
        }
        result = {
            name: sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            )
            for name, module in groups.items()
        }
        if getattr(self.planner, "robot_state_projector", None) is not None:
            result["robot_state_projector"] = sum(
                p.numel()
                for p in self.planner.robot_state_projector.parameters()
                if p.requires_grad
            )
        result["total"] = sum(result.values())
        return result
