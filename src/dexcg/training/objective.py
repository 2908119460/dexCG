"""Joint contact-planner and SMP diffusion training objective."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn
from torch.nn import functional as F

from dexcg.common.typing import ContactPlan
from dexcg.models.dexcg import DexCG
from dexcg.models.smp.losses import router_alignment_loss, sticky_gate_loss


class DexCGTrainingObjective(nn.Module):
    def __init__(
        self,
        model: DexCG,
        state_min: torch.Tensor,
        state_max: torch.Tensor,
        diffusion_config: Mapping[str, object],
        loss_config: Mapping[str, float],
        train_contact_planner: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_contact_planner = bool(train_contact_planner)
        if not self.train_contact_planner:
            self.model.contact_planner.requires_grad_(False)
            self.model.contact_planner.eval()
        self.register_buffer("state_min", state_min.float())
        self.register_buffer("state_max", state_max.float())
        self.loss_config = dict(loss_config)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(diffusion_config["num_train_timesteps"]),
            beta_start=float(diffusion_config["beta_start"]),
            beta_end=float(diffusion_config["beta_end"]),
            beta_schedule=str(diffusion_config["beta_schedule"]),
            prediction_type=str(diffusion_config["prediction_type"]),
            clip_sample=False,
        )

    def train(self, mode: bool = True) -> "DexCGTrainingObjective":
        super().train(mode)
        if not self.train_contact_planner:
            self.model.contact_planner.eval()
        return self

    def normalize_observation(
        self, observation: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = dict(observation)
        value_range = self.state_max - self.state_min
        safe_range = torch.where(value_range < 1e-4, torch.ones_like(value_range), value_range)
        normalized = 2.0 * (observation["agent_pos"] - self.state_min) / safe_range - 1.0
        normalized = torch.where(value_range < 1e-4, torch.zeros_like(normalized), normalized)
        result["agent_pos"] = normalized
        return result

    def _mixed_contact_plan(
        self,
        observation: Mapping[str, torch.Tensor],
        languages: list[str],
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        teacher_forcing_probability: float,
    ) -> tuple[ContactPlan, torch.Tensor]:
        batch_size, target_length = target_ids.shape
        use_prediction = torch.rand(batch_size, device=target_ids.device).ge(
            teacher_forcing_probability
        )
        if not use_prediction.any():
            return ContactPlan(target_ids, target_mask), use_prediction

        selected = use_prediction.nonzero(as_tuple=False).flatten()
        predicted = self.model.contact_planner.plan(
            observation["point_cloud"][selected, -1],
            [languages[index] for index in selected.tolist()],
        )
        predicted_ids = torch.full_like(
            target_ids,
            self.model.contact_planner.contact_tokenizer.joint_end_id,
        )
        predicted_mask = torch.zeros_like(target_mask)
        copy_length = min(target_length, predicted.token_ids.shape[1])
        predicted_ids[selected, :copy_length] = predicted.token_ids[:, :copy_length]
        predicted_mask[selected, :copy_length] = predicted.attention_mask[:, :copy_length]
        mixed_ids = torch.where(use_prediction[:, None], predicted_ids, target_ids)
        mixed_mask = torch.where(use_prediction[:, None], predicted_mask, target_mask)
        return ContactPlan(mixed_ids, mixed_mask), use_prediction

    def forward(
        self,
        batch: Mapping[str, object],
        teacher_forcing_probability: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        observation = batch["observation"]
        action = batch["action"].float()
        languages = list(batch["language"])
        target_ids = batch["contact_token_ids"].long()
        target_mask = batch["contact_token_mask"].bool()

        if self.train_contact_planner:
            contact_loss, contact_metrics = self.model.contact_planner.training_loss(
                observation["point_cloud"][:, -1],
                languages,
                target_ids,
                target_mask,
            )
            contact_plan, predicted_rows = self._mixed_contact_plan(
                observation,
                languages,
                target_ids,
                target_mask,
                teacher_forcing_probability,
            )
        else:
            contact_loss = action.new_zeros(())
            contact_metrics = {
                "correct": target_mask.new_zeros((), dtype=torch.long),
                "count": target_mask.new_zeros((), dtype=torch.long),
            }
            contact_plan = ContactPlan(target_ids, target_mask)
            predicted_rows = target_mask.new_zeros(target_mask.shape[0])
        observation_feature = self.model.observation_encoder(
            self.normalize_observation(observation)
        )
        contact_feature = self.model.encode_contact(contact_plan)
        basis_bias = self.model.physgraph(observation, contact_plan.token_ids)
        targets = self.model.smp.build_training_targets(
            observation_feature,
            contact_feature,
            action,
            basis_bias,
        )

        clean_coefficients = targets["coefficient_target"]
        noise = torch.randn_like(clean_coefficients)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action.shape[0],),
            device=action.device,
        )
        noisy_coefficients = self.noise_scheduler.add_noise(
            clean_coefficients, noise, timesteps
        )
        prediction = self.model.smp.denoise(
            noisy_coefficients,
            timesteps,
            observation_feature,
            contact_feature,
        )
        coefficient_loss = F.mse_loss(prediction, noise, reduction="none").sum((1, 2)).mean()
        reconstruction_loss = (
            F.mse_loss(targets["reconstructed_action"], action, reduction="none").sum((1, 2))
            / (2.0 * float(self.loss_config["action_likelihood_std"]) ** 2)
        ).mean()
        gate_loss = sticky_gate_loss(
            targets["global_concentration"],
            targets["posterior_concentration"],
            alpha=float(self.loss_config["gate_alpha"]),
            alpha0=float(self.loss_config["gate_alpha0"]),
            kappa=float(self.loss_config["gate_kappa"]),
        )
        alignment_loss = router_alignment_loss(
            targets["posterior_concentration"], targets["prior_concentration"]
        )
        total = (
            float(self.loss_config["coefficient"]) * coefficient_loss
            + float(self.loss_config["reconstruction"]) * reconstruction_loss
            + float(self.loss_config["gate"]) * gate_loss
            + float(self.loss_config["alignment"]) * alignment_loss
            + float(self.loss_config["contact"]) * contact_loss
        )
        metrics = {
            "loss": total.detach(),
            "loss_coefficient": coefficient_loss.detach(),
            "loss_reconstruction": reconstruction_loss.detach(),
            "loss_gate": gate_loss.detach(),
            "loss_alignment": alignment_loss.detach(),
            "loss_contact": contact_loss.detach(),
            "contact_correct": contact_metrics["correct"].detach(),
            "contact_count": contact_metrics["count"].detach(),
            "predicted_contact_rows": predicted_rows.sum().detach(),
            "batch_rows": torch.tensor(action.shape[0], device=action.device),
        }
        return total, metrics

    @torch.no_grad()
    def predict_action(
        self,
        observation: Mapping[str, torch.Tensor],
        languages: list[str],
        num_inference_steps: int,
        action_steps: int,
    ) -> torch.Tensor:
        contact_plan = self.model.plan_contact(observation, languages)
        observation_feature = self.model.observation_encoder(
            self.normalize_observation(observation)
        )
        contact_feature = self.model.encode_contact(contact_plan)
        basis_bias = self.model.physgraph(observation, contact_plan.token_ids)
        basis = self.model.smp.basis(observation_feature, basis_bias)
        gate = self.model.smp.route(observation_feature, contact_feature)["prior_gate"]
        coefficients = torch.randn(
            observation_feature.shape[0],
            self.model.smp.action_horizon,
            self.model.smp.num_experts,
            device=observation_feature.device,
            dtype=observation_feature.dtype,
        )
        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(num_inference_steps, device=observation_feature.device)
        for timestep in scheduler.timesteps:
            prediction = self.model.smp.denoise(
                coefficients,
                timestep,
                observation_feature,
                contact_feature,
            )
            coefficients = scheduler.step(prediction, timestep, coefficients).prev_sample
        action = self.model.smp.decode(basis, gate, coefficients)
        start = self.model.observation_encoder.obs_horizon - 1
        return action[:, start : start + action_steps]
