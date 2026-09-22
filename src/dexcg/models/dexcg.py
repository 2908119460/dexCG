"""Top-level dexCG network."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from dexcg.common.config import ProjectConfig
from dexcg.common.typing import ContactPlan, DexCGOutput
from dexcg.models.contact.coordinates import (
    CONTACT_MAX_POSITION,
    CONTACT_MIN_POSITION,
    CONTACT_POSITION_BINS,
    robot_base_point_cloud,
)
from dexcg.models.contact.planner import ContactPlanner
from dexcg.models.contact.token_encoder import ContactTokenEncoder
from dexcg.models.observation.dp3_encoder import DP3ObservationEncoder
from dexcg.models.physgraph import PhysGraphBasisBias, load_robot_graph_spec
from dexcg.models.smp.coefficients import coefficient_targets
from dexcg.models.smp.model import ContactConditionedSMP
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS


class DexCG(nn.Module):
    """Observation + language -> contact-conditioned SMP representation."""

    def __init__(
        self,
        observation_encoder: DP3ObservationEncoder,
        contact_planner: ContactPlanner,
        contact_encoder: ContactTokenEncoder,
        smp: ContactConditionedSMP,
        physgraph: PhysGraphBasisBias | None = None,
    ) -> None:
        super().__init__()
        self.observation_encoder = observation_encoder
        self.contact_planner = contact_planner
        self.contact_encoder = contact_encoder
        self.smp = smp
        self.physgraph = physgraph
        # SMP owns its input lookup. Switching a finetuned planner must not
        # change the meaning of identical contact IDs under a fixed policy.
        tokenizer = contact_planner.contact_tokenizer
        ids = torch.tensor(sorted(set((*tokenizer.link_token_ids,
            *tokenizer.position_token_ids, tokenizer.joint_start_id, tokenizer.joint_end_id))))
        device = next(contact_planner.parameters()).device
        with torch.no_grad():
            values = contact_planner.embed_contact_tokens(ids.to(device)).detach().float().clone()
        self.register_buffer("contact_embedding_ids", ids.to(device))
        self.register_buffer("contact_embedding_values", values)

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        project_root: str | Path,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "DexCG":
        observation_config = dict(config.observation)
        observation_encoder = DP3ObservationEncoder(**observation_config)

        planner_config = dict(config.contact_planner)
        for key, expected in (
            ("min_position", CONTACT_MIN_POSITION),
            ("max_position", CONTACT_MAX_POSITION),
            ("position_bins", CONTACT_POSITION_BINS),
        ):
            if planner_config.get(key, expected) != expected:
                raise ValueError(f"robot-base contact contract requires {key}={expected}")
        checkpoint = Path(project_root) / planner_config.pop("checkpoint")
        base_model = Path(project_root) / planner_config.pop("base_model")
        prompt_template = (
            (Path(project_root) / planner_config.pop("prompt")).read_text(encoding="utf-8").strip()
        )
        contact_planner = ContactPlanner.from_dexter_checkpoint(
            checkpoint=checkpoint,
            base_model=base_model,
            prompt_template=prompt_template,
            torch_dtype=torch_dtype,
            **planner_config,
        )
        contact_encoder = ContactTokenEncoder(
            llm_dim=contact_planner.hidden_size,
            **config.contact_encoder,
        )

        smp_config = dict(config.smp)
        expert_config = smp_config.pop("experts")
        smp = ContactConditionedSMP(
            observation_dim=observation_encoder.output_dim,
            contact_dim=contact_encoder.output_dim,
            expert_down_dims=expert_config["down_dims"],
            expert_timestep_dim=expert_config["diffusion_step_embed_dim"],
            expert_kernel_size=expert_config["kernel_size"],
            expert_groups=expert_config["num_groups"],
            **smp_config,
        )

        physgraph_config = dict(config.physgraph)
        robot_config = Path(project_root) / physgraph_config.pop("robot_config")
        graph_spec = load_robot_graph_spec(
            config_path=robot_config,
            project_root=project_root,
            contact_link_names=tuple(link.dexart_link for link in ALLEGRO_CONTACT_LINKS),
        )
        contact_token_ids = tuple(
            contact_planner.contact_tokenizer.link_to_id[link.token]
            for link in ALLEGRO_CONTACT_LINKS
        )
        physgraph = PhysGraphBasisBias(
            spec=graph_spec,
            contact_token_ids=contact_token_ids,
            num_experts=smp.num_experts,
            observation_horizon=observation_encoder.obs_horizon,
            **physgraph_config,
        )
        return cls(observation_encoder, contact_planner, contact_encoder, smp, physgraph)

    def plan_contact(
        self,
        observation: Mapping[str, torch.Tensor],
        languages: list[str],
        previous_plan: ContactPlan | None = None,
    ) -> ContactPlan:
        point_cloud = self.contact_planner_input(observation)
        robot_inputs = self.contact_planner_robot_input(observation)
        if previous_plan is None:
            plan = self.contact_planner.plan(point_cloud, languages, **robot_inputs)
        else:
            plan = self.contact_planner.plan(
                point_cloud, languages, previous_plan=previous_plan, **robot_inputs
            )
        return plan

    def contact_planner_robot_input(self, observation):
        if getattr(self.contact_planner, "robot_state_projector", None) is None:
            return {}
        return {
            "robot_qpos": observation["agent_pos"][:, -1, :22],
            "palm_pose_robot_base": observation["palm_pose_robot_base"][:, -1],
        }

    @staticmethod
    def contact_planner_input(
        observation: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        separate_cloud = "planner_point_cloud" in observation
        separate_mask = "planner_object_point_mask" in observation
        if separate_cloud != separate_mask:
            raise ValueError("planner point cloud and object mask must be supplied together")
        prefix = "planner_" if separate_cloud else ""
        point_cloud = observation[f"{prefix}point_cloud"][:, -1]
        object_mask = observation[f"{prefix}object_point_mask"][:, -1]
        if separate_cloud and not object_mask.bool().all():
            raise ValueError("dedicated planner cloud must contain only object points")
        return robot_base_point_cloud(point_cloud, object_mask)

    def encode_contact(self, contact_plan: ContactPlan) -> torch.Tensor:
        indices = torch.searchsorted(self.contact_embedding_ids, contact_plan.token_ids.contiguous())
        indices = indices.clamp_max(len(self.contact_embedding_ids) - 1)
        known = self.contact_embedding_ids[indices] == contact_plan.token_ids
        if (contact_plan.attention_mask.bool() & ~known).any():
            raise ValueError("Contact ID is outside the fixed SMP vocabulary")
        embeddings = self.contact_embedding_values[indices] * known.unsqueeze(-1)
        encoder_dtype = self.contact_encoder.projector.network[1].weight.dtype
        tokenizer = self.contact_planner.contact_tokenizer
        return self.contact_encoder(
            embeddings.to(dtype=encoder_dtype),
            contact_plan.attention_mask,
            token_ids=contact_plan.token_ids,
            link_token_ids=torch.as_tensor(
                tokenizer.link_token_ids, device=contact_plan.token_ids.device
            ),
            position_token_ids=torch.as_tensor(
                tokenizer.position_token_ids, device=contact_plan.token_ids.device
            ),
            joint_start_id=tokenizer.joint_start_id,
            joint_end_id=tokenizer.joint_end_id,
        )

    def forward_with_contact(
        self,
        observation: Mapping[str, torch.Tensor],
        contact_plan: ContactPlan,
        noisy_coefficients: torch.Tensor | None = None,
        timestep: torch.Tensor | int | None = None,
        action: torch.Tensor | None = None,
    ) -> DexCGOutput:
        observation_feature = self.observation_encoder(observation)
        contact_feature = self.encode_contact(contact_plan)
        basis_bias = None
        if self.physgraph is not None:
            basis_bias = self.physgraph(observation, contact_plan.token_ids)
        basis = self.smp.basis(observation_feature, basis_bias)
        routing = self.smp.route(observation_feature, contact_feature, action)

        coefficient_prediction = None
        if noisy_coefficients is not None:
            coefficient_prediction = self.smp.denoise(
                noisy_coefficients, timestep, observation_feature, contact_feature
            )

        coefficient_target = None
        reconstructed_action = None
        if action is not None:
            coefficient_target, reconstruction_coefficients = coefficient_targets(
                basis,
                action,
                routing["posterior_gate"],
                self.smp.coefficient_eps,
            )
            reconstructed_action = self.smp.decode(
                basis, routing["posterior_gate"], reconstruction_coefficients
            )

        return DexCGOutput(
            observation_feature=observation_feature,
            contact_feature=contact_feature,
            basis=basis,
            prior_concentration=routing["prior_concentration"],
            prior_gate=routing["prior_gate"],
            coefficient_prediction=coefficient_prediction,
            posterior_concentration=routing.get("posterior_concentration"),
            posterior_gate=routing.get("posterior_gate"),
            coefficient_target=coefficient_target,
            reconstructed_action=reconstructed_action,
        )

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        languages: list[str],
        noisy_coefficients: torch.Tensor | None = None,
        timestep: torch.Tensor | int | None = None,
        action: torch.Tensor | None = None,
    ) -> DexCGOutput:
        contact_plan = self.plan_contact(observation, languages)
        return self.forward_with_contact(
            observation,
            contact_plan,
            noisy_coefficients=noisy_coefficients,
            timestep=timestep,
            action=action,
        )
