"""DextER-derived PartField/Qwen planner with contact-only generation."""

from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer, LogitsProcessorList, Qwen2ForCausalLM

from dexcg.common.typing import ContactPlan
from dexcg.models.contact.grammar import ContactGrammarLogitsProcessor
from dexcg.models.contact.partfield import PartFieldEncoder
from dexcg.models.contact.partfield.encoder import PartFieldConfig
from dexcg.models.contact.projector import PointCloudProjector
from dexcg.models.contact.tokenizer import (
    VISION_TOKENS,
    AllegroContactTokenizer,
    dexter_checkpoint_tokens,
)

DEFAULT_PROMPT = (
    "First predict which Allegro hand links should contact where on the object "
    "to satisfy the grasp instruction. Return only the contact block. Query: {language}"
)


class ContactPlanner(nn.Module):
    """Slow branch: PartField observation tokens + Qwen language tokens -> contacts."""

    def __init__(
        self,
        point_encoder: PartFieldEncoder,
        point_projector: PointCloudProjector,
        language_model: Qwen2ForCausalLM,
        contact_tokenizer: AllegroContactTokenizer,
        max_new_tokens: int = 66,
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        super().__init__()
        self.point_encoder = point_encoder
        self.point_projector = point_projector
        self.language_model = language_model
        self.contact_tokenizer = contact_tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_template = prompt_template

    @classmethod
    def from_dexter_checkpoint(
        cls,
        checkpoint: str | Path,
        base_model: str | Path,
        position_bins: int = 256,
        min_position: float = -0.4,
        max_position: float = 0.4,
        max_new_tokens: int = 66,
        temperature: float = 0.0,
        partfield: Mapping[str, object] | None = None,
        prompt_template: str = DEFAULT_PROMPT,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ContactPlanner":
        checkpoint = Path(checkpoint)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        base_tokenizer.padding_side = "right"
        base_tokenizer.add_special_tokens(
            {"additional_special_tokens": dexter_checkpoint_tokens(position_bins=position_bins)}
        )

        qwen_config = AutoConfig.from_pretrained(base_model, local_files_only=True)
        language_model = Qwen2ForCausalLM(qwen_config).to(dtype=torch_dtype)
        language_model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)

        point_encoder = PartFieldEncoder(PartFieldConfig(**(partfield or {}))).to(dtype=torch_dtype)
        point_projector = PointCloudProjector(
            input_dim=point_encoder.point_feat_dim,
            output_dim=qwen_config.hidden_size,
        ).to(dtype=torch_dtype)

        state = load_file(checkpoint / "model.safetensors", device="cpu")
        point_encoder.load_state_dict(
            {
                name.removeprefix("pc_encoder."): value
                for name, value in state.items()
                if name.startswith("pc_encoder.")
            }
        )
        point_projector.load_state_dict(
            {
                name.removeprefix("pc_proj."): value
                for name, value in state.items()
                if name.startswith("pc_proj.")
            }
        )
        language_model.load_state_dict(
            {
                name.removeprefix("vlm."): value
                for name, value in state.items()
                if name.startswith("vlm.")
            },
            strict=False,
        )
        language_model.tie_weights()
        del state

        contact_tokenizer = AllegroContactTokenizer.build(
            base_tokenizer,
            model=language_model,
            position_bins=position_bins,
            min_position=min_position,
            max_position=max_position,
        )
        return cls(
            point_encoder=point_encoder,
            point_projector=point_projector,
            language_model=language_model,
            contact_tokenizer=contact_tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_template=prompt_template,
        )

    @property
    def hidden_size(self) -> int:
        return self.language_model.config.hidden_size

    def embed_contact_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.get_input_embeddings()(token_ids)

    def training_loss(
        self,
        point_cloud: torch.Tensor,
        languages: list[str],
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Teacher-force contact tokens after the multimodal prompt."""
        tokenized = self._tokenize_languages(languages, point_cloud.device)
        prefix_embeddings, prefix_mask, prefix_groups = self._embed_prefix(
            point_cloud, tokenized["input_ids"], tokenized["attention_mask"]
        )
        target_ids = target_ids.to(device=point_cloud.device, dtype=torch.long)
        target_mask = target_mask.to(device=point_cloud.device, dtype=torch.bool)
        input_ids = target_ids[:, :-1]
        input_mask = target_mask[:, :-1]
        token_embeddings = self.language_model.get_input_embeddings()(input_ids)
        embeddings = torch.cat((prefix_embeddings, token_embeddings), dim=1)
        padding_mask = torch.cat((prefix_mask, input_mask), dim=1)
        target_groups = torch.ones_like(input_ids)
        attention_groups = torch.cat((prefix_groups, target_groups), dim=1)
        position_ids = padding_mask.long().cumsum(dim=1).sub(1).clamp_min(0)
        outputs = self.language_model(
            inputs_embeds=embeddings,
            attention_mask=self._attention_mask(
                padding_mask, attention_groups, embeddings.dtype
            ),
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        prefix_length = prefix_embeddings.shape[1]
        logits = outputs.logits[:, prefix_length : prefix_length + input_ids.shape[1]]
        labels = target_ids[:, 1:]
        label_mask = target_mask[:, 1:]
        loss = F.cross_entropy(logits[label_mask].float(), labels[label_mask])
        with torch.no_grad():
            correct = logits[label_mask].argmax(dim=-1).eq(labels[label_mask]).sum()
            count = label_mask.sum()
        return loss, {"correct": correct, "count": count}

    def _tokenize_languages(
        self, languages: list[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        tokenizer = self.contact_tokenizer.base_tokenizer
        texts = []
        vision_placeholder = VISION_TOKENS[1] + VISION_TOKENS[0] + VISION_TOKENS[2]
        for language in languages:
            message = vision_placeholder + self.prompt_template.format(language=language.strip())
            texts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": message}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        return tokenizer(texts, padding=True, return_tensors="pt").to(device)

    def _embed_prefix(
        self,
        point_cloud: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_tokens = self.point_encoder(point_cloud, return_point_features=False)
        point_tokens = self.point_projector(
            point_tokens.to(dtype=self.point_projector.proj.weight.dtype)
        )

        vision_id = self.contact_tokenizer.vision_token_id
        vision_columns = (input_ids == vision_id).nonzero(as_tuple=False)[:, 1]
        vision_column = int(vision_columns[0])
        repeated_vision = input_ids[:, vision_column : vision_column + 1].expand(
            -1, point_tokens.shape[1]
        )
        interpolated_ids = torch.cat(
            (input_ids[:, :vision_column], repeated_vision, input_ids[:, vision_column + 1 :]),
            dim=1,
        )
        interpolated_mask = torch.cat(
            (
                attention_mask[:, :vision_column],
                attention_mask[:, vision_column : vision_column + 1].expand(
                    -1, point_tokens.shape[1]
                ),
                attention_mask[:, vision_column + 1 :],
            ),
            dim=1,
        )
        embeddings = self.language_model.get_input_embeddings()(interpolated_ids)
        vision_mask = interpolated_ids == vision_id
        embeddings = embeddings.masked_scatter(
            vision_mask.unsqueeze(-1).expand_as(embeddings),
            point_tokens.to(dtype=embeddings.dtype),
        )
        attention_groups = torch.ones_like(interpolated_ids)
        attention_groups.masked_fill_(vision_mask, 0)
        return embeddings, interpolated_mask, attention_groups

    @staticmethod
    def _attention_mask(
        padding_mask: torch.Tensor,
        attention_groups: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        groups = attention_groups.cumsum(dim=1)
        causal = groups[:, None, :] <= groups[:, :, None]
        valid = padding_mask.bool()
        allowed = causal & valid[:, None, :] & valid[:, :, None]
        visible = torch.zeros((), dtype=dtype, device=padding_mask.device)
        blocked = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=padding_mask.device)
        return torch.where(allowed[:, None], visible, blocked)

    @torch.no_grad()
    def plan(self, point_cloud: torch.Tensor, languages: list[str]) -> ContactPlan:
        tokenized = self._tokenize_languages(languages, point_cloud.device)
        embeddings, padding_mask, groups = self._embed_prefix(
            point_cloud, tokenized["input_ids"], tokenized["attention_mask"]
        )
        position_ids = padding_mask.long().cumsum(dim=1) - 1
        outputs = self.language_model.model(
            inputs_embeds=embeddings,
            attention_mask=self._attention_mask(padding_mask, groups, embeddings.dtype),
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        batch_size, prefix_length = padding_mask.shape
        start_tokens = torch.full(
            (batch_size, 1),
            self.contact_tokenizer.joint_start_id,
            dtype=torch.long,
            device=point_cloud.device,
        )
        generation_mask = torch.cat((padding_mask.long(), torch.ones_like(start_tokens)), dim=1)
        generation_kwargs = {
            "input_ids": start_tokens,
            "attention_mask": generation_mask,
            "past_key_values": outputs.past_key_values,
            "cache_position": torch.tensor([prefix_length], device=point_cloud.device),
            "max_new_tokens": self.max_new_tokens - 1,
            "eos_token_id": self.contact_tokenizer.joint_end_id,
            "pad_token_id": self.contact_tokenizer.joint_end_id,
            "logits_processor": LogitsProcessorList(
                [ContactGrammarLogitsProcessor(self.contact_tokenizer)]
            ),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        token_ids = self.language_model.generate(**generation_kwargs)
        attention_mask = token_ids.ne(self.contact_tokenizer.joint_end_id)
        attention_mask[:, 0] = True
        first_end = token_ids.eq(self.contact_tokenizer.joint_end_id).int().argmax(dim=1)
        attention_mask.scatter_(1, first_end.unsqueeze(1), True)
        return ContactPlan(token_ids=token_ids, attention_mask=attention_mask)

    forward = plan
