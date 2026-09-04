"""Gemma-3 multi-view grasp-instruction annotation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

OUTPUT_FIELDS = (
    "class_name",
    "grasped_object_part",
    "low_level_grasp_instruction",
    "high_level_grasp_instruction",
)


class AnnotationFormatError(ValueError):
    """Gemma output cannot be accepted as a structured annotation."""


def parse_annotation(raw: str) -> dict[str, str]:
    """Parse Gemma JSON, allowing the code fence it commonly adds."""
    candidate = raw.strip()
    lines = candidate.splitlines()
    if len(lines) >= 3 and lines[0].strip().lower() in ("```", "```json"):
        if lines[-1].strip() != "```":
            raise AnnotationFormatError(f"Unclosed annotation code fence: {raw[:500]!r}")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        annotation = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise AnnotationFormatError(
            f"Gemma returned invalid annotation JSON: {raw[:500]!r}"
        ) from error
    if not isinstance(annotation, dict):
        raise AnnotationFormatError(
            f"Annotation must be a JSON object, received {type(annotation).__name__}"
        )
    if set(annotation) != set(OUTPUT_FIELDS):
        raise AnnotationFormatError(f"Unexpected annotation fields: {sorted(annotation)}")
    if not all(isinstance(annotation[field], str) for field in OUTPUT_FIELDS):
        raise AnnotationFormatError("Every annotation field must be a string")
    return annotation


class GemmaGraspAnnotator:
    def __init__(
        self,
        checkpoint: str | Path,
        prompt_path: str | Path,
        device: str,
        max_new_tokens: int = 512,
        revision: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.prompt_path = Path(prompt_path)
        self.prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        self.prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        self.max_new_tokens = max_new_tokens
        self.revision = revision
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, revision=revision, local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            revision=revision,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()
        self.device = torch.device(device)

    @property
    def model_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)

    @torch.inference_mode()
    def annotate(
        self,
        images: np.ndarray,
        contact_links: Sequence[str],
        object_id: str,
        task_id: str,
    ) -> tuple[dict[str, str], str]:
        context = (
            f"task_id: {task_id}\n"
            f"object_identifier: {object_id}\n"
            f"contact_links: {', '.join(contact_links)}"
        )
        content = [{"type": "image", "image": Image.fromarray(image)} for image in images]
        content.append({"type": "text", "text": f"{self.prompt}\n\n{context}"})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        annotation = parse_annotation(raw)
        return annotation, raw

    def generation_metadata(self) -> Mapping[str, object]:
        return {
            "model_checkpoint": str(self.checkpoint),
            "model_revision": self.model_revision or self.revision,
            "prompt_sha256": self.prompt_hash,
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "n_queries": 1,
        }
