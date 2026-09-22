"""Contact-only tokenizer and fixed DextER-to-Allegro embedding transfer."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from dexcg.robots.allegro import ALLEGRO_CONTACT_TOKENS, SHADOW_TO_ALLEGRO_TOKEN

DEXTER_SHADOW_CONTACT_NAMES = (
    "rh_palm",
    "rh_ffdistal",
    "rh_ffmiddle",
    "rh_ffproximal",
    "rh_ffknuckle",
    "rh_mfdistal",
    "rh_mfmiddle",
    "rh_mfproximal",
    "rh_mfknuckle",
    "rh_rfdistal",
    "rh_rfmiddle",
    "rh_rfproximal",
    "rh_rfknuckle",
    "rh_lfdistal",
    "rh_lfmiddle",
    "rh_lfproximal",
    "rh_lfknuckle",
    "rh_thdistal",
    "rh_thhub",
    "rh_thmiddle",
    "rh_thproximal",
)

VISION_TOKENS = ("<|vision_pad|>", "<|vision_start|>", "<|vision_end|>")
JOINT_START = "<|joint_start|>"
JOINT_END = "<|joint_end|>"


def dexter_checkpoint_tokens(action_bins: int = 256, position_bins: int = 256) -> list[str]:
    """Special-token order used by the released DextER checkpoint."""
    tokens = list(VISION_TOKENS)
    tokens.extend(f"<action_bin_{index}>" for index in range(action_bins))
    tokens.extend(("<|action_start|>", "<|action_end|>"))
    tokens.extend(f"<pos_bin_{index}>" for index in range(position_bins))
    tokens.extend(("<|pos_start|>", "<|pos_end|>"))
    tokens.extend(f"<{name}>" for name in DEXTER_SHADOW_CONTACT_NAMES)
    tokens.extend((JOINT_START, JOINT_END))
    return tokens


class AllegroContactTokenizer:
    """Tokenize metric robot-base contacts with the DextER boundary convention."""

    def __init__(
        self,
        base_tokenizer,
        position_bins: int = 256,
        min_position: float = -1.0,
        max_position: float = 1.2,
    ) -> None:
        self.base_tokenizer = base_tokenizer
        self.position_bins = position_bins
        self.min_position = min_position
        self.max_position = max_position
        self.position_boundaries = np.linspace(
            min_position, max_position, position_bins, dtype=np.float32
        )
        self.position_centers = (self.position_boundaries[:-1] + self.position_boundaries[1:]) / 2.0

        vocabulary = base_tokenizer.get_vocab()
        self.link_to_id = {token: vocabulary[token] for token in ALLEGRO_CONTACT_TOKENS}
        self.id_to_link = {token_id: token for token, token_id in self.link_to_id.items()}
        self.position_token_ids = np.asarray(
            [vocabulary[f"<pos_bin_{index}>"] for index in range(position_bins)],
            dtype=np.int64,
        )
        self.position_id_to_bin = {
            int(token_id): index for index, token_id in enumerate(self.position_token_ids)
        }
        self.joint_start_id = vocabulary[JOINT_START]
        self.joint_end_id = vocabulary[JOINT_END]
        self.vision_token_id = vocabulary[VISION_TOKENS[0]]

    @classmethod
    def build(
        cls,
        base_tokenizer,
        model=None,
        position_bins: int = 256,
        min_position: float = -1.0,
        max_position: float = 1.2,
    ) -> "AllegroContactTokenizer":
        """Prepare DextER's checkpoint vocabulary, then add the final Allegro tokens."""
        base_tokenizer.add_special_tokens(
            {"additional_special_tokens": dexter_checkpoint_tokens(position_bins=position_bins)}
        )
        checkpoint_vocab_size = len(base_tokenizer)
        base_tokenizer.add_special_tokens(
            {"additional_special_tokens": list(ALLEGRO_CONTACT_TOKENS)}
        )

        if model is not None:
            model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
            embedding = model.get_input_embeddings().weight
            vocabulary = base_tokenizer.get_vocab()
            with torch.no_grad():
                for shadow_token, allegro_token in SHADOW_TO_ALLEGRO_TOKEN.items():
                    embedding[vocabulary[allegro_token]].copy_(embedding[vocabulary[shadow_token]])
            model.tie_weights()

        instance = cls(base_tokenizer, position_bins, min_position, max_position)
        instance.checkpoint_vocab_size = checkpoint_vocab_size
        return instance

    @property
    def link_token_ids(self) -> tuple[int, ...]:
        return tuple(self.link_to_id.values())

    def encode(self, contacts: Mapping[str, Sequence[Sequence[float]]]) -> list[int]:
        ids = [self.joint_start_id]
        for link_token in ALLEGRO_CONTACT_TOKENS:
            token_name = link_token[1:-1]
            for position in contacts.get(token_name, ()):
                clipped = np.asarray(position, dtype=np.float32)
                if clipped.shape != (3,) or not np.isfinite(clipped).all():
                    raise ValueError("contact position must be a finite XYZ vector")
                if np.any(clipped < self.min_position) or np.any(clipped > self.max_position):
                    raise ValueError("robot-base contact position exceeds the token range")
                bins = (
                    np.digitize(clipped, self.position_boundaries).clip(1, self.position_bins) - 1
                )
                ids.append(self.link_to_id[link_token])
                ids.extend(self.position_token_ids[bins].tolist())
        ids.append(self.joint_end_id)
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        position_offset: Sequence[float] | None = None,
    ) -> dict[str, list[list[float]]]:
        contacts: dict[str, list[list[float]]] = {}
        values = list(map(int, token_ids))
        offset = (
            np.zeros(3, dtype=np.float32)
            if position_offset is None
            else np.asarray(position_offset, dtype=np.float32)
        )
        if offset.shape != (3,):
            raise ValueError(f"position_offset must have shape (3,), received {offset.shape}")
        if not np.all(offset == 0):
            raise ValueError("robot-base contact tokens must not receive a position offset")
        start = values.index(self.joint_start_id) + 1
        end = values.index(self.joint_end_id, start)
        cursor = start
        while cursor < end:
            link_token = self.id_to_link[values[cursor]]
            bins = [
                self.position_id_to_bin[values[cursor + coordinate_offset]]
                for coordinate_offset in (1, 2, 3)
            ]
            bins = np.clip(bins, 0, len(self.position_centers) - 1)
            local_position = self.position_centers[bins]
            contacts.setdefault(link_token[1:-1], []).append((local_position + offset).tolist())
            cursor += 4
        return contacts
