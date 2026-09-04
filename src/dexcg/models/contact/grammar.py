"""Finite-state grammar for contact-only autoregressive decoding."""

import torch
from transformers import LogitsProcessor


class ContactGrammarLogitsProcessor(LogitsProcessor):
    """Allow only `(Allegro link, x, y, z)*` after a forced start token."""

    def __init__(self, tokenizer: "AllegroContactTokenizer") -> None:
        self.link_ids = torch.tensor(tokenizer.link_token_ids, dtype=torch.long)
        self.position_ids = torch.tensor(tokenizer.position_token_ids, dtype=torch.long)
        self.start_id = tokenizer.joint_start_id
        self.end_id = tokenizer.joint_end_id
        self.max_contacts = len(tokenizer.link_token_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        for row, sequence in enumerate(input_ids):
            start_positions = (sequence == self.start_id).nonzero(as_tuple=False)
            emitted = sequence.numel() - int(start_positions[-1]) - 1
            phase = emitted % 4
            if phase == 0:
                if emitted // 4 < self.max_contacts:
                    allowed[row, self.link_ids.to(scores.device)] = True
                allowed[row, self.end_id] = True
            else:
                allowed[row, self.position_ids.to(scores.device)] = True
        return scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)


from dexcg.models.contact.tokenizer import AllegroContactTokenizer  # noqa: E402
