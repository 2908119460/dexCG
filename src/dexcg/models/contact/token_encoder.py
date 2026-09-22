"""Set encoder for DextER's generated contact tokens."""

import torch
from torch import nn

from dexcg.common.tensors import masked_mean
from dexcg.models.contact.projector import ContactProjector


class ContactTokenEncoder(nn.Module):
    """Encode contact items while preserving each link's XYZ roles.

    Parse canonical ``(link, x, y, z)`` groups before set aggregation.
    Tokenizer metadata is required: silently pooling raw tokens would lose
    coordinate roles and link/position associations.
    """

    def __init__(
        self,
        llm_dim: int,
        feature_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projector = ContactProjector(llm_dim, feature_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim),
            enable_nested_tensor=False,
        )
        self.role_embedding = nn.Embedding(4, feature_dim)
        self.contact_projector = nn.Sequential(
            nn.LayerNorm(4 * feature_dim),
            nn.Linear(4 * feature_dim, feature_dim),
            nn.GELU(),
        )
        self.output_dim = feature_dim

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        link_token_ids: torch.Tensor,
        position_token_ids: torch.Tensor,
        joint_start_id: int,
        joint_end_id: int,
    ) -> torch.Tensor:
        projected = self.projector(token_embeddings)
        if token_ids.shape != attention_mask.shape or token_ids.shape[:2] != projected.shape[:2]:
            raise ValueError("token IDs, attention mask, and embeddings must share [batch, length]")
        if token_ids.shape[1] < 2:
            raise ValueError("contact sequence requires START and END tokens")

        max_contacts = max(1, (token_ids.shape[1] - 1) // 4)
        starts = 1 + 4 * torch.arange(max_contacts, device=token_ids.device)
        offsets = torch.arange(4, device=token_ids.device)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1).clamp_max(token_ids.shape[1] - 1)
        groups = projected.index_select(1, indices).reshape(
            token_ids.shape[0], max_contacts, 4, self.output_dim
        )
        group_ids = token_ids.index_select(1, indices).reshape(
            token_ids.shape[0], max_contacts, 4
        )
        group_mask = attention_mask.index_select(1, indices).reshape(
            token_ids.shape[0], max_contacts, 4
        ).all(dim=-1)
        link_mask = group_ids[:, :, 0, None].eq(link_token_ids.reshape(1, 1, -1)).any(dim=-1)
        coordinate_mask = group_ids[:, :, 1:, None].eq(
            position_token_ids.reshape(1, 1, 1, -1)
        ).any(dim=-1).all(dim=-1)
        contact_mask = group_mask & link_mask & coordinate_mask

        # Accept both dataset masks (END is valid) and generated masks (END
        # is masked); reject partial tuples, interior padding, or invalid IDs.
        counts = contact_mask.sum(dim=1)
        end_indices = 1 + 4 * counts
        columns = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
        payload = columns < end_indices[:, None]
        expected_groups = torch.arange(max_contacts, device=token_ids.device)[None] < counts[:, None]
        valid_sequence = (
            token_ids[:, 0].eq(joint_start_id)
            & (end_indices < token_ids.shape[1])
            & token_ids.gather(1, end_indices.clamp_max(token_ids.shape[1] - 1)[:, None]).squeeze(1).eq(joint_end_id)
            & (contact_mask == expected_groups).all(dim=1)
            & (attention_mask.bool() | ~payload).all(dim=1)
            & (~attention_mask.bool() | (columns <= end_indices[:, None])).all(dim=1)
        )
        if not valid_sequence.all():
            raise ValueError("malformed contact sequence: expected START, (link,x,y,z)*, END")

        roles = self.role_embedding.weight.reshape(1, 1, 4, self.output_dim)
        groups = (groups + roles).reshape(token_ids.shape[0], max_contacts, -1)
        contact_features = self.contact_projector(groups)
        # Keep an unmasked dummy slot for empty plans to avoid all-masked
        # attention NaNs, while preserving a DDP-compatible computation graph.
        safe_mask = contact_mask.clone()
        safe_mask[:, 0] |= ~contact_mask.any(dim=1)
        contact_features = contact_features.masked_fill(~contact_mask[:, :, None], 0)
        encoded = self.encoder(contact_features, src_key_padding_mask=~safe_mask)
        return masked_mean(encoded, contact_mask, dim=1)
