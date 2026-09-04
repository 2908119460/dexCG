"""Set encoder for DextER's generated contact tokens."""

import torch
from torch import nn

from dexcg.common.tensors import masked_mean
from dexcg.models.contact.projector import ContactProjector


class ContactTokenEncoder(nn.Module):
    """Project, self-attend, and pool a contact graph token set."""

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
        self.output_dim = feature_dim

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        projected = self.projector(token_embeddings)
        encoded = self.encoder(projected, src_key_padding_mask=~attention_mask.bool())
        return masked_mean(encoded, attention_mask, dim=1)
