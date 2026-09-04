import torch

from dexcg.models.contact.token_encoder import ContactTokenEncoder


def test_contact_token_encoder_pools_valid_tokens() -> None:
    encoder = ContactTokenEncoder(
        llm_dim=16,
        feature_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
    )
    embeddings = torch.randn(2, 6, 16)
    mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, False, False, False]]
    )
    assert encoder(embeddings, mask).shape == (2, 8)
