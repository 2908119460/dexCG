import torch
import pytest

from dexcg.models.contact.token_encoder import ContactTokenEncoder


def make_encoder():
    torch.manual_seed(123)
    return ContactTokenEncoder(
        llm_dim=16,
        feature_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=32,
    ).eval()


def encode(encoder, ids, mask=None, embedding=None):
    if embedding is None:
        embedding = torch.arange(64 * 16, dtype=torch.float32).reshape(64, 16).sin()
    if mask is None:
        mask = ids.ne(3)
    return encoder(
        embedding[ids], mask, token_ids=ids, link_token_ids=torch.tensor([10, 11]),
        position_token_ids=torch.arange(20, 64), joint_start_id=2, joint_end_id=3,
    )


def test_axis_and_link_coordinate_associations_are_preserved():
    encoder = make_encoder()
    ids = torch.tensor([
        [2, 10, 20, 21, 22, 11, 23, 24, 25, 3],
        [2, 10, 21, 20, 22, 11, 23, 24, 25, 3],
        [2, 10, 23, 24, 25, 11, 20, 21, 22, 3],
        [2, 11, 23, 24, 25, 10, 20, 21, 22, 3],
    ])
    features = encode(encoder, ids)
    assert (features[0] - features[1]).abs().max() > 1e-4
    assert (features[0] - features[2]).abs().max() > 1e-4
    torch.testing.assert_close(features[0], features[3], atol=1e-6, rtol=1e-5)


def test_padding_and_end_mask_do_not_change_contact_features():
    encoder = make_encoder()
    ids = torch.tensor([[2, 10, 20, 21, 22, 3]])
    result = encode(encoder, ids)
    torch.testing.assert_close(result, encode(encoder, ids, torch.ones_like(ids, dtype=torch.bool)))
    padded = torch.cat((ids, torch.full((1, 60), 3)), dim=1)
    torch.testing.assert_close(result, encode(encoder, padded), atol=1e-6, rtol=1e-5)


def test_empty_contacts_are_finite_and_keep_trainable_graph():
    encoder = make_encoder()
    result = encode(encoder, torch.tensor([[2, 3], [2, 3]]))
    assert torch.equal(result, torch.zeros_like(result))
    result.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


@pytest.mark.parametrize("sequence", [[2, 10, 20, 21, 3], [2, 10, 20, 11, 22, 3], [2, 10, 20, 21, 22, 20]])
def test_malformed_contacts_are_rejected(sequence):
    with pytest.raises(ValueError, match="malformed"):
        encode(make_encoder(), torch.tensor([sequence]))
