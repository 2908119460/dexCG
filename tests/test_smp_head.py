import torch

from dexcg.models.smp.model import ContactConditionedSMP


def make_smp() -> ContactConditionedSMP:
    return ContactConditionedSMP(
        observation_dim=12,
        contact_dim=8,
        action_dim=6,
        action_horizon=8,
        num_experts=4,
        condition_dim=16,
        basis_hidden_dim=16,
        gate_hidden_dim=16,
        expert_down_dims=(8, 16),
        expert_timestep_dim=8,
        expert_groups=4,
    )


def test_basis_and_contact_conditioned_routing_shapes() -> None:
    smp = make_smp()
    observation = torch.randn(2, 12)
    contact = torch.randn(2, 8)
    action = torch.randn(2, 8, 6)
    targets = smp.build_training_targets(observation, contact, action)

    assert targets["basis"].shape == (2, 6, 4)
    assert targets["prior_gate"].shape == (2, 8, 4)
    assert targets["posterior_gate"].shape == (2, 8, 4)
    assert targets["coefficient_target"].shape == (2, 8, 4)
    assert targets["reconstructed_action"].shape == action.shape
    gram = targets["basis"].transpose(-2, -1) @ targets["basis"]
    assert torch.allclose(gram, torch.eye(4).expand_as(gram), atol=1e-5)


def test_each_expert_predicts_one_coefficient_channel() -> None:
    smp = make_smp()
    prediction = smp.denoise(
        noisy_coefficients=torch.randn(2, 8, 4),
        timestep=torch.tensor([3, 7]),
        observation=torch.randn(2, 12),
        contact=torch.randn(2, 8),
    )
    assert prediction.shape == (2, 8, 4)


def test_basis_bias_changes_only_the_orthogonal_basis() -> None:
    torch.manual_seed(7)
    smp = make_smp().eval()
    observation = torch.randn(2, 12)
    contact = torch.randn(2, 8)
    action = torch.randn(2, 8, 6)
    noisy = torch.randn(2, 8, 4)
    basis_bias = torch.randn(2, 6, 4)

    unbiased_basis = smp.basis(observation)
    biased_basis = smp.basis(observation, basis_bias)
    unbiased_route = smp.route(observation, contact, action)
    biased_route = smp.route(observation, contact, action)
    unbiased_denoising = smp.denoise(noisy, 3, observation, contact)
    biased_denoising = smp.denoise(noisy, 3, observation, contact)

    assert not torch.allclose(unbiased_basis, biased_basis)
    gram = biased_basis.transpose(-2, -1) @ biased_basis
    assert torch.allclose(gram, torch.eye(4).expand_as(gram), atol=1e-5)
    assert torch.equal(unbiased_route["prior_gate"], biased_route["prior_gate"])
    assert torch.equal(unbiased_route["posterior_gate"], biased_route["posterior_gate"])
    assert torch.equal(unbiased_denoising, biased_denoising)
