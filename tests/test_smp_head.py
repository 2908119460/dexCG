import torch
import pytest
from torch import nn
from types import SimpleNamespace

from dexcg.models.smp.model import ContactConditionedSMP
from dexcg.models.smp.losses import router_alignment_loss, sticky_gate_loss
from dexcg.training.objective import DexCGTrainingObjective
from dexcg.common.typing import ContactPlan


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


def test_full_action_mask_preserves_gate_loss_scales():
    posterior = torch.rand(2, 8, 4) + 0.5
    prior = torch.rand(2, 8, 4) + 0.5
    global_q = torch.ones(4)
    mask = torch.ones(2, 8, dtype=torch.bool)
    torch.testing.assert_close(
        router_alignment_loss(posterior, prior), router_alignment_loss(posterior, prior, mask)
    )
    torch.testing.assert_close(
        sticky_gate_loss(global_q, posterior, 2, .5, 20),
        sticky_gate_loss(global_q, posterior, 2, .5, 20, mask),
    )


class _MaskObservation(nn.Module):
    def forward(self, observation):
        return observation["agent_pos"][:, -1]


class _MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.smp = make_smp()
        self.observation_encoder = _MaskObservation()
        self.contact_planner = nn.Embedding(8, 8)
        self.contact_planner.contact_tokenizer = SimpleNamespace(joint_end_id=3)
        self.contact_encoder = nn.Linear(8, 8)
        self.plan_calls = 0

    def encode_contact(self, plan):
        return self.contact_encoder(self.contact_planner(plan.token_ids).mean(dim=1))

    def physgraph(self, observation, ids):
        return observation["agent_pos"].new_zeros((len(ids), 6, 4))

    def plan_contact(self, observation, languages):
        self.plan_calls += 1
        ids = torch.tensor([[2, 4, 3]]).expand(len(languages), -1)
        return ContactPlan(ids, torch.ones_like(ids, dtype=torch.bool))


def _masked_objective():
    return DexCGTrainingObjective(
        _MaskModel(), torch.zeros(12), torch.ones(12),
        {"num_train_timesteps": 10, "beta_start": .0001, "beta_end": .02,
         "beta_schedule": "linear", "prediction_type": "epsilon"},
        {"coefficient": 1, "reconstruction": 1, "gate": 1, "alignment": 1,
         "contact": 0, "action_likelihood_std": 1, "gate_alpha": 2,
         "gate_alpha0": .5, "gate_kappa": 20}, train_contact_planner=False,
    )


def _masked_batch():
    return {
        "observation": {"agent_pos": torch.rand(2, 2, 12)},
        "action": torch.randn(2, 8, 6),
        "action_valid_mask": torch.arange(8)[None] < torch.tensor([[1], [5]]),
        "contact_token_ids": torch.tensor([[2, 5, 3], [2, 5, 3]]),
        "contact_token_mask": torch.ones(2, 3, dtype=torch.bool),
        "language": ["test", "test"],
    }


def test_invalid_actions_cannot_change_loss_or_gradients():
    torch.manual_seed(42)
    objective, batch = _masked_objective(), _masked_batch()
    def run(action):
        objective.zero_grad(set_to_none=True)
        action = action.detach().requires_grad_(True)
        torch.manual_seed(77)
        loss, metrics = objective({**batch, "action": action}, 1.0)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.equal(action.grad[~batch["action_valid_mask"]],
                           torch.zeros_like(action.grad[~batch["action_valid_mask"]]))
        return metrics, {n: p.grad.clone() for n, p in objective.named_parameters() if p.grad is not None}
    baseline, baseline_grad = run(batch["action"])
    changed = batch["action"].masked_fill(~batch["action_valid_mask"][:, :, None], float("nan"))
    modified, modified_grad = run(changed)
    for name in baseline:
        torch.testing.assert_close(baseline[name], modified[name], atol=0, rtol=0)
    for name in baseline_grad:
        torch.testing.assert_close(baseline_grad[name], modified_grad[name], atol=0, rtol=0)


def test_frozen_planner_can_supply_predictions_independently_of_training():
    objective, batch = _masked_objective(), _masked_batch()
    loss, metrics = objective(batch, 0.0)
    loss.backward()
    assert metrics["predicted_contact_rows"].item() == 2
    assert objective.model.plan_calls == 1
    assert all(p.grad is None for p in objective.model.contact_planner.parameters())
    assert objective.model.contact_encoder.weight.grad is not None


def test_ema_after_broadcast_and_rank_mismatch_detection(monkeypatch):
    from scripts.train import EMA
    torch.manual_seed(42)
    rank0 = nn.Linear(3, 2)
    torch.manual_seed(43)
    rank1 = nn.Linear(3, 2)
    assert not torch.equal(rank0.weight, rank1.weight)
    rank1.load_state_dict(rank0.state_dict())  # DDP's initial broadcast.
    ema0, ema1 = EMA(rank0, .9998), EMA(rank1, .9998)
    ema0.update(rank0)
    ema1.update(rank1)
    for name, value in ema0.state_dict().items():
        torch.testing.assert_close(value, ema1.state_dict()[name], atol=0, rtol=0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object",
                        lambda result, value, group=None: result.__setitem__(slice(None), [value, value]))
    ema0.assert_synchronized()
    monkeypatch.setattr(torch.distributed, "all_gather_object",
                        lambda result, value, group=None: result.__setitem__(slice(None), [value, "different"]))
    with pytest.raises(RuntimeError, match="EMA differs"):
        ema0.assert_synchronized()


def test_ema_shares_only_frozen_planner():
    from scripts.train import EMA
    source = nn.Module()
    source.model = nn.Module()
    source.model.contact_planner = nn.Linear(3, 2).requires_grad_(False)
    source.model.policy = nn.Linear(3, 2)
    ema = EMA(source, .5)
    assert ema.module.model.contact_planner is source.model.contact_planner
    assert ema.module.model.policy is not source.model.policy
    assert source.model.policy.weight.requires_grad
    assert not ema.module.model.policy.weight.requires_grad
    assert all("contact_planner" not in name for name in ema.state_dict())


def test_dual_ranking_deduplicates_and_prunes_only_unranked_weights(monkeypatch):
    from pathlib import Path
    import json
    from scripts import train
    manifests, files = {}, {}
    root = Path("in-memory")
    monkeypatch.setattr(Path, "exists", lambda p: str(p) in manifests)
    monkeypatch.setattr(Path, "read_text", lambda p: json.dumps(manifests[str(p)]))
    monkeypatch.setattr(Path, "glob", lambda p, pattern: list(files))
    monkeypatch.setattr(Path, "unlink", lambda p: files.pop(p))
    monkeypatch.setattr(train, "write_json_atomic", lambda value, path: manifests.__setitem__(str(path), value))
    monkeypatch.setattr(train, "save_atomic", lambda value, path: files.__setitem__(path, value))
    class DummyEMA:
        def state_dict(self):
            return {"policy": torch.ones(1)}
    for epoch, old, new in ((900, .8, .2), (1000, .7, .9), (1100, .1, .8), (1200, .9, .95)):
        train.save_ranked(root, DummyEMA(), epoch,
            {"old": {"mean_success_rate": old}, "new": {"mean_success_rate": new}}, {}, 2)
    rankings = manifests[str(root / "rankings.json")]
    assert [r["epoch"] for r in rankings["old"]] == [1200, 900]
    assert [r["epoch"] for r in rankings["new"]] == [1200, 1000]
    assert set(files) == {
        root / "epoch=0900-new_seen_score=0.200-old_seen_score=0.800.ckpt",
        root / "epoch=1000-new_seen_score=0.900-old_seen_score=0.700.ckpt",
        root / "epoch=1200-new_seen_score=0.950-old_seen_score=0.900.ckpt",
    }
    assert all(root / row["checkpoint"] in files for rows in rankings.values() for row in rows)
