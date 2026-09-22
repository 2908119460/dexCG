import torch
from torch import nn

from dexcg.training.planner_objective import ContactPlannerTrainingObjective


class _Planner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Linear(3, 4)
        self.point_projector = nn.Linear(4, 4)
        self.language_model = nn.Embedding(32, 4)

    def training_loss(
        self,
        point_cloud,
        languages,
        target_ids,
        target_mask,
        previous_ids,
        previous_mask,
        sample_weight,
    ):
        del languages, target_mask, previous_mask
        points = self.point_projector(self.point_encoder(point_cloud)).mean((1, 2))
        tokens = self.language_model(target_ids).mean((1, 2))
        history = self.language_model(previous_ids).mean((1, 2))
        per_sample = (points + tokens + history).square()
        loss = (per_sample * sample_weight).mean()
        count = torch.tensor(target_ids.numel())
        return loss, {"correct": count.new_zeros(()), "count": count}


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.contact_planner = _Planner()
        self.observation_encoder = nn.Linear(2, 2)
        self.contact_encoder = nn.Linear(2, 2)
        self.smp = nn.Linear(2, 2)
        self.physgraph = nn.Linear(2, 2)


def test_objective_freezes_downstream_and_backpropagates_through_planner() -> None:
    model = _Model()
    objective = ContactPlannerTrainingObjective(model)
    objective.train()
    batch = {
        "point_cloud": torch.randn(2, 5, 3),
        "object_point_mask": torch.ones(2, 5, dtype=torch.bool),
        "object_center": torch.zeros(2, 3),
        "target_ids": torch.tensor([[1, 2], [3, 4]]),
        "target_mask": torch.ones(2, 2, dtype=torch.bool),
        "previous_ids": torch.tensor([[5, 6], [7, 8]]),
        "previous_mask": torch.ones(2, 2, dtype=torch.bool),
        "sample_weight": torch.ones(2),
    }

    loss, _ = objective(batch, ["a", "b"])
    loss.backward()

    for module in (
        model.observation_encoder,
        model.contact_encoder,
        model.smp,
        model.physgraph,
    ):
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.grad is None for parameter in module.parameters())
    for module in (
        model.contact_planner.point_encoder,
        model.contact_planner.point_projector,
        model.contact_planner.language_model,
    ):
        assert all(parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.grad is not None for parameter in module.parameters())


def test_frozen_partfield_stays_identical_while_projector_learns():
    model = _Model()
    objective = ContactPlannerTrainingObjective(model, freeze_partfield=True)
    model.contact_planner.point_encoder.register_buffer("probe", torch.ones(1))
    original = {k: v.clone() for k, v in model.contact_planner.point_encoder.state_dict().items()}
    projection = model.contact_planner.point_projector.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        objective.eval()
        objective.train()
        assert not model.contact_planner.point_encoder.training
        assert model.contact_planner.point_projector.training
        optimizer.zero_grad(set_to_none=True)
        loss = model.contact_planner.point_projector(
            model.contact_planner.point_encoder(torch.randn(2, 10, 3))).square().mean()
        loss.backward()
        optimizer.step()
    assert all(p.grad is None for p in model.contact_planner.point_encoder.parameters())
    for k, v in model.contact_planner.point_encoder.state_dict().items():
        assert torch.equal(v, original[k])
    assert not torch.equal(projection, model.contact_planner.point_projector.weight)
