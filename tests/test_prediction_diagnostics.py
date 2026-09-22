import torch
from torch import nn

from dexcg.common.typing import ContactPlan
from dexcg.training.objective import DexCGTrainingObjective


class _ObservationEncoder(nn.Module):
    obs_horizon = 2

    def forward(self, observation):
        return observation["agent_pos"][:, -1]


class _SMP(nn.Module):
    action_horizon = 4
    num_experts = 2

    @staticmethod
    def basis(observation, bias):
        value = observation.new_tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        return value.expand(observation.shape[0], -1, -1) + bias

    @staticmethod
    def route(observation, contact):
        gate = observation.new_ones((observation.shape[0], 4, 2))
        return {"prior_gate": gate}

    @staticmethod
    def denoise(coefficients, timestep, observation, contact):
        return torch.zeros_like(coefficients)

    @staticmethod
    def decode(basis, gate, coefficients):
        return torch.einsum("bdk,btk->btd", basis, gate * coefficients)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.observation_encoder = _ObservationEncoder()
        self.smp = _SMP()
        self.plan_calls = 0

    def plan_contact(self, observation, languages, previous_plan=None):
        self.plan_calls += 1
        self.previous_plan = previous_plan
        batch_size = observation["agent_pos"].shape[0]
        return ContactPlan(
            token_ids=torch.ones(batch_size, 2, dtype=torch.long),
            attention_mask=torch.ones(batch_size, 2, dtype=torch.bool),
        )

    @staticmethod
    def encode_contact(contact_plan):
        return torch.zeros(contact_plan.token_ids.shape[0], 2)

    @staticmethod
    def physgraph(observation, token_ids):
        return torch.zeros(observation["agent_pos"].shape[0], 3, 2)


def _objective() -> DexCGTrainingObjective:
    return DexCGTrainingObjective(
        _Model(),
        state_min=torch.zeros(3),
        state_max=torch.ones(3),
        diffusion_config={
            "num_train_timesteps": 4,
            "beta_start": 1.0e-4,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "prediction_type": "epsilon",
        },
        loss_config={},
    ).eval()


def test_prediction_diagnostics_returns_outputs_from_one_plan() -> None:
    objective = _objective()
    observation = {"agent_pos": torch.rand(1, 2, 3)}

    prediction = objective.predict_action_with_diagnostics(
        observation, ["test instruction"], num_inference_steps=2, action_steps=2
    )

    assert objective.model.plan_calls == 1
    assert prediction.actions.shape == (1, 2, 3)
    assert prediction.basis.shape == (1, 3, 2)
    assert prediction.contact_plan.token_ids.shape == (1, 2)


def test_predict_action_preserves_tensor_return_type() -> None:
    objective = _objective()
    observation = {"agent_pos": torch.rand(1, 2, 3)}

    actions = objective.predict_action(
        observation, ["test instruction"], num_inference_steps=2, action_steps=2
    )

    assert isinstance(actions, torch.Tensor)
    assert actions.shape == (1, 2, 3)
    assert objective.model.plan_calls == 1


def test_prediction_passes_previous_contact_to_planner() -> None:
    objective = _objective()
    observation = {"agent_pos": torch.rand(1, 2, 3)}
    previous = ContactPlan(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.bool))

    objective.predict_action_with_diagnostics(
        observation,
        ["test instruction"],
        num_inference_steps=2,
        action_steps=2,
        previous_contact_plan=previous,
    )

    assert objective.model.previous_plan is previous


def test_prediction_executes_current_action_without_skipping(monkeypatch):
    objective = _objective()
    sequence = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    monkeypatch.setattr(objective.model.smp, "decode", lambda *args: sequence)
    result = objective.predict_action(
        {"agent_pos": torch.rand(1, 2, 3)}, ["test"], 2, 2
    )
    torch.testing.assert_close(result, sequence[:, :2])
