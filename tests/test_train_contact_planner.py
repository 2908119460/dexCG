import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_contact_planner.py"
SPEC = importlib.util.spec_from_file_location("train_contact_planner", SCRIPT)
train_contact_planner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_contact_planner)


def _config():
    return {
        "epochs": 10,
        "batch_size_per_gpu": 4,
        "num_workers": 2,
        "prefetch_factor": 2,
        "pin_memory": True,
        "mixed_precision": "bf16",
        "optimizer": {
            "scheduler": "cosine",
            "learning_rate": 1.0e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "epsilon": 1.0e-8,
        },
        "validation": {"enabled": True, "interval_epochs": 1},
        "checkpoint": {"keep_validation_best": 2},
    }


@pytest.mark.parametrize("name", train_contact_planner.REQUIRED_BENCHMARK_SETTINGS)
def test_preflight_rejects_unbenchmarked_setting(name: str) -> None:
    config = _config()
    config[name] = None
    with pytest.raises(ValueError, match=name):
        train_contact_planner.validate_training_config(config)


def test_preflight_accepts_complete_benchmark_settings() -> None:
    train_contact_planner.validate_training_config(_config())


def test_preflight_rejects_zero_validation_interval() -> None:
    config = _config()
    config["validation"]["interval_epochs"] = 0
    with pytest.raises(ValueError, match="validation.interval_epochs"):
        train_contact_planner.validate_training_config(config)


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.norm = nn.LayerNorm(4)
        self.feed_forward = nn.Linear(4, 4, bias=False)
        self.projection = nn.Linear(4, 8, bias=False)
        self.projection.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding


class _Planner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = nn.Linear(3, 4)
        self.point_projector = nn.Linear(4, 4)
        self.language_model = _LanguageModel()


class _Objective(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.planner = _Planner()


def test_grouped_optimizer_has_exact_coverage_and_expected_hyperparameters() -> None:
    objective = _Objective()
    config = {
        "learning_rates": {
            "qwen": 1.0e-5,
            "partfield": 5.0e-6,
            "point_projector": 1.0e-4,
        },
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "epsilon": 1.0e-8,
    }
    optimizer = train_contact_planner.optimizer_for(objective, config)

    assigned = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in objective.parameters() if parameter.requires_grad]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == set(expected)
    settings = {
        group["name"]: (group["lr"], group["weight_decay"]) for group in optimizer.param_groups
    }
    assert settings == {
        "partfield.decay": (5.0e-6, 0.01),
        "partfield.no_decay": (5.0e-6, 0.0),
        "point_projector.decay": (1.0e-4, 0.01),
        "point_projector.no_decay": (1.0e-4, 0.0),
        "qwen.decay": (1.0e-5, 0.01),
        "qwen.no_decay": (1.0e-5, 0.0),
    }
    embedding_id = id(objective.planner.language_model.embedding.weight)
    embedding_groups = [
        group for group in optimizer.param_groups if embedding_id in map(id, group["params"])
    ]
    assert len(embedding_groups) == 1
    assert embedding_groups[0]["name"] == "qwen.no_decay"


def test_early_stopping_state_tracks_improvements_and_resume_state() -> None:
    state = (float("inf"), 0, 0)
    state = train_contact_planner.update_early_stopping(2.0, 1, *state, min_delta=0.01)
    assert state == (2.0, 1, 0)
    state = train_contact_planner.update_early_stopping(1.995, 2, *state, min_delta=0.01)
    assert state == (2.0, 1, 1)
    state = train_contact_planner.update_early_stopping(1.8, 3, *state, min_delta=0.01)
    assert state == (1.8, 3, 0)


def test_latest_checkpoint_preserves_early_stopping_state(tmp_path) -> None:
    objective = _Objective()
    optimizer = torch.optim.AdamW(objective.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "latest.ckpt"
    training_state = {
        "best_validation_loss": 1.25,
        "best_validation_epoch": 2,
        "bad_validation_epochs": 1,
    }

    train_contact_planner.save_latest(
        path,
        objective,
        optimizer,
        scheduler,
        epoch=3,
        global_step=12,
        training_state=training_state,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 12
    assert checkpoint["training_state"] == training_state


def test_validation_checkpoints_retain_only_lowest_losses(tmp_path) -> None:
    objective = _Objective()
    for epoch, loss in enumerate((3.0, 2.0, 4.0, 1.0), start=1):
        train_contact_planner.save_validation_checkpoint(
            tmp_path, objective, epoch=epoch, loss=loss, keep=2
        )

    names = sorted(path.name for path in tmp_path.glob("*.ckpt"))
    assert names == [
        "epoch=0002-deployment_loss=2.000000.ckpt",
        "epoch=0004-deployment_loss=1.000000.ckpt",
    ]


def test_frozen_full_checkpoint_loads_without_training_flags():
    source = _Objective()
    source.freeze_partfield = True
    source.planner.point_encoder.requires_grad_(False)
    source.planner.point_encoder.register_buffer("probe", torch.tensor([3.0]))
    state = train_contact_planner.compact_planner_state(source)
    assert "point_encoder.weight" in state and "point_encoder.probe" in state
    target = _Objective()
    target.planner.point_encoder.register_buffer("probe", torch.zeros(1))
    target.planner.language_model.tie_weights = lambda: None
    train_contact_planner.load_compact_planner_state(target, state)
    for name, value in target.planner.state_dict().items():
        assert torch.equal(value, state[name])
    optimizer = train_contact_planner.optimizer_for(source, {
        "learning_rates": {"qwen": 1e-5, "point_projector": 1e-4},
        "weight_decay": 0.01, "betas": [0.9, 0.95], "epsilon": 1e-8})
    frozen = {id(p) for p in source.planner.point_encoder.parameters()}
    assert not any(id(p) in frozen for g in optimizer.param_groups for p in g["params"])


def test_monitor_requires_continuous_free_memory_regardless_of_utilization():
    valid = [(25601, 100)] * 5
    histories = {i: valid.copy() for i in range(4)}
    assert len(train_contact_planner.eligible_gpus(histories)) == 4
    histories[0][-1] = (25600, 0)
    assert len(train_contact_planner.eligible_gpus(histories)) == 3
    histories[0][-1] = (40000, 100)
    assert len(train_contact_planner.eligible_gpus(histories)) == 4
    histories[0] = valid[:4]
    assert len(train_contact_planner.eligible_gpus(histories)) == 3


def test_monitor_selects_four_cards_by_free_memory_only():
    histories = {i: [(30000 + i * 1000, 100)] * 5 for i in range(6)}
    assert train_contact_planner.eligible_gpus(histories) == [5, 4, 3, 2]
    histories[5][1] = (25000, 0)
    assert train_contact_planner.eligible_gpus(histories) == [4, 3, 2, 1]
