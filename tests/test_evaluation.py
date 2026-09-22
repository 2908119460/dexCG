from types import SimpleNamespace
from io import StringIO
from pathlib import Path

import numpy as np
import torch

from dexcg.evaluation import dexart as evaluation


def test_single_rank_evaluation_does_not_require_distributed_initialization(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("single-rank CPU evaluation must not call CUDA or collectives")
    monkeypatch.setattr(torch.distributed, "all_gather_object", unexpected)
    monkeypatch.setattr(torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(evaluation, "_evaluate_task", lambda *args: {"success_rate": .75})
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: StringIO())
    result = evaluation.evaluate_seen_tasks(
        None, {"evaluation": {"split": "seen", "tasks": {"bucket": {}}},
               "diffusion": {"num_inference_steps": 10}},
        1, Path("in-memory"), torch.device("cpu"), 0, 1,
    )
    assert result["mean_success_rate"] == .75


def test_preflight_does_not_write_official_success_rates(monkeypatch):
    def forbid_files(*args, **kwargs):
        raise AssertionError("Preflight must not create official evaluation files")
    monkeypatch.setattr(evaluation, "_evaluate_task", lambda *args: {"success_rate": 0.})
    monkeypatch.setattr(Path, "mkdir", forbid_files)
    monkeypatch.setattr(Path, "open", forbid_files)
    result = evaluation.evaluate_seen_tasks(None,
        {"evaluation": {"split": "seen", "tasks": {"bucket": {}}, "write_results": False},
         "diffusion": {"num_inference_steps": 10}},
        0, Path("not-created"), torch.device("cpu"), 0, 1)
    assert result["mean_success_rate"] == 0.


def _sample():
    points = np.random.default_rng(7).uniform(-.4, .4, (10000, 3)).astype(np.float32)
    return {"point_cloud": points, "object_point_mask": np.ones(10000, dtype=bool),
            "object_center": (points.min(0) + points.max(0)) / 2,
            "agent_pos": np.zeros(33, dtype=np.float32),
            "palm_pose_robot_base": np.eye(4, dtype=np.float32)}


class _Adapter:
    def __init__(self, invalid_initial=False, invalid_after_step=False):
        self.invalid_initial = invalid_initial
        self.invalid_after_step = invalid_after_step
        self.steps = 0
        self.seeds = []
        self.environment = SimpleNamespace(seed=self.seeds.append)
        self.closed = False
        self.resolution = None
        self.sample = _sample()

    def enable_high_resolution_object_cloud(self, resolution):
        assert self.point_count == 10000
        self.resolution = resolution

    @property
    def is_success(self):
        return self.steps == 2

    def reset(self):
        self.steps = 0
        return {}

    def observe(self):
        return {}

    def observation(self, raw):
        if self.invalid_initial or (self.invalid_after_step and self.steps == 1):
            raise evaluation.InvalidObjectObservation("Insufficient distinct object pixels: 9999 < 10000")
        return self.sample

    def step(self, action):
        self.steps += 1
        return {}, 0., self.is_success, {}

    def close(self):
        self.closed = True


class _Objective:
    def __init__(self):
        from dexcg.models.observation.dp3_encoder import DP3ObservationEncoder
        self.model = SimpleNamespace(
            contact_planner=SimpleNamespace(point_encoder=torch.nn.Identity()),
            observation_encoder=DP3ObservationEncoder(point_mlp_dims=(8,), point_feature_dim=4,
                                                      state_mlp_dims=(4,), feature_dim=8),
        )
        self.observations = []

    def predict_action_with_diagnostics(self, observation, *args, **kwargs):
        from dexcg.models.dexcg import DexCG
        self.observations.append(observation)
        self.model.contact_planner.point_encoder(DexCG.contact_planner_input(observation))
        self.model.observation_encoder(observation)
        return SimpleNamespace(actions=torch.zeros(1, 1, 22), contact_plan=None)


def _run(monkeypatch, adapter, planner_count=10000, episodes=1):
    monkeypatch.setattr(evaluation.RobotBaseCollectionAdapter, "create", lambda *a, **k: adapter)
    monkeypatch.setattr(evaluation, "robot_geometry", lambda q: np.zeros((96, 7), dtype=np.float32))
    def forbid_files(*args, **kwargs):
        raise AssertionError("This test must not create any files")
    monkeypatch.setattr(evaluation, "_VideoWriter", forbid_files)
    objective = _Objective()
    result = evaluation._evaluate_task(
        objective, "faucet", {"instruction": "test", "max_steps": 4},
        {"episodes_per_seed": episodes, "seeds": [1000], "split": "seen",
         "num_inference_steps": 1, "action_steps": 1,
         "planner_point_count": planner_count, "record_video": False},
        1, Path("not-created"), torch.device("cpu"),
    )
    assert adapter.closed
    assert adapter.resolution == [1024, 1024]
    assert not objective.model.contact_planner.point_encoder._forward_pre_hooks
    assert not objective.model.observation_encoder.encoder._forward_pre_hooks
    return result, objective


def test_invalid_object_observation_counts_as_failed_episode(monkeypatch):
    result, objective = _run(monkeypatch, _Adapter(invalid_initial=True))
    assert result["successes"] == 0
    assert result["episodes"] == 1
    assert result["invalid_observation_episodes"] == 1
    assert result["invalid_observation_frames"] == 9
    assert result["episode_results"][0]["failure_reason"] == "invalid_initial_observation"
    assert result["point_input_contract"]["partfield_actual_point_count"] is None
    assert not objective.observations


def test_invalid_rollout_frame_ends_episode_without_stale_history(monkeypatch):
    adapter = _Adapter(invalid_after_step=True)
    result, objective = _run(monkeypatch, adapter)
    assert len(objective.observations) == 1
    assert adapter.steps == 1
    assert result["successes"] == 0
    assert result["episodes"] == 1
    assert result["invalid_observation_frames"] == 1
    assert result["episode_results"][0]["failure_reason"] == "invalid_rollout_observation"


import pytest


@pytest.mark.parametrize("planner_count", [1024, 10000])
def test_closed_loop_separate_inputs_reach_networks(monkeypatch, planner_count):
    adapter = _Adapter()
    result, objective = _run(monkeypatch, adapter, planner_count, episodes=2)
    assert result["successes"] == 2
    counts = result["point_input_contract"]
    assert counts["partfield_actual_point_count"] == planner_count
    assert counts["dp3_actual_object_point_count"] == 1024
    assert counts["dp3_actual_robot_point_count"] == 96
    assert counts["partfield_calls"] == counts["dp3_calls"] == 4
    assert len(set(adapter.seeds)) == 2
    for obs in objective.observations:
        assert obs["planner_point_cloud"].shape == (1, 2, planner_count, 3)
        assert obs["point_cloud"].shape == (1, 2, 1024, 3)
        full_rows = {tuple(row) for row in adapter.sample["point_cloud"]}
        subset = obs["point_cloud"][0, -1].numpy()
        assert len(np.unique(subset, axis=0)) == 1024
        assert all(tuple(row) in full_rows for row in subset)
        if planner_count == 10000:
            np.testing.assert_array_equal(obs["planner_point_cloud"][0, -1], adapter.sample["point_cloud"])
        else:
            torch.testing.assert_close(obs["point_cloud"], obs["planner_point_cloud"])


def test_both_variants_use_identical_subsets_and_episode_seeds(monkeypatch):
    a, b = _Adapter(), _Adapter()
    _, old = _run(monkeypatch, a, 1024, episodes=2)
    _, new = _run(monkeypatch, b, 10000, episodes=2)
    assert a.seeds == b.seeds
    for x, y in zip(old.observations, new.observations, strict=True):
        torch.testing.assert_close(x["point_cloud"], y["point_cloud"])


@pytest.mark.parametrize("corruption", ["short", "duplicate", "nonfinite", "environment"])
def test_split_rejects_invalid_capture(monkeypatch, corruption):
    sample = _sample()
    if corruption == "short":
        sample["point_cloud"] = sample["point_cloud"][:-1]
    elif corruption == "duplicate":
        sample["point_cloud"][1] = sample["point_cloud"][0]
    elif corruption == "nonfinite":
        sample["point_cloud"][0, 0] = np.nan
    else:
        sample["object_point_mask"][0] = False
    with pytest.raises(evaluation.InvalidObjectObservation):
        evaluation._split_object_observation(sample, 10000, np.random.default_rng(42))


def test_network_guards_reject_wrong_counts_and_are_removed_on_error():
    objective = _Objective()
    counts = {"partfield_calls": 0, "dp3_calls": 0}
    with pytest.raises(ValueError, match="PartField"):
        with evaluation._point_input_guards(objective, 10000, counts):
            objective.model.contact_planner.point_encoder(torch.zeros(1, 1024, 3))
    assert not objective.model.contact_planner.point_encoder._forward_pre_hooks
    with pytest.raises(ValueError, match="DP3"):
        with evaluation._point_input_guards(objective, 10000, counts):
            objective.model.observation_encoder.encoder({
                "point_cloud": torch.zeros(2, 10000, 3),
                "imagin_robot": torch.zeros(2, 96, 7), "agent_pos": torch.zeros(2, 33)})
    assert not objective.model.observation_encoder.encoder._forward_pre_hooks
