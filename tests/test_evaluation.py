from types import SimpleNamespace

import numpy as np
import torch

from dexcg.evaluation import dexart as evaluation


class _Adapter:
    environment = SimpleNamespace(seed=lambda seed: None)
    is_success = False

    def reset(self):
        return {
            "instance_1-rgb": np.zeros((2, 2, 3), dtype=np.float32),
            "instance_1-depth": np.zeros((2, 2), dtype=np.float32),
            "instance_1-point_cloud": np.zeros((4, 3), dtype=np.float32),
            "instance_1-seg_gt": np.zeros((4, 4), dtype=np.float32),
            "imagination_robot": np.zeros((2, 3), dtype=np.float32),
            "state": np.zeros(33, dtype=np.float32),
        }

    def step(self, action):
        raise AssertionError("an invalid initial observation must not reach the policy")

    def observe(self):
        return self.reset()

    def close(self):
        pass


class _Writer:
    def __init__(self, *args, **kwargs):
        pass

    def append(self, frame):
        pass

    def close(self):
        pass


def test_invalid_object_observation_counts_as_failed_episode(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(evaluation.DexArtAdapter, "create", lambda *args, **kwargs: _Adapter())
    monkeypatch.setattr(evaluation, "_VideoWriter", _Writer)
    objective = SimpleNamespace(
        predict_action=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the policy must not receive an invalid object observation")
        )
    )

    result = evaluation._evaluate_task(
        objective=objective,
        task="faucet",
        task_config={"instruction": "test", "max_steps": 1},
        evaluation_config={
            "episodes_per_seed": 1,
            "seeds": [1000],
            "split": "seen",
            "num_inference_steps": 1,
            "action_steps": 1,
        },
        epoch=1,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert result["successes"] == 0
    assert result["invalid_observation_episodes"] == 1
    assert result["invalid_observation_frames"] == 9


class _TransientInvalidAdapter:
    environment = SimpleNamespace(seed=lambda seed: None)

    def __init__(self):
        self.steps = 0

    @property
    def is_success(self):
        return self.steps == 2

    @staticmethod
    def _raw(has_object):
        segmentation = np.zeros((4, 4), dtype=np.float32)
        if has_object:
            segmentation[0, 0] = 1.0
        return {
            "instance_1-rgb": np.zeros((2, 2, 3), dtype=np.float32),
            "instance_1-depth": np.zeros((2, 2), dtype=np.float32),
            "instance_1-point_cloud": np.zeros((4, 3), dtype=np.float32),
            "instance_1-seg_gt": segmentation,
            "imagination_robot": np.zeros((2, 3), dtype=np.float32),
            "state": np.zeros(33, dtype=np.float32),
        }

    def reset(self):
        return self._raw(True)

    def observe(self):
        return self._raw(True)

    def step(self, action):
        self.steps += 1
        return self._raw(self.steps == 2), 0.0, self.steps == 2, {}

    def close(self):
        pass


def test_transient_invalid_frame_is_skipped(monkeypatch, tmp_path) -> None:
    adapter = _TransientInvalidAdapter()
    monkeypatch.setattr(evaluation.DexArtAdapter, "create", lambda *args, **kwargs: adapter)
    monkeypatch.setattr(evaluation, "_VideoWriter", _Writer)
    observations = []

    def predict_action(observation, *args, **kwargs):
        observations.append(observation)
        assert observation["object_point_mask"].any(dim=-1).all()
        return torch.zeros((1, 1, 2))

    result = evaluation._evaluate_task(
        objective=SimpleNamespace(predict_action=predict_action),
        task="faucet",
        task_config={"instruction": "test", "max_steps": 2},
        evaluation_config={
            "episodes_per_seed": 1,
            "seeds": [1000],
            "split": "seen",
            "num_inference_steps": 1,
            "action_steps": 1,
        },
        epoch=1,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert len(observations) == 2
    assert result["successes"] == 1
    assert result["invalid_observation_episodes"] == 1
    assert result["invalid_observation_frames"] == 1
