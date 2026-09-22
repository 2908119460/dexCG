import numpy as np
import pytest
import torch

from dexcg.evaluation.planner import ContactMetricAccumulator, contact_metrics_for_pair


class _Tokenizer:
    joint_start_id = 1
    joint_end_id = 2
    id_to_link = {10: "<link_a>", 11: "<link_b>", 12: "<link_c>"}
    position_token_ids = np.asarray([20, 21, 22, 23])
    position_id_to_bin = {20: 0, 21: 1, 22: 2, 23: 3}
    position_centers = np.asarray([-0.3, -0.1, 0.1])


def test_contact_pair_metrics_cover_graph_position_count_and_saturation() -> None:
    tokenizer = _Tokenizer()
    predicted = [1, 10, 20, 21, 22, 11, 21, 21, 21, 2]
    target = [1, 10, 21, 21, 22, 12, 22, 22, 22, 2]

    result = contact_metrics_for_pair(predicted, target, tokenizer).result()

    assert result["samples"] == 1
    assert result["link_precision"] == 0.5
    assert result["link_recall"] == 0.5
    assert result["link_f1"] == 0.5
    assert result["mean_absolute_contact_count_error"] == 0.0
    assert result["exact_graph_accuracy"] == 0.0
    assert result["matched_link_xyz_distance"] == pytest.approx(0.2)
    assert result["boundary_bin_saturation"] == pytest.approx(1 / 6)


def test_metric_accumulator_reports_breakdowns_and_invalid_predictions() -> None:
    tokenizer = _Tokenizer()
    accumulator = ContactMetricAccumulator(tokenizer)
    predicted = torch.tensor([[1, 10, 21, 21, 22, 2], [1, 10, 21, 2, 2, 2]])
    predicted_mask = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, False, False]]
    )
    target = torch.tensor([[1, 10, 21, 21, 22, 2], [1, 12, 22, 22, 22, 2]])
    target_mask = torch.ones_like(target, dtype=torch.bool)

    accumulator.update(
        predicted,
        predicted_mask,
        target,
        target_mask,
        {"task": ["faucet", "bucket"], "history": ["present", "absent"]},
    )
    result = accumulator.result()

    assert result["overall"]["samples"] == 2
    assert result["overall"]["invalid_predictions"] == 1
    assert result["breakdown"]["task"]["faucet"]["exact_graph_accuracy"] == 1.0
    assert result["breakdown"]["task"]["bucket"]["invalid_predictions"] == 1
