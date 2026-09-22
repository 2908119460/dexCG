"""Metrics for free-running contact-planner predictions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import torch

from dexcg.models.contact.tokenizer import AllegroContactTokenizer
from dexcg.models.contact.coordinates import robot_base_point_cloud


@dataclass
class ContactMetricTotals:
    samples: int = 0
    invalid_predictions: int = 0
    link_true_positive: int = 0
    link_false_positive: int = 0
    link_false_negative: int = 0
    absolute_contact_count_error: int = 0
    exact_graphs: int = 0
    matched_link_distance_sum: float = 0.0
    matched_links: int = 0
    boundary_position_tokens: int = 0
    predicted_position_tokens: int = 0

    def merge(self, other: "ContactMetricTotals") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def result(self) -> dict[str, float | int]:
        precision_denominator = self.link_true_positive + self.link_false_positive
        recall_denominator = self.link_true_positive + self.link_false_negative
        precision = self.link_true_positive / max(precision_denominator, 1)
        recall = self.link_true_positive / max(recall_denominator, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
        return {
            "samples": self.samples,
            "invalid_predictions": self.invalid_predictions,
            "link_precision": precision,
            "link_recall": recall,
            "link_f1": f1,
            "mean_absolute_contact_count_error": self.absolute_contact_count_error
            / max(self.samples, 1),
            "exact_graph_accuracy": self.exact_graphs / max(self.samples, 1),
            "matched_link_xyz_distance": self.matched_link_distance_sum
            / max(self.matched_links, 1),
            "matched_links": self.matched_links,
            "boundary_bin_saturation": self.boundary_position_tokens
            / max(self.predicted_position_tokens, 1),
            "predicted_position_tokens": self.predicted_position_tokens,
        }


def _parse_contacts(
    token_ids: Sequence[int],
    tokenizer: AllegroContactTokenizer,
) -> tuple[dict[int, list[np.ndarray]], int, int, bool]:
    values = list(map(int, token_ids))
    try:
        start = values.index(tokenizer.joint_start_id) + 1
        end = values.index(tokenizer.joint_end_id, start)
    except ValueError:
        return {}, 0, 0, False
    payload = values[start:end]
    if len(payload) % 4:
        return {}, 0, 0, False

    contacts: dict[int, list[np.ndarray]] = defaultdict(list)
    boundary_tokens = 0
    position_tokens = 0
    final_bin = len(tokenizer.position_token_ids) - 1
    for offset in range(0, len(payload), 4):
        link_id = payload[offset]
        if link_id not in tokenizer.id_to_link:
            return {}, 0, 0, False
        try:
            bins = [
                tokenizer.position_id_to_bin[payload[offset + coordinate]]
                for coordinate in (1, 2, 3)
            ]
        except KeyError:
            return {}, 0, 0, False
        boundary_tokens += sum(bin_index in (0, final_bin) for bin_index in bins)
        position_tokens += 3
        decoded_bins = np.clip(bins, 0, len(tokenizer.position_centers) - 1)
        contacts[link_id].append(tokenizer.position_centers[decoded_bins])
    return dict(contacts), boundary_tokens, position_tokens, True


def contact_metrics_for_pair(
    predicted_ids: Sequence[int],
    target_ids: Sequence[int],
    tokenizer: AllegroContactTokenizer,
) -> ContactMetricTotals:
    predicted, boundary_tokens, position_tokens, valid = _parse_contacts(predicted_ids, tokenizer)
    target, _, _, target_valid = _parse_contacts(target_ids, tokenizer)
    if not target_valid:
        raise ValueError("contact target does not satisfy the contact-token grammar")

    predicted_links = set(predicted)
    target_links = set(target)
    matched_links = predicted_links & target_links
    distance_sum = 0.0
    for link_id in matched_links:
        distances = [
            float(np.linalg.norm(predicted_position - target_position))
            for predicted_position in predicted[link_id]
            for target_position in target[link_id]
        ]
        distance_sum += min(distances)
    predicted_count = sum(map(len, predicted.values()))
    target_count = sum(map(len, target.values()))
    return ContactMetricTotals(
        samples=1,
        invalid_predictions=int(not valid),
        link_true_positive=len(matched_links),
        link_false_positive=len(predicted_links - target_links),
        link_false_negative=len(target_links - predicted_links),
        absolute_contact_count_error=abs(predicted_count - target_count),
        exact_graphs=int(valid and predicted_links == target_links),
        matched_link_distance_sum=distance_sum,
        matched_links=len(matched_links),
        boundary_position_tokens=boundary_tokens,
        predicted_position_tokens=position_tokens,
    )


class ContactMetricAccumulator:
    """Accumulate global metrics plus arbitrary categorical breakdowns."""

    def __init__(self, tokenizer: AllegroContactTokenizer) -> None:
        self.tokenizer = tokenizer
        self.overall = ContactMetricTotals()
        self.groups: dict[str, dict[str, ContactMetricTotals]] = defaultdict(dict)

    def update(
        self,
        predicted_ids: torch.Tensor,
        predicted_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        categories: Mapping[str, Sequence[Any]],
    ) -> None:
        rows = int(predicted_ids.shape[0])
        if any(len(values) != rows for values in categories.values()):
            raise ValueError("metric categories must have one value per prediction")
        for row in range(rows):
            predicted = predicted_ids[row][predicted_mask[row].bool()].detach().cpu().tolist()
            target = target_ids[row][target_mask[row].bool()].detach().cpu().tolist()
            totals = contact_metrics_for_pair(predicted, target, self.tokenizer)
            self.overall.merge(totals)
            for category, values in categories.items():
                value = str(values[row])
                group = self.groups[category].setdefault(value, ContactMetricTotals())
                group.merge(totals)

    def result(self) -> dict[str, Any]:
        return {
            "overall": self.overall.result(),
            "breakdown": {
                category: {value: totals.result() for value, totals in sorted(groups.items())}
                for category, groups in sorted(self.groups.items())
            },
        }


@torch.inference_mode()
def evaluate_contact_condition(planner, dataset, indices, language_family, device, batch_size=3):
    """Compare fixed samples using unweighted frame CE and free generation."""
    planner.eval()
    accumulator = ContactMetricAccumulator(planner.contact_tokenizer)
    totals = {"loss_sum": 0.0, "samples": 0, "correct": 0, "tokens": 0}
    task_results = {}
    started = time.perf_counter()
    for task_index, task in enumerate(dataset.tasks):
        local = dict.fromkeys(totals, 0)
        selected = [i for i in indices if dataset.samples[i].episode.task_index == task_index]
        for start in range(0, len(selected), batch_size):
            cpu_batch = torch.utils.data.default_collate(
                [dataset[i] for i in selected[start:start + batch_size]])
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in cpu_batch.items()}
            cloud = robot_base_point_cloud(batch["point_cloud"], batch["object_point_mask"])
            languages = dataset.languages_for_family(cpu_batch, language_family)
            robot = ({key: batch[key] for key in ("robot_qpos", "palm_pose_robot_base")}
                     if planner.robot_state_projector is not None else {})
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = planner.training_loss(
                    cloud, languages, batch["target_ids"], batch["target_mask"],
                    sample_weight=None, **robot)
                plan = planner.plan(cloud, languages, **robot)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss")
            local["loss_sum"] += loss.item() * len(languages)
            local["samples"] += len(languages)
            local["correct"] += metrics["correct"].item()
            local["tokens"] += metrics["count"].item()
            accumulator.update(plan.token_ids, plan.attention_mask,
                               batch["target_ids"], batch["target_mask"], {"task": batch["task"]})
        if local["samples"]:
            task_results[task] = {"loss": local["loss_sum"] / local["samples"],
                                  "token_accuracy": local["correct"] / local["tokens"], **local}
        for key in totals:
            totals[key] += local[key]
    if not totals["samples"]:
        raise ValueError("Empty diagnostic sample selection")
    return {"teacher_forcing": {"loss": totals["loss_sum"] / totals["samples"],
            "token_accuracy": totals["correct"] / totals["tokens"], **totals},
            "teacher_forcing_by_task": task_results, "generation": accumulator.result(),
            "seconds": time.perf_counter() - started}


def composite_contact_score(results, position_scale_m=0.10):
    """Equal mean over language families and the three user-approved components."""
    if position_scale_m <= 0 or set(results) != {"low_level", "high_level", "deployment"}:
        raise ValueError("Composite score requires all three language families and a positive scale")
    scores = {}
    for family, result in results.items():
        generation = result["generation"]["overall"]
        accuracy = float(result["teacher_forcing"]["token_accuracy"])
        f1 = float(generation["link_f1"])
        error = float(generation["matched_link_xyz_distance"])
        if not (np.isfinite([accuracy, f1, error]).all()
                and 0 <= accuracy <= 1 and 0 <= f1 <= 1 and error >= 0):
            raise ValueError("Invalid metric values for composite score")
        # No matched links means location is unmeasurable, not a perfect zero error.
        position_score = (1 / (1 + error / position_scale_m)
                          if generation["matched_links"] > 0 else 0.0)
        scores[family] = (accuracy + f1 + position_score) / 3
    return {"score": float(np.mean(list(scores.values()))), "by_language": scores,
            "position_scale_m": position_scale_m}
