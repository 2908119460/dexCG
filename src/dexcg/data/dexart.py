"""DexArt trajectory representation and Zarr storage contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dexcg.models.contact.coordinates import (
    CONTACT_COORDINATE_CONTRACT,
    OBJECT_CENTER_DEFINITION,
    object_aabb_center_numpy,
)
from dexcg.models.contact.tokenizer import AllegroContactTokenizer
from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

TARGET_TOKEN_IDS = "contact_target_token_ids"
TARGET_TOKEN_MASK = "contact_target_token_mask"


@dataclass
class DexArtEpisode:
    observations: dict[str, list[np.ndarray]]
    actions: list[np.ndarray]
    raw_contact_points: list[np.ndarray]
    raw_contact_masks: list[np.ndarray]
    stable_contact_step: int
    stable_contact_points: np.ndarray
    stable_contact_mask: np.ndarray
    object_id: str
    task_id: str
    annotation_views: np.ndarray
    camera_extrinsics: np.ndarray
    language: Mapping[str, str]
    annotation_raw: str

    def contact_targets(self) -> tuple[np.ndarray, np.ndarray]:
        points = np.stack(self.raw_contact_points)
        masks = np.stack(self.raw_contact_masks)
        target_points = points.copy()
        target_masks = masks.copy()
        target_points[: self.stable_contact_step + 1] = self.stable_contact_points
        target_masks[: self.stable_contact_step + 1] = self.stable_contact_mask
        return target_points, target_masks


def encode_contact_graphs(
    tokenizer: AllegroContactTokenizer,
    points: np.ndarray,
    masks: np.ndarray,
    centers: np.ndarray,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.full((len(points), max_length), tokenizer.joint_end_id, dtype=np.int64)
    attention_mask = np.zeros((len(points), max_length), dtype=np.bool_)
    for row, (graph_points, graph_mask, center) in enumerate(
        zip(points, masks, centers, strict=True)
    ):
        local_points = graph_points - center
        valid_positions = local_points[graph_mask]
        if valid_positions.size and (
            np.any(valid_positions < tokenizer.min_position)
            or np.any(valid_positions > tokenizer.max_position)
        ):
            minimum = float(valid_positions.min())
            maximum = float(valid_positions.max())
            raise ValueError(
                "Object-centered contact coordinate would be clipped: "
                f"range [{minimum:.6f}, {maximum:.6f}], tokenizer range "
                f"[{tokenizer.min_position}, {tokenizer.max_position}]"
            )
        contacts = {
            link.token_name: [local_points[index]]
            for index, link in enumerate(ALLEGRO_CONTACT_LINKS)
            if graph_mask[index]
        }
        encoded = tokenizer.encode(contacts)
        if len(encoded) > max_length:
            raise ValueError(f"Contact graph requires {len(encoded)} tokens; limit is {max_length}")
        token_ids[row, : len(encoded)] = encoded
        attention_mask[row, : len(encoded)] = True
    return token_ids, attention_mask


def episode_coordinate_arrays(
    episode: DexArtEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return current centers, validity, and target-frame centers for one episode."""
    point_cloud = np.stack(episode.observations["point_cloud"])
    object_mask = np.stack(episode.observations["object_point_mask"]).astype(np.bool_)
    valid = object_mask.any(axis=-1)
    centers = np.full((len(point_cloud), 3), np.nan, dtype=np.float32)
    if valid.any():
        centers[valid] = object_aabb_center_numpy(point_cloud[valid], object_mask[valid])
    if not valid.all():
        invalid = np.flatnonzero(~valid).tolist()
        raise ValueError(f"episode has no observed object points at steps {invalid}")
    if not 0 <= episode.stable_contact_step < len(centers):
        raise ValueError(
            f"stable contact step {episode.stable_contact_step} is outside the episode"
        )
    target_centers = centers.copy()
    target_centers[: episode.stable_contact_step + 1] = centers[episode.stable_contact_step]
    return centers, valid, target_centers


def _stack_steps(episodes: Sequence[DexArtEpisode], key: str) -> np.ndarray:
    return np.concatenate([np.stack(episode.observations[key]) for episode in episodes])


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def write_dexart_dataset(
    output_path: str | Path,
    episodes: Sequence[DexArtEpisode],
    tokenizer: AllegroContactTokenizer,
    max_token_length: int,
    attributes: Mapping[str, Any],
) -> None:
    import numcodecs
    import zarr

    if not episodes:
        raise ValueError("At least one successful episode is required")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coordinate_arrays = [episode_coordinate_arrays(item) for item in episodes]
    object_centers = np.concatenate([item[0] for item in coordinate_arrays])
    object_center_valid = np.concatenate([item[1] for item in coordinate_arrays])
    target_centers = np.concatenate([item[2] for item in coordinate_arrays])
    raw_points = np.concatenate([np.stack(item.raw_contact_points) for item in episodes])
    raw_masks = np.concatenate([np.stack(item.raw_contact_masks) for item in episodes])
    target_graphs = [item.contact_targets() for item in episodes]
    target_points = np.concatenate([item[0] for item in target_graphs])
    target_masks = np.concatenate([item[1] for item in target_graphs])
    target_ids, target_token_masks = encode_contact_graphs(
        tokenizer, target_points, target_masks, target_centers, max_token_length
    )

    root = zarr.group(str(output_path))
    data = root.create_group("data")
    metadata = root.create_group("meta")
    annotation = root.create_group("annotation")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    time_chunk = min(max(len(item.actions) for item in episodes), len(raw_points))

    step_arrays = {
        "img": _stack_steps(episodes, "img"),
        "depth": _stack_steps(episodes, "depth"),
        "point_cloud": _stack_steps(episodes, "point_cloud"),
        "object_point_mask": _stack_steps(episodes, "object_point_mask"),
        "imagin_robot": _stack_steps(episodes, "imagin_robot"),
        "state": _stack_steps(episodes, "state"),
        "agent_pos": _stack_steps(episodes, "agent_pos"),
        "action": np.concatenate([np.stack(item.actions) for item in episodes]),
        "object_center": object_centers,
        "object_center_valid": object_center_valid,
        "contact_raw_points": raw_points,
        "contact_raw_mask": raw_masks,
        "contact_target_points": target_points,
        "contact_target_mask": target_masks,
        TARGET_TOKEN_IDS: target_ids,
        TARGET_TOKEN_MASK: target_token_masks,
    }
    for name, values in step_arrays.items():
        chunks = (time_chunk, *values.shape[1:])
        data.create_dataset(name, data=values, chunks=chunks, compressor=compressor)

    episode_lengths = np.asarray([len(item.actions) for item in episodes], dtype=np.int64)
    metadata.create_dataset("episode_ends", data=np.cumsum(episode_lengths), compressor=compressor)
    metadata.create_dataset(
        "stable_contact_steps",
        data=np.asarray([item.stable_contact_step for item in episodes], dtype=np.int32),
        compressor=compressor,
    )
    text_codec = numcodecs.VLenUTF8()
    text_fields = {
        "object_id": [item.object_id for item in episodes],
        "task_id": [item.task_id for item in episodes],
        "class_name": [item.language["class_name"] for item in episodes],
        "grasped_object_part": [item.language["grasped_object_part"] for item in episodes],
        "low_level_grasp_instruction": [
            item.language["low_level_grasp_instruction"] for item in episodes
        ],
        "high_level_grasp_instruction": [
            item.language["high_level_grasp_instruction"] for item in episodes
        ],
        "annotation_raw": [item.annotation_raw for item in episodes],
    }
    for name, values in text_fields.items():
        metadata.create_dataset(
            name,
            data=_string_array(values),
            object_codec=text_codec,
            compressor=compressor,
        )

    views = np.stack([item.annotation_views for item in episodes])
    extrinsics = np.stack([item.camera_extrinsics for item in episodes])
    annotation.create_dataset(
        "multiview_rgb",
        data=views,
        chunks=(1, 1, *views.shape[2:]),
        compressor=compressor,
    )
    annotation.create_dataset(
        "camera_extrinsics",
        data=extrinsics,
        chunks=(1, *extrinsics.shape[1:]),
        compressor=compressor,
    )
    dataset_attributes = dict(attributes)
    dataset_attributes.update(
        {
            "format": "dexcg.dexart.v2",
            "contact_coordinate_contract": CONTACT_COORDINATE_CONTRACT,
            "point_cloud_frame": "robot_base",
            "contact_point_frame": "robot_base",
            "contact_token_frame": "object_aabb_center",
            "object_center_definition": OBJECT_CENTER_DEFINITION,
            "contact_target_center_rule": (
                "stable_frame_through_stable_step_else_current_frame"
            ),
        }
    )
    root.attrs.update(json.loads(json.dumps(dataset_attributes)))
