"""Rendering helpers for online DexCG rollout diagnostics."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping

import cv2
import numpy as np

from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 10
BG = (24, 23, 21)
PANEL_BG = (38, 36, 33)
BORDER = (75, 71, 66)
TEXT = (238, 238, 238)
MUTED = (170, 170, 170)
ACCENT = (70, 205, 255)
RAW_CONTACT = (255, 210, 70)
PREDICTED_CONTACT = (70, 220, 255)
ROBOT = (235, 90, 220)

# ColorBrewer RdBu_r: blue for negative values and red for positive values.
_RDBU_RGB = np.asarray(
    [
        (5, 48, 97),
        (33, 102, 172),
        (67, 147, 195),
        (146, 197, 222),
        (209, 229, 240),
        (247, 247, 247),
        (253, 219, 199),
        (244, 165, 130),
        (214, 96, 77),
        (178, 24, 43),
        (103, 0, 31),
    ],
    dtype=np.float32,
)


def put_text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def put_wrapped_text(
    image: np.ndarray,
    value: str,
    origin: tuple[int, int],
    width: int,
    scale: float = 0.42,
    color: tuple[int, int, int] = TEXT,
    line_height: int = 18,
    max_lines: int = 3,
) -> int:
    approximate_columns = max(12, int(width / max(7.0, 15.0 * scale)))
    lines = textwrap.wrap(str(value), width=approximate_columns) or [""]
    y = origin[1]
    for line in lines[:max_lines]:
        put_text(image, line, (origin[0], y), scale, color)
        y += line_height
    return y


def panel(image: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BG, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), BORDER, 1)
    put_text(image, title, (x + 10, y + 22), 0.5, ACCENT)


def fit_image(
    source: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    source_height, source_width = source.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_size = (
        max(1, int(source_width * scale)),
        max(1, int(source_height * scale)),
    )
    resized = cv2.resize(source, resized_size, interpolation=interpolation)
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    x = (width - resized_size[0]) // 2
    y = (height - resized_size[1]) // 2
    canvas[y : y + resized_size[1], x : x + resized_size[0]] = resized
    return canvas


def project(points: np.ndarray) -> np.ndarray:
    angle = math.radians(42.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rotated_x = cosine * x - sine * y
    rotated_y = sine * x + cosine * y
    return np.stack((rotated_x, z + 0.28 * rotated_y, rotated_y), axis=-1)


def compute_projection_bounds(
    point_clouds: np.ndarray, robots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate(
        (
            np.asarray(point_clouds)[..., :3].reshape(-1, 3),
            np.asarray(robots)[..., :3].reshape(-1, 3),
        ),
        axis=0,
    )
    projected = project(points)
    finite = np.all(np.isfinite(projected[:, :2]), axis=1)
    if not finite.any():
        raise ValueError("rollout has no finite points for projection")
    low = np.percentile(projected[finite, :2], 0.5, axis=0)
    high = np.percentile(projected[finite, :2], 99.5, axis=0)
    padding = np.maximum((high - low) * 0.08, 1e-3)
    return low - padding, high + padding


def pixel_coordinates(
    projected: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    span = np.maximum(high - low, 1e-6)
    scale = min((width - 24) / span[0], (height - 24) / span[1])
    rendered_span = span * scale
    offset = (np.asarray((width, height)) - rendered_span) * 0.5
    x = ((projected[:, 0] - low[0]) * scale + offset[0]).astype(int)
    y = (height - ((projected[:, 1] - low[1]) * scale + offset[1])).astype(int)
    return x, y


def draw_contact_points(
    canvas: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    color: tuple[int, int, int],
    predicted: bool,
) -> None:
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        return
    projected = project(np.asarray(points)[active, :3])
    x, y = pixel_coordinates(projected, bounds[0], bounds[1], canvas.shape[1], canvas.shape[0])
    for px, py in zip(x, y, strict=True):
        if not (0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]):
            continue
        if predicted:
            cv2.line(canvas, (px - 5, py - 5), (px + 5, py + 5), color, 2)
            cv2.line(canvas, (px - 5, py + 5), (px + 5, py - 5), color, 2)
        else:
            cv2.circle(canvas, (px, py), 5, color, 2)


def render_point_cloud(
    point_cloud: np.ndarray,
    robot: np.ndarray,
    raw_points: np.ndarray,
    raw_mask: np.ndarray,
    predicted_points: np.ndarray,
    predicted_mask: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    projected = project(np.asarray(point_cloud)[..., :3])
    finite = np.all(np.isfinite(projected), axis=1)
    projected = projected[finite]
    x, y = pixel_coordinates(projected, bounds[0], bounds[1], width, height)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    for index in np.argsort(projected[:, 2]):
        if inside[index]:
            cv2.circle(canvas, (x[index], y[index]), 1, (205, 205, 205), -1)

    robot_projected = project(np.asarray(robot)[..., :3])
    robot_finite = np.all(np.isfinite(robot_projected), axis=1)
    rx, ry = pixel_coordinates(
        robot_projected[robot_finite], bounds[0], bounds[1], width, height
    )
    for px, py in zip(rx, ry, strict=True):
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(canvas, (px, py), 2, ROBOT, -1)

    draw_contact_points(canvas, raw_points, raw_mask, bounds, RAW_CONTACT, predicted=False)
    draw_contact_points(
        canvas,
        predicted_points,
        predicted_mask,
        bounds,
        PREDICTED_CONTACT,
        predicted=True,
    )
    return canvas


def basis_heatmap_matrix(bases: np.ndarray, current_step: int) -> np.ndarray:
    values = np.asarray(bases, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected bases [N, D, K], received {values.shape}")
    if values.shape[0] < 1 or values.shape[1:] != (22, 4):
        raise ValueError(f"Expected bases [N, 22, 4], received {values.shape}")
    if not 0 <= current_step < values.shape[0]:
        raise ValueError(
            f"current_step {current_step} is outside [0, {values.shape[0]})"
        )
    return values[current_step]


def rd_bu(values: np.ndarray, limit: float) -> np.ndarray:
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("heatmap limit must be finite and positive")
    normalized = np.clip(np.asarray(values, dtype=np.float32) / limit, -1.0, 1.0)
    positions = (normalized + 1.0) * 0.5 * (_RDBU_RGB.shape[0] - 1)
    low = np.floor(positions).astype(np.int64)
    high = np.minimum(low + 1, _RDBU_RGB.shape[0] - 1)
    weight = (positions - low)[..., None]
    rgb = _RDBU_RGB[low] * (1.0 - weight) + _RDBU_RGB[high] * weight
    return np.rint(rgb[..., ::-1]).astype(np.uint8)


def render_basis_heatmap(
    bases: np.ndarray,
    current_step: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    num_steps = bases.shape[0]
    if not 0 <= current_step < num_steps:
        raise ValueError(f"current_step {current_step} is outside [0, {num_steps})")
    matrix = basis_heatmap_matrix(bases, current_step)
    limit = float(np.max(np.abs(bases)))
    if limit < 1e-8:
        limit = 1.0

    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    label_width = 34
    scale_width = 42
    header_height = 16
    plot_width = width - label_width - scale_width
    plot_height = height - header_height
    colors = rd_bu(matrix, limit)
    heatmap = cv2.resize(colors, (plot_width, plot_height), interpolation=cv2.INTER_NEAREST)
    canvas[header_height:, label_width : label_width + plot_width] = heatmap

    for column in range(5):
        x = label_width + round(column / 4 * plot_width)
        cv2.line(canvas, (x, header_height), (x, height - 1), BORDER, 1)

    arm_boundary = header_height + round(6 / 22 * plot_height)
    cv2.line(
        canvas,
        (label_width, arm_boundary),
        (label_width + plot_width - 1, arm_boundary),
        TEXT,
        1,
    )
    for row in range(22):
        label = f"A{row}" if row < 6 else f"H{row - 6}"
        y = header_height + round((row + 0.72) / 22 * plot_height)
        put_text(canvas, label, (1, y), 0.25, MUTED)
    for column in range(4):
        center = label_width + round((column + 0.5) / 4 * plot_width)
        put_text(canvas, f"B{column + 1}", (center - 8, 11), 0.28, MUTED)

    bar_x = label_width + plot_width + 8
    bar_values = np.linspace(1.0, -1.0, plot_height, dtype=np.float32)[:, None]
    bar = rd_bu(bar_values, 1.0)
    canvas[header_height:, bar_x : bar_x + 10] = np.repeat(bar, 10, axis=1)
    put_text(canvas, f"{limit:.2f}", (bar_x + 12, header_height + 7), 0.25, MUTED)
    put_text(canvas, "0", (bar_x + 12, header_height + plot_height // 2 + 3), 0.25, MUTED)
    put_text(canvas, f"-{limit:.2f}", (bar_x + 12, height - 2), 0.25, MUTED)
    return canvas, limit


def active_raw_links(mask: np.ndarray) -> str:
    names = [
        link.token_name.removeprefix("allegro_")
        for link, active in zip(ALLEGRO_CONTACT_LINKS, mask, strict=True)
        if active
    ]
    return ", ".join(names) if names else "none"


def active_predicted_links(link_indices: np.ndarray, mask: np.ndarray) -> str:
    names = []
    for index in np.asarray(link_indices)[np.asarray(mask, dtype=bool)].tolist():
        name = ALLEGRO_CONTACT_LINKS[int(index)].token_name.removeprefix("allegro_")
        if name not in names:
            names.append(name)
    return ", ".join(names) if names else "none"


def render_rollout_frame(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    frame_index: int,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    length = len(arrays["img"])
    if not 0 <= frame_index < length:
        raise ValueError(f"frame_index {frame_index} is outside [0, {length})")
    inference_index = int(arrays["inference_index"][frame_index])
    stable_frame = int(metadata["stable_frame"])

    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BG, dtype=np.uint8)
    put_text(frame, f"DexCG online rollout | {metadata['task']}", (20, 34), 0.72, TEXT, 2)
    put_text(
        frame,
        f"object {metadata['object_id']} | frame {frame_index + 1:03d}/{length:03d} | "
        f"inference {inference_index:02d}/{len(arrays['basis']) - 1:02d}",
        (720, 34),
        0.46,
        MUTED,
    )

    rgb_rect = (20, 52, 340, 332)
    cloud_rect = (380, 52, 880, 332)
    basis_rect = (20, 404, 820, 296)
    info_rect = (860, 404, 400, 296)
    panel(frame, rgb_rect, "RGB observation")
    panel(frame, cloud_rect, "Point cloud | raw=o | predicted token=x")
    panel(frame, basis_rect, "Current final B(s, contact) | columns=[B1 B2 B3 B4]")
    panel(frame, info_rect, "Contact and rollout diagnostics")

    rgb = cv2.cvtColor(arrays["img"][frame_index], cv2.COLOR_RGB2BGR)
    frame[84:368, 35:345] = fit_image(rgb, 310, 284)

    predicted_points = arrays["predicted_contact_points"][inference_index]
    predicted_mask = arrays["predicted_contact_mask"][inference_index]
    frame[84:368, 395:1245] = render_point_cloud(
        arrays["point_cloud"][frame_index],
        arrays["imagin_robot"][frame_index],
        arrays["raw_contact_points"][frame_index],
        arrays["raw_contact_mask"][frame_index],
        predicted_points,
        predicted_mask,
        bounds,
        850,
        284,
    )

    heatmap, _ = render_basis_heatmap(arrays["basis"], inference_index, 790, 244)
    frame[438:682, 35:825] = heatmap

    phase = (
        "stable/post-contact"
        if stable_frame >= 0 and frame_index >= stable_frame
        else "pre-contact"
    )
    y = 440
    put_text(frame, f"task: {metadata['task']}", (875, y), 0.43, TEXT)
    put_text(frame, f"object ID: {metadata['object_id']}", (875, y + 22), 0.43, TEXT)
    put_text(frame, f"phase: {phase}", (875, y + 44), 0.43, TEXT)
    put_text(
        frame,
        f"predicted contacts: {int(np.asarray(predicted_mask).sum())}",
        (875, y + 66),
        0.43,
        TEXT,
    )
    y = put_wrapped_text(
        frame,
        f"Raw links: {active_raw_links(arrays['raw_contact_mask'][frame_index])}",
        (875, y + 92),
        370,
        color=RAW_CONTACT,
        max_lines=3,
    )
    y = put_wrapped_text(
        frame,
        "Predicted links: "
        + active_predicted_links(
            arrays["predicted_contact_link_indices"][inference_index], predicted_mask
        ),
        (875, y + 4),
        370,
        color=PREDICTED_CONTACT,
        max_lines=3,
    )
    put_wrapped_text(
        frame,
        f"Instruction: {metadata['instruction']}",
        (875, y + 8),
        370,
        max_lines=4,
    )
    return frame


def render_planner_frame(
    image: np.ndarray,
    point_cloud: np.ndarray,
    robot: np.ndarray,
    raw_points: np.ndarray,
    raw_mask: np.ndarray,
    predicted_points: np.ndarray,
    predicted_mask: np.ndarray,
    predicted_link_indices: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    task: str,
    variant: str,
    object_id: str,
    instruction: str,
    episode_index: int,
    frame_index: int,
    frame_count: int,
    prediction_step: int,
    stable_step: int,
    success: bool | None = None,
) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BG, dtype=np.uint8)
    title = "DexCG VLM" if success is None else "DexCG VLM+SMP"
    put_text(frame, f"{title} | {variant} | {task}", (20, 34), 0.72, TEXT, 2)
    put_text(
        frame,
        f"episode {episode_index:03d} | object {object_id} | "
        f"frame {frame_index + 1:03d}/{frame_count:03d}",
        (800, 34), 0.43, MUTED,
    )
    panel(frame, (20, 52, 500, 478), "RGB observation")
    panel(frame, (540, 52, 720, 478), "Point cloud | raw=o | predicted token=x")
    panel(frame, (20, 550, 1240, 150), "Contact and trajectory diagnostics")
    frame[86:510, 35:505] = fit_image(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 470, 424
    )
    frame[86:510, 555:1245] = render_point_cloud(
        point_cloud, robot, raw_points, raw_mask,
        predicted_points, predicted_mask, bounds, 690, 424,
    )
    put_text(
        frame,
        f"task: {task} | object ID: {object_id} | episode: {episode_index} "
        f"| prediction step: {prediction_step} | "
        f"{'stable/post-contact' if stable_step >= 0 and frame_index >= stable_step else 'pre-contact'}"
        + ("" if success is None else f" | {'SUCCESS' if success else 'FAILURE'}"),
        (35, 586), 0.43, TEXT,
    )
    put_wrapped_text(
        frame, f"Raw links: {active_raw_links(raw_mask)}",
        (35, 608), 580, color=RAW_CONTACT, max_lines=2,
    )
    put_wrapped_text(
        frame, "Predicted links: "
        + active_predicted_links(predicted_link_indices, predicted_mask),
        (635, 608), 600, color=PREDICTED_CONTACT, max_lines=2,
    )
    put_wrapped_text(
        frame, f"Instruction: {instruction}", (35, 670), 1190, max_lines=2,
    )
    return frame
