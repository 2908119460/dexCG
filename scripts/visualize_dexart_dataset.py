#!/usr/bin/env python3
"""Render one data-quality review video per collected DexArt episode."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
import zarr

from dexcg.robots.allegro import ALLEGRO_CONTACT_LINKS

TASKS = ("faucet", "bucket", "laptop", "toilet")
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
TARGET_CONTACT = (70, 220, 255)
ROBOT = (235, 90, 220)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--variant", default="expert")
    parser.add_argument("--data-root", type=Path, default=Path("data/dexart"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data_visualize/dexart")
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crf", type=int, default=23)
    return parser.parse_args()


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
    scale: float = 0.46,
    color: tuple[int, int, int] = TEXT,
    line_height: int = 20,
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


def compute_task_scale(group: zarr.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_cloud = group["data"]["point_cloud"]
    robot = group["data"]["imagin_robot"]
    depth = group["data"]["depth"]
    indices = np.linspace(0, len(point_cloud) - 1, min(96, len(point_cloud)), dtype=int)
    scene_points = np.asarray(point_cloud.oindex[indices])[..., :3].reshape(-1, 3)
    robot_points = np.asarray(robot.oindex[indices])[..., :3].reshape(-1, 3)
    projected = project(np.concatenate((scene_points, robot_points), axis=0))
    finite = np.all(np.isfinite(projected[:, :2]), axis=1)
    low = np.percentile(projected[finite, :2], 0.5, axis=0)
    high = np.percentile(projected[finite, :2], 99.5, axis=0)
    padding = np.maximum((high - low) * 0.08, 1e-3)

    depth_values = np.asarray(depth.oindex[indices]).reshape(-1)
    valid_depth = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
    depth_range = (
        np.percentile(valid_depth, (1.0, 99.0))
        if valid_depth.size
        else np.array((0.0, 1.0))
    )
    return low - padding, high + padding, depth_range


def render_depth(
    depth: np.ndarray,
    depth_range: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    low, high = depth_range
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if high > low:
        values = 1.0 - np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized[valid] = np.asarray(values[valid] * 255, dtype=np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return fit_image(colored, width, height)


def pixel_coordinates(
    projected: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    span = np.maximum(high - low, 1e-6)
    x = ((projected[:, 0] - low[0]) / span[0] * (width - 24) + 12).astype(int)
    y = ((1.0 - (projected[:, 1] - low[1]) / span[1]) * (height - 24) + 12).astype(int)
    return x, y


def draw_contact_points(
    canvas: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    color: tuple[int, int, int],
    target: bool,
) -> None:
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        return
    projected = project(np.asarray(points)[active, :3])
    x, y = pixel_coordinates(projected, bounds[0], bounds[1], canvas.shape[1], canvas.shape[0])
    for px, py in zip(x, y, strict=True):
        if not (0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]):
            continue
        if target:
            cv2.line(canvas, (px - 5, py - 5), (px + 5, py + 5), color, 2)
            cv2.line(canvas, (px - 5, py + 5), (px + 5, py - 5), color, 2)
        else:
            cv2.circle(canvas, (px, py), 5, color, 2)


def render_point_cloud(
    point_cloud: np.ndarray,
    robot: np.ndarray,
    raw_points: np.ndarray,
    raw_mask: np.ndarray,
    target_points: np.ndarray,
    target_mask: np.ndarray,
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

    draw_contact_points(canvas, raw_points, raw_mask, bounds, RAW_CONTACT, target=False)
    draw_contact_points(canvas, target_points, target_mask, bounds, TARGET_CONTACT, target=True)
    return canvas


def action_heatmap(actions: np.ndarray, frame_index: int, width: int, height: int) -> np.ndarray:
    values = np.clip(actions.T, -1.0, 1.0)
    red = np.clip(values, 0.0, 1.0) * 215
    blue = np.clip(-values, 0.0, 1.0) * 215
    neutral = 48 + (1.0 - np.abs(values)) * 42
    heatmap = np.stack((neutral + blue, neutral, neutral + red), axis=-1)
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    cursor = int(frame_index / max(1, len(actions) - 1) * (width - 1))
    cv2.line(resized, (cursor, 0), (cursor, height - 1), (255, 255, 255), 2)
    return resized


def active_links(mask: np.ndarray) -> str:
    names = [
        link.token_name.removeprefix("allegro_")
        for link, active in zip(ALLEGRO_CONTACT_LINKS, mask, strict=True)
        if active
    ]
    return ", ".join(names) if names else "none"


def episode_metadata(group: zarr.Group, episode_index: int) -> dict[str, str]:
    fields = (
        "object_id",
        "task_id",
        "class_name",
        "grasped_object_part",
        "low_level_grasp_instruction",
        "high_level_grasp_instruction",
    )
    return {field: str(group["meta"][field][episode_index]) for field in fields}


def build_intro_frame(
    task: str,
    episode_index: int,
    group: zarr.Group,
    metadata: dict[str, str],
) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BG, dtype=np.uint8)
    put_text(
        frame,
        f"DexArt data quality review | {task} | episode {episode_index:03d}",
        (20, 36),
        0.75,
        TEXT,
        2,
    )
    views = np.asarray(group["annotation"]["multiview_rgb"][episode_index])
    labels = ("+X", "-X", "+Y", "-Y", "+Z")
    for index, (view, label) in enumerate(zip(views, labels, strict=True)):
        x = 20 + index * 248
        put_text(frame, label, (x, 65), 0.48, ACCENT)
        image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        frame[75:307, x : x + 232] = fit_image(image, 232, 232, cv2.INTER_AREA)

    y = 345
    put_text(
        frame,
        f"object={metadata['object_id']}  task={metadata['task_id']}  "
        f"class={metadata['class_name']}  part={metadata['grasped_object_part']}",
        (20, y),
        0.5,
        MUTED,
    )
    y = put_wrapped_text(
        frame,
        f"Low-level: {metadata['low_level_grasp_instruction']}",
        (20, y + 38),
        1220,
        scale=0.55,
        line_height=26,
        max_lines=3,
    )
    put_wrapped_text(
        frame,
        f"High-level: {metadata['high_level_grasp_instruction']}",
        (20, y + 18),
        1220,
        scale=0.55,
        line_height=26,
        max_lines=3,
    )
    put_text(frame, "Five robot-base annotation views used by Gemma-3-12B", (20, 695), 0.45, MUTED)
    return frame


def build_timeline_frame(
    task: str,
    episode_index: int,
    frame_index: int,
    episode: dict[str, np.ndarray],
    metadata: dict[str, str],
    stable_step: int,
    task_scale: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BG, dtype=np.uint8)
    length = len(episode["img"])
    put_text(frame, f"DexArt data quality review | {task}", (20, 34), 0.72, TEXT, 2)
    put_text(
        frame,
        f"episode {episode_index:03d} | frame {frame_index + 1:03d}/{length:03d} | "
        f"stable step {stable_step:03d}",
        (730, 34),
        0.5,
        MUTED,
    )

    rgb_rect = (20, 52, 360, 342)
    depth_rect = (400, 52, 360, 342)
    cloud_rect = (780, 52, 480, 342)
    action_rect = (20, 414, 600, 276)
    info_rect = (640, 414, 620, 276)
    panel(frame, rgb_rect, "RGB observation")
    panel(frame, depth_rect, "Depth observation")
    panel(frame, cloud_rect, "Point cloud | raw=o | target=x")
    panel(frame, action_rect, "Action history | 22 dimensions | blue=-1 red=+1")
    panel(frame, info_rect, "Contact and language diagnostics")

    rgb = cv2.cvtColor(episode["img"][frame_index], cv2.COLOR_RGB2BGR)
    frame[84:378, 35:365] = fit_image(rgb, 330, 294)
    frame[84:378, 415:745] = render_depth(
        episode["depth"][frame_index], task_scale[2], 330, 294
    )
    frame[84:378, 795:1245] = render_point_cloud(
        episode["point_cloud"][frame_index],
        episode["imagin_robot"][frame_index],
        episode["contact_raw_points"][frame_index],
        episode["contact_raw_mask"][frame_index],
        episode["contact_target_points"][frame_index],
        episode["contact_target_mask"][frame_index],
        task_scale[:2],
        450,
        294,
    )
    frame[450:668, 35:605] = action_heatmap(
        episode["action"], frame_index, 570, 218
    )

    raw_mask = episode["contact_raw_mask"][frame_index]
    target_mask = episode["contact_target_mask"][frame_index]
    raw_tokens = int(episode["contact_raw_token_mask"][frame_index].sum())
    target_tokens = int(episode["contact_target_token_mask"][frame_index].sum())
    y = 450
    put_text(
        frame,
        f"phase={'stable/post-contact' if frame_index >= stable_step else 'pre-contact'} | "
        f"raw tokens={raw_tokens} | target tokens={target_tokens}",
        (655, y),
        0.43,
        TEXT,
    )
    y = put_wrapped_text(
        frame,
        f"Raw links: {active_links(raw_mask)}",
        (655, y + 25),
        590,
        scale=0.42,
        color=RAW_CONTACT,
        line_height=18,
        max_lines=2,
    )
    y = put_wrapped_text(
        frame,
        f"Target links: {active_links(target_mask)}",
        (655, y + 5),
        590,
        scale=0.42,
        color=TARGET_CONTACT,
        line_height=18,
        max_lines=2,
    )
    y = put_wrapped_text(
        frame,
        f"Low: {metadata['low_level_grasp_instruction']}",
        (655, y + 8),
        590,
        scale=0.42,
        line_height=18,
        max_lines=3,
    )
    put_wrapped_text(
        frame,
        f"High: {metadata['high_level_grasp_instruction']}",
        (655, y + 8),
        590,
        scale=0.42,
        line_height=18,
        max_lines=3,
    )
    return frame


def open_encoder(output_path: Path, crf: int) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render_episode(
    task: str,
    episode_index: int,
    group: zarr.Group,
    start: int,
    end: int,
    task_scale: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    crf: int,
) -> None:
    data = group["data"]
    keys = (
        "img",
        "depth",
        "point_cloud",
        "imagin_robot",
        "action",
        "contact_raw_points",
        "contact_raw_mask",
        "contact_target_points",
        "contact_target_mask",
        "contact_raw_token_mask",
        "contact_target_token_mask",
    )
    episode = {key: np.asarray(data[key][start:end]) for key in keys}
    metadata = episode_metadata(group, episode_index)
    stable_step = int(group["meta"]["stable_contact_steps"][episode_index])
    intro = build_intro_frame(task, episode_index, group, metadata)
    encoder = open_encoder(output_path, crf)
    try:
        assert encoder.stdin is not None
        for _ in range(2 * FPS):
            encoder.stdin.write(intro.tobytes())
        for frame_index in range(end - start):
            frame = build_timeline_frame(
                task,
                episode_index,
                frame_index,
                episode,
                metadata,
                stable_step,
                task_scale,
            )
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        output_path.unlink(missing_ok=True)
        raise
    if return_code != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output_path} with code {return_code}")


def main() -> None:
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    data_root = args.data_root
    output_root = args.output_root
    if not data_root.is_absolute():
        data_root = args.project_root / data_root
    if not output_root.is_absolute():
        output_root = args.project_root / output_root
    for task in args.tasks:
        source = data_root / f"{task}_{args.variant}.zarr"
        output_dir = output_root / f"{task}_{args.variant}"
        if not source.exists():
            raise FileNotFoundError(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(source), mode="r")
        episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
        total = len(episode_ends) if args.limit is None else min(args.limit, len(episode_ends))
        task_scale = compute_task_scale(group)
        print(f"[{task}] rendering {total} episodes", flush=True)
        for episode_index in range(total):
            output_path = output_dir / f"episode_{episode_index:03d}.mp4"
            if output_path.exists() and not args.overwrite:
                print(f"  skip {output_path.name}", flush=True)
                continue
            start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
            end = int(episode_ends[episode_index])
            print(
                f"  {episode_index + 1:03d}/{total:03d} {output_path.name} ({end - start} frames)",
                flush=True,
            )
            render_episode(
                task,
                episode_index,
                group,
                start,
                end,
                task_scale,
                output_path,
                args.crf,
            )


if __name__ == "__main__":
    main()
