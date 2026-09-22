import cv2
import numpy as np

from dexcg.visualization import (
    PANEL_BG,
    PREDICTED_CONTACT,
    basis_heatmap_matrix,
    draw_contact_points,
    pixel_coordinates,
    rd_bu,
    render_basis_heatmap,
    render_planner_frame,
)


def test_basis_heatmap_selects_current_four_basis_columns() -> None:
    bases = np.arange(2 * 22 * 4, dtype=np.float32).reshape(2, 22, 4)

    matrix = basis_heatmap_matrix(bases, current_step=1)

    assert matrix.shape == (22, 4)
    np.testing.assert_array_equal(matrix, bases[1])


def test_point_cloud_projection_preserves_aspect_ratio() -> None:
    projected = np.asarray([[0.0, 0.0], [2.0, 1.0]], dtype=np.float32)

    x, y = pixel_coordinates(
        projected,
        low=np.asarray([0.0, 0.0]),
        high=np.asarray([2.0, 1.0]),
        width=400,
        height=300,
    )

    assert x[1] - x[0] == 2 * (y[0] - y[1])


def test_rd_bu_is_diverging_around_zero() -> None:
    colors = rd_bu(np.asarray([-1.0, 0.0, 1.0]), 1.0)

    assert colors[0, 0] > colors[0, 2]
    assert abs(int(colors[1, 0]) - int(colors[1, 2])) < 2
    assert colors[2, 2] > colors[2, 0]


def test_basis_heatmap_has_stable_dimensions() -> None:
    bases = np.zeros((3, 22, 4), dtype=np.float32)
    bases[:, 0, 0] = 1.0

    image, limit = render_basis_heatmap(bases, current_step=1, width=790, height=244)

    assert image.shape == (244, 790, 3)
    assert image.dtype == np.uint8
    assert limit == 1.0


def test_predicted_contact_uses_cross_marker() -> None:
    canvas = np.full((80, 80, 3), PANEL_BG, dtype=np.uint8)
    point = np.zeros((1, 3), dtype=np.float32)
    mask = np.ones(1, dtype=np.bool_)
    projected = np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)

    draw_contact_points(
        canvas,
        point,
        mask,
        (projected[0], projected[1]),
        PREDICTED_CONTACT,
        predicted=True,
    )

    center = canvas[40, 40]
    assert np.linalg.norm(center.astype(int) - np.asarray(PREDICTED_CONTACT)) < 10
    assert cv2.countNonZero(cv2.inRange(canvas, PREDICTED_CONTACT, PREDICTED_CONTACT)) > 1


def test_planner_frame_keeps_rgb_and_contact_panels_without_basis() -> None:
    frame = render_planner_frame(
        image=np.full((24, 24, 3), 255, dtype=np.uint8),
        point_cloud=np.zeros((1, 3), dtype=np.float32),
        robot=np.zeros((1, 3), dtype=np.float32),
        raw_points=np.zeros((16, 3), dtype=np.float32),
        raw_mask=np.zeros(16, dtype=np.bool_),
        predicted_points=np.zeros((16, 3), dtype=np.float32),
        predicted_mask=np.zeros(16, dtype=np.bool_),
        predicted_link_indices=np.full(16, -1, dtype=np.int16),
        bounds=(np.asarray([-1.0, -1.0]), np.asarray([1.0, 1.0])),
        task="bucket",
        variant="qwen-10000",
        object_id="100431",
        instruction="Lift the bucket by its rim.",
        episode_index=0,
        frame_index=0,
        frame_count=100,
        prediction_step=0,
        stable_step=76,
    )

    assert frame.shape == (720, 1280, 3)
    np.testing.assert_array_equal(frame[200, 200], [255, 255, 255])
