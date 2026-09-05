import numpy as np

from dexcg.data.dexart import DexArtEpisode, episode_coordinate_arrays


def test_stable_contact_fills_targets_through_grasp_transition() -> None:
    raw_points = [np.zeros((16, 3), dtype=np.float32) for _ in range(4)]
    raw_masks = [np.zeros(16, dtype=np.bool_) for _ in range(4)]
    raw_points[3][4] = [0.3, 0.2, 0.1]
    raw_masks[3][4] = True
    stable_points = np.zeros((16, 3), dtype=np.float32)
    stable_mask = np.zeros(16, dtype=np.bool_)
    stable_points[4] = [0.1, 0.2, 0.3]
    stable_mask[4] = True

    episode = DexArtEpisode(
        observations={},
        actions=[np.zeros(22, dtype=np.float32) for _ in range(4)],
        raw_contact_points=raw_points,
        raw_contact_masks=raw_masks,
        stable_contact_step=2,
        stable_contact_points=stable_points,
        stable_contact_mask=stable_mask,
        object_id="object",
        task_id="task",
        annotation_views=np.empty((5, 1, 1, 3), dtype=np.uint8),
        camera_extrinsics=np.empty((5, 4, 4), dtype=np.float32),
        language={},
        annotation_raw="{}",
    )

    target_points, target_masks = episode.contact_targets()

    np.testing.assert_array_equal(target_masks[:3], np.broadcast_to(stable_mask, (3, 16)))
    np.testing.assert_array_equal(target_points[:3], np.broadcast_to(stable_points, (3, 16, 3)))
    np.testing.assert_array_equal(target_masks[3], raw_masks[3])
    np.testing.assert_array_equal(target_points[3], raw_points[3])


def test_target_coordinates_use_the_stable_object_center_before_contact() -> None:
    centers = [0.5, 0.7, 0.9]
    point_clouds = [
        np.array(
            [[center - 0.2, 0.0, 0.0], [center + 0.2, 0.0, 0.0], [9.0, 9.0, 9.0]],
            dtype=np.float32,
        )
        for center in centers
    ]
    episode = DexArtEpisode(
        observations={
            "point_cloud": point_clouds,
            "object_point_mask": [np.array([True, True, False])] * 3,
        },
        actions=[np.zeros(22, dtype=np.float32) for _ in centers],
        raw_contact_points=[np.zeros((16, 3), dtype=np.float32) for _ in centers],
        raw_contact_masks=[np.zeros(16, dtype=np.bool_) for _ in centers],
        stable_contact_step=1,
        stable_contact_points=np.zeros((16, 3), dtype=np.float32),
        stable_contact_mask=np.zeros(16, dtype=np.bool_),
        object_id="object",
        task_id="task",
        annotation_views=np.empty((5, 1, 1, 3), dtype=np.uint8),
        camera_extrinsics=np.empty((5, 4, 4), dtype=np.float32),
        language={},
        annotation_raw="{}",
    )

    object_centers, valid, target_centers = episode_coordinate_arrays(episode)

    np.testing.assert_allclose(object_centers[:, 0], centers)
    np.testing.assert_array_equal(valid, [True, True, True])
    np.testing.assert_allclose(target_centers[:, 0], [0.7, 0.7, 0.9])
