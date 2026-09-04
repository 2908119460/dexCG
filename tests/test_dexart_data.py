import numpy as np

from dexcg.data.dexart import DexArtEpisode


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
