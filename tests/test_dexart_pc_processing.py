import numpy as np
from dexart.env.rl_env.pc_processing import process_pc


def test_full_resolution_object_center_is_independent_of_random_sampling() -> None:
    cloud = np.array(
        [
            [0.2, -0.4, -0.1],
            [0.8, 0.2, -0.3],
            [0.4, 0.0, -0.2],
            [0.6, -0.2, -0.25],
            [1.2, 0.5, -0.2],
        ],
        dtype=np.float32,
    )
    segmentation = np.array([[10], [20], [10], [20], [99]], dtype=np.float32)
    grouping_info = {
        "handle": 10,
        "instance_body": [20],
        "palm": [30],
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
    }
    camera_pose = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)

    first, first_center = process_pc(
        "faucet",
        cloud,
        camera_pose,
        num_points=2,
        np_random=np.random.RandomState(1),
        grouping_info=grouping_info,
        segmentation=segmentation,
        return_object_center=True,
    )
    second, second_center = process_pc(
        "faucet",
        cloud,
        camera_pose,
        num_points=2,
        np_random=np.random.RandomState(2),
        grouping_info=grouping_info,
        segmentation=segmentation,
        return_object_center=True,
    )

    assert first.shape == second.shape == (2, 7)
    np.testing.assert_allclose(first_center, [0.5, -0.1, 0.2])
    np.testing.assert_array_equal(first_center, second_center)
