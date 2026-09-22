import numpy as np
from dexart.env.task_setting import BOUND_CONFIG


def segmentation_masks(labels: np.ndarray, grouping_info: dict):
    labels = np.asarray(labels)
    handle_mask = labels == grouping_info['handle']
    instance_body_mask = labels == grouping_info['instance_body'][0]
    for instance_body_id in grouping_info['instance_body']:
        instance_body_mask = np.logical_or(labels == instance_body_id, instance_body_mask)

    arm_mask = np.logical_and(
        labels < grouping_info['palm'][0], labels > grouping_info['palm'][0] - 8
    )
    hand_mask = labels == grouping_info['palm'][0]
    for instance in (
        grouping_info['palm']
        + grouping_info['thumb']
        + grouping_info['index']
        + grouping_info['middle']
        + grouping_info['ring']
    ):
        hand_mask = np.logical_or(labels == instance, hand_mask)
    return handle_mask, instance_body_mask, hand_mask, arm_mask


def process_pc(task_name: str, cloud: np.ndarray, camera_pose: np.ndarray, num_points: int,
               np_random: np.random.RandomState, noise_level=0, grouping_info=None, segmentation=None,
               color=None, return_object_center=False):
    """
    1. only sample pc to num_points
    noise_level=0, group_info=None, segmentation=None

    2. sample pc and add noise
    noise_level!=0, group_info=None, segmentation=None

    3. sample pc,  add noise and use segmentation
    (output : num_points x (3 + NUM_CLASS))
    noise_level!=0, group_info!=None, segmentation!=None (H,W,1)

    other input params combination may cause undefined behavior
    """

    # first remove points with z < -0.05 (remove artifacts)
    within_bound_r = (cloud[..., 2] < -0.05)

    pc = cloud[..., :3]
    pc = pc @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    bound = BOUND_CONFIG[task_name]

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z, within_bound_r)))[
        0]

    object_center = np.full(3, np.nan, dtype=np.float32)
    if segmentation is not None and grouping_info is not None:
        if len(segmentation) != len(pc):
            raise ValueError("segmentation and point cloud must contain the same number of points")
        handle_mask, body_mask, _, _ = segmentation_masks(segmentation, grouping_info)
        object_mask = np.logical_or(handle_mask, body_mask).reshape(-1)
        visible_object_indices = within_bound[object_mask[within_bound]]
        if len(visible_object_indices) > 0:
            object_points = pc[visible_object_indices]
            object_center = np.asarray(
                0.5 * (object_points.min(axis=0) + object_points.max(axis=0)),
                dtype=np.float32,
            )

    num_index = len(within_bound)
    if num_index == 0:
        result = np.zeros([num_points, 3])
        return (result, object_center) if return_object_center else result
    if num_index < num_points:
        indices = np.concatenate([within_bound, np.ones(num_points - num_index, dtype=np.int32) * within_bound[0]])
        if noise_level != 0:
            multiplicative_noise = 1 + np_random.randn(num_index)[:, None] * 0.01 * noise_level  # (num_index, 1)
            multiplicative_noise = np.concatenate(
                [multiplicative_noise, np.ones([num_points - num_index, 1]) * multiplicative_noise[0]], axis=0)
    else:
        indices = within_bound[np_random.permutation(num_index)[:num_points]]
        if noise_level != 0:
            multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)

    if noise_level != 0:
        pc = pc[indices, :] * multiplicative_noise
        cloud = np.concatenate([pc, cloud[indices, 3:]], axis=1)
    else:
        pc = pc[indices, :]
        cloud = np.concatenate([pc, cloud[indices, 3:]], axis=1)

    if color is not None:
        color = color[indices, :]
        cloud = np.concatenate([cloud, color], axis=1)
        
    if segmentation is not None:
        labels = segmentation[indices, :]   # N x 1
        handle_mask, instance_body_mask, hand_mask, arm_mask = segmentation_masks(
            labels, grouping_info
        )
        cloud = np.concatenate([cloud, handle_mask, instance_body_mask, hand_mask, arm_mask], axis=1)
        # (N, 7) == (N, xyz + 4masks)

    return (cloud, object_center) if return_object_center else cloud

def add_gaussian_noise(cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = 1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level  # (n, 1)
    return cloud * multiplicative_noise
