# DexArt object-only collection: coordinate contract

This collection targets 250 accepted trajectories for each of faucet, bucket,
laptop and toilet, using every object in the DexArt `seen` lists. The first
three tasks contain 11 objects (22 or 23 trajectories each); toilet contains
17 objects (14 or 15 each). Actual completion is recorded in `distribution.json`.

## Base frame

`robot_base` means the articulation pose returned by `environment.robot.get_pose()`.
The robot URDF calls its root link `world`, but that name does **not** mean the
simulation world frame. The URDF fixed joint from `world` to `link_base` has zero
translation and zero rotation. Their poses were also compared in the simulator.

The task source sets the world position of this base to (-0.5, 0, 0) for
faucet/laptop, (-0.5, 0, 0.3) for bucket, and (-0.5, 0, 0.4) for toilet.
All four base rotations are identity. Collection reads the actual pose instead
of hardcoding these values. Distances are metres, angles are radians, and time is
seconds. The axes are those of the DexArt robot articulation; no axis permutation,
object centering or scale normalization is applied to saved Cartesian data.

For a world point, `p_base = R_world_from_base.T @ (p_world - t_world_from_base)`.
Vectors such as linear/angular velocity use only the inverse rotation.

## Stored quantities

| Field | Meaning |
|---|---|
| `data/point_cloud` | 1024 XYZ points of the visible target object, in base metres |
| `data/object_center` | Midpoint of the full visible object AABB, in base metres |
| `data/contact_raw_points`, `contact_target_points` | Base-frame contact XYZ, metres; masks determine active entries |
| `data/contact_raw_token_ids`, `contact_target_token_ids` | Quantization of those same base coordinates, with **no center subtraction** |
| `data/state[0:22]` | Original robot joint positions, radians |
| `data/state[22:25]` | Palm linear velocity, base axes, m/s |
| `data/state[25:28]` | Palm angular velocity, base axes, rad/s |
| `data/state[28:31]` | Palm position, base metres |
| bucket `data/state[31]` | Palm local +X direction dotted with base +Z |
| Last state element | Step divided by task horizon, dimensionless |
| `data/agent_pos` | State padded to 33 dimensions; non-bucket index 31 is zero |
| `data/palm_pose_robot_base` | Homogeneous transform from palm-local to robot-base coordinates |
| `data/action[0:3]` | Normalized end-link **center-of-mass** linear velocity, base axes |
| `data/action[3:6]` | Normalized end-link angular velocity, base axes |
| `data/action[6:22]` | DexArt normalized hand joint position commands |
| `data/img`, `data/depth` | Camera RGB and optical-axis depth in metres; these remain image observations |
| `data/camera_to_robot_base` | Transform from OpenGL camera XYZ to base XYZ |
| `data/camera_intrinsics` | Pixel calibration; pixel centers are (column+0.5, row+0.5) |
| `annotation/camera_extrinsics` | SAPIEN camera-local (+X forward) to robot-base transform |
| `meta/world_from_robot_base` | Explicit provenance transform; this is a frame mapping, not a world-frame target |

Action values are bounded to [-1, 1]. Per-episode velocity limits, joint limits
and control timestep are saved, so the physical commands can be recovered.
Finite differences confirm that the linear rows of DexArt's Jacobian refer to
the end-link center of mass, not its link-frame origin. The local COM offset and
Jacobian errors are recorded in `action_frame_audit.json`.
Joint coordinates, image pixels, scalar depth, IDs and normalized commands are
not mislabeled as Cartesian XYZ.

## Point-cloud extraction

The renderer's link-level actor IDs are compared with all `instance_links` of
the current target object. This includes both the body and moving parts. Hand,
arm, table and background pixels are excluded. Extraction uses the full rendered
image, not the expert's cropped scene sample. There is no additional point noise.
Only finite points beyond the 0.05 m camera near cutoff are accepted. If fewer
than 1024 object pixels are visible, points are repeated from that object;
no background or zero padding is introduced. Empty observations reject the episode.

Each stored point retains its actor ID and source pixel index. The available
object-pixel count is saved per frame, and the allowed actor IDs per episode.
Robot imagination clouds are not saved in this object-only dataset.

The pretrained expert still receives its original native DexArt observation:
world-frame state and noisy scene point cloud. Object sampling uses a separate
random generator and does not change the expert's observation or RNG sequence.

## Contact quantization

The shared range is [-1.0, 1.2] metres on each axis. The existing tokenizer uses
256 boundaries and midpoint decoding, giving a maximum per-axis error of
2.2 / 255 / 2 = 0.004313726 m. Coordinates outside the range stop collection;
they are never silently clipped. Raw and target contact graphs are both encoded.
Decoded token positions require no object-center offset.

The dataset declares `contact_coordinate_contract=dexart_robot_base_metric_v1`.
Existing training readers that require object-centered tokens intentionally reject
this contract. They must be adapted explicitly before training on this collection;
an old object-centered checkpoint must not silently reinterpret these token values.

## Evidence and acceptance checks

Source definitions checked: `dexart/env/rl_env/base.py` camera pose composition and
arm action execution; the four task `get_robot_state` and reset methods;
`dexart/utils/kinematics_helper.py`; and the XArm/Allegro URDF fixed root joint.
The adapter transforms simulator contact positions into the robot base before
encoding. State conversion is independent of the expert input.

`*_preflight.json` records initial-observation checks for all 50 seen objects
and one successful expert trajectory per task. `action_frame_audit.json` records
the finite-difference Jacobian and action-normalization checks. `pilot_audit.json`
records independent depth reconstruction and token decoding of the first two
saved trajectories. These are limited pilot results, not evidence that all 1000
trajectories have completed.

Every collected frame checks camera transform composition, point/world and
palm/world round trips, velocity rotation, camera pixel reprojection, actor-ID
membership, and simulator contact/world round trips. Each accepted episode checks
contact range and token decoding. After each task reaches its quota, the collector
reads every stored frame to reconstruct XYZ from depth, verify actor IDs and
decode both raw and target tokens. Only a successful final audit produces
`*_audit.json` and marks the task complete.

Episodes are committed individually. Restarting the collection command truncates
an interrupted append to the last committed episode and resumes missing quotas.
Attempt seeds and counts are recorded. A task stops and reports a deficit if an
object reaches 200 attempts before meeting its quota.
