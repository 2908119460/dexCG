# dexCG

dexCG combines DextER-style embodied contact planning with an SMP diffusion
policy for xArm6 + Allegro manipulation. The public model input contains only
the current observation history and a language instruction. There is no task
ID input.

## Architecture

```text
slow branch
  point_cloud[-1] -> PartField -> point tokens -> Qwen2.5-0.5B
  language ------------------------------^         |
                                                    v
       <joint_start>(<Allegro link><x><y><z>)*<joint_end>
                                                    |
                    projector -> self-attention -> pooled contact feature c

fast branch
  point_cloud + imagin_robot -> PointNet --+
  agent_pos -> MLP -------------------------+-> observation feature s

embodiment branch
  URDF + qpos history -> robot link nodes --+
  point_cloud + object mask -> object node --+-> physically biased graph transformer
  planned contact links --------------------+                  |
                                                               v
  B(s, contact) = QR[U(s) + physical basis residual] -> orthonormal skill basis
  z(noisy coefficient, t,s,c)  -> one scalar diffusion expert per basis vector
  p(g | s,c)                   -> deployment gate
  q(g | s,action,c)            -> training posterior gate
  action = B(s, contact) [g elementwise-multiplied by z]
```

The contact graph is represented by an unordered set of contact nodes. Each
node is one Allegro contact link paired with an XYZ contact position relative
to the axis-aligned bounding-box midpoint of the observed object points. The
same center is subtracted from the planner point cloud. `ContactPlan.object_center`
retains the origin needed to decode generated positions back into the
robot-base frame. Metric scale is unchanged.

The basis branch additionally constructs a physical graph from all URDF links
and one object supernode. VLM-planned contact links add dynamic edges to that
object node; these are target relations, not sensor measurements of current
contact. Its graph transformer combines the four PhysGraph terms: clipped
shortest-path spatial bias, self/joint/contact edge-type bias, RBF geometric
proximity from batched forward kinematics, and head-specific serial/synergy
anatomical bias. The graph output is projected to an action-by-expert residual,
added before the skill basis QR factorization, and initialized with zero output
scale. It is not passed to the diffusion experts or either gate.

## Fixed Allegro contract

The vocabulary follows the exact links used by DP3/DexArt for contact and
robot imagination: palm; three thumb links; and four links each for index,
middle, and ring fingers. This produces 16 link tokens. `link_13.0`, the
ShadowHand little finger, and `rh_thhub` are not planner outputs.

The released DextER checkpoint is loaded with its original vocabulary first.
The 16 Allegro embedding rows are then initialized once from their fixed
ShadowHand semantic counterparts. Contact-only grammar masking makes action
tokens and disabled ShadowHand tokens unreachable during decoding.

The DextER weights and the Qwen tokenizer/configuration snapshot are both
stored under `checkpoints/`; model construction does not read either sibling
repository or a global model cache.

## Observation tensors

Default DexArt shapes are:

```text
point_cloud: [B, 2, 1024, 3]
object_point_mask: [B, 2, 1024]
imagin_robot: [B, 2, 96, 7]
agent_pos: [B, 2, 33]
language: list[str] with length B
```

`object_point_mask` selects the handle and instance-body segmentation classes
for the object supernode. When loading older observations without this field,
the model falls back to all scene points rather than failing.

The fast encoder uses only XYZ from `imagin_robot`; its remaining four
semantic channels are preserved at the data boundary for compatibility with
DP3. The action space is 22-dimensional: xArm6 (6) + Allegro (16).

## Setup and inspection

```bash
uv sync --extra test
uv run pytest
uv run python scripts/inspect_model.py
```

`inspect_model.py` loads the copied DextER checkpoint from
`checkpoints/dexter-qwen2.5-0.5B-dexgys` and prints the composed model and
parameter counts. The checkpoint remains outside Git through `.gitignore`.

## DexArt demonstration collection

DexArt and its customized Stable-Baselines3 implementation are vendored under
`third_party/dexart`. Collection does not import the sibling DP3 checkout. The
four experts are the exact `nopretrain_0` checkpoints selected by DP3's
released collection script.

Each successful trajectory stores the original DP3 observation/action fields,
the object-point mask, raw SAPIEN contacts, contact targets, contact token IDs,
and episode metadata.
The native DexArt `state` is retained unchanged. The model-facing `agent_pos`
is always 33-dimensional: bucket's native 33D state is copied directly, while
the 32D faucet, laptop, and toilet states receive a zero immediately before
their final time-progress value.
The first transition into DexArt stage 3 is the grasp boundary. Its physical
contact graph fills all targets through that transition; later targets use the
current simulator contact graph. Contact manifold points belonging to the same
Allegro link are stored in the robot-base frame for traceability. During data
collection they are also encoded into object-centered target tokens. The Zarr
contract records the center estimator and both coordinate frames; the training
loader rejects datasets without that contract.

Language annotations use five `896x896` robot-base views and
`google/gemma-3-12b-it` with deterministic decoding. `task_id` is provided only
to the annotation model. dexCG receives only the generated low-level physical
instruction.

After placing the gated Gemma checkpoint in `checkpoints/gemma-3-12b-it`, run
one task per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/collect_dexart.py --task faucet --device cuda:0
CUDA_VISIBLE_DEVICES=1 uv run python scripts/collect_dexart.py --task bucket --device cuda:0
CUDA_VISIBLE_DEVICES=2 uv run python scripts/collect_dexart.py --task laptop --device cuda:0
CUDA_VISIBLE_DEVICES=3 uv run python scripts/collect_dexart.py --task toilet --device cuda:0
```
