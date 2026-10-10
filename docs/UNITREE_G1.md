# Unitree G1 (29-DoF) in HOPE

The Unitree G1 is HOPE's second robot model. It is **additive and selected
explicitly**: the Agibot A3 remains the hardware-validated default line and
nothing in the A3 path changes when the G1 files are present. What ships for
the G1:

| Area | Shipped | Not shipped |
| --- | --- | --- |
| Robot description | Vendored Unitree `g1_29dof_rev_1_0` + generated racket-equipped URDF/MJCF ([`unitree/`](../unitree/README.md)) | — |
| Isaac Lab training | Asset prep, articulation config, `HOPE-PingPong-UnitreeG1-v0` and `HOPE-HitterPingPong-UnitreeG1-v0` tasks, `HOPEPingPongUnitreeG1` task YAML, 104-D `hitter_pure_unitree_g1` contract, exporter `--robot unitree_g1`, derived G1 reference clips (joint-space transfer of the A3 pair) | A kinematically faithful (keypoint/IK) G1 retarget of the swings (see 3.3) |
| MuJoCo sim-to-sim | `mujoco_eval_onnx.py --robot unitree_g1`, reference runner `--robot unitree_g1`, G1 runtime + action-adapter YAMLs, `RobotProfile UNITREE_G1` | — |
| Deploy | — | No G1 hardware runner (the C++ runner and the Gate scripts are A3-only) |

## 1. Where things live

```text
unitree/URDF/g1_description/                 verbatim upstream (BSD-3-Clause), SHA256SUMS
unitree/URDF/g1_29dof_pingpang/              generated racket-equipped URDF + MJCF (+ blade meshes)
unitree/scripts/build_g1_pingpang_models.py  the racket mount, single source of truth
hope_training/config/joint_order_unitree_g1.yaml                     canonical 29-joint order
hope_training/whole_body_tracking/scripts/prepare_g1_isaac_asset.py  Isaac asset prep
hope_training/whole_body_tracking/scripts/check_g1_asset_isaac.py    joint-order pinning
hope_training/.../robots/g1.py                                       articulation + name constants
hope_training/.../tasks/tracking/config/g1/hope_env_cfg.py           the two G1 HOPE env configs
hope_training/.../utils/g1_retarget.py                               A3 -> G1 name retarget
hope_training/whole_body_tracking/cfg/task/HOPEPingPongUnitreeG1.yaml training recipe
a3_deploy/a3_deploy_example/config/hope_pingpong_runtime_unitree_g1.yaml   MuJoCo runtime
a3_deploy/a3_deploy_example/config/action_adapter_unitree_g1.yaml          shared action adapter
a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/robot_profile.py       RobotProfile (A3 default, G1)
```

## 2. The racket mount (modelling assumption)

The racket keeps the A3 link names (`right_hand_pingpang_Link`,
`pingpang_red_Link`, `pingpang_black_Link`) and the A3 convention **racket-local
+Y = red / forehand face normal**, so `RacketTargetCommandCfg` defaults,
contact-exclusion regexes and motion sidecars carry over. The blade centre is
at `(0.210, 0.015, 0.010)` m in the `right_wrist_yaw_link` frame
(`G1_MOUNT_OFFSET` in `robots/g1.py`, asserted equal to the generated URDF by a
test).

No physical G1 racket adapter has been measured. The mount reproduces the A3's
21 cm wrist-to-blade reach with the handle through the rubber fist. Before any
hardware work: measure the adapter, edit the constants in
`unitree/scripts/build_g1_pingpang_models.py`, regenerate, and retune the
racket-target sampling boxes in `HOPEPingPongUnitreeG1.yaml`.

```bash
python3 unitree/scripts/build_g1_pingpang_models.py            # regenerate URDF + MJCF
python3 unitree/scripts/build_g1_pingpang_models.py --check    # committed == regenerated
python3 -m pytest -q unitree/tests
```

## 3. Isaac Lab

All commands inside the `grasping` Distrobox
([DISTROBOX_SETUP.md](DISTROBOX_SETUP.md)), from `hope_training/whole_body_tracking/`.

### 3.1 Prepare the asset

```bash
hope_isaac_py scripts/prepare_g1_isaac_asset.py --force
hope_isaac_py scripts/prepare_g1_isaac_asset.py --check
```

Writes the git-ignored `assets/unitree_g1/{urdf/model.urdf,meshes/}` (37 meshes,
flat) exactly like the A3 prep does for `assets/agibot_a3/`.

### 3.2 Pin the joint order (once)

```bash
hope_isaac_py scripts/check_g1_asset_isaac.py
```

Spawns the articulation in a bare scene (no clip needed), prints
`robot.joint_names`, and fails if they differ from
`hope_training/config/joint_order_unitree_g1.yaml`. The YAML is seeded with the
Isaac/PhysX **breadth-first** enumeration (the order the A3 motion sidecars
also record), which is *not* the Unitree SDK / MJCF actuator order. If Isaac
enumerates differently on your build, update the YAML **and**
`a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/joint_order_unitree_g1.py`
together (`tests/test_policy_contract_g1.py` pins them equal), regenerate the
derived G1 clips (`scripts/retarget_a3_clip_to_g1.py`; their `joint_pos`
columns follow the YAML) and re-run. `scripts/export_onnx.py --robot unitree_g1`
hard-gates on this equality.

### 3.3 Reference motions

The published `hope_forehand.npz` / `hope_backhand.npz` are retargeted to the
**A3** (31 joints, 32 bodies); the motion loader rejects them on the G1 with a
message naming both joint counts. GMR cannot be re-run either: it retargets the
human SMPL-X source motion, which is not published.

What ships instead is `g1_hope_forehand.npz` / `g1_hope_backhand.npz` (+ YAML
sidecars) in `hope_training/motions/preprocessed/`, produced by

```bash
python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py          # host: mujoco, numpy, pyyaml
python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py --check
```

a **joint-space transfer** of the A3 pair: the two robots share joint names,
hinge axes and zero conventions, so every G1 joint takes the same-named A3
angle (head dropped, clamped to the G1 limits — no clamping was needed), the
root orientation is copied, the root displacement is scaled by the stand-height
ratio and offset so the lowest sole rests on the floor, and the 14 tracked-body
poses and all velocities are recomputed by MuJoCo forward kinematics on the G1.
The compact 14-body schema is used; the phase structure (frames, strike frame)
is unchanged, so the task YAML's `strike_phase_per_clip` still applies.

**What this does not give you:** the G1 is about 30 % shorter with different
limb ratios, so the A3's paddle contact point, reach and stride geometry are
not preserved — only the joint-space swing style. That is adequate as an
imitation prior (the racket-target command, not the clip, defines where the
paddle must be), not as a faithful retarget. For a faithful one, drive a
keypoint/IK retarget (e.g. the A3 body trajectories as scaled keypoints, or a
GMR run from re-recorded human motion with GMR's G1 target) and emit the same
schema with the sidecar fields `joint_order` = the Isaac order pinned in 3.2,
`anchor_body: torso_link`, `root_body: pelvis`,
`racket_link: right_wrist_yaw_link`, `racket_body: pingpang_red_Link`.

### 3.4 Train / evaluate / export

On the Slurm cluster the whole Isaac side runs inside the pinned image
(`~/containers/hope_grasping.sif`); the job scripts live in
[`hope_training/slurm/`](../hope_training/slurm/README.md):

| script | what |
|---|---|
| `g1_isaac_smoke.sbatch` | asset prep, joint-order pin, cfg-name test, Isaac-vs-MuJoCo plant replay, 2-iteration train of each G1 recipe, export + 2-serve MuJoCo eval |
| `g1_train.sbatch` | the full run on `long` (requeue + resume from the newest `*_<run_name>` checkpoint), `logger=wandb` project `hope-g1`, W&B run id `hope-g1-<run_name>-<jobid>`; knobs `HOPE_G1_RUN_NAME`, `HOPE_G1_TASK`, `HOPE_G1_NUM_ENVS` (4096 on the 11 GB 2080 Ti), `HOPE_G1_EXTRA`, `HOPE_G1_VIDEO` |
| `g1_rollout_sidecar.sbatch` | self-rescheduling (default every 2 h): newest checkpoint → `export_onnx.py` → `mujoco_eval_onnx.py --video-out --json-out` → `wandb_log_rollout.py` into the companion run `<run_name>-rollouts` (same project, grouped), x-axis = iteration |

`HOPE_G1_VIDEO=1` (in-Isaac `RecordVideo`) is off by default because Isaac Sim 5.1's RTX
renderer segfaults at startup on NVIDIA driver 610.43 (`librtx.scenedb.plugin.so`, upstream
isaac-sim/IsaacSim#650); the sidecar's MuJoCo-rendered rollouts are the mp4 source until the
image moves to an Isaac Sim that supports the driver.

Two shutdown gotchas, both handled in the shipped scripts: a bare `SimulationContext` script
must `clear_all_callbacks()`/`clear_instance()` before `simulation_app.close()` (otherwise Isaac
Lab's timeline-STOP handler spins forever), and a pytest module that imports
`whole_body_tracking` must launch a headless `AppLauncher` first (`isaaclab.envs` needs
`omni.timeline`).

```bash
hope_isaac_py scripts/train.py task=HOPEPingPongUnitreeG1 headless=true   # num_envs=16 max_iterations=2 for a smoke
# (defaults to the derived g1_hope_* clips; override with motion_file=... motion_file_2=...)
hope_isaac_py scripts/evaluate.py --task HOPE-HitterPingPong-UnitreeG1-v0 \
  --experiment-name unitree_g1_hitter_pingpong --checkpoint <run>/model_<iter>.pt
hope_isaac_py scripts/export_onnx.py --robot unitree_g1 --task HOPE-HitterPingPong-UnitreeG1-v0 \
  --experiment-name unitree_g1_hitter_pingpong --checkpoint <run>/model_<iter>.pt \
  --motion-file hope_training/motions/preprocessed/g1_hope_forehand.npz \
  --motion-file-2 hope_training/motions/preprocessed/g1_hope_backhand.npz
```

The export embeds `joint_order` = the 29 G1 names and `contract =
hitter_pure_unitree_g1` (104-D) in the ONNX metadata and records
`action_adapter_unitree_g1.yaml` in the manifest, so a G1 policy can never be
loaded by an A3 runner and vice versa.

`HOPEPingPongUnitreeG1.yaml` is the A3 `HOPEPingPong.yaml` recipe with the
robot swapped (marked `G1:` inline): no passive-head reward, generic
actuator-gain DR instead of the A3 message/passive kd cohort, G1 contact body
names, and the sampling boxes / reward stds copied unchanged from the A3 reach
(retune for the G1). `config/g1/hope_env_cfg.py` reuses the A3 configclass
chain verbatim and retargets every body name mechanically
(`utils/g1_retarget.py`), failing loudly if any A3 name survives.

## 4. MuJoCo sim-to-sim

Host Python with `mujoco`, `onnxruntime`, `numpy`, `pyyaml`
(`a3_deploy/a3_deploy_example/reference/requirements.txt`). From the repo root:

```bash
MUJOCO_GL=egl python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 \
  --onnx <exported>/policy.onnx --num-serves 50 --json-out g1_eval.json \
  --video-out g1_eval.mp4          # optional offscreen mp4 (imageio + imageio-ffmpeg)
python -m a3_deploy_onnx_ref_pingpong --robot unitree_g1 --onnx <exported>/policy.onnx --view --realtime
```

`--robot unitree_g1` selects `hope_pingpong_runtime_unitree_g1.yaml`
(`policy.joint_order: unitree_g1` is load-bearing), the `UNITREE_G1` profile
(29 joints, 104-D, no passive columns, MJCF names), the generated G1 MJCF and
per-joint training-plant PD gains. The robot is never inferred from a joint
count; a G1 runtime YAML loaded with `--robot agibot_a3` is rejected. The
example serve bands start equal to the A3's; override with
`--serve-y-fh/--serve-y-bh/--serve-z LO HI` or edit `SERVE_RANGES`.

## 5. Tests

Host (no Isaac):

```bash
python3 -m pytest -q unitree/tests \
  hope_training/whole_body_tracking/tests/test_prepare_g1_isaac_asset.py \
  hope_training/whole_body_tracking/tests/test_g1_retarget.py \
  hope_training/whole_body_tracking/tests/test_robot_profile.py \
  hope_training/whole_body_tracking/tests/test_policy_contract_g1.py \
  hope_training/whole_body_tracking/tests/test_action_adapter_parity_g1.py \
  hope_training/whole_body_tracking/tests/test_passive_head_feedback_g1.py \
  hope_training/whole_body_tracking/tests/test_g1_motion_assets.py
```

Isaac (`grasping`): `hope_isaac_py -m pytest -q tests/test_g1_hope_env_cfg_names.py`
instantiates both G1 env configs and asserts no A3 name survives; then run
`scripts/check_g1_asset_isaac.py` (3.2).
