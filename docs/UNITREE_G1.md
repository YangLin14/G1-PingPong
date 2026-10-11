# Unitree G1 (29-DoF) in HOPE

The Unitree G1 is HOPE's second robot model. It is **additive and selected
explicitly**: the Agibot A3 remains the hardware-validated default line and
nothing in the A3 path changes when the G1 files are present. What ships for
the G1:

| Area | Shipped | Not shipped |
| --- | --- | --- |
| Robot description | Vendored Unitree `g1_29dof_rev_1_0` + generated racket-equipped URDF/MJCF ([`unitree/`](../unitree/README.md)) | — |
| Isaac Lab training | Asset prep, articulation config, `HOPE-PingPong-UnitreeG1-v0` and `HOPE-HitterPingPong-UnitreeG1-v0` tasks, the **recipe v2** task YAMLs `HOPEPingPongUnitreeG1` (variant A, joint-space clips) and `HOPEPingPongUnitreeG1_IK` (variant B, keypoint/IK clips), 104-D `hitter_pure_unitree_g1` contract, exporter `--robot unitree_g1` with an ONNX-vs-torch parity check, two derived G1 clip pairs (see 3.3), Slurm jobs ([`hope_training/slurm/`](../hope_training/slurm/README.md)) | A GMR retarget from re-recorded human motion (the SMPL-X source is not published) |
| MuJoCo sim-to-sim | `mujoco_eval_onnx.py --robot unitree_g1` with per-serve diagnostics, reference runner `--robot unitree_g1`, generated G1 runtime + action-adapter YAMLs (per-joint training gains), `RobotProfile UNITREE_G1`, the Isaac-vs-MuJoCo plant replay gate (see 2b) | — |
| Deploy | — | No G1 hardware runner (the C++ runner and the Gate scripts are A3-only) |

## 1. Where things live

```text
unitree/URDF/g1_description/                 verbatim upstream (BSD-3-Clause), SHA256SUMS
unitree/URDF/g1_29dof_pingpang/              generated racket-equipped URDF + MJCF (+ blade meshes)
unitree/scripts/build_g1_pingpang_models.py  the racket mount, single source of truth (+ armature / torque limits)
unitree/scripts/g1_actuation.py              pure-Python mirror of robots/g1.py gains, armature, limits, init pose
unitree/scripts/generate_g1_runtime_yamls.py generates the MuJoCo runtime + action-adapter YAMLs (--check)
unitree/scripts/sync_g1_variant_yaml.py      generates the variant B task YAML from variant A (--check)
hope_training/config/joint_order_unitree_g1.yaml                     canonical 29-joint order
hope_training/whole_body_tracking/scripts/prepare_g1_isaac_asset.py  Isaac asset prep
hope_training/whole_body_tracking/scripts/check_g1_asset_isaac.py    joint-order pinning
hope_training/.../robots/g1.py                                       articulation + name constants
hope_training/.../tasks/tracking/config/g1/hope_env_cfg.py           the two G1 HOPE env configs
hope_training/.../tasks/tracking/config/g1/g1_recipe.py              recipe v2: the A3 numbers rescaled to the G1
hope_training/whole_body_tracking/scripts/sim2sim_replay_g1.py       Isaac-vs-MuJoCo plant replay gate
hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py     joint-space clip transfer (variant A clips)
hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py  keypoint/IK clip retarget (variant B clips)
hope_training/.../utils/g1_retarget.py                               A3 -> G1 name retarget
hope_training/whole_body_tracking/cfg/task/HOPEPingPongUnitreeG1.yaml    training recipe v2, variant A
hope_training/whole_body_tracking/cfg/task/HOPEPingPongUnitreeG1_IK.yaml variant B (generated; IK clips)
hope_training/slurm/                                                  smoke / train / rollout-sidecar Slurm jobs
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

## 2b. Plant parity: the MuJoCo G1 is the Isaac G1

The first G1 training run (Slurm job 4047, 24.5k iterations, W&B `triton-pupper/hope-g1`)
scored 0 % in every MuJoCo rollout while Isaac reported a 0.3 % fall rate: the robot fell over
in the first second of each rollout. Three things keep that from happening again:

* **One actuation table.** `unitree/scripts/g1_actuation.py` mirrors `robots/g1.py`
  `G1_CYLINDER_CFG` (motor armature, 10 Hz / ζ = 2 PD gains, the recipe's effort limits, the init
  pose). The MJCF generator writes the per-joint `armature` (the upstream MJCF has none; the
  Isaac implicit drive has it) with zero implicit damping/friction, and `ctrlrange` /
  `actuatorfrcrange` = the **training recipe's** effort limits (50 N·m at the ankles and waist
  roll/pitch, where the vendor URDF says 35). `generate_g1_runtime_yamls.py` writes the runtime
  YAML's **per-joint** PD gains and the action adapter from the same table (the leg group alone
  spans 28.5–99.1 N·m/rad; group-averaged gains topple the robot). The Isaac-side test
  `tests/test_g1_hope_env_cfg_names.py` pins the table to the articulation, the host test
  `tests/test_g1_mujoco_plant.py` pins the MJCF and YAMLs to the table.
* **The replay gate.** `scripts/sim2sim_replay_g1.py` drives one scripted joint-target sequence
  (2 s hold, 4 s of slow arm/waist/squat sinusoids) through a bare Isaac scene (`--isaac`, in the
  container) and through the evaluator's MuJoCo scene (`--mujoco --compare`, host) and compares
  joint and pelvis trajectories. Measured (job 4193 / smoke 4205): RMS joint error 0.023 rad and
  pelvis difference 0.047 m over the first second (0.073 rad without armature), fall onset 1.34 s
  vs 1.40 s. Yes, fall: a *static* PD hold with the recipe's ankle gains sags and topples in
  **both** simulators — the policy balances actively — which is why the gate is evaluated over the
  pre-fall window and on the fall onset.
* **Export parity.** `scripts/export_onnx.py` runs the exported graph on real observations from
  the env and compares with the torch actor after every export (`--skip-verify` to disable).

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

### 3.3 Reference motions (two derived pairs)

The published `hope_forehand.npz` / `hope_backhand.npz` are retargeted to the
**A3** (31 joints, 32 bodies); the motion loader rejects them on the G1 with a
message naming both joint counts. GMR cannot be re-run either: it retargets the
human SMPL-X source motion, which is not published. Two G1 pairs are derived
from the A3 pair instead (both in `hope_training/motions/preprocessed/`, compact
14-body schema, phase structure unchanged):

**Variant A — `g1_hope_*` (joint-space transfer)**, produced by

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
not preserved — only the joint-space swing style (adequate as an imitation
prior: the racket-target command, not the clip, defines where the paddle must be).

**Variant B — `g1ik_hope_*` (keypoint/IK retarget)**, produced by

```bash
python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py          # host: mujoco, numpy, pyyaml
python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py --check
```

keeps the A3 arm **shape** instead: per frame, the A3 elbow / wrist /
racket-centre positions and blade plane (torso frame, scaled by the measured
G1/A3 segment ratios 0.673 upper arm and 0.782 forearm about the G1's own
shoulder) are tracked by damped least-squares IK on each 7-joint G1 arm
(MuJoCo Jacobians, URDF limits, two warm starts per frame, face sign free so
the black-face backhand solves); legs, waist and root come from the joint-space
transfer. Tracking error is recorded in the sidecars (`ik_mean_errors_m`,
`ik_strike_frame_errors`): forehand 5.5 cm mean racket error / 5° plane,
backhand 4.0 cm / 6° (8.4 cm and 4.8 cm at the strike frames; the two robots'
blade-in-wrist geometry differs, so the centre cannot be matched exactly while
the arm shape is kept). `tests/test_g1_ik_retarget.py` reproduces the
strike-frame error by FK and bounds it. A GMR run from re-recorded human motion
with GMR's G1 target would be the next step up; emit the same schema with
`joint_order` = the Isaac order pinned in 3.2, `anchor_body: torso_link`,
`root_body: pelvis`, `racket_link: right_wrist_yaw_link`,
`racket_body: pingpang_red_Link`.

### 3.4 Train / evaluate / export

On the Slurm cluster the whole Isaac side runs inside the pinned image
(`~/containers/hope_grasping.sif`); the job scripts live in
[`hope_training/slurm/`](../hope_training/slurm/README.md):

| script | what |
|---|---|
| `g1_isaac_smoke.sbatch` | asset prep, joint-order pin, cfg-name test, Isaac-vs-MuJoCo plant replay, 2-iteration train of each G1 recipe, export + 2-serve MuJoCo eval |
| `g1_train.sbatch` | the full run on `long` (requeue + resume from the newest `*_<run_name>` checkpoint), `logger=wandb` project `hope-g1`, W&B run id `hope-g1-<run_name>-<jobid>`; knobs `HOPE_G1_RUN_NAME`, `HOPE_G1_TASK`, `HOPE_G1_EXPERIMENT`, `HOPE_G1_MOTION_FH`/`_BH`, `HOPE_G1_NUM_ENVS` (4096 on the 11 GB 2080 Ti), `HOPE_G1_EXTRA`, `HOPE_G1_VIDEO`; the same job runs the original A3 recipe as the reference line (`hope_training/slurm/README.md`) |
| `g1_rollout_sidecar.sbatch` | self-rescheduling (default every 2 h): newest checkpoint → `export_onnx.py` → `mujoco_eval_onnx.py --video-out --json-out` → `wandb_log_rollout.py` into the companion run `<run_name>-rollouts` (same project, grouped), x-axis = iteration; `HOPE_G1_ROBOT=agibot_a3` for the A3 reference run |

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

#### Recipe v2 (what differs from the A3 recipe, and why)

`HOPEPingPongUnitreeG1.yaml` is the A3 `HOPEPingPong.yaml` recipe with the
robot swapped (marked `G1:` inline): no passive-head reward, generic
actuator-gain DR instead of the A3 message/passive kd cohort, G1 contact body
names. `config/g1/hope_env_cfg.py` reuses the A3 configclass chain verbatim,
retargets every body name mechanically (`utils/g1_retarget.py`, failing loudly
if any A3 name survives) and then applies `config/g1/g1_recipe.py`.

That module exists because the first run (job 4047) proved the A3 numbers
unreachable for the G1: the strike matrix showed a systematic −9 cm x / −10 cm z
racket position error at high targets, and `rally_ready_root_height_debt`
demanded a 0.98 m pelvis from a robot whose pelvis stands at 0.76 m (a constant
−1.2/episode debt, the run's largest negative term). The rule, from the two
URDFs (`tests/test_g1_recipe_scaling.py` re-derives it): horizontal reach ×0.79
(shoulder-to-blade 0.62 m vs 0.786 m), heights shoulder-relative (shoulders at
1.05 m vs 1.36 m), pelvis-height terms × 0.76/1.068, stance × 0.87 (hip spacing
0.233 m vs 0.269 m). Everything world-scale or robot-agnostic (racket velocity
boxes, strike window and phases, station steps, DR, PPO) stays the A3 value.

| knob | A3 | G1 v2 |
|---|---|---|
| strike plane x (`racket_pos_range_per_clip`) | 0.58 | 0.46 |
| forehand / backhand y bands | (−0.48, −0.40) / (−0.13, −0.05) | (−0.38, −0.32) / (−0.10, −0.04) |
| strike z box | (0.85, 1.30) | (0.80, 1.10) |
| venue tuple contact z (`venue_tuple_contact_z_range`, new cfg field) | (0.98, 1.26) | (0.84, 1.10) |
| `rally_ready_root_height_debt_min_height` | 0.98 | 0.70 |
| `terminations.base_too_low.minimum_height` | 0.5 | 0.40 |
| `ready_stance_width` / `stance_width` lo, hi | 0.25, 0.35 | 0.22, 0.30 |
| `hold_ready_reach` | 0.3 | 0.24 |
| runtime `lifecycle.ready_reach_{x,y,z}` | 0.40, 0.20, −0.05 | 0.32, 0.16, −0.04 |

Note that `train.py` applies only the `env` / `motion` / `domain_rand` /
`rewards` / `overrides` sections of a task YAML: the `racket:` block documents
the values the env cfg sets in code, and the scaling test pins both to the
module. `HOPEPingPongUnitreeG1_IK.yaml` (variant B) is generated from variant A
by `unitree/scripts/sync_g1_variant_yaml.py` and differs only in the two clips.

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
example serve bands are the A3's scaled to the G1 reach (`SERVE_RANGES`;
override with `--serve-y-fh/--serve-y-bh/--serve-z LO HI`), and the table
placement / strike plane default to each robot's training recipe
(`TABLE_GEOMETRY`: edge 0.50 m, plane +0.08 m A3 / −0.04 m G1).

The JSON now carries per-serve diagnostics next to `success_rate`:
`contact_rate`, `net_clear_rate`, `on_opponent_rate`, `fall_rate`,
`mean_min_racket_ball_distance_m` and a `serves` list. **Read them before
believing a 0 %**: the published A3 `model_21800` scores 0 returns in this
planner-less harness (the old default placement put the ball 0.28 m closer than
training, and even at the recipe geometry its racket never comes within 0.14 m
of the ball), so `success_rate` from this script is not a pass/fail signal for
either robot. The in-Isaac `scripts/evaluate.py` number and the Gate 3
closed loop are; this harness is for standing, swinging and plant checks.

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
  hope_training/whole_body_tracking/tests/test_g1_motion_assets.py \
  hope_training/whole_body_tracking/tests/test_g1_ik_retarget.py \
  hope_training/whole_body_tracking/tests/test_g1_mujoco_plant.py \
  hope_training/whole_body_tracking/tests/test_g1_recipe_scaling.py
```

Isaac (`grasping`): `hope_isaac_py -m pytest -q tests/test_g1_hope_env_cfg_names.py`
instantiates both G1 env configs, asserts no A3 name survives, that the hitter
cfg carries recipe v2 and that the host actuation table equals the articulation;
then run `scripts/check_g1_asset_isaac.py` (3.2). The Slurm smoke job runs all
of it plus the plant replay, a 2-iteration train of both recipes, an export and
a 2-serve MuJoCo eval.

## 6. Training runs

| run | recipe | clips | job / W&B | status |
|---|---|---|---|---|
| `g1_v1` | A3 numbers verbatim | joint-space | 4047 / `hope-g1-g1_v1-4047` | 24.5k iterations, exact-strike composite 0.60, reach-limited (−9 cm x), 0 % MuJoCo (plant fell); cancelled 2026-10-10 |
| `g1_v2a` | recipe v2 (variant A) | joint-space `g1_hope_*` | see `squeue -n hope-g1-train` / W&B project `hope-g1` | running |
| `g1_v2b` | recipe v2 (variant B) | IK `g1ik_hope_*` | idem | running |

**A3 baseline for comparison (in-Isaac `scripts/evaluate.py`, model_21800, 256 envs x 4000
steps, job 4257):** 5286 strikes, contacted 100 % (racket within 10 cm of the target at the
strike frame), mean contact error 2.7 cm (median 2.6 cm), net-clear 0, success_rate 0.0. The
shipped analytic metric models the return as a no-spin ball leaving with the racket velocity, which
at 1.5-2.6 m/s from 0.1-0.5 m above the table never reaches net height, so its `success_rate` is 0
for the published A3 policy too. Compare G1 checkpoints on `attempts`, `contacted` and
`mean_contact_error_m` (printed by `evaluate.py` since recipe v2), on the exact-strike W&B
metrics, and on the MuJoCo sidecar's contact / fall rates.

Metrics to read on W&B: `Metrics/racket_target/strike_composite_success_exact`
(position ∧ velocity ∧ normal at the exact strike frame; `strike_success` is a
known-broken window-exit value), `exact_strike_pos_success_10cm`,
`Episode_Termination/base_fell_tilt`, `Episode_Reward/rally_ready_root_height_debt`,
and in the companion `<run>-rollouts` run `Eval/mujoco_contact_rate` /
`Eval/mujoco_fall_rate` with the mp4. Adjustment rules while a run trains
(applied as a resume from the newest checkpoint): root-height debt still below
−0.3/episode → lower `min_height` by 0.03; strike-matrix `position_error_signed_x`
below −0.04 m → move the strike plane 0.04 m closer; `velocity_pass` under half
of `position_pass` at velocity stage 2 → scale the velocity boxes ×0.85; fall
rate above 2 % → `post_swing_tilt_debt_weight` −1.0.

## 7. Forehand cross-court line (Triton Droids)

A forehand-only recipe with an apex-strike, cross-court question bank, a club-specific success
metric, a video -> G1 reference pipeline and a container-free cluster setup build on this port:
[`TRITON_G1_FOREHAND.md`](TRITON_G1_FOREHAND.md) ([繁體中文](TRITON_G1_FOREHAND.zh-TW.md)).
