# Unitree G1 description and the HOPE racket-equipped variant

This folder vendors the Unitree G1 (29-DoF, `rev_1_0`) robot description and
adds the HOPE racket-equipped variant that the Isaac Lab training path and the
MuJoCo sim-to-sim evaluation path load. It mirrors the role `agibot/URDF/`
plays for the Agibot A3. Nothing here is a deploy runner: the G1 has **no
hardware runner** in this repository (see [docs/UNITREE_G1.md](../docs/UNITREE_G1.md)).

## Folder map

| Path | Contents | Origin |
| --- | --- | --- |
| [`URDF/g1_description/`](URDF/g1_description/) | `LICENSE`, `urdf/g1_29dof_rev_1_0.urdf`, `mjcf/g1_29dof_rev_1_0.xml`, the 35 referenced STL meshes, `SHA256SUMS`. | **Verbatim** subset of upstream `unitreerobotics/unitree_ros`, `robots/g1_description/` (BSD-3-Clause). Never edit. |
| [`URDF/g1_29dof_pingpang/`](URDF/g1_29dof_pingpang/) | `urdf/g1_29dof_pingpang.urdf`, `mjcf/g1_29dof_pingpang.xml`, the two blade meshes. | **Generated** by `scripts/build_g1_pingpang_models.py` from the upstream files (HOPE, Apache-2.0). The blade meshes are copied from `agibot/URDF/A3T2.5-URDF-std-pingpang/meshes/`. |
| [`scripts/build_g1_pingpang_models.py`](scripts/build_g1_pingpang_models.py) | The single source of truth for the racket mount; regenerates both variant files. | HOPE |
| [`scripts/g1_actuation.py`](scripts/g1_actuation.py) | Pure-Python mirror of `robots/g1.py` `G1_CYLINDER_CFG` (motor armature, PD gains, effort limits, init pose) used by every host-side generator and test. | HOPE |
| [`scripts/generate_g1_runtime_yamls.py`](scripts/generate_g1_runtime_yamls.py) | Generates (`--check`) the MuJoCo runtime YAML (per-joint training gains) and the action-adapter YAML under `a3_deploy/a3_deploy_example/config/`. | HOPE |
| [`scripts/sync_g1_variant_yaml.py`](scripts/sync_g1_variant_yaml.py) | Generates (`--check`) the variant B task YAML from the variant A recipe. | HOPE |
| [`tests/`](tests/) | Checksum, regeneration, structure and MuJoCo load tests. | HOPE |

## Upstream provenance

| Field | Value |
| --- | --- |
| Repository | <https://github.com/unitreerobotics/unitree_ros> |
| Path | `robots/g1_description/` |
| Commit | `5994d4faef0a9cadd3287f8de0199a67eeb2a259` (fetched 2026-10-07) |
| Variant | `g1_29dof_rev_1_0` (29 actuated joints: 6 per leg, 3 waist, 7 per arm; 4010 wrist motors; `right_rubber_hand` / `left_rubber_hand` end effectors) |
| License | BSD-3-Clause, retained in `URDF/g1_description/LICENSE` and listed in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) |

Only the files referenced by that variant were vendored (the URDF, its MJCF
export, the 35 STL meshes they reference, and the license). The
`SHA256SUMS` file records the checksum of every vendored upstream file;
`tests/test_g1_pingpang_models.py` verifies them so the "verbatim" claim stays
true. To re-fetch, use the raw URL form

```text
https://raw.githubusercontent.com/unitreerobotics/unitree_ros/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/<path>
```

## The racket-equipped variant

`g1_29dof_pingpang` is the upstream model plus a paddle on the right wrist,
named and oriented exactly like the Agibot A3 ping-pong asset so every HOPE
racket default carries over unchanged:

```text
right_wrist_yaw_link
  └─ right_hand_pingpang_joint (fixed, xyz 0)        → right_hand_pingpang_Link  (handle capsule, racket inertial)
       ├─ pingpang_red_joint   (fixed, mount xyz)    → pingpang_red_Link         (red / forehand face mesh)
       └─ pingpang_black_joint (fixed, mount xyz)    → pingpang_black_Link       (black / backhand face mesh)
```

* Blade centre (`G1_RACKET_MOUNT_XYZ`): **(0.210, 0.015, 0.010) m** in the
  `right_wrist_yaw_link` frame, rotation `(0, 0, 0)`.
* Blade plane = racket-local XZ; **racket-local +Y is the red (forehand) face
  normal** (`mount_normal_axis = 1`, identity `mount_quat`).
* The MJCF adds the `pelvis_free_joint` free joint name, a `right_racket` site
  at the blade centre, `right_racket_collision` (blade cylinder),
  `right_racket_handle_collision`, the diagnostic
  `right_racket_clearance_proxy` box, a `pelvis_imu_gyro` sensor, torque
  `<motor>` actuators with `ctrlrange` = ± the **training recipe's** effort
  limit (`g1_actuation.py`; 50 N·m at the ankles / waist roll-pitch where the
  vendor URDF says 35), per-joint `armature` equal to the Isaac implicit drive
  (no implicit damping / friction), and a grounded `stand` keyframe (key 0) equal
  to the Isaac init pose. These are the names
  `hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py` and the
  reference runner resolve.

**Modelling assumption.** No physical G1 racket adapter has been measured.
The mount reproduces the A3's 21 cm wrist-to-blade reach and runs the handle
through the centre of the G1 rubber fist (the blade's near edge therefore
overlaps the distal part of the fist mesh visually). Re-measure the mount
against the real adapter before any hardware work, update the constants in
`scripts/build_g1_pingpang_models.py`, regenerate, and retune the
racket-target sampling ranges in the G1 task configs.

## Regenerate / verify

From the repository root (plain Python 3, no third-party packages):

```bash
python3 unitree/scripts/build_g1_pingpang_models.py            # rewrite the committed variant files
python3 unitree/scripts/build_g1_pingpang_models.py --check    # committed == fresh generation ?
python3 -m pytest -q unitree/tests                              # checksums, structure, MuJoCo load (if installed)
```

Consumers:

* Isaac Lab: `hope_training/whole_body_tracking/scripts/prepare_g1_isaac_asset.py`
  copies this variant into the git-ignored training asset directory
  (`assets/unitree_g1/`), the same way the A3 is prepared.
* MuJoCo: `a3_deploy/a3_deploy_example/config/hope_pingpong_runtime_unitree_g1.yaml`
  points at `URDF/g1_29dof_pingpang/mjcf/g1_29dof_pingpang.xml`.

See [docs/UNITREE_G1.md](../docs/UNITREE_G1.md) for the end-to-end G1 workflow.
