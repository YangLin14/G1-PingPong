# g1_29dof_pingpang (generated)

Racket-equipped Unitree G1. **Do not edit `urdf/` or `mjcf/` by hand**: both
files are written by `unitree/scripts/build_g1_pingpang_models.py` from the
verbatim upstream description in `../g1_description/`, and
`unitree/tests/test_g1_pingpang_models.py` fails when the committed files
differ from a fresh generation.

```bash
python3 unitree/scripts/build_g1_pingpang_models.py            # regenerate
python3 unitree/scripts/build_g1_pingpang_models.py --check    # verify
```

## Mount geometry (right_wrist_yaw_link frame, metres)

| Item | Value |
| --- | --- |
| Blade centre (`pingpang_red_joint` / `pingpang_black_joint` origin, `right_racket` site) | `(0.210, 0.015, 0.010)`, rpy `(0, 0, 0)` |
| Face normal | racket-local **+Y** = red (forehand) face; **−Y** = black (backhand) face |
| Handle capsule | from `(0.060, 0.015, 0.010)` to `(0.145, 0.015, 0.010)`, radius 0.014 |
| Racket mass | 0.18 kg lumped at the blade centre (thin disc, r = 0.08) on `right_hand_pingpang_Link` (URDF) / `right_racket_collision` (MJCF) |
| Blade collision cylinder (MJCF) | centre `(0.206, 0.0084, 0.006)`, radius 0.081, half-thickness 0.007 (the blade STLs are not perfectly centred on their origin) |

Link names (`right_hand_pingpang_Link`, `pingpang_red_Link`,
`pingpang_black_Link`) are the A3's, so `RacketTargetCommandCfg.racket_body_name`,
the contact-exclusion regexes and the motion sidecar `racket_body` field apply
without change. The vendor's `pingbang_ball_Link` calibration sphere is
deliberately **not** reproduced.

## Mesh references

* Body meshes: `../../g1_description/meshes/<file>.STL` (upstream, verbatim).
* Blade meshes: `../meshes/pingpang_red_Link.STL`, `../meshes/pingpang_black_Link.STL`
  (copied from `agibot/URDF/A3T2.5-URDF-std-pingpang/meshes/`).

The MJCF uses `meshdir="../../g1_description/meshes"` and refers to the blade
meshes relative to that directory, so the model loads from a fresh clone with
`mujoco.MjModel.from_xml_path(...)` and no copying.

## MJCF additions over upstream

| Element | Purpose |
| --- | --- |
| `pelvis_free_joint` (renamed from `floating_base_joint`) | first 7 qpos entries = base pose, as the evaluator expects |
| `right_racket` site, `right_racket_collision`, `right_racket_handle_collision`, `right_racket_clearance_proxy` | racket FK / ball contact / clearance diagnostics |
| `pelvis_site`, sensors `pelvis_imu_gyro`, `pelvis_framepos`, `right_racket_framepos`, `right_racket_framequat` | base gyro + racket telemetry |
| `<motor ... ctrllimited="true" ctrlrange="-E E">` | torque actuators clipped at the URDF effort limit; the harness computes the PD torque |
| `<option timestep="0.001">`, `collision` default class | same physics step and contact class as `a3_pingpong.xml` |
| `stand` keyframe (key 0) | grounded stand at the BeyondMimic G1 default joint pose, pelvis z = 0.7585 |
