# Extending HOPE

Everything shipped here — motions, task recipe, reward, observation contract, side selection,
physics, policy — is documented and meant to be replaced or extended. This page lists the extension
points and the exact files behind each.

## Bring your own motions

Replace the two shipped Build clips with your own forehand/backhand motions. The `.npz` + `.yaml`
schema and the CLI to point training at them are in
[REPLACE_MOTIONS.md](REPLACE_MOTIONS.md).

## Compose your own task

Tasks are Hydra YAMLs under `hope_training/whole_body_tracking/cfg/task/`, layered with a
`defaults:` chain — the shipped `HOPEPingPong.yaml` composes the shared `cfg/base/*` defaults
and states the full recipe. To experiment, add a new YAML that inherits `HOPEPingPong` (start
it with `defaults: [HOPEPingPong, _self_]`) and override fields (reward weights, DR ranges,
question-bank mix, `gym_task` id). **The task YAML overrides the Python env-cfg** — when a
value exists in both, edit the YAML. The base dataclasses live in
`hope_training/.../tasks/tracking/config/agibot_a3/hope_env_cfg.py`.

## Bring your own observation contract

Actor observation layouts are a registry, not scattered constants:
`hope_training/.../tasks/tracking/actor_observation_contract.py` defines the named contracts
(`full`=180, `deploy_parity`=175, `hitter_footwork`=177, `hitter_pure`=110, `hitter_pure_v15`=118)
term by term, with each term's deploy-side data source. A task selects one via its
`actor_obs_contract` key, the exporter stamps the contract name into the ONNX metadata, and the
deploy loader fails closed on a mismatch — so a new contract means: add it to the registry, build
its terms in `tasks/tracking/mdp/hope_observations.py`, reference it from your task YAML, and teach
the runner to assemble it. See [POLICY_INTERFACE.md](POLICY_INTERFACE.md).

## Bring your own reward

- Term functions: `hope_training/.../tasks/tracking/mdp/hope_rewards.py`
- Weights/wiring: the task YAML (see above) on top of the `RewardsCfg` in
  `hope_training/.../tasks/tracking/config/agibot_a3/hope_env_cfg.py`

Edit weights, add or remove terms, or write new term functions. The evaluation story
([TRAIN_POLICY.md](TRAIN_POLICY.md#evaluation) — Isaac evaluation, MuJoCo sim-to-sim,
closed-loop rehearsal) is independent of the reward, so you can reshape the reward freely without
changing how success is measured.

## Bring your own action realization

The policy emits a 31-D raw action, realized as `q_des = default_q + raw_action * action_scale`
plus a deterministic joint clamp:

- Training side: `hope_training/.../tasks/tracking/mdp/hope_actions.py`
  (`ClampedJointPositionAction` and the V11+ safe-clamp variants).
- Deploy side: the constants (default pose, scale, clamps, PD gains) are **exported with the
  policy** — `policy.deploy.json` / `params/deploy.yaml` plus fail-closed ONNX metadata — and the
  C++ runner validates them at load. Change the training values and re-export; do not hand-edit
  the deploy side.

See [POLICY_INTERFACE.md](POLICY_INTERFACE.md#action-31-dims).

## Bring your own side selector

Forehand/backhand is chosen **inside the planner** from where the predicted ball crosses the hit
plane, with hysteresis so alternating rallies don't flap:

- Parameters: `swing_side_split_y` and `swing_side_hysteresis_y`, declared in
  `hope_ws/src/hope_planner_cpp/src/planner_node.cpp` and configured in
  `hope_ws/src/hope_planner_cpp/config/model21800_hardware.yaml`.
- Logic: the authoritative implementation is the C++ Planner; the retired
  Python source may be used only as an offline comparison oracle.

Replace the lateral split with your own rule. The chosen side travels on the wire as `swing_sign`
in `/racket/command_flat` ([PLANNER_INTERFACE.md](PLANNER_INTERFACE.md)) and drives the runner's
engage machine — the policy observation itself has **no side term**, so no training change is
needed to alter side selection.

## Bring your own ball physics

The no-spin ball model is a small set of configs fitted from real data:

- `configs/ball_physics.yaml` — the generic model (read by training, planner, and eval).
- `configs/ball_physics_venue.yaml` — the real venue fit (e.g. `drag_k = 0.1261`).
- `configs/incoming_ball_venue.yaml` — the measured serve envelope for that venue.
- Fitting code: `hope_training/ball_physics_fit/` (stage 1/2 fits plus `falsify/` checks). It fits
  your own captured trajectory CSVs — capture with `hope_bag_to_csv`, then re-fit and copy the
  constants into the configs; all consumers pick them up.

The published command stays no-spin — `[x, y, z, vx, vy, vz]` only (the C++ planner's optional
spin *shadow* estimator is diagnostics-only).

## Bring your own policy

Any actor that honors the contract — `observation[110] -> raw_action[31]`, 50 Hz, no observation
normalization ([POLICY_INTERFACE.md](POLICY_INTERFACE.md)) — is a drop-in, **provided the ONNX
carries the fail-closed deploy metadata** the loader checks. Retrain with your changes and
re-export with the shipped exporters; the runner and evaluators depend only on the contract, not on
how the policy was produced.

## Bring your own robot

The stack targets the Agibot A3 (31 DOF); the Unitree G1 (29 DOF) is the worked example of a
second robot ([UNITREE_G1.md](UNITREE_G1.md)). The places a robot diverges, with the G1 file next
to each:

| Concern | A3 | G1 (worked example) |
| --- | --- | --- |
| Description + racket | `agibot/URDF/A3T2.5-URDF-std-pingpang/` (vendor, racket included) | `unitree/URDF/g1_description/` (verbatim) + `unitree/scripts/build_g1_pingpang_models.py` generating `g1_29dof_pingpang` with the A3 racket link names and `+Y` face normal |
| Isaac asset prep | `scripts/prepare_a3_isaac_asset.py` | `scripts/prepare_g1_isaac_asset.py` |
| Articulation + name constants | `robots/agibot_a3.py` | `robots/g1.py` (`G1_*` mirrors of the `A3_*` constants) |
| Canonical joint order | `hope_training/config/joint_order_agibot_a3.yaml` | `hope_training/config/joint_order_unitree_g1.yaml`, pinned once with `scripts/check_g1_asset_isaac.py` |
| Env configs | `config/agibot_a3/hope_env_cfg.py` | `config/g1/hope_env_cfg.py`: subclasses the A3 configs and retargets every body name mechanically (`utils/g1_retarget.py`), no passive joints, generic PD DR |
| Actor contract | `hitter_pure` (110-D) | `hitter_pure_contract(num_joints, name=...)` factory → `hitter_pure_unitree_g1` (104-D) |
| Shared action adapter | `a3_deploy/.../config/action_adapter.yaml` | `action_adapter_unitree_g1.yaml`; both loaders are keyed by `robot=` |
| Export / eval | `scripts/export_onnx.py`, `scripts/mujoco_eval_onnx.py` | the same scripts with `--robot unitree_g1` |
| Deploy reference runner | `RobotProfile AGIBOT_A3` (default) | `RobotProfile UNITREE_G1`; the robot is selected explicitly, never inferred from a joint count |
| Reference motions | published A3 pair | must be retargeted to the new robot (the loader rejects a mismatched joint count) |

Keep the A3 racket link names and face convention on a new robot and the racket command, contact
exclusions and motion sidecars carry over unchanged.
