# What changed in the code (branch `g1-forehand-rally`)

[繁體中文](changes.zh-TW.md) · [Docs index](../README.md)

### Training code

| File | Change |
|---|---|
| `tasks/tracking/mdp/hope_commands.py` | New `RacketTargetCommandCfg.active_clip_ids` (clips this recipe trains; empty = both). With a single side: the ability gate checks only that side (otherwise the never-sampled backhand keeps it locked forever), both `fixed_balanced_bank_v1` tuple cohorts are assigned to that side (otherwise `RuntimeError: permanent physical tuple cohort disagrees with MotionCommand clip`), and the bank loader requires rows only for that side. Default behaviour is unchanged. |
| `cfg/task/HOPEPingPongUnitreeG1_FH.yaml` | Forehand-only variant A recipe: `eval_clip_sequence=[0]`, `fixed_clip_env_fraction_per_clip=0` (no permanent backhand cohort), `active_clip_ids=[0]`. Still uses the A3 bank (baseline). |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC.yaml` | `_FH` + the cross-court apex bank at 100 % mix. **Main recipe.** |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC_RALLY.yaml` | `_FH_XC` + the faster-feed bank + rally pace (`rally_pace_prob` 0.6) + ±30 ms timing jitter, ramped over 72 000 env steps (3000 iterations). Recipe of the `g1_fh_rally` run (D14). |
| `tasks/tracking/mdp/commands.py` | `MotionCommand` rally pace (`rally_pace_prob`, `rally_pace_ramp_steps`, `rally_pace_start_tts_range`, `rally_pace_post_strike_range`): when a swing follows a swing, the clip is entered late (time-to-strike 0.4–0.83 s, no hold) and the previous follow-through is cut 0.7–1.0 s after its strike. Metrics `rally_pace_swing`, `rally_pace_prob`. Off by default. |
| `tasks/tracking/mdp/hope_commands.py` (also) | `vb_timing_jitter_s` / `vb_timing_jitter_ramp_steps`: at the strike frame the virtual ball is moved ballistically by τ ~ U(−s, s), so the policy must hit a ball that is slightly early or late. Metric `virtual_timing_jitter_s`. Off by default. |
| `scripts/train.py`, `cfg/train.yaml` | `restore_step_counter` (D15). |
| `utils/forehand_success.py` | The [club rules](../overview.md) success definition (`classify_strikes`, `ForehandSuccessTally` with misses in the denominator, `TARGET_ZONES`, `FEED_ZONE`, feed origin). The tally is a funnel (contact → over the net → on the opponent's half → in the zone); `ZoneSuccessReport` splits it per zone. Isaac-free. |
| `tasks/tracking/mdp/events.py`, `scripts/train.py` | `push_by_setting_velocity_ramped` (Isaac's push with its range ramped on `env.common_step_counter`); task-YAML `domain_rand.push_velocity_range` / `push_interval_s` / `push_ramp_steps` / `push_ramp_start_steps` install `events.push_robot` (absent = unchanged). Tests: `tests/test_domain_rand_overrides.py`. |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL.yaml` | `_FH_XC_RALLY` + shoves (D19). |
| `unitree/scripts/build_g1_pingpang_models.py` | Right fist (D20): generates `unitree/URDF/g1_29dof_pingpang/meshes/right_fist.STL` (`FIST_*` constants) and uses it as the `right_rubber_hand` visual in the URDF and MJCF; `--check` compares it too. |

### Tools (`hope_training/whole_body_tracking/scripts/`)

| File | What |
|---|---|
| `make_g1_forehand_bank.py` | Generates the question bank ([question bank](question_bank.md)). `--vin-x/--vin-y/--vin-z/--spin-rev-s` set the feed range. |
| `evaluate_forehand.py` (+ `cfg/evaluate_forehand.yaml`) | Club success rate per zone in Isaac: deterministic policy, 512 envs × 60 s, zone from each question's intended landing. |
| `mujoco_forehand_bank.py` | Bank questions for MuJoCo: feed start state at the far end line, NumPy venue flight model, closed-loop apex planner that re-solves the racket velocity after the real bounce. |
| `mujoco_eval_onnx.py` (extended) | `--serve-bank` (real ball fed from our bank), `--zones`, `--feed-lead-s`, `--rally-pace` / `--rally-gap-s` (next feed 0.7–1.0 s after our contact, announced only at launch); club scoring per ball and per zone. Default mode unchanged. |
| `mujoco_pingpong_scene.py` (extended) | Opt-in Magnus force, target-zone drawing, ball spin, and the venue table-bounce model (MuJoCo's own bounce put the apex 20–40 cm too deep). |
| `check_rally_pace.py` | Isaac check of the rally-pace clock and timing jitter (swing entry time, strike-to-strike interval, falls). |
| `mujoco_eval_onnx.py` (more) | `--racket-contact venue` (training paddle model instead of MuJoCo's soft contact), `--aim-offset DX DY` (D18), `--trace-out` (per-tick log incl. racket normal, ball spin). |
| `onnx_from_checkpoint.py` | policy.onnx for a checkpoint without Isaac: copies the actor weights into an exported graph of the same task (CPU). Verified identical to a real export. |
| `ball_perception.py`, `perception_sweep.py` | Camera model for `mujoco_eval_onnx.py --perception` (rate, latency, noise, dropout, HITTER-style fit, spin unknown / known / read with error) and the sweep over camera systems. `--reset-after-fall` stands the robot up between balls. |
| `mujoco_eval_onnx.py --push V_XY W_RPY`, `--push-window-s` | Balance test: one shove per ball, a random base velocity change of up to V_XY m/s (x, y) and W_RPY rad/s (roll, pitch, yaw) at a random moment 0–1 s after the feed. Own RNG stream, so a seed gives the same balls with or without pushes. Per-ball record `push`. |
| `view_feed_bank.py` | MuJoCo visualizer of the bank: diagonal feed from the opponent's right → bounce → apex → return, with the feed origin, all zones (current target highlighted) and racket target; interactive or MP4, `--camera behind|top`. |
| `gvhmr_to_g1_headless.py` | GMR's GVHMR → G1 retarget without its viewer (cluster nodes have no display). |
| `gmr_video_clip_to_g1.py` | GMR output → HOPE G1 clip: trim, 30 → 50 Hz, heading/xy normalization, joint reorder + limit clamp, loop-closure blend, MuJoCo FK of the 14 tracked bodies, strike frame (peak racket speed or manual), optional MP4 preview. |

### Assets, jobs, tests

| Path | What |
|---|---|
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank.{npz,receipt.json}` | 50 000-row bank (seed 0, reproducible, 12 500 rows per target zone) + receipt with the full contract and statistics. |
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide.{npz,receipt.json}` | Faster-feed bank for `_FH_XC_RALLY` ([question bank](question_bank.md)). Force-added: `assets/venue_tuple/*` is gitignored. |
| `hope_training/slurm/pip_env/` | Container-free install, smoke, training and generic run jobs for SDSC ([README](../../../hope_training/slurm/pip_env/README.md)). `train.sbatch` adds `INIT_CHECKPOINT` (warm start from another run), a healthy-GPU fallback and `restore_step_counter`; `status.sh`, `progress.sh`, `eval_checkpoint.sh` (MuJoCo videos: examples, bank, rally pace), `eval_forehand.sh` (Isaac per-zone). Usage in the [runbook](../operations/runbook.md). |
| `hope_training/slurm/video_to_g1/` | Video → G1 clip install script and job ([README](../../../hope_training/slurm/video_to_g1/README.md)). |
| `tests/test_forehand_success.py` | 12 tests: the four zones (three aligned at the end line, no overlap), feed zone and origin, an apex ball *can* be returned into the zone (grid search), falling ball / missed racket / soft tap / down-the-line all fail, misses count in the denominator, funnel counts, per-zone report. |
| `tests/test_make_g1_forehand_bank.py` | 4 tests: loader schema; feed/apex/strike-box/speed constraints; the feed comes diagonally from the opponent's right; equal rows per zone and **every row is a success into its own zone**. |
| `tests/test_gmr_video_clip_to_g1.py` | 3 tests: round trip of a published G1 clip through a fake GMR file (joints < 0.03 rad, bodies < 1 cm, strike frame ±1), sidecar schema, loop blend. |
| `tests/test_mujoco_forehand_bank.py` | 3 tests: the MuJoCo-side flight and planner reproduce the bank's apex and racket velocity. |
| `a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/{lifecycle,racket_command,ros_command_source,config,runner}.py` | Opt-in `clock: strike_time` (D17): the observed time-to-strike is `strike_time − now`. Default `countdown` keeps the original behaviour. |
| `tests/test_lifecycle_strike_time.py` | 5 tests: countdown unchanged, strike_time clock exact for fresh and late messages, stamping without a deadline, fallback without `now`, ROS field [10]. |
| `tests/test_ball_perception.py` | 4 tests: noise-free fit and latency, bounce segmentation, presets, per-ball spin-reading error. |
| `docs/g1_forehand/plans/serve_side_backspin.md` | The shelved serve plan (D4). |

