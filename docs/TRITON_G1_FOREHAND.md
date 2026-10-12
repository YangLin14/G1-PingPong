# Triton Droids: Unitree G1 forehand cross-court rally

[繁體中文版](TRITON_G1_FOREHAND.zh-TW.md) · Day-to-day commands: [RUNBOOK_G1_FOREHAND.md](RUNBOOK_G1_FOREHAND.md)

This document records what the Triton Droids team built on top of HOPE and the club's G1 port,
why each decision was made, how the whole system works end to end, and how to run every piece.
Status as of **2026-10-10**: simulation only (the G1 has not arrived); everything is designed so the
sim-to-real path stays open.

---

## 1. Context

| Repository | What it is |
|---|---|
| [`hitchopen/HOPE`](https://github.com/hitchopen/HOPE) | Upstream HOPE: the open humanoid table-tennis platform for the **Agibot A3** (Isaac Lab training, ball planner, MuJoCo sim-to-sim, C++ deploy runner). |
| `YangLin14/G1-PingPong` | Our fork of HOPE (this checkout, `origin`). |
| [`triton-droids/HOPE`](https://github.com/triton-droids/HOPE) branch `g1-port` | The club's G1 port by Oren Gershony: G1 URDF/MJCF with a racket, recipe v2 (A3 numbers rescaled to the G1), two derived G1 clip pairs, MuJoCo plant parity, Slurm jobs. Documented in [`UNITREE_G1.md`](UNITREE_G1.md). |
| `triton-droids/pingpong` | An earlier, separate prototype (DLS IK + PPO residual on a G1, Isaac Lab); not used here. |

This work lives on branch **`g1-forehand-rally`**, branched from `g1-port`.

HOPE's design follows **HITTER** (UC Berkeley, [arXiv 2508.21043](https://arxiv.org/abs/2508.21043)):
a model-based planner predicts the ball and decides *where, when and how fast* the racket must be,
and a reinforcement-learned whole-body controller makes the robot get there while imitating a
human reference swing. HITTER ran exactly this on a **Unitree G1** (96 % hit rate, 92 % returns,
rallies up to 106 shots), using references made by filming a human, lifting the video with
**GVHMR** and retargeting with **GMR**.

---

## 2. Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Build on HOPE/HITTER rather than the hand-crafted `pingpong` prototype. | HITTER is proven on the G1; HOPE gives a complete, tested training/eval/deploy stack. |
| D2 | Start from Oren's `g1-port`. | It already solved the G1 asset, racket mount, joint-order pinning, MuJoCo plant parity and recipe scaling. |
| D3 | Robot = Unitree G1, simulation first, keep sim-to-real in mind. | The robot has not arrived. |
| D4 | A backhand side-backspin **serve** was planned and **shelved**. | Forehand rally is easier and higher priority. Plan kept in [`plans/SERVE_SIDE_BACKSPIN_PLAN.md`](plans/SERVE_SIDE_BACKSPIN_PLAN.md). |
| D5 | Train **forehand only**. | One swing is a smaller learning problem; backhand comes later. |
| D6 | Define success by the club's rally rule (section 3), not HOPE's `success_rate`. | HOPE's analytic `success_rate` never clears the net (0 % even for the published A3 policy). |
| D7 | **Diagonal rally**: the feed is hit from the **opponent's right** to the robot's right; the robot returns from its right to the **opponent's** right. Four landing zones: one open-court and three lined up along the far end line. The feed bounce zone is fixed for now. | Club spec (section 3). Feeding along the same diagonal matches a real cross-court rally. |
| D8 | Replace every training question with a G1-specific **apex-strike, cross-court bank**. | The inherited A3 bank strikes falling balls (median incoming vz −0.77 m/s), aims across the full table width, uses A3 reach (0.58 m, z 0.85–1.30 m), and covers only 25 % of questions. |
| D9 | Make reference motions from **human video** (GVHMR + GMR), as HITTER did. | `g1-port` clips are joint-space copies of the A3 swing; at its strike frame the racket is at ~0.78 m (table height), and the first G1 run was reach-limited (−9 cm x, −10 cm z). |
| D10 | Do not calibrate metrics against the A3 `model_21800`. | It does not work in our setup; metrics are verified with unit tests on known-good/bad racket states instead. |
| D11 | Run Isaac on SDSC from a **pip** install instead of the 64 GB container. | Podman cannot run on the BeeGFS home; the pip route needs no sudo. Node workarounds are scripted (section 8). |
| D13 | **One policy for all landing zones; the zone is chosen by the planner.** Positions are selectable modes (one or several) at run time, not separate policies. | The policy tracks a racket target; the planner/bank turns the chosen landing point into that racket velocity. One policy shares the body skill across zones, costs a quarter of the GPU time, and switches zones between shots without changing controllers. Per-zone evaluation (`target_zone_id`) shows whether a zone needs more weight or fine-tuning. |
| D14 | **Train at real rally timing, in a separate run (`g1_fh_rally`, task `HOPEPingPongUnitreeG1_FH_XC_RALLY`), warm-started from `g1_fh_diag`.** Up to 60 % of follow-up swings enter the clip late (time-to-strike 0.4–0.83 s, no hold) and the previous follow-through ends 0.7–1.0 s after its strike; the virtual ball arrives ±30 ms off the clock; feeds are faster (4.5–8.4 m/s). All ramped in over 3000 iterations. | MuJoCo at real cadence (`--rally-pace`) dropped contact from 88 % to 38 % with falls on 44 % of balls: training gave 0.83 s of lead and ~3 s between strikes, a rally gives 0.4–0.8 s and ~1.5 s. Late entry is what the deploy reference clock does when the planner's time-to-strike is short, so no clip time-warp is needed. Slower feeds are impossible with the middle-third bounce and a fixed strike plane (the apex would be in front of the robot); they need forward footwork. The main run keeps going untouched as the comparison. |
| D12 | A lean HITTER-style reward (upper-body imitation + racket goal + base position + regularization) will be A/B-tested against the 53-term HOPE recipe. | **Pending**: needs the video clip first. |

---

## 3. Task specification (club rules)

All coordinates are the **env frame, robot's view**: robot root at the origin facing **+x** (toward
the table), **+y = robot's left**, z up. Near table edge at x = 0.5 m, table surface at z = 0.76 m,
table centred on y = 0, net at x = 1.87 m, far end at x = 3.24 m.

A forehand return is a **success** only if all hold:

1. **Timing.** The ball is struck at the **apex** after bouncing on the robot's side: incoming
   vertical speed at contact |vz| ≤ 0.3 m/s (about ±30 ms around the top).
2. **Contact.** Racket centre within 9.5 cm of the ball, moving into it faster than 0.3 m/s (the
   training capture gate).
3. **Legal.** The ball clears the net (centre above net top + ball radius) and first lands on the
   opponent's half.
4. **Zone.** That first bounce is inside the target zone.

Diagonal geometry (both directions use the same diagonal):

* **Feed**: hit from the **opponent's right** (the robot's +y half, crossing the far end line 5–45 cm
  above the table), clears the net and bounces in the feed zone on the **robot's right** (−y).
* **Return**: the robot hits from **its right** to the **opponent's right**, which is the robot's
  **left** half of the far table (+y).

| Zone | Depth (fraction of a half, from the net) | Lateral (fraction of half width, +y) | Env-frame box |
|---|---|---|---|
| **Feed origin** (far end line) | at x = 3.24 | +0.25 … +0.95 (opponent's right) | y 0.191–0.724, z 0.81–1.21 |
| **Feed** (incoming bounce, robot's half) | middle third | −0.75 … −0.25 (robot's right) | x 0.957–1.413, y −0.572…−0.191 |
| `opp_right_mid`: open court | middle third | +0.25 … +0.75 | x 2.327–2.783, y 0.191–0.572 |
| `center_line_deep`: beside the centre line (opponent's right side of it), at the end line | 0.78 … 0.98 | +0.03 … +0.23 | x 2.939–3.213, y 0.023–0.175 |
| `opp_right_deep`: middle of the right half, at the end line | 0.78 … 0.98 | +0.25 … +0.75 | x 2.939–3.213, y 0.191–0.572 |
| `opp_right_corner`: end line × side line | 0.78 … 0.98 | 0.78 … 0.98 | x 2.939–3.213, y 0.595–0.747 |

The three edge zones are **lined up along the end line**: they share the back edge band (the last 2–22 % of
the half) and go centre line → middle of the right half → corner. The centre-line zone sits just on the
opponent's-right side of the line: its inner edge is 2.3 cm off the line (more than a ball radius),
so a landing in it never touches the line. A side-line zone at middle depth
(`opp_right_side`) is kept, commented out, in `TARGET_ZONES` for later.
The zones do not overlap. The training bank draws the four targets equally (25 % each). The
policy does not need to be told the target, because the bank solves the racket velocity that sends
the ball to it, and the policy tracks that velocity.

---

## 4. How the system works

```text
                 QUESTION (one per swing)                              POLICY (50 Hz)
  bank row: station_xy, racket contact point,      ──►   104-D obs: proprioception, racket target
  racket velocity (= face normal), incoming ball         pos/vel, time-to-strike, base target ...
  velocity/spin at contact, intended landing             ──► 29-D action ──► joint PD targets (G1)
                                                                    │
                                                 Isaac Lab / PhysX  ▼  (200 Hz physics)
  REWARD (per step)                                       robot moves; reference clock runs
   * track the reference swing clip (BeyondMimic)
   * racket position / velocity / normal at the exact strike frame
   * VIRTUAL BALL at the strike frame: achieved racket state ──► spin contact model ──►
     Magnus/drag flight ──► net clearance + landing vs the row's intended landing point
   * stability / regularization terms
```

* **Reference clip.** A 50 Hz swing (ready → strike → recover) in the G1 Isaac joint order with 14
  tracked bodies; the strike frame is marked in the sidecar YAML. The policy is rewarded for following
  it, which gives a human-like motion prior (BeyondMimic tracking inherited from HOPE).
* **Question.** HOPE's `RacketTargetCommand` gives the policy a racket target at a known
  time-to-strike. In `fixed_balanced_bank_v1` mode the target comes from a precomputed bank.
  `HOPEPingPongUnitreeG1_FH_XC` sets the bank share to 100 %, so every question is a physically
  consistent apex strike that returns cross-court.
* **Virtual ball.** The ball is not simulated as a PhysX body in training. At the exact strike frame
  the venue-fitted model ([`configs/ball_physics_venue.yaml`](../configs/ball_physics_venue.yaml),
  [`virtual_ball.py`](../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/virtual_ball.py))
  computes the return from the **achieved** racket state:
  - flight: `a = g − k_d|v|v + k_m(ω × v)` (k_d = 0.1261 m⁻¹, k_m = 0.00444);
  - racket and table contact: an impulse model with tangential grip (spin generation) and
    speed-dependent normal restitution `e(u_n) = 0.759·exp(−0.0441|u_n|)`;
  - fitted on real venue data, valid to about 7 m/s and 15 rev/s.
* **Same physics everywhere.** The bank generator, the success metric, the training reward and the
  visualizer all call this one module, so they cannot disagree.

---

## 5. What changed (branch `g1-forehand-rally`)

### Training code

| File | Change |
|---|---|
| `tasks/tracking/mdp/hope_commands.py` | New `RacketTargetCommandCfg.active_clip_ids` (clips this recipe trains; empty = both). With a single side: the ability gate checks only that side (otherwise the never-sampled backhand keeps it locked forever), both `fixed_balanced_bank_v1` tuple cohorts are assigned to that side (otherwise `RuntimeError: permanent physical tuple cohort disagrees with MotionCommand clip`), and the bank loader requires rows only for that side. Default behaviour is unchanged. |
| `cfg/task/HOPEPingPongUnitreeG1_FH.yaml` | Forehand-only variant A recipe: `eval_clip_sequence=[0]`, `fixed_clip_env_fraction_per_clip=0` (no permanent backhand cohort), `active_clip_ids=[0]`. Still uses the A3 bank (baseline). |
| `cfg/task/HOPEPingPongUnitreeG1_FH_XC.yaml` | `_FH` + the cross-court apex bank at 100 % mix. **Main recipe.** |
| `utils/forehand_success.py` | The section 3 success definition (`classify_strikes`, `ForehandSuccessTally` with misses in the denominator, `TARGET_ZONES`, `FEED_ZONE`, feed origin). Isaac-free. |

### Tools (`hope_training/whole_body_tracking/scripts/`)

| File | What |
|---|---|
| `make_g1_forehand_bank.py` | Generates the question bank (section 6). |
| `view_feed_bank.py` | MuJoCo visualizer of the bank: diagonal feed from the opponent's right → bounce → apex → return, with the feed origin, all zones (current target highlighted) and racket target; interactive or MP4, `--camera behind|top`. |
| `gvhmr_to_g1_headless.py` | GMR's GVHMR → G1 retarget without its viewer (cluster nodes have no display). |
| `gmr_video_clip_to_g1.py` | GMR output → HOPE G1 clip: trim, 30 → 50 Hz, heading/xy normalization, joint reorder + limit clamp, loop-closure blend, MuJoCo FK of the 14 tracked bodies, strike frame (peak racket speed or manual), optional MP4 preview. |

### Assets, jobs, tests

| Path | What |
|---|---|
| `assets/venue_tuple/g1_fh_crosscourt_apex_bank.{npz,receipt.json}` | 50 000-row bank (seed 0, reproducible, 12 500 rows per target zone) + receipt with the full contract and statistics. |
| `hope_training/slurm/pip_env/` | Container-free install, smoke, training and generic run jobs for SDSC ([README](../hope_training/slurm/pip_env/README.md)). |
| `hope_training/slurm/video_to_g1/` | Video → G1 clip install script and job ([README](../hope_training/slurm/video_to_g1/README.md)). |
| `tests/test_forehand_success.py` | 9 tests: the four zones (three aligned at the end line, no overlap), feed zone and origin, an apex ball *can* be returned into the zone (grid search), falling ball / missed racket / soft tap / down-the-line all fail, misses count in the denominator. |
| `tests/test_make_g1_forehand_bank.py` | 4 tests: loader schema; feed/apex/strike-box/speed constraints; the feed comes diagonally from the opponent's right; equal rows per zone and **every row is a success into its own zone**. |
| `tests/test_gmr_video_clip_to_g1.py` | 3 tests: round trip of a published G1 clip through a fake GMR file (joints < 0.03 rad, bodies < 1 cm, strike frame ±1), sidecar schema, loop blend. |
| `docs/plans/SERVE_SIDE_BACKSPIN_PLAN.md` | The shelved serve plan (D4). |

---

## 6. The question bank

`make_g1_forehand_bank.py` builds each row in four steps:

1. **Feed.** A bounce point is drawn uniformly in the feed zone, with a pre-bounce velocity
   (vx −5.5…−2.5, vy −2.4…−0.3, vz −3.5…−1.5 m/s) and a random spin up to 5 rev/s. The flight is
   integrated **backward** from the bounce, and the row is kept only if it crossed the far end line
   over the opponent's right (y 0.19–0.72 m, 5–45 cm above the table) and cleared the net by 3 cm.
   That makes it a diagonal feed from the opponent's right. The venue table contact gives the
   post-bounce velocity and spin.
2. **Apex.** RK4 flight (2 ms steps) runs to the apex (vz = 0). The row is kept only if the apex is
   inside the G1 forehand strike box: x 0.42–0.50 m in front of the station, z 0.80–1.10 m. The
   station y follows from the recipe's forehand racket offset (−0.38…−0.32 m).
3. **Return.** One of the four target zones is drawn (recorded in `target_zone_id`), then a
   landing point uniformly inside it. The racket velocity (face normal along it, the V14 contract)
   is solved by damped least squares on (landing x, landing y, net clearance − 0.25 m), weakly
   regularized (λ = 1e-3) toward (1.9, 0.3, 1.0) m/s. The deep and edge targets need fast, flat
   swings, which a stronger pull would forbid.
4. **Accept.** The row is kept if the landing error is ≤ 5 cm, net clearance is within
   0.25 ± 0.10 m (and ≥ 3 cm), racket speed ≤ 3.5 m/s, and `classify_strikes` calls it a success
   for its own zone. Generation continues until every zone has its quota (equal rows per zone).

Statistics of the committed bank (50 000 rows, seed 0):

| Quantity | min | mean | p95 | max |
|---|---|---|---|---|
| racket speed (m/s) | 1.15 | 2.73 | 3.38 | 3.50 |
| apex height above table (m) | 0.13 | 0.24 | 0.32 | 0.34 |
| incoming speed at contact (m/s) | 1.99 | 3.03 | 3.51 | 3.78 |
| net clearance (m) | 0.23 | 0.26 | 0.26 | 0.27 |
| landing residual (m) | 0.000 | 0.002 | 0.003 | 0.047 |

12 500 rows per zone, from 4.6 million candidate feeds. Most are rejected because their apex falls
outside the G1 strike box. The deep/edge zones push the racket speed up to the 3.5 m/s limit.

Regenerate (about 15 min on a laptop CPU) with:
`python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`.
Changing the seed or physics changes the rows, so **retrain after regenerating**.

---

## 7. End-to-end pipeline

```text
 phone video ──► GVHMR (SMPL-X) ──► GMR (G1) ──► gmr_video_clip_to_g1 ──► G1 reference clip (.npz/.yaml)  [pending video]
                                                                                   │
 club rules ──► forehand_success.py ──► make_g1_forehand_bank ──► question bank ───┤
                                                                                   ▼
                                     Isaac Lab PPO (task HOPEPingPongUnitreeG1_FH_XC, SDSC, ~45 h)
                                                                                   │ checkpoints
                                                                                   ▼
            evaluation: forehand_success metric (Isaac)  ·  export_onnx --robot unitree_g1
                                                                                   ▼
                          MuJoCo sim-to-sim (mujoco_eval_onnx.py --robot unitree_g1)
                                                                                   ▼
                       hardware: G1 runner + mocap + planner  [not built yet; C++ runner is A3-only]
```

---

## 8. Running it

### Local (macOS)

```bash
cd G1-PingPong
python3 -m venv .venv && .venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
.venv/bin/python -m pytest -q unitree/tests hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py

# Watch the ball feeding. MuJoCo windows on macOS need mjpython. Because this path has a space,
# call mjpython through the venv python (its shebang breaks on the space).
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 30 --speed 0.5
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --camera top
.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4
```

In the viewer: purple is the feed origin (opponent's right, at the far end line), orange the feed
bounce zone, faint green the four target zones and bright green the current question's target. The
yellow trail is the ball; the red disc and arrow are the racket target (face and velocity).

### Cluster (SDSC Atlantis, `ssh sdsc-scc`)

* Account `tritondroids`. The whole club shares **32 CPUs / 96 GB / 8 GPUs**. Isaac needs an RTX
  GPU, so use `--gres=rtx2080ti:1` (node `zixian`, 11 GB, driver 610.43).
* Layout: `~/hope/{HOPE (this repo), isaac-venv, IsaacLab, syslibs, mocap, slurm_logs}`.
* Submit every job **from the repository root**.

```bash
sbatch hope_training/slurm/pip_env/install_isaac.sbatch                    # once
bash   hope_training/slurm/video_to_g1/install_mocap.sh                    # once (+ SMPL/SMPL-X, see its README)
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_xc hope_training/slurm/pip_env/train.sbatch
squeue -u $USER; tail -f hope-g1-train-<jobid>.out
# TensorBoard on the laptop:
ssh -L 6006:localhost:6006 sdsc-scc '~/hope/isaac-venv/bin/tensorboard --port 6006 \
  --logdir ~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong'
```

Watch `strike_composite_success_exact`, `exact_strike_pos_success_10cm`, `virtual_legal_rate_forehand`,
`virtual_landing_error_m_forehand`, `virtual_contact_rate_forehand` and `Episode_Termination/base_fell_tilt`. Do not watch `success_rate`.

Node workarounds (each was a real failure; all are handled by `install_isaac.sbatch` and the
rendered `setup_train_env.local.sh`):

| Symptom | Fix |
|---|---|
| Kit `No device could be created` | `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` |
| Kit hangs after URDF import (`Neuray::LoadModule`) | `libGLU.so.1` extracted from the `mesa-libGLU` RPM onto `LD_LIBRARY_PATH` |
| `PermissionError` on `/tmp/isaaclab`, `/tmp/IsaacLab` | per-user `TMPDIR` + a one-line Isaac Lab patch |
| `isaaclab.sh` needs sudo; `flatdict` build fails | `pip install -e source/*`; prebuild `flatdict` with `setuptools<70` |
| GVHMR: `get_yaml_loader` error; no `ffmpeg` | `omegaconf==2.3.0`; imageio-ffmpeg static binary |
| GMR viewer needs a display | `gvhmr_to_g1_headless.py` |
| MuJoCo EGL picks a broken device | `hope_training/slurm/pick_egl_device.py` |

### Recording the reference video

Static camera at about 45° front-side, whole body and racket in frame, 30–60 fps, a full forehand from
the ready pose back to the ready pose, 5–10 takes. Then follow
[`slurm/video_to_g1/README.md`](../hope_training/slurm/video_to_g1/README.md). SMPL/SMPL-X are
licensed per person and must never be committed (they are in `.git/info/exclude` locally).

---

## 9. Results so far (2026-10-10)

* Environment: Isaac Sim 5.1 + Isaac Lab 0.54.2 boot and train on the 2080 Ti. The G1 smoke passes
  all steps. The Isaac/MuJoCo plant replay matches `g1-port`'s numbers (0.023 rad, 0.047 m, fall
  onset 1.34 s vs 1.40 s).
* Video pipeline: verified on GVHMR's example video (person height 1.73 m, 10 m/s peak racket
  speed); converter round trip within 0.012 rad / 1.6 mm.
* Training runs on `rtx2080ti`, 4096 envs, about 8 s/iteration with two runs sharing a node:

  | Job | Task | Status at last check |
  |---|---|---|
  | 4354 `g1_fh` | `_FH` (A3 bank, baseline) | iter 602: pos-within-10 cm 7.8 %, composite 0, falls 39 % of terminations |
  | 4367 `g1_fh_xc` | `_FH_XC` (**main**) | iter 240: early; no strikes yet |

* HOPE upstream issues found along the way: commit `00e2e96` made the Gate 3 table in
  `a3_pingpong.xml` collide with the robot (it pins the racket arm in the standalone evaluator), and
  the evaluator's default geometry does not match Gate 3 (0.30 m vs 0.50 m to the table edge).
  `g1-port` already moves the evaluator to recipe geometry.

---

## 10. Next steps

1. Record the forehand video, convert it with `video_to_g1`, and inspect the preview.
2. Train `_FH_XC` with the human clip (`motion_file=...`; update `strike_phase_per_clip`).
3. Add the lean HITTER reward recipe and A/B it against the HOPE recipe (D12).
4. ~~Isaac evaluation with `forehand_success`, per zone~~: done (2026-10-11),
   `scripts/evaluate_forehand.py` / `eval_forehand.sh`; results in
   [`TRAINING_STATUS_G1_FOREHAND.md`](TRAINING_STATUS_G1_FOREHAND.md).
5. Zone selection (D13): a `target_zone_ids` filter on the bank for training/eval/play, and the
   same choice exposed in the deploy planner. Then widen
   the feed distribution.
6. Close the real-ball gap found by the MuJoCo bank evaluation
   ([`TRAINING_STATUS_G1_FOREHAND.md`](TRAINING_STATUS_G1_FOREHAND.md) section 4): the policy strikes
   15–40 ms before the apex, MuJoCo's racket contact returns the ball lower than the venue paddle
   model, and the swing clip needs ~0.83 s of lead while a fast feed flies 0.5–0.8 s. Candidates:
   timing noise/delay on the virtual strike in training, a racket-contact model check against MuJoCo,
   a later swing start (or a shorter pre-strike clip).
7. Before hardware: measure the real racket adapter, refit ball physics on our table, and write a
   G1 deploy runner (the C++ runner is A3-only).
7. Then balance and footwork (free station), then backhand, then the serve plan.
