# Status log (newest first)

[繁體中文](log.zh-TW.md) · [Docs index](../README.md)


### 2026-10-14 17:00 — balance test, balance run prepared, right hand closed into a fist; docs reorganized
* **Cluster:** zixian still down (since 10-13 20:15, "Not responding"). 4382 `g1_fh_diag` queued at
  15 756 / 20 000; Oren's 4218 and his sidecar 5652 also wait. Nothing can train.
* **Balance test** (new `mujoco_eval_onnx.py --push V_XY W_RPY`): one random shove per ball, 0–1 s after
  the feed, a base velocity change of up to V_XY m/s sideways/forward and W_RPY rad/s of rotation.
  120 balls per row (3 seeds × 40), deploy defaults, robot stood back up after a fall:

  | policy | shove up to | falls | hits the ball | legal return | club success |
  |---|---:|---:|---:|---:|---:|
  | `g1_fh_diag` `model_15700` | none | 0 % | 100 % | 86 % | 6.7 % |
  | | 0.25 | 0 % | 98 % | 82 % | 5.0 % |
  | | 0.5 | 5.8 % | 82 % | 61 % | 4.2 % |
  | | 0.75 | 16.7 % | 61 % | 43 % | 2.5 % |
  | | 1.0 | 25.8 % | 52 % | 37 % | 1.7 % |
  | `g1_fh_rally` `model_5900` | none | 0.8 % | 91 % | 70 % | 1.7 % |
  | | 0.25 | 10.0 % | 85 % | 65 % | 1.7 % |
  | | 0.5 | 22.5 % | 71 % | 52 % | 1.7 % |
  | | 0.75 | 33.3 % | 58 % | 42 % | 0.0 % |
  | | 1.0 | 38.3 % | 48 % | 34 % | 0.8 % |

  Reading: even when it does not fall, a shove costs the ball (hits drop from 100 % to 82 % at
  0.5 m/s). The rally policy is much less stable. Cause: **the training has no pushes at all** (the
  HOPE recipe copies HITTER, which has none); only mass, friction, CoM and PD-gain randomization.
* **Balance run prepared (D19):** `HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL` = rally recipe + shoves that
  grow from 0 to ±0.5 m/s (x, y), ±0.1 (z), ±0.4 rad/s (roll, pitch), ±0.6 (yaw) every 3–6 s over 3000
  iterations. Warm start from the latest `g1_fh_diag` checkpoint. Not submitted: zixian is down and
  Isaac cannot be smoke-tested elsewhere. Success check afterwards: the push table above, same seeds.
* **Right hand closed into a fist (D20):** the open rubber hand is replaced by a generated fist round
  the handle ([picture](../assets/right_fist.png), before: [picture](../assets/right_hand_open_before.png)). Looks only: same
  racket pose, mass, inertia and collisions (checked: identical MuJoCo model mass/inertia), so no retraining.
* **Docs** split into `docs/g1_forehand/` (overview, decisions, implementation, operations, status,
  research, plans).

### 2026-10-14 — why the returns miss, and aim compensation (in-zone 12 % → 35 %)
* **Where the misses go** (`model_15700`, deploy defaults, 120 balls): the balls that miss the table go
  off the **side line** (12 of 120 off the side, 5 into the net/short, 0 past the end line); with the
  old clock it was 43 of 120 off the side. Balls that land on the table land **short of the aim point**
  (median −30 to −45 cm), then fly on past the end after the bounce, which in the video looks "long".
* **Cause, measured at each contact** (`--trace-out`, now also logs racket normal and ball spin). Median
  landing error vs the aim point when the same contact is pushed through:

  | model of the hit | landing error |
  |---|---:|
  | training (venue) paddle model, racket exactly as commanded | +5 cm |
  | training paddle model, racket as the robot moved it | −13 cm |
  | MuJoCo's own racket contact (what happened) | −42 cm |

  1. **MuJoCo's racket contact (~30 cm):** same exit speed (6.2 m/s) but much flatter (vz +0.9 vs
     +2.3 m/s). The training model has rubber grip (the ball is dragged along an upward swing); MuJoCo's
     soft contact (friction 0.6, ~20 ms time constant) does not. The training model was fitted on real
     racket hits, so MuJoCo is the less realistic one. New option `--racket-contact venue` replaces it
     with the training model, applied analytically like the table bounce.
  2. **The swing (~15 cm):** the racket reaches 94 % of the commanded speed and is tilted upward: velocity
     error (achieved − commanded) is a repeatable bias of (−0.19, −0.08, +0.18) m/s with a scatter of only
     (0.09, 0.14, 0.09) m/s.
* **What did not work:** scaling the racket speed up (moves the median but in-zone stays ~10 % and more
  balls go out), and correcting the racket command for the bias (linear or bias-only): the policy does
  not follow commands outside what it was trained on, and results got worse. Removed.
* **What works: aim compensation.** The planner solves the racket velocity for the target moved by the
  measured mean miss (`--aim-offset DX DY`). Held-out balls (seeds 3–5, 120 per row):

  | racket contact | notice | aim offset | legal | in zone = success | median miss |
  |---|---|---|---:|---:|---:|
  | training paddle model | 0.9 s | none | 83 % | 11.7 % | 38 cm |
  | training paddle model | 0.9 s | (+0.40, −0.10) m | 89 % | **35.0 %** | 19 cm |
  | MuJoCo contact | 0.9 s | none | 77 % | 5.0 % | 46 cm |
  | MuJoCo contact | 0.9 s | (+0.63, −0.06) m | 90 % | **25.8 %** | 25 cm |
  | training paddle model | rally pace | none | 87 % | 11.7 % | 29 cm |
  | training paddle model | rally pace | (+0.33, −0.08) m | 78 % | **34.2 %** | 16 cm |

  Per zone (training paddle model, 0.9 s, +0.40/−0.10): middle 55 %, centre line 30 %, deep 44 %,
  corner 14 %. Video: `artifacts/evals/g1_fh_diag_model_15700/eval_bank_aim.mp4`.
* **The right offset depends on the racket** (0.40 m with the training model, 0.63 m with MuJoCo's
  contact), so on the real robot it has to be measured with the real racket, like the timing offset.
* What remains is scatter (~20 cm median miss), about the size of the end-line zones. Shrinking it needs
  a more precise swing (training), not more calibration.

### 2026-10-14 13:10 — zixian still down (17 h)
* zixian has been "not responding" since 10-13 20:15. Nothing of ours is training: 4382 waits at
  15 756 / 20 000 (`model_15700`), `g1_fh_rally` at `model_5900`. Oren's 4218 is also waiting.

### 2026-10-14 — +40 ms timing calibration is a config parameter
* `lifecycle.tts_offset_s` (runtime YAML, default 0) is added to every command's time left and deadline,
  so the policy is told the strike is that much later; the contact phase moves with it. The G1 YAML sets
  `tts_offset_s: 0.04`; the A3 keeps 0. `mujoco_eval_onnx.py --tts-offset-s` now overrides this value
  instead of adding to the commands.
* With the G1 defaults (`strike_time` + 40 ms) a plain MuJoCo run is ball-for-ball identical to the
  earlier "strike_time + 40 ms" test (main run `model_15400`: apex OK 90 %, legal 77 %, club success 10 %).
* From now on G1 MuJoCo numbers include the calibration. Older runs: `--clock countdown --tts-offset-s 0`.
  The cause of the 40 ms is still open; on the real robot the value must be measured again.

### 2026-10-14 — the G1 now uses the `strike_time` clock by default
* `hope_pingpong_runtime_unitree_g1.yaml` (generated by `unitree/scripts/generate_g1_runtime_yamls.py`)
  sets `clock: strike_time`, `max_message_age_s: 0.1`. The A3 config keeps `countdown`.
* Clock guard: every command carries both "time left" and the absolute deadline, so the runner can
  compute how old the message is. If that age is negative or over 0.1 s, the planner's clock and the
  runner's clock disagree; the deadline is ignored, the command is stamped on arrival, and
  `clock_rejections` counts it. This keeps an unsynced clock from silently shifting every swing.
* **Comparability:** all MuJoCo numbers before this entry used `countdown`. `eval_checkpoint.sh` and
  `mujoco_eval_onnx.py` now default to `strike_time` for the G1; add `--clock countdown` to reproduce
  older results (checked: the default run is ball-for-ball identical to `--clock strike_time`).

### 2026-10-14 — `strike_time` clock implemented in the deploy reference
* `lifecycle.py` has a second clock mode, `clock: strike_time` (default stays `countdown`, so the A3
  behaviour is unchanged). The policy then sees `strike_time − now`: the time left from the state just
  measured, as in training, whatever the message age. `RacketCommand` carries `strike_time`, the ROS
  parser reads it from field [10] (0 = missing), the runner passes the wall clock in real-time mode, and
  `mujoco_eval_onnx.py --clock strike_time` uses the simulation clock. Tests:
  `tests/test_lifecycle_strike_time.py` (5).
* Result (MuJoCo, 3 seeds × 40 balls per row):

  | policy, mode | clock | + offset | apex OK | median vz | legal | in zone | club success |
  |---|---|---:|---:|---:|---:|---:|---:|
  | `g1_fh_diag` `model_15400`, 0.9 s notice | countdown (original) | 0 | 0 % | +0.48 | 57 % | 10 % | 0 % |
  | | `strike_time` | 0 | 13 % | +0.37 | 65 % | 9 % | 1.7 % |
  | | `strike_time` | +20 ms | 52 % | +0.27 | 72 % | 12 % | 5.0 % |
  | | `strike_time` | **+40 ms** | **90 %** | +0.16 | **77 %** | 11 % | **10.0 %** |
  | `g1_fh_rally` `model_5900`, rally pace | countdown (original) | 0 | 2 % | +0.48 | 61 % | 6 % | 0 % |
  | | `strike_time` | 0 | 9 % | +0.42 | 65 % | 6 % | 0 % |
  | | `strike_time` | +20 ms | 47 % | +0.28 | 62 % | 3 % | 0.8 % |
  | | `strike_time` | **+40 ms** | **72 %** | +0.22 | 60 % | 3 % | 1.7 % |
* The clock fix removes exactly the 20 ms it should (same effect as the earlier +20 ms offset). The
  remaining ~40 ms still needs a calibration offset, and its cause is still open.

### 2026-10-14 00:30 — why the robot strikes early (solved in simulation); third zixian outage
* **Outage:** zixian went "not responding" again at 20:15 on 10-13 (munge errors, like 10-10 and 10-12).
  4382 is queued at iteration 15 756 and resumes from `model_15700` when the node is back.
* **Symptom:** in MuJoCo every checkpoint hits the ball while it is still rising (vz +0.40 to +0.50 m/s,
  ~45 ms before the apex), so the club's apex rule is almost never met. In Isaac the same policy is on
  time: at the strike frame its racket is −0.35 cm (x) and −0.6 cm (z) from the target.
* **Measurement** (`mujoco_eval_onnx.py --trace-out`, per-tick log): the racket centre reaches the target
  point **53 ms (median, 12 of 13 swings between 47 and 65 ms) before the command clock reaches zero**,
  and 3–4 cm high. The ball has not arrived yet, so the racket meets it early.
* **Cause, part 1 (20 ms):** the deploy clock (`a3_deploy_onnx_ref_pingpong/lifecycle.py`) copies the
  planner's time-to-strike and subtracts one tick (20 ms) in the same update, before the policy sees it.
  In Isaac the observed time-to-strike describes the state just simulated; here it is 20 ms short, so the
  policy thinks the strike is sooner and moves early.
  *Why the subtraction exists:* the runner must keep its own countdown between planner messages (the
  planner may send slower than 50 Hz or drop a packet), so every tick it does `tts -= dt`. The code does
  "copy the new value, then count down" in the same tick, so a fresh value is aged once before use. The
  intent is not documented. It is only correct if the message is already one tick old when it arrives
  (possible on hardware over ROS, never in our MuJoCo loop, where the planner runs in the same tick).
  The robust fix is to time-stamp commands: the wire format already carries an absolute `strike_time`
  (currently "informational"), so the runner can compute `tts = strike_time − now` when it builds the
  observation, whatever the message age.
* **Cause, part 2 (~30 ms):** not explained yet. Ruled out: a different racket point (both use 0.21,
  0.015, 0.01 m from the wrist), action delay in training (`qdes_delay` 0). Likely a sim-to-sim dynamics
  difference; needs an Isaac-vs-MuJoCo replay of the same swing when the cluster is back.
* **Fix test** (`--tts-offset-s`: add a constant to every commanded time-to-strike), 80–120 balls per row:

  | policy, mode | offset | apex OK | median vz at contact | legal | in zone | club success |
  |---|---:|---:|---:|---:|---:|---:|
  | `g1_fh_diag` `model_15400`, 0.9 s notice | 0 | 0 % | +0.48 | 57 % | 10 % | 0 % |
  | | +20 ms | 11 % | +0.37 | 61 % | 8 % | |
  | | +40 ms | 54 % | +0.27 | 69 % | 9 % | |
  | | +50 ms | 78 % | +0.20 | 72 % | 8 % | |
  | | **+60 ms** | **91 %** | +0.14 | **78 %** | 9 % | **7.5 %** |
  | | +80 ms | 100 % | +0.04 | 79 % | 1 % | |
  | `g1_fh_rally` `model_5900`, rally pace | 0 | 2 % | +0.48 | 61 % | 6 % | 0 % |
  | | +40 ms | 47 % | +0.28 | 62 % | 3 % | 1 % |
  | | **+60 ms** | **72 %** | +0.22 | 60 % | 3 % | 2 % |
  | | +80 ms | 81 % | +0.11 | 36 % | 2 % | 1 % |
* **Conclusion:** the early strike is a timing offset between training and the deploy pipeline, not a
  skill the policy lacks. A +60 ms calibration fixes it in MuJoCo. The next bottleneck is **aim**: only
  3–10 % of balls land in the target zone. The real robot will need its own calibration (motor and
  sensing latency), measured the same way.

### 2026-10-13 19:40 — MuJoCo check of `g1_fh_diag` `model_15400`
* zixian cannot start Isaac for the usual ONNX export, so the ONNX was built on the laptop with
  `scripts/onnx_from_checkpoint.py` (copies the checkpoint's actor weights into an exported graph of the
  same task; checked against the real export of `model_5000`: identical outputs).
* MuJoCo, default bank, robot stood up after a fall, 3 seeds × 40 = 120 balls per row:

  | checkpoint | mode | contact | apex OK | over net | legal | in zone | falls | median vz at contact |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | `g1_fh_diag` `model_5000` | 0.9 s notice | 78 % | 9 % | 57 % | 38 % | 4 % | 2 % | +0.41 m/s |
  | `g1_fh_diag` `model_5000` | rally pace | 75 % | 14 % | 52 % | 44 % | 8 % | 1 % | +0.40 m/s |
  | `g1_fh_diag` `model_15400` | 0.9 s notice | **92 %** | 1 % | **90 %** | 58 % | **12 %** | 3 % | +0.47 m/s |
  | `g1_fh_diag` `model_15400` | rally pace | 82 % | 9 % | 60 % | 37 % | 4 % | 2 % | +0.44 m/s |
  | `g1_fh_rally` `model_5900` | 0.9 s notice | 93 % | 1 % | 77 % | **62 %** | 10 % | 0 % | +0.50 m/s |
  | `g1_fh_rally` `model_5900` | rally pace | 88 % | 2 % | 67 % | **61 %** | 6 % | 0 % | +0.48 m/s |
* `model_15400` is the best checkpoint so far under its training condition (90 % over the net, 12 % in
  zone), but at rally pace it drops to 37 % legal. `g1_fh_rally` `model_5900` keeps 61 % at rally pace:
  the rally training (D14) works.
* All three still strike the ball while it is rising (vz +0.40 to +0.50 m/s, about 45 ms before the
  apex), and more training has not changed that, so club success stays ~0. This looks systematic (swing
  timing in the reference clip, or MuJoCo vs Isaac contact) rather than undertraining. Next to investigate.
* Videos: `artifacts/evals/g1_fh_diag_model_15400/{eval,eval_bank,eval_rally}.mp4` (12 balls each).

### 2026-10-13 19:20 — main run at 15 405; zixian still cannot start new CUDA jobs
* `g1_fh_diag` (4382) iteration 15 405 / 20 000: racket ≤10 cm 99.9 %, all OK 64 %, contact 74 %,
  legal 50 %, landing error 0.31 m, falls 0.25 %. Flat since ~13 700 (all OK 65 %, legal 55 % then).
  About 10 h left at 7.9 s/iteration.
* A 1-GPU probe on index 0 (`3D:00.0`) still fails with `CUDA unknown error`, so `g1_fh_rally` stays
  stopped at `model_5900` until SDSC resets GPU index 5 or reboots zixian. `matthew` is back up, but its
  MI210/P100 GPUs cannot run Isaac Sim (it needs an RTX card).

### 2026-10-13 evening — perception study: spin matters more than the camera
* Added a camera model to the MuJoCo evaluator and ran 25 conditions × 120 balls with `model_5900`
  at rally pace. Without spin, legal returns drop from 61 % to 10 % even with perfect tracking; with
  spin known, mocap 56 % and iPhone-level 54 %; a rough spin estimate (±50 %, ±40°) 48–49 %.
  Full table and caveats: [`research/perception.md`](../research/perception.md) (decision D16).
* Side finding: in continuous MuJoCo sessions a fallen robot never got up, so one fall decided every
  later ball. Sweeps now use `--reset-after-fall`.

### 2026-10-13 15:15 — `g1_fh_rally` stopped by a failed GPU; new CUDA jobs cannot start on zixian
* `g1_fh_diag` (4382) iteration 13 704: racket ≤10 cm 99.8 %, all OK 65 %, contact 75 %, legal 55 %,
  landing error 0.32 m, falls 0.35 %.
* `g1_fh_rally` (5584) crashed on 10-12 at 17:22 at iteration 5985 (`CUDA error: unspecified launch
  failure`, Vulkan `ERROR_DEVICE_LOST`) and then hung for 22 h without exiting. Its idle second GPU,
  Slurm index 5 = `0000:B2:00.0`, has fallen off the bus (`nvidia-smi`: "Unable to determine the device
  handle … Unknown Error"). The crash came while `eval_checkpoint.sh` was rendering in that job, most
  likely on that idle GPU.
* Since then, a fresh CUDA context fails on zixian even on healthy GPUs (index 0 / `3D:00.0`: `CUDA
  unknown error`, while `nvidia-smi` sees the card). Already-running jobs (4382, Oren's 4218) are not
  affected. Resubmission 5640 (1 GPU, index 0) failed at start for this reason. Needs SDSC to reset the
  GPU or reboot the node; `g1_fh_rally` resumes from `model_5900` afterwards.
* `train.sbatch` now refuses to start on Slurm GPU index 5 (`AVOID_GPU_IDX`), without touching it.

### 2026-10-12 17:30 — both runs healthy; MuJoCo at rally pace no longer falls
* `g1_fh_diag` iteration 5099: racket ≤10 cm 98 %, all OK 46 %, contact 73 %, legal 49 %, falls 1.9 %.
* `g1_fh_rally` iteration 5979 (rally pace 42 % of swings, jitter at the full 30 ms): racket ≤10 cm 95 %,
  all OK 30 %, contact 60 %, legal 37 %, falls 2.1 %. The ramps finish around iteration 5600–5900.
* MuJoCo, 12 bank balls each (`eval_checkpoint.sh`; videos in `~/hope/evals/<run>/<model>/`):

  | checkpoint | mode | contact | apex OK | over net | legal | in zone | falls | strike-to-strike |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | `g1_fh_diag` `model_5000` | bank (0.9 s notice) | 75 % | 8 % | 58 % | 33 % | 8 % | 0 % | 2.3 s |
  | `g1_fh_diag` `model_5000` | rally pace | 83 % | 17 % | 33 % | 33 % | 8 % | 0 % | 1.6 s |
  | `g1_fh_rally` `model_5900` | bank (0.9 s notice) | 83 % | 0 % | 75 % | 50 % | 0 % | 0 % | 2.3 s |
  | `g1_fh_rally` `model_5900` | rally pace | **92 %** | 0 % | **83 %** | **75 %** | 8 % | 0 % | 1.5 s |
* Falls at rally pace are gone for both (44 % with `model_2400`). The remaining problem is timing: both
  strike while the ball is still rising (median vz +0.36 to +0.44 m/s, about 40 ms before the apex).

### 2026-10-12 12:40 — second outage; 4382 was being preempted every 31 min
* All GPU nodes went "not responding" at 02:29 (munge errors again, like 10-10). 5560 was cancelled
  rather than requeued; resubmitted as **5584**, which resumes from its own `model_4600`. 4382 is queued.
* `g1_fh_rally` before the outage (iteration 4677, rally pace 42 %, jitter 21 ms): racket ≤10 cm 93 %,
  all OK 29 %, contact 66 %, legal return 43 %, falls 3.3 %. It recovered from the warm-start dip
  (contact 23 %, falls 57 % at 2609) while the rally pace ramped up, and is now level with `g1_fh_diag`.
* `g1_fh_diag` (iteration 3774): racket ≤10 cm 89 %, all OK 24 %, contact 63 %, legal 40 %, falls 6.8 %.
* From 21:18 to 02:25, 4382 was requeued every ~31 min: a 2-GPU job from another club member
  (`hope-g1-sidecar`, QOS `normal`) preempts our `long`-QOS jobs, fails after 45 s, and is resubmitted.
  4382 lost about half its throughput (≈100 iterations per cycle instead of ≈230). Worth a word with them.
* Found while checking: a resume reset `env.common_step_counter` to 0, so step-keyed ramps (rally pace,
  timing jitter) would restart after every requeue. `train.sbatch` now restores it on a same-run resume
  (`restore_step_counter=true`; not for `INIT_CHECKPOINT`). The all-zero rows in 4382's `progress.sh`
  table (2700–3600) are the first iterations after each requeue, not a collapse.

### 2026-10-11 21:10 — rally pace measured; new run `g1_fh_rally` started
* MuJoCo `--rally-pace` (next feed hit 0.7–1.0 s after our contact, known only at launch), `model_2400`,
  same 16 questions:

  | condition | contact | timing OK | over net | falls | lead | strike-to-strike |
  |---|---:|---:|---:|---:|---:|---:|
  | announced 0.9 s ahead (training condition) | 88 % | 38 % | 19 % | 0 % | 0.90 s | 2.3 s |
  | announced at launch | 69 % | 50 % | 19 % | 0 % | 0.59 s | 2.0 s |
  | **real rally pace** | **38 %** | 25 % | 12 % | **44 %** | 0.57 s | **1.5 s** |
* Training changes (decision D14): rally-pace reference clock, ±30 ms virtual-ball timing jitter, a
  faster-feed bank (4.4–8.5 m/s, end line → apex 0.38–0.85 s). Slower feeds are impossible with the
  middle-third bounce and fixed strike plane. `check_rally_pace.py` in Isaac (pace forced to 100 %):
  swing entry at time-to-strike 0.40–0.82 s (median 0.62), strike-to-strike median 1.52 s, jitter 30 ms.
* First submission 5554 died: zixian's GPU `0000:61:00.0` no longer answers the driver since the
  outage and Slurm hands it out first. `train.sbatch` now falls back to a healthy device; 5560 asks for
  2 GPUs. Worth reporting to SDSC.

### 2026-10-11 20:40 — first per-zone evaluations (Isaac and MuJoCo)
* New tools: `eval_forehand.sh` (Isaac, club success per zone), `progress.sh`, and `--serve-bank` in
  the MuJoCo evaluator (`eval_checkpoint.sh` now also writes `eval_bank.{json,mp4}`).
* `g1_fh_diag model_2300`, Isaac: 11.1 % overall; contact 92 %, legal return 79 %.
* Baseline `g1_fh model_2100` on the same questions: 4.6 % (main run 2.4× better at the same age).
* MuJoCo real ball: `model_2400` contact 10/12, over the net 3/12, 0 falls; strikes early.
* Sim2real items found: the 0.83 s swing lead vs 0.5–0.8 s feed flight; MuJoCo's table contact
  needed the fitted bounce.

### 2026-10-11 19:42 — 4382 resumed
* zixian came back at 17:08; 4382 restarted on its own (Slurm requeue) from `model_1300` and kept the
  20 000 total. About 74 iterations (1300–1374) were redone.
* Iteration 2302: racket ≤10 cm 72.9 %, contact 50.1 %, legal return 27.1 %, landing error 0.71 m,
  falls 18 % of episode ends.
* 4354 was not restarted (submitted before `train.sbatch` had `--requeue`). Left stopped at
  `model_2100` as the baseline.

### 2026-10-11 14:36 — cluster outage
* All GPU nodes (zixian, gauri, matthew, monster) `down*`. Cause: the Slurm authentication daemon
  (munge) on zixian failed at 10-10 23:37; both training jobs were killed at that moment.
* 4382 was requeued and waiting (`ReqNodeNotAvail`) at iteration 1374, `model_1300`.
* 4354 ended `FAILED` at iteration 2196.

### 2026-10-10 evening — first checkpoints
* `g1_fh_diag model_200` in MuJoCo: falls on every ball (fall rate 1.00). Expected this early.
* `g1_fh model_900` in MuJoCo: stands (fall rate 0); racket within 10 cm ~12 % in training.

### 2026-10-10 20:12 — main run started
* 4382 `g1_fh_diag` on the final bank (50 000 questions, 12 500 per zone), 4096 environments,
  ~8 s per iteration, 20 000 iterations (~45 h).
* Earlier attempts 4367, 4379, 4381 were cancelled as the feed and zones changed that evening.

