# Status log (newest first)

[繁體中文](log.zh-TW.md) · [Docs index](../README.md)

### 2026-10-17 00:15 — zixian fixed (rebooted); balance-first runs (D21) started
* **Node:** zixian was rebooted on 10-16 at 15:56 (Slurm `BootTime`). A plain torch CUDA probe (5856) works on
  idx 1. We still never use idx 0 or 5. Oren's `hope-g1-sidecar` jobs fail every 30 min since the reboot
  (exit 1). We cannot read his logs, so we don't know if that is the GPU.
* **Smoke 5857** (`_FH_FIST_STABLE_SWING`, idx 1): every step passes. Isaac now really runs the new terms:
  `root_height_ceiling` and `com_support` show up in the reward table and in `Episode_Reward/*`.
  Kit prints "Skipping NVIDIA GPU due CUDA being in bad state" (not in 5719). This is the renderer. It sees
  both allocated GPUs, but CUDA only sees one (`CUDA_VISIBLE_DEVICES`). Physics runs on `cuda:0`
  (= the chosen GPU) as usual.
* **Started:** 5858 `g1_fh_stable` (idx 1) and 5859 `g1_fh_stable_swing` (idx 3). Both warm-start from `model_32299`
  and run to 40 300. Speed is 5.5 / 6.5 s per iteration, so each takes ~15 h (ETA 10-17 ~15:00).








### 2026-10-16 13:10 — balance-first runs ready (D21); zixian cannot start CUDA, nothing submitted
* **Why the robot stands tall:** in MuJoCo (new per-ball `posture` record) `model_32299` keeps the pelvis at
  0.715–0.73 m, the CoM at ~0.66 m, and the CoM 1–4.5 cm off the line between the feet. The fist clip crouches at
  0.675 m, but the recipe's `rally_ready_root_height_debt` (minimum 0.70 m, an A3 value rescaled) taxes
  anything lower, so the policy was held up above its own reference.
* **Reach is not a reason to stand tall:** upper-body IK puts the fist racket on 100 % of 200 bank strike
  points for a pelvis at 0.71, 0.68, 0.66, 0.63 and 0.60 m.
* **Built (D21):** two new reward terms (`root_height_ceiling`, `com_support`; formulas checked by hand on test
  numbers) and two tasks: `_FH_FIST_STABLE` (balance only) and `_FH_FIST_STABLE_SWING` (balance + sharper racket
  rewards), both from `model_32299` to 40 300 (~8000 iterations, ~18 h each, run side by side).
* **Smoke test:** the startup log of 5798 lists every new setting (ceiling −1, CoM −2, upright −3, roll/pitch rate
  −0.15, tilt −1.5, floor 0.58 m, racket std 0.10 / 0.40, shoves 0.8 m/s every 2–5 s). But Isaac never ran:
  **CUDA cannot start on zixian** (`CUDA error 999: unknown error` at the first context), first on GPU idx 0
  (5798), then on idx 1 too (5800). So the new terms have not run in Isaac yet.
* **What happened to the GPUs:** 5719 printed a PhysX CUDA memory-allocator warning on its last iteration
  (10-16 11:35), saved `model_32299`, then hung (cancelled 11:45). Since then no new CUDA context starts.
  Same pattern as idx 5 on 10-12, which ended with the node out for ~21 h. Oren's 4218 (started before) is still running.
* **Precautions:** idx 0 added to the avoid list (`AVOID_GPU_IDX=0,5`); jobs can now take 2 GPUs and hide the
  bad one from CUDA. **No further jobs submitted** so the node is not touched more; SDSC has to look at it.
* Next, when CUDA works again: smoke test, then the two runs (runbook §4). Judge by MuJoCo: falls at 0.75 / 1.0 m/s,
  pelvis height, then hits / in-zone with the aim offset.

### 2026-10-16 12:20 — aim offset for the fist policy `model_32299`
* Method (as D18): measure the median miss (landing − aim) on seeds 0–2 (120 balls), set the offset to
  minus that, test on seeds 3–5 (120 other balls). Fist model, deploy defaults.
* Measured offsets: **(+0.14, −0.03) m** with the training paddle model (`--racket-contact venue`),
  **(+0.19, +0.04) m** with MuJoCo's contact. Much smaller than the old grip (+0.40 / +0.63 m): the fist
  policy already hits much closer to the aim point.
* Held-out balls (seeds 3–5, 120 per row, no shoves, 0 falls everywhere):

  | racket contact | aim offset | hits | at apex | legal | in zone | club success | median miss |
  |---|---|---:|---:|---:|---:|---:|---:|
  | training paddle model | none | 100 % | 93 % | 85 % | 29.2 % | 27.5 % | 17 cm |
  | training paddle model | (+0.14, −0.03) | 100 % | 98 % | 80 % | **42.5 %** | **41.7 %** | 14 cm |
  | MuJoCo contact | none | 100 % | 95 % | 81 % | 18.3 % | 18.3 % | 24 cm |
  | MuJoCo contact | (+0.19, +0.04) | 100 % | 95 % | 68 % | **35.8 %** | **34.2 %** | 16 cm |

  Per zone (training paddle model, with offset): middle 52 %, deep 52 %, corner 34 %, centre line 30 %.
* For comparison, the old-grip best (`model_15700`, same method) was 35 % / 25.8 %: the fist policy is
  now the best one in both contact models.
* Cost: aiming deeper sends more balls long (legal 85 → 80 %, 81 → 68 % with MuJoCo's contact); the
  remaining error is scatter (~15 cm median), not bias.
* Video (16 balls, seed 3, training paddle model + offset): 16/16 hits, 13 legal, 6 in zone, 0 falls —
  `artifacts/evals/g1_fh_fist_model_32299/eval_bank_aim.mp4`.

### 2026-10-16 11:50 — fist run finished; best checkpoint `model_32299`; balance check
* 5719 reached 32 299 / 32 300 and saved `model_32299` (10-16 11:35), then hung while Isaac shut down
  (the process kept running with no output). Cancelled at 11:45 after checking the file was complete.
  The GPU is free; no job of ours is running.
* Isaac, last 8 000 iterations: flat. Velocity pass 34–37 %, virtual contact 65–66 %; falls
  (`base_fell_tilt`) 0.01–0.08 % of episodes even with training shoves (x/y up to 0.5 m/s, roll/pitch up to 0.4 rad/s).
* MuJoCo push test (fist model, 120 balls per row, seeds 0–2, no aim offset):

  | checkpoint | shove | falls | hits | legal | in zone | club success |
  |---|---:|---:|---:|---:|---:|---:|
  | `model_26300` | none | 0 % | 98 % | 76 % | 12.5 % | 10.8 % |
  | `model_28300` | none | 0 % | 98 % | 82 % | 16.7 % | 16.7 % |
  | `model_30300` | none | 0 % | 93 % | 71 % | 5.0 % | 5.0 % |
  | `model_32299` | none | 0 % | **100 %** | **92 %** | **34.2 %** | **32.5 %** |
  | `model_28300` | 0.5 | 0 % | 91 % | 71 % | 10.8 % | 9.2 % |
  | `model_30300` | 0.5 | 0 % | 79 % | 58 % | 10.0 % | 8.3 % |
  | `model_32299` | 0.5 | 0 % | 97 % | 79 % | 21.7 % | 18.3 % |
  | `model_28300` | 1.0 | 7.5 % | 71 % | 48 % | 10.8 % | 10.0 % |
  | `model_30300` | 1.0 | 0.8 % | 71 % | 49 % | 4.2 % | 2.5 % |
  | `model_32299` | 1.0 | 5.0 % | 82 % | 58 % | 13.3 % | 10.0 % |

  (0.25 and 0.75 rows are in the scratch results; 32299 at 0.75: 0.8 % falls, 93 % hits.)
* Re-check on new seeds 3–5 (120 balls, unpushed): `model_32299` hits 100 %, apex 95 %, legal 81 %,
  in zone 18.3 %; `model_28300` 98 % / 96 % / 78 % / 7.5 %. So over 240 balls 32299 is about
  **100 % hits, 86 % legal, 26 % in zone, with no aim offset**. (The diag `model_15700` reached 35 % only with the aim offset.)
* **Checkpoints next to each other differ a lot:** `model_32000` hits 73 % (in zone 0.8 %), `model_32200`
  75 % (12.5 %), `model_32299` 100 %. Isaac averages hide this. Rule kept: pick by the MuJoCo test.
* **Balance verdict:**
  * Unpushed: 0 falls in 960+ balls over all fist checkpoints. Up to 0.5 m/s: 0 falls. 0.75 m/s: ≤ 1 %.
  * At 1.0 m/s (twice the training range) 5 % falls (diag: 26 %, rally: 38 %). Almost all falls come
    from shoves that are large in **both** sideways speed and roll/pitch rate (0.6–1.0 rad/s), mostly
    before the hit; the ball is then missed.
  * By the criterion we set (falls ≤ 2 % and hits ≥ 95 % at 0.5 m/s), `model_32299` passes:
    **no new balance reward needed**. What pushes cost now is accuracy (hits 100 → 97 %, in zone 34 → 22 % at 0.5 m/s), not falls.
* Next: measure the aim offset for `model_32299` (D18), then video and deploy notes.

### 2026-10-15 20:30 — fist run at 26 300: three times more in-zone returns in MuJoCo
* 5719 at 26 307 / 32 300, 7.95 s/it, ETA ~13 h. New checkpoints go to a new directory on each resume
  (`2026-10-15_13-48-31_g1_fh_fist`).
* Isaac (progress.sh): flat since the strike-mark change (legal 43–47 %, velocity pass ~35 %).
* MuJoCo (fist model, 120 balls per row, same seeds; no aim offset):

  | checkpoint | shove | falls | hits | legal | in zone | club success |
  |---|---:|---:|---:|---:|---:|---:|
  | `model_23400` | none | 0 % | 93 % | 84 % | 3.3 % | 2.5 % |
  | `model_26300` | none | 0 % | **98 %** | 76 % | **12.5 %** | **10.8 %** |
  | `model_23400` | 0.5 | 0 % | 88 % | 66 % | 4.2 % | 4.2 % |
  | `model_26300` | 0.5 | 0 % | **95 %** | 65 % | 8.3 % | 7.5 % |
  | `model_23400` | 1.0 | 3.3 % | 69 % | 44 % | 2.5 % | 1.7 % |
  | `model_26300` | 1.0 | 9.2 % | 71 % | 48 % | 5.0 % | 4.2 % |

* Where the balls go (unpushed): 26 300 hits deeper: into the net / short 15 → 8 of the hits, the
  on-table miss is 25 cm short instead of 42 cm; but 22 hits (was 3) go over the net and off the table.
  Apex timing unchanged (91–93 %). So the returns are faster and longer; the aim offset (D18) has to be
  re-measured for the fist policy, which should turn many of the off-table balls into hits.
* Balance at ≤ 0.5 m/s still 0 % falls; at 1 m/s 9 % (was 3 %): watch it at the next check.

### 2026-10-15 15:10 — fist run continues with the strike marked at frame 43
* `HOPEPingPongUnitreeG1_FH_FIST` now sets `commands.racket_target.strike_phase(_per_clip)` to 0.4057 (frame 43)
  through `overrides` (the `racket:` block of the YAML is documentation only; the code value was 0.3868).
* 5686 cancelled at 23 7xx; **5719** resumes the same run from `model_23700` (GPU idx 0), step counter
  restored (153 624), to 32 300 (~19 h). Check: the Isaac velocity pass rate (37 % before) and the
  MuJoCo test at ~25 700.

### 2026-10-15 14:30 — why the fist run plateaus: racket speed at the strike
* Isaac strike checks at the latest iteration (pass rates at the strike frame):

  | run | position | velocity | face | all three | legal |
  |---|---:|---:|---:|---:|---:|
  | `g1_fh_diag` 19 999 (old grip, normal pace) | 98 % | 65 % | 100 % | 64 % | 57 % |
  | `g1_fh_rally` 5 985 (old grip, rally pace) | 80 % | 38 % | 99 % | 30 % | 37 % |
  | `g1_fh_fist` 23 416 (fist, rally pace + shoves) | 91 % | **37 %** | 100 % | 34 % | 47 % |

  The bottleneck is the **racket speed at the strike** (error 2.1 m/s). It is mostly the rally recipe
  (faster feeds, less time; the old grip drops the same way), not the fist grip.
* **One fixable cause in the fist clip:** at its marked strike frame (41) the reference racket moves at
  only 1.41 m/s (old clips: 2.3–2.4 m/s; bank targets: median 2.5 m/s). The fast part of the fist swing
  runs from frame 41 to 46, so the strike mark sits at its very start and the imitation reward teaches a
  slow racket at contact. Fix candidate: mark the strike at frame 43 (`strike_phase_per_clip[0]` 0.3868 →
  0.4057) for the fist clip. Scaling the A3 racket path down did not help (the IK solution and the speed
  hardly change), so that was dropped.

### 2026-10-15 13:00 — diag finished; fist run balances much better; late diag checkpoints regress in MuJoCo
* **4382 `g1_fh_diag` finished** (20 000, `model_19999`, Isaac legal 57 %, falls 0.2 %).
  **5686 `g1_fh_fist`** at 23 416 / 32 300, 7.9 s/it, ETA ~20 h. Oren's 4218 runs; GPU idx 5 still avoided.
* Isaac metrics of the fist run (progress.sh): legal return 58 % → 42–45 % and "all OK" 61 % → 31–34 %
  while rally pace, timing jitter and shoves ramped in (to 20 300); flat since, falls ≤ 0.2 %.
* **Push test** (MuJoCo, 120 balls per row, same seeds as before; fist policy on the fist model):

  | policy | shove up to | falls | hits the ball | legal return |
  |---|---:|---:|---:|---:|
  | `g1_fh_fist` `model_23400` | none | **0 %** | 93 % | 84 % |
  | | 0.5 | **0 %** | 88 % | 66 % |
  | | 0.75 | 2.5 % | 72 % | 53 % |
  | | 1.0 | 3.3 % | 69 % | 44 % |
  | `g1_fh_diag` `model_15700` (before) | 0.5 / 1.0 | 5.8 % / 25.8 % | 82 % / 52 % | 61 % / 37 % |

  Shoves in training work: falls at 1 m/s drop from 26 % to 3 %, and more balls are still hit after a
  shove. Without shoves it is about as good as the old best (84 % vs 86 % legal), with the new grip.
  In-zone is low (3 %) because the aim offset (D18) was measured for the old grip; it must be re-measured.
* **Finding: the last diag checkpoints get worse in MuJoCo** even though Isaac says they are fine
  (unpushed, 120 balls): `model_15700` falls 0 %, hits 100 %, legal 86 %; `model_17300` 1.7 % / 92 % / 70 %;
  `model_19999` **14 % / 68 % / 53 %**. The policy keeps fitting Isaac details that MuJoCo (and so
  probably the real robot) does not share. Rule from now on: **pick checkpoints by the MuJoCo test, not by
  "latest"**; for diag, `model_15700` stays the best. Watch the fist run the same way (test every ~2000).
* Balance reward: not needed for now (decision rule in the 10-14 answer: falls ≤ 2 % at 0.5 m/s → met).

### 2026-10-14 21:40 — fist grip done properly: racket perpendicular to the forearm; retraining started
* The first fist (17:00 entry) only changed the hand's look and kept the racket along the forearm. A real
  fist holds the handle **across the palm**, so the racket stands perpendicular to the forearm (shakehand
  grip photos). Revised D20: a separate model `g1_29dof_pingpang_fist` (handle across the fist, red face on
  the palm side, blade origin (0.083, 0.0226, 0.14) m in the wrist frame); the original model is back to
  the open hand and is still what `g1_fh_diag` trains on. Pictures: [fist grip](../assets/right_fist.png),
  [clip at the strike, original vs fist](../assets/clip_strike_forearm_vs_fist.png).
* **Retraining is required:** `g1_fh_diag` `model_15700` on the fist model in MuJoCo hits 0 of 6 balls.
* **Reach check (no retraining needed for the bank):** whole upper-body IK (waist 3 + arm 7) from the
  station puts the fist racket on 100 % of 200 sampled bank strike points (position < 3 cm, face < 15°),
  same as the old mount. With the arm alone, the fist racket reaches ~10 cm less far out along the arm.
* **Reference clips:** the A3 clips retargeted by IK on the fist model (`g1ikfist_hope_*`): face error
  6° mean, 8° at the strike; racket-centre error 15 cm (the A3 clip reaches further out than a fist
  racket can with the clip's legs and waist; the bank, not the clip, sets the strike point).
* **Smoke test** 5685 (GPU idx 0): all 7 steps OK; the log shows the fist asset `assets/unitree_g1_fist`,
  the fist mount, the shoves and the fist clips.
* **Training job 5686 `g1_fh_fist`** (`HOPEPingPongUnitreeG1_FH_FIST`: rally pace + shoves + fist grip),
  warm start from `g1_fh_diag` `model_17300`, to 32 300 (15 000 iterations, ~33 h at 7.8 s/it).
  `g1_fh_bal` (shoves on the old mount) is not submitted: the fist run already includes the shoves.
* `g1_smoke.sbatch` now also refuses GPU idx 5.

### 2026-10-14 21:05 — zixian back; all GPUs except idx 5 healthy
* zixian came back at ~17:05 (outage ~21 h). 4382 `g1_fh_diag` resumed on GPU idx 3 (14th requeue) and is
  at iteration 17 261 / 20 000 (`model_17200`), 7.8 s per iteration, ETA ~6 h. Oren's 4218 runs on idx 2;
  his sidecar 5677 waits (BeginTime).
* GPU probe (six 1-GPU jobs, each runs `nvidia-smi` + a CUDA matmul; a job handed idx 5 exits without
  touching it): idx 0 (3D:00.0), 1 (3E:00.0), 4 (B1:00.0), 6 (DA:00.0), 7 (DB:00.0) all **CUDA OK**,
  33–36 °C; idx 2 and 3 are working (running training). The node-wide "CUDA unknown error" of 10-13 is
  gone. **idx 5 (B2:00.0) not tested on purpose** (it fell off the bus on 10-12); keep avoiding it.

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
  the handle (superseded at 21:40: this version kept the racket along the forearm, see the 21:40 entry). Looks only: same
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

