# Status log (newest first)

[繁體中文](log.zh-TW.md) · [Docs index](../README.md)

### 2026-10-21 13:00 — why the backhand returns land short, why the torso moves, and D25 (`g1_bh_quiet`)
* **Short returns: mostly the MuJoCo paddle, partly the robot.** Same 40 balls (seed 3), per contact:

  | | racket speed / command | racket vs strike point | ball out vz | over the net |
  |---|---:|---:|---:|---:|
  | `g1_bh_scratch` 9999, MuJoCo paddle contact | 85 % | 6 cm behind | 0.74 m/s | 15 % |
  | same, venue (measured) paddle model | 86 % | 6 cm behind | 1.47 m/s | 95 % |

  1. **Paddle model (main cause in the videos).** The feed arrives with ~21 rev/s topspin (−133 rad/s). The
     venue model (fitted to real ball–paddle measurements; the one training uses) turns that spin into lift.
     MuJoCo's soft contact does not. On identical racket states the venue model gives vz 2.2 m/s, MuJoCo
     0.5 m/s, so in MuJoCo the ball goes into the net. Videos should use `--racket-contact venue`.
  2. **Robot.** The racket arrives at 85 % of the commanded speed and ~6 cm behind the strike point, so even
     with the venue model it lands ~35 cm short. The planner aim offset (D18, +0.30 m for this model) hides
     that: 11/12 over the net in the new video `artifacts/evals/g1_bh_scratch_model_9999/backhand_12balls_venue.mp4`.
     The training-side fix is D25.
* **Upper body.** New per-ball torso metrics in `mujoco_eval_onnx.py` (torso roll/pitch/yaw range, peak
  angular speed). The reference clip turns the torso 23° (peak 1.2 rad/s). Policies (venue contact, 40 balls):

  | policy | pelvis | torso yaw range | torso roll / pitch range | peak torso ang. speed |
  |---|---:|---:|---:|---:|
  | reference backhand clip | — | 23° | 2° / 4° | 1.2 rad/s |
  | `g1_fh_diag` 2300 (forehand) | **0.76 m** | 34° | 10° / 6° | 2.8 rad/s |
  | `g1_bh_scratch` 9999 | 0.61 m | 46° | 6° / 5° | 3.2 rad/s |
  | `g1_bh_still` 50299 | 0.63 m | 42° | 10° / 10° | 3.8 rad/s |
  | `g1_bh_stand` 57500 | 0.70 m | 47° | 7° / 10° | 3.7 rad/s |

  At the end of training `g1_bh_scratch` had a torso orientation error of 10.7° and an angular-velocity
  error of 1.5 rad/s. The racket terms (weight 14 each) outweigh torso imitation (0.5–1.0), so the torso
  is used to chase the racket target.
* **Paper (PingpongDuo, arXiv 2610.20579, two G1s in Isaac Lab):** what applies to us.
  (a) Fit the ball–paddle contact to real measurements (speed-dependent restitution, tangential response);
  our venue model already is, MuJoCo's is not, so judge returns with the venue model.
  (b) Reward the outcome (contact, net-clearance margin, landing), not only racket tracking. They found
  smaller racket-tracking error does not mean better returns.
  (c) A bounded residual on the 7 racket-arm joints, gated around contact and trained on the outcome
  with the base policy frozen: +12 points of contact and +11 of legal returns. A good next step after D25.
  (d) Separate footwork and striking policies, each trained from states the other produces: matches our
  plan for a separate stepping policy.
* **D25 started:** `g1_bh_quiet` (task `HOPEPingPongUnitreeG1_BH_FIST_QUIET`), a new model, 10 000 iterations,
  16 384 envs, session `ses-6Z99Q0N89591SRSM2HHQMRVX52`. Code: `add_outcome_terms` (G1 env, off by
  default), `virtual_landing` now aims at the bank row's intended point (`vb_intended_xy`).

### 2026-10-21 11:30 — D24 result: upright stance works, but the run blew up at the very end
* **Training collapsed after iteration 57 781.** Up to 57 750 it was healthy (episodes 420–450 steps,
  falls ~3 %, virtual hit ~86 %, torso 0.74 m vs reference 0.80 m). Then single batches had huge actions:
  `action_rate_l2` reached −2.7×10¹⁸, the value loss went to `inf` at 57 856, and by 58 299 every episode
  fell within 17 steps. **`model_58299` is unusable** (MuJoCo: 12/12 falls). The action noise std had been
  creeping up the whole run (0.42 → 0.68). Next time: clip actions and/or lower the learning rate / entropy
  coefficient so one bad batch cannot wreck the critic. The last checkpoint before the blow-up is
  **`model_57500`**; it is the one evaluated below (`artifacts/evals/g1_bh_stand_model_57500/`).
* **MuJoCo, 120 held-out balls per row** (seeds 3–5 × 40, training paddle contact `--racket-contact venue`,
  aim offset fitted on seeds 0–2; `model_57500` offset (+0.345, −0.026)):

  | policy | shove | legal | in zone | falls | pelvis | foot slide p90 | foot lift > 1 cm |
  |---|---|---:|---:|---:|---:|---:|---:|
  | `g1_bh_still` 50299 (crouch 0.63) | none | 100 % | 62 % | 0 % | 0.628 m | 0.2 cm | 0 % |
  | | 0.3 m/s | 98 % | 58 % | 1 % | | 0.9 cm | 6 % |
  | | 0.5 m/s | 74 % | 25 % | 13 % | | 2.6 cm | 41 % |
  | `g1_bh_scratch` 9999 (crouch, new model) | none | 92 % | 42 % | 0 % | 0.610 m | 0.1 cm | 0 % |
  | | 0.3 m/s | 88 % | 35 % | 1 % | | 0.7 cm | 2 % |
  | | 0.5 m/s | 64 % | 24 % | 12 % | | 2.6 cm | 41 % |
  | **`g1_bh_stand` 57500 (upright)** | none | 95 % | 47 % | 0 % | **0.700 m** | 0.5 cm | 0 % |
  | | 0.3 m/s | 94 % | 44 % | 1 % | | 1.3 cm | 13 % |
  | | 0.5 m/s | 56 % | 21 % | 22 % | | 6.0 cm | 44 % |

* **Reading:** standing taller (pelvis 0.63 → 0.70 m, knees much straighter) costs some aim (in zone
  62 → 47 %) and makes hard shoves worse (0.5 m/s: 22 % falls vs 13 %), as expected from a higher centre of
  mass. Without shoves and with small shoves it still stands still: no falls, feet slide < 1.5 cm. It did not
  reach the reference height (0.70 vs 0.755 m); the 8 000 iterations were cut short by the collapse.
* Video: `artifacts/evals/g1_bh_stand_model_57500/backhand_12balls.mp4` (12 balls, no aim offset,
  MuJoCo contact: 12/12 contact, 0 falls, none over the net, same as the crouched policies in that setting).

### 2026-10-20 22:10 — D24: backhand drill back to a normal upright stance (`g1_bh_stand`)
* The user: the pelvis can be higher; the knees are bent inward instead of over the toes, and the deep bend
  loads the knees. Asked for the normal pelvis height and a normal straight stance.
* New clip `g1ikfist_hope_backhand_stand` (the G1 stand keyframe's height and feet; `plant_clip_feet.py` gained
  `--foot-yaw-deg` and `--foot-x`). Per clip (mean over the swing): A3 backhand 0.675 m / knees 71° / hip yaw 13° /
  ~23 N·m; flat063 0.630 m / 82° / 11° / ~27 N·m; **stand 0.755 m / 42° / 2° / ~18 N·m**. Ankle roll ≤ 0.017 rad,
  CoM 8 cm inside the feet. Reach: 100 % of a 3000-row bank sample from the stand clip's strike pose.
* `g1_bh_stand`: task `HOPEPingPongUnitreeG1_BH_FIST_STAND` (design: [decisions D24](../decisions.md)), warm start
  from `g1_bh_still` `model_50299` (slim checkpoint works as a warm start), 16 384 envs, to 58 300.

### 2026-10-20 21:30 — D23 result: both backhand policies stand still; the warm start aims better (62 % in zone)
* Runs done: `g1_bh_still` (warm start from `model_40299`) at 50 299, `g1_bh_scratch` (new model) at 9 999.
* MuJoCo (fist_sole model, robot reset in the flat063 clip pose at the station, continuous, 40 balls × seeds;
  aim offset fitted on seeds 0–2, tested on held-out seeds 3–5 = 120 balls per row; new per-ball foot metrics
  `foot_slide_max_m` / `foot_lift_max_m` in `mujoco_eval_onnx.py`):

  | policy | contact model | shove | legal | **in zone** | falls | foot slide p90 | foot lift > 1 cm | pelvis |
  |---|---|---|---:|---:|---:|---:|---:|---:|
  | warm start | venue (= training) | none | **100 %** | **62 %** | 0 % | 0.2 cm | 0 % | 0.628 m |
  | warm start | venue | 0.3 m/s | 98 % | 58 % | 1 % | 0.9 cm | 6 % | 0.629 m |
  | warm start | venue | 0.5 m/s | 74 % | 25 % | 13 % | 2.6 cm | 41 % | 0.635 m |
  | new model | venue | none | 92 % | 42 % | 0 % | 0.1 cm | 0 % | 0.611 m |
  | new model | venue | 0.3 m/s | 88 % | 35 % | 1 % | 0.7 cm | 2 % | 0.610 m |
  | new model | venue | 0.5 m/s | 64 % | 24 % | 12 % | 2.6 cm | 41 % | 0.614 m |
  | warm start | MuJoCo rigid | none | 3 % | 3 % | 0 % | 0.2 cm | 0 % | 0.629 m |
  | new model | MuJoCo rigid | none | 1 % | 0 % | 0 % | 0.1 cm | 0 % | 0.612 m |

  Aim offsets (venue contact): warm start (+0.26, −0.06) m, new model (+0.30, −0.00) m.
* **Feet:** without shoves neither policy moves its feet (largest slide 2 mm, no lifts); at 0.3 m/s
  almost never; at 0.5 m/s ~40 % of balls see a foot lift > 1 cm and 12–13 % falls (never trained with shoves).
* **Swing:** at contact the racket direction matches the command (cos 1.00) and the face is 1° off; the speed is
  88 % of the command (1.65 vs 1.93 m/s), which is why the returns land ~0.3 m short before the offset.
* **Contact model gap:** with MuJoCo's own rigid ball–racket contact only 1–3 % of returns are legal (median
  0.6–0.7 m short, mostly into the net), against 92–100 % with the venue contact model that training, the bank
  and the planner use. The policy is not the problem there; the two physics models disagree on how hard a
  1.6 m/s backhand block sends the ball (the forehand, at ~2.5 m/s, was much less affected). Which one matches a
  real paddle has to be measured (research recommendation 3a: re-fit restitution).
* Warm start wins on aim (62 % vs 42 % in zone) with equal balance; the new model stands 2 cm lower.

### 2026-10-20 16:55 — D23 status + first backhand MuJoCo videos
* `g1_bh_still` finished at 50 299 (final checkpoint home: the session-idles fix works). Last 1000 iterations
  flat: episodes ~420–460, virtual hit 0.86, velocity pass 0.17, landed in Isaac 0.61, no falls, `feet_planted`
  ends ~24 %, foot max slide 3.9 cm, pelvis 0.635 m. `g1_bh_scratch` at 7 698: episodes 463, hit 0.84,
  velocity pass 0.19, landed 0.60, no falls, `feet_planted` 16 %, slide 3.4 cm, pelvis 0.633 m (finishes ~20:30).
* `mujoco_eval_onnx.py`: the swing side now comes from the bank row's clip (backhand banks), and
  `--stand-clip NPZ` resets the robot in that clip's first frame at the bank's single station (as training
  does) instead of the stand keyframe at the origin. `mujoco_pingpong_scene.set_robot_pose`.
* Videos (12 balls, seed 1, no shoves, no aim offset, flat-sole model): `artifacts/evals/g1_bh_still_model_50299/`
  and `artifacts/evals/g1_bh_scratch_model_7500/` `backhand_12balls.mp4`. Both: 12/12 contacts, 0 falls, pelvis
  0.63 m. Over the net: warm start 1/12, new model 5/12; nothing in the centre zone yet. The returns land short
  (into the net), as the forehand did before its aim offset; the offset and the full comparison come next.

### 2026-10-20 12:40 — laptop disk full: checkpoints slimmed
* 1.6 GB free; `artifacts/antioch/` held 22 GB. Of each 320 MB checkpoint, 314 MB is the exact-resume simulator
  snapshot, only needed to continue the same run bit-exactly. `scripts/slim_checkpoint.py` keeps weights,
  optimizer, iteration and the env step counter (6.6 MB); the ONNX exported before and after is byte-identical.
  All 80 local checkpoints slimmed (21.2 GB freed), finished-run logs gzipped, the stuck `g1_bh_still_scratch`
  checkpoints deleted (log kept). `collect.sh` now slims each checkpoint on arrival. Free: 24 GB. Full
  checkpoints of the two running jobs stay on their sessions until release.

### 2026-10-20 04:40 — D23 new-model run restarted with an early-termination cost (`g1_bh_scratch`)
* `g1_bh_still_scratch` was stuck: at iteration 2 842 every episode still ended after ~15 steps, almost all by
  `ee_body_pos` (a foot > 25 cm off the reference height), and the action noise had already shrunk to 0.15.
  15 steps = the 0.3 s settle window: the policy kicked a foot up exactly when the foot costs start. For a
  policy that cannot stand yet, ending the episode cost nothing, so it learned to end it. Stopped and released.
* Fix: `rewards.early_termination` (Isaac Lab `is_terminated`, any end except the time limit), installed off;
  task `HOPEPingPongUnitreeG1_BH_FIST_STILL_SCRATCH` = the backhand drill + weight −200 (−4 once per early end).
  The warm-start run is unchanged.
* `g1_bh_scratch` (0 → 10 000, 16 384 envs): at iteration 383 episodes are 342 / 500 steps, 67 % reach the time
  limit, falls 8 %, `feet_planted` 25 %, pelvis 0.648 m, action noise 0.42 (healthy). No hits yet.
* Warm start `g1_bh_still` at 43 094: episodes 468, virtual hit 0.90, `feet_planted` 19 % (rising with the
  rally-pace ramp), no falls, pelvis 0.632 m.
* `g1_fh_planted16k_nopush` also ended without its final checkpoint (session failed at the end, as above);
  44 000 is home.

### 2026-10-20 00:05 — D23: second backhand run from a NEW model (`g1_bh_still_scratch`)
* The user asked why the backhand run starts from the forehand `model_40299` instead of a new model, and asked
  to train a new one in parallel. Same task `HOPEPingPongUnitreeG1_BH_FIST_STILL`, no checkpoint, 0 → 10 000,
  16 384 envs (session `ses-76G0YF4V73MZACAFAMFB92ZZPZ`). The two will be compared in MuJoCo at matching
  iteration counts (foot motion, falls, centre-zone rate).
* Warm start trade-off: it keeps the forehand's balance and whole-body control, but also its habit of stepping
  (the forehand station kept changing) and a forehand arm; a new model has no habit but learns standing,
  tracking and hitting from zero.
* First readings: warm start at 40 669 (370 iterations in): episodes 463 / 500 steps, `feet_planted` ends 12 %
  (100 % at the start), virtual hit 0.75. New model at iteration 32: episodes 16 steps, `feet_planted` 67 %,
  no hits yet (expected this early).

### 2026-10-20 00:00 — D23 backhand standing-still drill started (`g1_bh_still`)
* The user put the three D22 jobs aside and asked for a new approach: backhand, robot left of the table
  centre, one target in the middle of the opponent's half, full focus on balance, feet not moving, no
  sinking into the floor, pelvis low; stepping becomes a separate policy later. Design in
  [decisions D23](../decisions.md).
* **Feet in the floor, measured (MuJoCo):** the upstream foot touches the floor through four 5 mm spheres
  while the visible sole is flat and longer. A foot tilted 15° shows its sole up to 5.7 mm under the floor
  (pitch) / 2.0 mm (roll); with the new sole box: 0. Standing still, the soft contact sinks 0.65 mm; the
  sole box with `solref` 0.005 sinks 0.37 mm.
* **Old backhand clip:** left foot rolled 10–12° (on its edge), right sole 3 mm under the floor. The new clip
  `g1ikfist_hope_backhand_flat063`: both feet flat and fixed (error < 0.001 mm), pelvis 0.63 m, CoM ≥ 6.6 cm
  inside the support area. With the source stance (0.344 m) the left ankle roll sat 3.3° from its soft
  limit; 0.32 m leaves ≥ 6.6° both ways.
* **Bank:** 52 000 backhand balls, station y +0.25 m, target `opp_center_mid`; racket 1.1–2.5 m/s
  (mean 1.76), feed 4.5–6.2 m/s before the bounce, 0.50–0.77 s from the far end line to the apex. IK: 100 %
  reachable from the clip's strike pose with waist + arm (a wider test box drops to 43 % at 0.58–0.62 m
  ahead, so the filter does reject unreachable balls).
* **Smoke tests (Antioch, 1024 envs):** (1) `feet_planted` ended 100 % of episodes within ~15 steps and the
  sole corners went 1.2 cm into the floor: the reset noise (root z ±1 cm, roll/pitch ±0.1 rad) put the flat
  soles into the floor and the depenetration kicked the feet away. Fixed: z only upward (+0.4..1.5 cm),
  roll/pitch ±0.02 rad, 0.3 s settle time (`foot_settle_s`), every episode from the clip pose (the default
  stand is narrower than the clip). (2) After the fix: deepest sole corner 0.0 mm, highest 0.5 mm, sole
  tilt ≤ 0.07 rad, pelvis 0.638 m; `feet_planted` still ends the episodes but they grow fast
  (23 → 79 steps in 65 iterations): the forehand policy is unlearning its steps.
* **Antioch:** when `train.sh` ends, the service command exits and Antioch marks the session failed at once,
  so its files are gone: the final checkpoints of `ss16k_nopush` / `still16k_nopush` (44 299) were lost;
  44 000 is home. The service now idles after training; `collect.sh` releases the session. Max 3 sessions.
* **Main run:** `g1_bh_still`, 16 384 envs, `model_40299` → 50 300.

### 2026-10-19 22:20 — no-shove jobs: hit rate drops as rally pace ramps back in (expected); `still` falls at full pace
* All three are warm starts from `model_40299`, so the rally-pace and timing-jitter ramps start over (D15): the share of
  rally-pace swings goes 0 → 0.6 over 3000 iterations (40 300 → 43 300). The Isaac virtual hit rate follows it
  down, as it did in D21 (0.99 → ~0.75):

  | iteration | rally-pace share | ss16k hit | still16k hit | still16k falls | planted16k hit | planted16k ended by foot rule |
  |---:|---:|---:|---:|---:|---:|---:|
  | 40 500 | 0.04 | 1.00 | 1.00 | 0.1 % | 1.00 | 38 % |
  | 41 500 | 0.24 | 1.00 | 0.99 | 1.4 % | 0.97 | 39 % |
  | 42 500 | 0.44 | 0.94 | 0.92 | 16 % | (0.94 at 42 250) | 40 % |
  | 43 500 | 0.60 | 0.80 | 0.79 | **21 %** | – | – |

* `ss16k_nopush`: still ~0 falls at full pace. `still16k_nopush`: falls climb with the pace to ~21 % of ended
  episodes. The late clip entry (time-to-strike 0.4–0.83 s, no hold) needs the feet to move, and the foot costs
  stop that. `planted16k_nopush`: the foot rule still ends ~40 % of episodes; it reaches full pace at ~43 300.
* The background collector had stopped with the previous Claude session (laptop process); restarted at 22:20.
  Checkpoints up to 42 000 (ss/still) and 41 500 (planted) were already home.
* ETA: ss/still ~23:10 PDT, planted ~02:40 PDT.

### 2026-10-19 17:20 — all three D22 jobs restarted WITHOUT shoves (user's decision)
* The user asked to stop shoving the robot in training. The three shoved runs (`g1_fh_ss16k` at ~41 800,
  `g1_fh_still16k_v2` and `g1_fh_planted16k_v2` just started) were stopped and their sessions released.
* Restarted from `model_40299` to 44 300, 16384 envs, with `task.domain_rand.push_velocity_range=null`
  (no `push_robot` event; the HOPE base config has none): `g1_fh_ss16k_nopush`, `g1_fh_still16k_nopush`,
  `g1_fh_planted16k_nopush`. Everything else as in D22 (still/planted keep the 0.8 s grace setting, which does
  nothing without shoves). The startup log confirms no push event.
* Script fixes: `pgrep -f scripts/train.py` inside `bash -c` matched its own command line, so `status.sh`
  reported a job that never started as "running" and `collect.sh` would never have released a finished
  session; now `pgrep -f "[s]cripts/train.py"`. A start whose uploads time out (slow network with three
  uploads at once) must be finished by hand: `SESSION=... ./push_ignored.sh`, `service cp` the checkpoint,
  `service exec -- touch /workspace/project/.train_go`.

### 2026-10-19 16:30 — D22 v1 still/planted fail under shoves; restarted with a recovery grace (v2)
* As the shoves ramped up (0 → 0.8 m/s over the first 2000 iterations), the two foot-constrained jobs got
  worse every 250 iterations, while `g1_fh_ss16k` stayed at 0 falls (share of ended episodes):

  | iteration | ss16k falls | still16k falls | planted16k ended by foot rule | planted16k episode length |
  |---:|---:|---:|---:|---:|
  | 40 500 | 0 % | 0.1 % | 38 % | 344 |
  | 41 000 | 0 % | 1.7 % | 58 % | 297 |
  | 41 250 | 0 % | 8 % | 77 % | 242 |
  | 41 500 | 0 % | **34 %** | **89 %** | 198 |

  Without a step the robot cannot absorb an 0.8 m/s shove: `still` (stepping costs) falls, `planted` breaks the
  foot rule. The hitting metric also dropped (velocity pass ~15 % vs 34 % in ss16k). This matches the survey:
  no team plants the feet, and stepping is what saves a robot from shoves.
* **v2 (user's choice):** v1 still/planted stopped at ~41 750 (checkpoints 40 500 and 41 000 kept). New setting
  `foot_push_grace_s: 0.8`: the foot costs and the planted rule are off for 0.8 s after each shove, and the feet's
  reference spot follows the feet then, so a recovery step is free and its landing spot becomes the new spot.
  Everywhere else the feet must stay still. Runs `g1_fh_still16k_v2` and `g1_fh_planted16k_v2`, again from
  `model_40299` to 44 300; `g1_fh_ss16k` continues.

### 2026-10-19 14:30 — deploy fix removes the stepping; three "feet still" jobs (D22) on Antioch
* **Fix (lifecycle `hold_last_target`, on for the G1):** after contact the policy now keeps seeing the last
  ball's target with time-to-strike running negative (floor −1.26 s) until the next ball, as in training.
  The ready reach (used only before the first ball) moved to where training targets are (0.45, −0.35, +0.33).
* **Feet** (`model_40299`, 16 balls, seed 3, per-tick log):

  | | steps per ball | balls with no step | foot travel per ball | max foot shift (median) |
  |---|---:|---:|---:|---:|
  | old deploy target | 3.5 | 0 % | 91 cm | 26 cm |
  | **with the fix** | **0.0** | **100 %** | **7 cm** | **2.5 cm** |

  The stumbling was almost all caused by the wrong deploy target. The robot now reaches every ball from where
  it stands (the pelvis still sways ~21 cm per ball).
* **Aim offset re-measured** (seeds 0–2, no shoves): **(+0.23, −0.01) m** (was +0.27, −0.08; the sideways bias
  is gone). Held-out seeds 3–5, 120 balls each, fist model, MuJoCo contact:

  | shove | fell | hit | legal | in zone (fix) | in zone (old target, old offset) |
  |---|---:|---:|---:|---:|---:|
  | none | 0 % | 100 % | 63 % | **30 %** | 37.5 % |
  | 0.5 m/s | 0 % | 100 % | 62 % | 21 % | 30 % |
  | 1.0 m/s | 0 % | 99 % | 61 % | 16 % | 21 % |

  Without the offset (seeds 0–2), legal returns rose with the fix at every shove level (+5 to +11 points).
  The zone differences are within the noise of 120 balls (±4 points per row) but all point the same way, so
  the fix may cost a little accuracy; the median miss is unchanged (~15 cm).
* **Three 16384-env jobs on Antioch (D22),** all from `model_40299` to 44 300: `g1_fh_ss16k` (current recipe),
  `g1_fh_still16k` (fixed station + foot costs), `g1_fh_planted16k` (+ episode ends if a foot slides 5 cm / lifts
  3 cm). ~5 s per iteration with 16384 envs (4× the samples of the cluster runs per iteration), ~5.5 h each.
  Early: planted job ends 36 % of episodes on the foot rule (learning to stop), no falls in any job.
* **Research** on SMASH / PKU / HITTER / PACE: [research/humanoid_table_tennis.md](../research/humanoid_table_tennis.md).

### 2026-10-19 12:45 — training moved to Antioch (cloud GPU); setup tested
* **Why:** the user switched training from the SDSC cluster (broken GPUs on zixian) to Antioch.
* **Setup** (`antioch_sim/`, usage in [antioch_sim/README.md](../../../antioch_sim/README.md)): engine `base` + our
  own Dockerfile with the cluster's exact stack (Isaac Sim 5.1, Isaac Lab v2.3.2, torch 2.7.0+cu128, rsl-rl 3.1.2;
  `cluster-constraints.txt` is the cluster venv's pip freeze). Antioch's own engines are Isaac Sim 6 / Isaac Lab
  3.0, which would need a port of HOPE and could change how checkpoints behave. GPU: 1× RTX PRO 6000 Blackwell
  (96 GB), 24 CPU cores, us-central. Launcher `hope_training/antioch/train.sh` = `train.sbatch` without Slurm.
* **Test:** `model_40299` resumed for 20 iterations on `_FH_FIST_STABLE_SWING`: runs, the D21 terms are active,
  value loss 0.05. The SMPL models, `artifacts/` and `.git` are not uploaded (checked on the session).
* **Speed:** 4096 envs = ~16k steps/s, 6.0 s per iteration: **the same as the 2080 Ti**. Collection takes 5.9 s
  of it whatever the GPU, so the step is bound by per-step overhead (many small Python/GPU calls), not GPU compute.
  16384 envs fit and run at 45-61k steps/s (6.4-8.7 s per iteration, 3-4× the samples) but change the PPO batch;
  not adopted without a decision.
* **Problems found and fixed:** (1) the base image points `uv` at its Python 3.12, so every install names the
  3.11 venv; (2) Isaac's USD libraries need `libgomp1` and `libatomic1`; (3) a download timeout from the PyTorch
  index (longer timeout + retries); (4) Antioch sync skips git-ignored files even when committed, so the four
  `assets/venue_tuple/g1_fh_*` bank files are copied with `antioch_sim/push_ignored.sh`; (5) after a Python error
  Isaac hung in shutdown and kept the GPU busy for 30 min (as 5719 did on the cluster): `train.sh` now has a
  watchdog that kills it 300 s after an error or the final save (tested).
* Session released; nothing running. Session files are deleted on release: copy checkpoints home first.

### 2026-10-19 02:30 — cause of the post-hit sideways step: the eval/deploy "ready" target is on the wrong side
* **Isaac does not do it.** `trace_base_motion.py` (job 6069, 256 envs × 30 s, training task with shoves,
  1706 strikes): after the strike the pelvis stays within ~1 cm until the clip ends (+1.3 s), then moves
  straight to the next station (−17 cm when it is to the right, +17 cm when to the left). Only 1 % of strikes
  move > 10 cm sideways by +1.4 s. (The run without shoves, job 6071, is not analysed yet.)
* **The MuJoCo eval feeds something training never shows.** `SwingLifecycle` (shared with deploy) switches to
  RECOVERY 0.6 s after the strike and then shows a "ready reach" target = pelvis + (0.32, side × 0.16, −0.04) m
  with time-to-strike pinned at 1.0 s. For the forehand (side +1) that is 16 cm to the robot's **left** and
  4 cm **below the pelvis**. Training forehand targets are 0.42–0.50 m ahead, 0.32–0.38 m to the **right** and
  0.9–1.1 m high. The values are the A3 example (0.40, 0.20, −0.05) × 0.81, not re-derived for the G1. The
  target is recomputed from the pelvis every tick, so the robot keeps chasing it to the left until the next
  ball is announced (+1.4 s).
* **Test** (same 16 balls, MuJoCo, base y change after the strike):

  | ready reach (x, y, z) m | +1.0 s | +1.4 s | > 10 cm left at +1.4 s |
  |---|---:|---:|---:|
  | (0.32, +0.16, −0.04) current G1 config | +13 cm | +22 cm | 12 / 15 |
  | (0.32, −0.16, −0.04) side flipped | +7 cm | +10 cm | 9 / 15 |
  | (0.45, −0.35, +0.33) where training targets are | +2 cm | +3 cm | 5 / 15 |

* **Hitting with the training-like ready target** (120 held-out balls, aim offset (+0.27, −0.08)): in zone 35.8 %
  (was 37.5 %), legal 59 % (62.5 %), no falls; at 1.0 m/s shoves 15.0 % (20.8 %), hits 99 % (97.5 %). The
  differences are within the noise of 120 balls (±4 points). The aim offset was fitted with the old ready target.
* **Not changed yet:** `hope_pingpong_runtime_unitree_g1.yaml` still has (0.32, 0.16, −0.04); all results in these
  docs use it. A more faithful fix is to keep showing the last strike target until the next ball, as training does.

### 2026-10-19 01:30 — why the robot leans/moves left after the hit (`model_40299`, MuJoCo)
Measured every physics step, 16 balls (seed 3), averaged around each hit; the old `model_32299` does the same.
* **0–0.4 s: follow-through.** The waist turns left (waist yaw +12° at +0.2 s) and the torso leans forward ~10°.
  The reference clip does exactly this (waist yaw +12.1° at +0.2 s), so it is the normal forehand finish.
* **0.8–1.7 s: a sideways shuffle.** The pelvis moves 25–35 cm to the left (15 of 16 hits, both feet step),
  often past the table centre line of the robot's area (up to +21 cm), with the CoM up to 13 cm left of the feet
  and the pelvis rolled 2–4° left. By ~2.0 s it walks back toward the next ball's station.
  * Not from the clip: its pelvis and feet move < 1 cm sideways after the strike.
  * Not from the eval's command: the base target is the ball's station (y −0.45…+0.12 m), never that far left.
  * Not a balance catch: the pelvis leads the CoM, so it is a deliberate step, not a fall being caught.
  * So it is learned behaviour from training. Cause not yet confirmed. Training also uses stations from the
    same bank (mean y −0.21 m) and a post-swing base-speed cost (0.20–1.55 s) that should discourage it.
    Next check: log the base target and base y after the strike in a short Isaac run.
* No falls come from it, but it wastes ~0.6 s of movement between balls and could matter at real rally pace.

### 2026-10-19 00:40 — D21 results: no falls up to 1.0 m/s; new best `stable_swing` `model_40299`
* **5859 `g1_fh_stable_swing`** finished 40 299 on 10-17 21:32. **5858 `g1_fh_stable`** crashed at 40 067:
  the value loss blew up to ~5e30, the action std became invalid (`normal expects all elements of std >= 0.0`),
  and the process hung until we cancelled it on 10-19. Its value loss had spiked (>100) 266 times since 32 700.
  The swing run had 18 spikes; the fist run 5719 had 9 over its last 8000 iterations.
  Checkpoints up to 40 000 are from before the crash.
* **Isaac:** as the shoves ramp up (first ~2000 iterations), the velocity pass rate falls from 0.59 to ~0.19 and
  the virtual hit rate from 0.99 to ~0.75; then it stays flat. The pelvis goes 0.692 → 0.685 m. Falls stay ~0.
* **MuJoCo** (fist model, 8 checkpoints screened, the best 3 on 120 balls per shove level; full table in
  [runs_and_results.md](runs_and_results.md)):
  * no falls at any shove level up to 1.0 m/s (`model_32299`: 5 %); at 1.0 m/s 93 % hits (82 %);
  * pelvis 0.66 m (was ~0.72), CoM closer to the feet;
  * **balls land short**: median 0.26–0.56 m short without an offset (old 0.21), so only 6–12 % in the zone;
  * with an aim offset fitted on seeds 0–2 and tested on 3–5, `stable_swing` `model_40299` with (+0.27, −0.08) m
    puts **38 %** in the zone (`model_32299` with its offset: 36 %), 30 % at 0.5 m/s and 21 % at 1.0 m/s
    (`model_32299`: 28 % and 17 %).
* **Conclusion:** balance-first worked without losing accuracy, provided the aim offset is re-fitted. The run
  with sharper racket rewards is better than balance alone (38 % vs 24 % in zone with offset).
  **New best policy: `g1_fh_stable_swing` `model_40299`.**

### 2026-10-17 00:15 — zixian fixed (rebooted); balance-first runs (D21) started
* **Node:** zixian was rebooted on 10-16 at 15:56 (Slurm `BootTime`). A plain torch CUDA probe (5856) works on
  idx 1. We still never use idx 0 or 5.
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

