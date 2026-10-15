# Runs, training curves and per-zone success

[繁體中文](runs_and_results.zh-TW.md) · [Docs index](../README.md)

## Runs

| Job | Run name | Task (questions) | Started | State | Last iteration | Note |
|---|---|---|---|---|---|---|
| 5686 | `g1_fh_fist` | `HOPEPingPongUnitreeG1_FH_FIST` (rally pace + shoves + **fist grip**, D19/D20) | 10-14 21:40 | **running** | warm start 17 300 → 32 300 | the run that matters for the real robot |
| – | `g1_fh_bal` | `HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL` (shoves on the old mount, D19) | – | not submitted | – | superseded by `g1_fh_fist` |
| 5640 | `g1_fh_rally` | as 5560 | 10-13 | failed at start ("CUDA unknown error" node-wide) | 5985 | resubmission of 5584; superseded by `g1_fh_bal` |
| 5584 | `g1_fh_rally` | as 5560 | 10-12 12:39 | crashed 10-12 17:22 (GPU idx 5 fell off the bus), cancelled | 5985 (`model_5900`) | continued 5560 with the step counter restored |
| 5560 | `g1_fh_rally` | `HOPEPingPongUnitreeG1_FH_XC_RALLY` (faster-feed bank + rally pace + timing jitter, D14) | 10-11 21:04 | killed by the 10-12 02:29 outage (not requeued) | 4677 | warm start from `g1_fh_diag` `model_2600`; 2 GPUs requested because of a broken GPU ([status log](log.md)) |
| 4382 | `g1_fh_diag` | `HOPEPingPongUnitreeG1_FH_XC` (our bank: diagonal feed, apex, 4 zones) | 10-10 20:12 | **queued** (nodes down) | 15 756 (`model_15700`) | **main run**; requeued 13 times (outages, preemption; [status log](log.md)) |
| 5554 | `g1_fh_rally` | as 5560 | 10-11 21:02 | failed at start | – | assigned the broken GPU |
| 4354 | `g1_fh` | `HOPEPingPongUnitreeG1_FH` (inherited A3-geometry questions) | 10-10 18:07 | failed (node crash), not resumed | 2196 | baseline; kept at `model_2100` |
| 4381 | `g1_fh_diag` | earlier bank version | 10-10 20:03 | cancelled after 3 min | – | superseded by the zone fix |
| 4379 | `g1_fh_diag` | earlier bank version | 10-10 19:54 | cancelled after 2 min | – | superseded |
| 4367 | `g1_fh_xc` | bank before the diagonal feed | 10-10 19:02 | cancelled after 47 min | – | feed changed to the diagonal |

---

## Training curves

`progress.sh 4382` (main run). Iterations 1300–1374 were trained twice because of the crash; the
table keeps the later values.

| iteration | reward | racket ≤10 cm | all OK | contact | legal return | landing err (m) | falls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -7.90 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | 0.0245 |
| 500 | 18.89 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | 0.3788 |
| 1000 | 22.80 | 0.0012 | 0.0000 | 0.0005 | 0.0000 | 1.7829 | 0.3736 |
| 1500 | 27.18 | 0.2713 | 0.0009 | 0.1261 | 0.0198 | 1.2790 | 0.2946 |
| 2000 | 34.41 | 0.6265 | 0.0349 | 0.4127 | 0.2070 | 0.7731 | 0.2363 |
| 2372 | 39.05 | 0.6967 | 0.0593 | 0.4758 | 0.2176 | 0.7565 | 0.1809 |

`progress.sh 4354` (baseline, its own questions; not directly comparable):

| iteration | reward | racket ≤10 cm | all OK | contact | legal return | landing err (m) | falls |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -7.91 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | – | 0.0243 |
| 500 | 20.60 | 0.1881 | 0.0000 | 0.0018 | 0.0000 | 2.6143 | 0.3900 |
| 1000 | 28.53 | 0.1578 | 0.0036 | 0.0006 | 0.0000 | 1.8976 | 0.3732 |
| 1500 | 31.74 | 0.4301 | 0.0195 | 0.0659 | 0.0254 | 0.8889 | 0.3410 |
| 2000 | 36.08 | 0.6023 | 0.0581 | 0.1347 | 0.0563 | 0.8602 | 0.3045 |
| 2196 | 33.66 | 0.6157 | 0.0773 | 0.1100 | 0.0459 | 0.8078 | 0.2785 |

Reading: the main run spent ~1000 iterations learning to stand and follow the reference swing, then
learned to reach the contact point between 1000 and 2000. It now captures about half the balls in
training and returns a fifth legally. Falls keep dropping.

---

## Club success rate per zone

### Isaac (virtual ball, 512 environments × 60 s, deterministic policy)

Rates are over all attempts and form a funnel (each column also requires the ones to its left);
success also requires the apex timing, which the bank guarantees in Isaac ([how we track](how_we_track.md)).

**`g1_fh_diag` `model_2300`** (2026-10-11, 8101 attempts, 22 lost to falls)

| zone | attempts | contact | + over net | + opponent half | + in zone = **success** |
|---|---:|---:|---:|---:|---:|
| `opp_right_mid` | 2056 | 93.9 % | 92.2 % | 85.6 % | **24.5 %** |
| `center_line_deep` | 2003 | 90.1 % | 88.1 % | 76.1 % | **6.2 %** |
| `opp_right_deep` | 2033 | 91.3 % | 89.6 % | 77.6 % | **11.3 %** |
| `opp_right_corner` | 2009 | 93.7 % | 92.3 % | 78.1 % | **2.3 %** |
| **all** | 8101 | 92.3 % | 90.6 % | 79.4 % | **11.1 %** |

**Baseline `g1_fh` `model_2100`** on the same questions (it was trained on the inherited A3-geometry
questions, never on this bank; 8117 attempts, 75 falls)

| zone | attempts | contact | + over net | + opponent half | + in zone = **success** |
|---|---:|---:|---:|---:|---:|
| `opp_right_mid` | 2030 | 85.3 % | 84.8 % | 79.0 % | **11.9 %** |
| `center_line_deep` | 2078 | 83.0 % | 77.8 % | 76.0 % | **2.4 %** |
| `opp_right_deep` | 2009 | 80.9 % | 72.2 % | 69.3 % | **3.5 %** |
| `opp_right_corner` | 2000 | 75.6 % | 66.1 % | 61.4 % | **0.7 %** |
| **all** | 8117 | 81.2 % | 75.3 % | 71.5 % | **4.6 %** |

Training on our bank pays off: at about the same age (2300 vs 2100 iterations) the main run is
2.4× better overall and 3× better in the corner, with fewer falls (22 vs 75).

Reading: in Isaac the policy already reaches and meets almost every ball and returns ~80 % legally;
what it cannot do yet is place the ball. The open middle zone is easiest; the three end-line zones,
especially the corner, need precise racket velocity, so they come last.

### MuJoCo (real ball, `eval_checkpoint.sh` → `eval_bank.{json,mp4}`)

Each serve is a bank question played with a real simulated ball: the feed is launched from the far
end line with its spin (Magnus lift on), the table bounce uses the venue-fitted table model (MuJoCo's
own soft contact sent the ball 20–40 cm too deep; with the fitted bounce the real ball reaches the
bank's apex within 1.3 cm median), a deploy-style planner predicts the apex from the real ball every
tick, and the racket–ball contact is MuJoCo's. Timing is measured on the real ball (|vz| at contact).
12 serves, seed 0, so both checkpoints face the same 12 questions.

| checkpoint | contact | apex timing OK | over net | opponent half | in zone | success | falls |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model_2300` | 6/12 | 1/12 | 3/12 | 3/12 | 1/12 | 0/12 | 3/12 |
| `model_2400` | 10/12 | 5/12 | 3/12 | 2/12 | 0/12 | 0/12 | 0/12 |
| `model_15700` (10-14) | 12/12 | 0/12 | 11/12 | 8/12 | 0/12 | 0/12 | 0/12 |
| `model_15700`, strike_time clock + 40 ms (10-14) | 12/12 | 12/12 | 11/12 | 10/12 | 0/12 | 0/12 | 0/12 |
| `g1_fh_rally` `model_5900` (10-14) | 10/12 | 0/12 | 9/12 | 6/12 | 0/12 | 0/12 | 0/12 |
| `g1_fh_rally` `model_5900`, strike_time clock + 40 ms (10-14) | 10/12 | 7/12 | 7/12 | 7/12 | 0/12 | 0/12 | 0/12 |

Reading — the important gap: Isaac says ~90 % of strikes clear the net, MuJoCo says 25 %. Causes
seen in the per-serve records:
* **early strikes**: contact while the ball is still rising (vz +0.14 to +0.41 m/s, i.e. 15–40 ms
  before the apex); in Isaac the ball cannot be early or late;
* **real racket contact**: MuJoCo's racket bounce (restitution 0.654, face geometry, contact on the
  rim) is not the venue paddle model the reward trains against, so the same swing returns lower/shorter;
* **lead time**: the swing clip needs ~0.83 s from start to strike, but a fast feed flies only
  0.50–0.82 s (median 0.59 s) from the far end line to the apex. The evaluation announces each
  question 0.9 s before the apex (as training does); with `FEED_LEAD_S=0` (deploy condition) the
  policy has less time than it was trained for. Before hardware, either the swing clip must start
  later or the planner must predict the opponent's shot earlier.

The old example serves (`eval.json`) are not our task: with them `model_2300` and `model_2400` never
touch the ball; use `eval_bank` for the forehand.

#### Latest, 2026-10-14: per zone, 120 balls (3 seeds × 40, robot stood up after a fall)

The Isaac evaluation could not be rerun (zixian down since 10-13 20:15); these are MuJoCo only. The
ONNX of `model_15700` was built on the laptop (`onnx_from_checkpoint.py`). Same funnel as the Isaac
tables; "apex OK" is shown separately and "success" needs apex OK + in zone.

**`g1_fh_diag` `model_15700`, G1 deploy defaults (strike_time clock + 40 ms), 0.9 s notice**

| zone | balls | contact | apex OK | + over net | + opponent half | + in zone | **success** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `opp_right_mid` | 27 | 100 % | 88.9 % | 70.4 % | 70.4 % | 29.6 % | **29.6 %** |
| `center_line_deep` | 33 | 100 % | 100 % | 100 % | 100 % | 0 % | **0 %** |
| `opp_right_deep` | 28 | 100 % | 100 % | 89.3 % | 89.3 % | 0 % | **0 %** |
| `opp_right_corner` | 32 | 100 % | 96.9 % | 93.8 % | 68.8 % | 0 % | **0 %** |
| **all** | 120 | 100 % | 96.7 % | 89.2 % | 82.5 % | 6.7 % | **6.7 %** |

**`g1_fh_rally` `model_5900`, deploy defaults, real rally pace**

| zone | balls | contact | apex OK | + over net | + opponent half | + in zone | **success** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `opp_right_mid` | 26 | 84.6 % | 38.5 % | 34.6 % | 34.6 % | 11.5 % | **3.8 %** |
| `center_line_deep` | 28 | 82.1 % | 64.3 % | 60.7 % | 60.7 % | 3.6 % | **3.6 %** |
| `opp_right_deep` | 33 | 90.9 % | 87.9 % | 60.6 % | 60.6 % | 0 % | **0 %** |
| `opp_right_corner` | 33 | 93.9 % | 90.9 % | 78.8 % | 78.8 % | 0 % | **0 %** |
| **all** | 120 | 88.3 % | 72.5 % | 60.0 % | 60.0 % | 3.3 % | **1.7 %** |

Totals for the other conditions (same 120 balls):

| checkpoint | condition | contact | apex OK | over net | legal | in zone | success | falls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `model_15700` | original clock, no calibration | 99.2 % | 1.7 % | 92.5 % | 60.0 % | 5.8 % | 0 % | 0 % |
| `model_15700` | deploy defaults, rally pace | 91.7 % | 84.2 % | 80.8 % | 75.0 % | 6.7 % | 5.8 % | 0 % |
| rally `model_5900` | original clock, no calibration | 93.3 % | 0.8 % | 76.7 % | 61.7 % | 10.0 % | 0 % | 0 % |
| rally `model_5900` | deploy defaults, 0.9 s notice | 90.8 % | 73.3 % | 60.0 % | 58.3 % | 1.7 % | 1.7 % | 0.8 % |

Reading:
* **Timing is fixed** by the clock + calibration (apex OK 2 % → 97 % for the main run), and the main
  run now reaches every ball and returns 82 % legally.
* **Aim is the new bottleneck, and it is systematic:** in every zone the ball lands **30–60 cm short**
  of the aim point (median; `model_15700`: −30 cm mid, −43 cm centre line, −49 cm deep, −52 cm corner;
  rally `model_5900`: −38 to −63 cm), with a tight spread (IQR about ±10 cm). Only the middle zone,
  which is deep enough to catch a short ball, ever scores. Sideways error is small (0–15 cm).
* Likely cause: MuJoCo's racket bounce (restitution 0.654, rubber-free face) returns the ball slower
  than the venue paddle model the policy is trained against, so the same swing flies shorter. Which
  of the two matches the real racket is unknown until we measure it (next steps: real racket model).
* At rally pace the main run (`model_15700`) now does better than the rally run's older checkpoint
  (75 % vs 60 % legal): the clock fix removed most of the timing problem that rally training was
  compensating for. The rally run has trained 3300 iterations; the main run 15 700.

