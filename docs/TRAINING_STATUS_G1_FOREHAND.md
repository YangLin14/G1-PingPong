# G1 forehand training: status report and tracking log

[繁體中文版](TRAINING_STATUS_G1_FOREHAND.zh-TW.md) · Commands: [`RUNBOOK_G1_FOREHAND.md`](RUNBOOK_G1_FOREHAND.md) ·
Design: [`TRITON_G1_FOREHAND.md`](TRITON_G1_FOREHAND.md)

This file is the running record of our training runs: what we measure, how we measure it, and every
status check so far. Add a new entry to section 5 each time you check a run (newest at the top), and
a new row to section 4 whenever you run the per-zone evaluation. All times are Pacific (cluster time).

**Latest (2026-10-11 20:40):** `g1_fh_diag` (job 4382) is running at iteration ~2400 / 20 000, about
38 h to go. Club success of `model_2300` in Isaac: **11.1 %** overall (24.5 % in the middle zone,
2.3 % in the corner). In MuJoCo with a real ball `model_2400` meets 10/12 balls but only 3/12 clear
the net, mostly because it strikes 15–40 ms early (section 4).

---

## 1. How we track a run

Three tools, from cheapest to most complete. All run on the cluster from `~/hope/HOPE`.

| Tool | What it answers | Cost | Output |
|---|---|---|---|
| `status.sh` | Is it running? Latest training metrics, newest checkpoint | instant | screen |
| `progress.sh <job> [every]` | How did the metrics move over the run? | instant | markdown table (paste into section 3) |
| `eval_forehand.sh --in-job auto <run>` | **Success rate per target zone under the club definition** (Isaac, virtual ball) | ~20 min, shares the training GPU | `~/hope/evals/<run>/<model_N>/forehand_zones.{json,txt}` |
| `eval_checkpoint.sh --in-job auto <run>` | Same questions with a **real ball** in another simulator (MuJoCo): video + club success per zone, including real timing | ~4 min | `~/hope/evals/<run>/<model_N>/eval_bank.{json,mp4}` |

```bash
ssh sdsc-scc
cd ~/hope/HOPE
bash hope_training/slurm/pip_env/status.sh                               # one screen
bash hope_training/slurm/pip_env/progress.sh 4382                        # history table
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job auto g1_fh_diag
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag
```

### Training metrics (`status.sh`, `progress.sh`)

These come from the training log. They are running averages over the last few thousand strikes,
with the training noise and domain randomization on, so they read lower than an evaluation.

| Metric | Meaning |
|---|---|
| racket ≤10 cm | at the strike frame, the racket centre is within 10 cm of the planned contact point |
| all OK | position, velocity and face direction all within tolerance at the strike frame |
| contact | the virtual ball is captured: racket within 9.5 cm and moving into the ball |
| legal return | contact, over the net, first bounce on the opponent half (any spot) |
| landing err | distance from the landing point to the question's intended landing point |
| falls | share of episode ends that were falls (the rest are time-outs) |

### Club success rate (`eval_forehand.sh`)

The definition we agreed on (`utils/forehand_success.py`). A return succeeds only if **all** hold:

1. **timing**: struck at the top of the bounce on our side (incoming |vz| ≤ 0.3 m/s);
2. **contact**: racket within 9.5 cm of the ball and moving into it;
3. **over the net**, and the first bounce is **on the opponent half**;
4. the first bounce is **inside the question's target zone** (one of four on the opponent's right).

How the evaluation counts:

* every question (one bank row: feed, strike point, target zone) issued after warm-up is one attempt;
  a ball the robot never reaches because it fell is a **failed** attempt of that zone, never dropped;
* no time-out during the evaluation, so only falls end an episode;
* the policy runs deterministically (no action noise); 512 environments × 60 s;
* the table shows a funnel: contact → + over the net → + on the opponent half → + in the zone;
  success also requires the timing check.

Limit: in Isaac the ball is virtual and is at the contact point exactly at the strike frame, and the
bank puts that frame at the apex. So "timing" here checks the question, not the robot. Robot-side
timing needs a real simulated ball (MuJoCo) or the real robot; that is a later step.

---

## 2. Runs

| Job | Run name | Task (questions) | Started | State | Last iteration | Note |
|---|---|---|---|---|---|---|
| 4382 | `g1_fh_diag` | `HOPEPingPongUnitreeG1_FH_XC` (our bank: diagonal feed, apex, 4 zones) | 10-10 20:12 | **running** | ~2370 | **main run**; requeued once (section 5) |
| 4354 | `g1_fh` | `HOPEPingPongUnitreeG1_FH` (inherited A3-geometry questions) | 10-10 18:07 | failed (node crash), not resumed | 2196 | baseline; kept at `model_2100` |
| 4381 | `g1_fh_diag` | earlier bank version | 10-10 20:03 | cancelled after 3 min | – | superseded by the zone fix |
| 4379 | `g1_fh_diag` | earlier bank version | 10-10 19:54 | cancelled after 2 min | – | superseded |
| 4367 | `g1_fh_xc` | bank before the diagonal feed | 10-10 19:02 | cancelled after 47 min | – | feed changed to the diagonal |

---

## 3. Training curves

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

## 4. Club success rate per zone

### Isaac (virtual ball, 512 environments × 60 s, deterministic policy)

Rates are over all attempts and form a funnel (each column also requires the ones to its left);
success also requires the apex timing, which the bank guarantees in Isaac (section 1).

**`g1_fh_diag` `model_2300`** (2026-10-11, 8101 attempts, 22 lost to falls)

| zone | attempts | contact | + over net | + opponent half | + in zone = **success** |
|---|---:|---:|---:|---:|---:|
| `opp_right_mid` | 2056 | 93.9 % | 92.2 % | 85.6 % | **24.5 %** |
| `center_line_deep` | 2003 | 90.1 % | 88.1 % | 76.1 % | **6.2 %** |
| `opp_right_deep` | 2033 | 91.3 % | 89.6 % | 77.6 % | **11.3 %** |
| `opp_right_corner` | 2009 | 93.7 % | 92.3 % | 78.1 % | **2.3 %** |
| **all** | 8101 | 92.3 % | 90.6 % | 79.4 % | **11.1 %** |

TODO_BASELINE_TABLE

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

---

## 5. Status log (newest first)

### 2026-10-11 20:40 — first per-zone evaluations (Isaac and MuJoCo)
* New tools: `eval_forehand.sh` (Isaac, club success per zone), `progress.sh`, and `--serve-bank` in
  the MuJoCo evaluator (`eval_checkpoint.sh` now also writes `eval_bank.{json,mp4}`).
* `g1_fh_diag model_2300`, Isaac: 11.1 % overall; contact 92 %, legal return 79 %.
* Baseline `g1_fh model_2100` on the same questions: TODO_BASELINE.
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

---

## 6. Adding an entry

1. `bash hope_training/slurm/pip_env/status.sh` and copy the main numbers into a new entry at the top of
   section 5 (date, time, job, iteration, what changed).
2. Every ~2000 iterations, or after a change: `eval_forehand.sh --in-job auto <run>`, then add a row
   to section 4 from `~/hope/evals/<run>/<model_N>/forehand_zones.txt`.
3. To refresh section 3: `progress.sh <job>` and replace the table.
4. Update the "Latest" line at the top, and the same parts of the Chinese version.
