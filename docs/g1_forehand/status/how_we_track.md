# How we track a run

[繁體中文](how_we_track.zh-TW.md) · [Docs index](../README.md)

## How we track a run

Three tools, from cheapest to most complete. All run on the cluster from `~/hope/HOPE`.

| Tool | What it answers | Cost | Output |
|---|---|---|---|
| `status.sh` | Is it running? Latest training metrics, newest checkpoint | instant | screen |
| `progress.sh <job> [every]` | How did the metrics move over the run? | instant | markdown table (paste into [training curves](runs_and_results.md)) |
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

## Adding an entry

1. `bash hope_training/slurm/pip_env/status.sh` and copy the main numbers into a new entry at the top of
   [status log](log.md) (date, time, job, iteration, what changed).
2. Every ~2000 iterations, or after a change: `eval_forehand.sh --in-job auto <run>`, then add a row
   to [per-zone success](runs_and_results.md) from `~/hope/evals/<run>/<model_N>/forehand_zones.txt`.
3. To refresh [training curves](runs_and_results.md): `progress.sh <job>` and replace the table.
4. Update the "Latest" line at the top, and the same parts of the Chinese version.
