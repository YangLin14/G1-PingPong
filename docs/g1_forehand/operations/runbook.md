# Runbook: G1 forehand training (Triton Droids)

[Docs index](../README.md)

[繁體中文版](runbook.zh-TW.md) · Background and design: [docs index](../README.md)

Copy-paste commands for daily work. Every command here was run on 2026-10-10. Job numbers in the
examples (4382, 4354) are the runs at that time; replace them with yours from `status.sh`. Never type
angle brackets in the shell: bash reads `<x>` as "read from file x".

* **Laptop**: the `G1-PingPong` checkout (macOS). Commands start in the repo root.
* **Cluster**: `ssh sdsc-scc`, repo at `~/hope/HOPE`. Cluster commands start with `cd ~/hope/HOPE`.

---

## 0. What is running

| Run name | Task | Purpose |
|---|---|---|
| `g1_fh_diag` | `HOPEPingPongUnitreeG1_FH_XC` | **Main run.** G1 forehand: diagonal feed from the opponent's right, apex strike, four landing zones. |
| `g1_fh_rally` | `HOPEPingPongUnitreeG1_FH_XC_RALLY` | Same task at real rally timing (late swing entry, short follow-through, ±30 ms timing jitter, faster feeds), warm-started from `g1_fh_diag` (decision D14). Job 5560. |
| `g1_fh` | `HOPEPingPongUnitreeG1_FH` | Baseline. Also a G1 forehand, but with HOPE's inherited A3-geometry questions (falling-ball strikes, landings anywhere). For comparison only. |

Both runs are on the **G1**. Neither trains the A3 robot.

---

## 1. Check training status

```bash
ssh sdsc-scc
cd ~/hope/HOPE && bash hope_training/slurm/pip_env/status.sh          # running jobs
bash hope_training/slurm/pip_env/status.sh --all                       # + jobs ended in the last 3 days
bash hope_training/slurm/pip_env/status.sh 4382                        # one job
bash hope_training/slurm/pip_env/progress.sh 4382                      # metrics every 500 iterations (markdown table)
```

It prints your jobs, the club's quota use, and for each run its iteration, key metrics and newest
checkpoint. What the metrics mean:

| Line | Meaning | Healthy direction |
|---|---|---|
| mean reward | total reward per step | rises, noisy |
| racket within 10 cm at strike | racket reached the target point at the strike frame | rises first (several hundred to a few thousand iterations) |
| pos+vel+normal all OK at strike | position, velocity and face all within tolerance | rises after the line above |
| virtual ball: contact rate | the strike would actually hit the ball | rises with the two above |
| virtual ball: legal return rate | over the net and onto the opponent's half | the real goal; rises last |
| virtual ball: landing error (m) | distance from the intended landing point | falls |
| falls (share of episode ends) | episodes that ended in a fall | starts near 1.0, should drop well below 0.5 |
| seconds per iteration | speed | ~5.3 s alone, ~8 s when two runs share the node |

A run takes 20 000 iterations, about 30–45 hours. Ignore `success_rate` in the raw log: HOPE's
analytic metric never clears the net.

Raw log and TensorBoard:

```bash
tail -f ~/hope/slurm_logs/hope-g1-fh-diag-4382.out    # the number is the job id
# on the laptop, then open http://localhost:6006
ssh -L 6006:localhost:6006 sdsc-scc '~/hope/isaac-venv/bin/tensorboard --port 6006 \
  --logdir ~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong'
```

---

## 2. Look at a checkpoint (export -> MuJoCo video)

```bash
cd ~/hope/HOPE
# inside the run's own job (found by run name): starts immediately, needs no extra quota (~1 min)
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag
# same, naming the job number yourself (from status.sh / squeue; type the number, not <...>)
bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job 4382 g1_fh_diag
# or as its own small job (8 GB), e.g. when the run has ended
sbatch hope_training/slurm/pip_env/eval_checkpoint.sh g1_fh_diag
```

This exports the newest `model_*.pt` of the run (or pass a checkpoint path as the 2nd argument), checks
that the ONNX matches the torch actor, and runs two MuJoCo sessions into `~/hope/evals/<run>/<model_N>/`:

* **`eval_bank.mp4` / `eval_bank.json` (watch this one)**: 12 questions from **our bank** with a real
  ball: diagonal feed with spin, apex strike. The four target zones are drawn on the far side, the
  current one bright. The JSON has contact, timing (vz at contact), landing point per serve and the
  club success per zone;
* `eval.mp4` / `eval.json`: the evaluator's own example serves (FH/BH, fixed landing). Not our task;
  kept for reference.

Knobs: `BANK_SERVES=24`, `ZONES="opp_right_mid opp_right_corner"` (only those zones),
`FEED_LEAD_S=0` (the question is only known once the ball is launched, as on the robot; default 0.9 s
like training). Example:
`BANK_SERVES=24 ZONES=opp_right_corner bash hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag`

Copy it to the laptop and watch:

```bash
mkdir -p artifacts/evals
scp -r sdsc-scc:hope/evals/g1_fh_diag/model_2000 artifacts/evals/g1_fh_diag_model_2000
open artifacts/evals/g1_fh_diag_model_2000/eval_bank.mp4
```

### 2b. Club success rate per target zone (Isaac)

```bash
cd ~/hope/HOPE
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job auto g1_fh_diag       # newest checkpoint, ~20 min
bash hope_training/slurm/pip_env/eval_forehand.sh --in-job 4382 g1_fh            # another run's checkpoint, on 4382's GPU
sbatch hope_training/slurm/pip_env/eval_forehand.sh g1_fh_diag                   # as its own job
```

Scores every bank question with our success definition (apex timing, contact, over the net, on the
opponent half, inside the question's zone) and prints one row per zone plus the total. Output:
`~/hope/evals/<run>/<model_N>/forehand_zones.{json,txt}`. Record the result in
[`status/runs_and_results.md`](../status/runs_and_results.md) (per-zone table).

---

## 3. Open MuJoCo on the laptop

One-time setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
```

macOS MuJoCo windows need `mjpython`. This repo path contains a space ("Triton Droids"), which breaks
the `mjpython` shebang, so **always start it through the venv's python**:
`.venv/bin/python .venv/bin/mjpython <script>`.

| What | Command |
|---|---|
| Ball feeding (the training questions), top view | `.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --camera top` |
| Ball feeding, behind the robot | `... view_feed_bank.py --camera behind --rows 30 --speed 0.5` |
| Ball feeding to MP4 (no window) | `.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4` |
| A trained G1 policy, swinging | `PYTHONPATH=a3_deploy/a3_deploy_example/reference .venv/bin/python .venv/bin/mjpython -m a3_deploy_onnx_ref_pingpong --robot unitree_g1 --onnx artifacts/evals/<dir>/policy.onnx --view --realtime` |
| A trained G1 policy against real balls | `.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 --onnx artifacts/evals/<dir>/policy.onnx --num-serves 12 --view` |

Feed viewer colours: purple = feed origin (opponent's right), orange = feed bounce zone, faint green =
the four target zones, bright green = this question's target, yellow = ball trail, red disc and
arrow = racket target (face and velocity).

---

## 4. Start, stop, resume

All from `cd ~/hope/HOPE` on the cluster (jobs must be submitted from the repo root).

```bash
# start (or resume: the same RUN_NAME continues from its newest checkpoint, total stays 20 000)
sbatch --output=$HOME/hope/slurm_logs/%x-%j.out --job-name=hope-g1-fh-diag \
  --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_diag hope_training/slurm/pip_env/train.sbatch
scancel <jobid>                    # stop (the newest saved checkpoint stays; it saves every 100 iterations)
squeue -u $USER                    # list
```

* Start **from scratch**: use a new `RUN_NAME`, or delete
  `hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_<run>` first.
* Knobs: `TASK`, `RUN_NAME`, `TOTAL_ITERS` (20000), `NUM_ENVS` (4096).
* The job runs on `long` with `--requeue`: if the node is preempted it restarts and resumes by itself.
* Running jobs keep the code they started with. After pulling new code, restart the job to use it.
* Smoke-test a new task before a long run:
  `sbatch --export=ALL,TASK=<task> hope_training/slurm/pip_env/g1_smoke.sbatch`.
* * **GPUs to avoid: idx 0 and 5** (`AVOID_GPU_IDX`, default `0,5`). Slurm gives the lowest free index, so submit
  with `--gres=rtx2080ti:2`: the job hides the bad GPU from CUDA and runs on the other one (log line
  `[train] using GPU index N`). With `:1` it stops if it gets idx 0.
* **Balance-first runs (D21), once CUDA works on zixian again:** smoke-test, then start both from `g1_fh_fist` `model_32299`:
  ```bash
  INIT=$(ls hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_g1_fh_fist/model_32299.pt)
  sbatch --gres=rtx2080ti:2 --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING hope_training/slurm/pip_env/g1_smoke.sbatch
  for v in STABLE:g1_fh_stable STABLE_SWING:g1_fh_stable_swing; do
    sbatch --gres=rtx2080ti:2 --job-name=hope-${v#*:} --output=$HOME/hope/slurm_logs/%x-%j.out \
      --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_FIST_${v%%:*},RUN_NAME=${v#*:},TOTAL_ITERS=40300,INIT_CHECKPOINT=$INIT \
      hope_training/slurm/pip_env/train.sbatch
  done
  ```
  The startup log must list `rewards.root_height_ceiling.weight = -1.0`, `rewards.com_support.weight = -2.0`,
  `rewards.upright.weight = -3.0` and the bigger `events.push_robot`. Check `root_height` going under ~0.68 m in progress.
* *(Superseded by `g1_fh_fist`, kept for reference)* **Balance run `g1_fh_bal` (D19), once zixian is back:** smoke-test the task, then warm-start it from the
  newest `g1_fh_diag` checkpoint (use the real number in place of `<N>`):
  ```bash
  sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL hope_training/slurm/pip_env/g1_smoke.sbatch
  sbatch --gres=rtx2080ti:1 --job-name=hope-g1-fh-bal --output=$HOME/hope/slurm_logs/%x-%j.out \
    --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY_BAL,RUN_NAME=g1_fh_bal,TOTAL_ITERS=<N+12000>,INIT_CHECKPOINT=$(ls -t hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/*_g1_fh_diag/model_<N>.pt | head -1) \
    hope_training/slurm/pip_env/train.sbatch
  ```
  The startup log must show `events.push_robot = ramped push ...`.

---

**Fist-grip runs (`g1_fh_fist`, D20):** always evaluate with the fist model, otherwise the racket is in the
wrong place: add `--model-xml unitree/URDF/g1_29dof_pingpang_fist/mjcf/g1_29dof_pingpang_fist.xml` to
`mujoco_eval_onnx.py`. `onnx_from_checkpoint.py` works unchanged (same network).

## 4b. Balance test in MuJoCo (laptop)

```bash
.venv/bin/python hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 \
  --onnx artifacts/evals/<dir>/policy.onnx \
  --serve-bank hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.npz \
  --eval-mode continuous --reset-after-fall --num-serves 40 --seed 0 \
  --push 0.5 0.5 --json-out /tmp/push.json        # shove up to 0.5 m/s and 0.5 rad/s once per ball
```
Compare `fall_rate` and the club numbers with the table in the status log (2026-10-14 17:00); use
seeds 0, 1, 2 and push levels 0.25 / 0.5 / 0.75 / 1.0.

---

## 5. Change code, then update the cluster

```bash
# laptop
.venv/bin/python -m pytest -q hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py \
  hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py unitree/tests
git commit ... && git push origin g1-forehand-rally
# cluster
cd ~/hope/HOPE && git pull --ff-only origin g1-forehand-rally
# then restart the affected job (section 4)
```

---

## 6. Change zones or the feed (regenerate the question bank)

1. Edit `TARGET_ZONES`, `FEED_ZONE` or `FEED_ORIGIN_*` in
   `hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/forehand_success.py`
   (and `BankCfg` in `scripts/make_g1_forehand_bank.py` for the feed speeds / target list).
2. Regenerate (~15 min on the laptop):
   `.venv/bin/python hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000`
3. Look at it (section 3, feed viewer) and run the tests (section 5).
4. Commit the bank (`git add -f hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.*`),
   push, pull on the cluster, and restart training **from scratch** (the questions changed).

---

## 7. Phone video -> G1 reference clip

Record: static camera ~45° front-side, whole body + racket in frame, 30–60 fps, ready → forehand →
ready, 5–10 takes.

```bash
scp forehand.mp4 sdsc-scc:hope/mocap/videos/
ssh sdsc-scc
cd ~/hope/HOPE && sbatch --export=ALL,VIDEO=$HOME/hope/mocap/videos/forehand.mp4,NAME=g1vid_forehand \
  hope_training/slurm/video_to_g1/video_to_g1.sbatch
# result: ~/hope/mocap/clips/g1vid_forehand/{g1vid_forehand.npz,.yaml,.mp4} (red border = strike frame)
```

Trim with `START=`/`END=` (source frames, 30 Hz) and override the strike with `STRIKE=` (output frame,
50 Hz), then resubmit. Details: [`slurm/video_to_g1/README.md`](../../../hope_training/slurm/video_to_g1/README.md).

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Job `PENDING (AssocGrpCpuLimit)` / `(AssocGrpMemLimit)` | The club's 32 CPU / 96 GB cap is full (`status.sh` shows who uses it). Wait, stop the baseline, or ask for less (`--mem`, `--cpus-per-task`; training needs ~8 GB RAM and ~5 GB GPU). |
| Want a quick eval but the quota is full | `eval_checkpoint.sh --in-job auto <run>` runs inside the run's existing allocation. |
| `status.sh` prints "errors in the log" | `grep -n -A30 Traceback <log>`. Isaac often keeps running after a Python error; `scancel` it. |
| Kit `No device could be created`, hang after URDF import, `/tmp` permission errors | Node workarounds missing: rerun `sbatch hope_training/slurm/pip_env/install_isaac.sbatch` (it renders `setup_train_env.local.sh`). |
| MuJoCo window does not open on the Mac | Use `.venv/bin/python .venv/bin/mjpython ...` (path-with-space issue); `--record out.mp4` needs no window. |
| `unable to open EGL` on the cluster | Set `MUJOCO_EGL_DEVICE_ID=$(MUJOCO_GL=egl ~/hope/isaac-venv/bin/python hope_training/slurm/pick_egl_device.py)`. |
| Job numbers | `sbatch` prints the id; logs are `~/hope/slurm_logs/<job-name>-<id>.out`. |

---

## 9. Where things are

| What | Where |
|---|---|
| Training logs (stdout) | cluster `~/hope/slurm_logs/` |
| Checkpoints, TensorBoard events | cluster `~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/<date>_<run>/` |
| Evaluations | cluster `~/hope/evals/<run>/<model_N>/` |
| Videos / clips from phone videos | cluster `~/hope/mocap/clips/<name>/` |
| Isaac venv, Isaac Lab, libGLU | cluster `~/hope/{isaac-venv, IsaacLab, syslibs}` |
| Laptop-only outputs (git-ignored) | `artifacts/` |
| Task recipes | `hope_training/whole_body_tracking/cfg/task/HOPEPingPongUnitreeG1_FH*.yaml` |
| Question bank | `hope_training/whole_body_tracking/assets/venue_tuple/g1_fh_crosscourt_apex_bank.*` |
| Zones and success rules | `.../whole_body_tracking/utils/forehand_success.py` |
| Cluster job scripts | `hope_training/slurm/pip_env/`, `hope_training/slurm/video_to_g1/` |


## 10. Training on Antioch (current, since 2026-10-19)

All commands and notes: [antioch_sim/README.md](../../../antioch_sim/README.md). In short, from `antioch_sim/`:
`session new --idle-timeout 1d` → `./push_ignored.sh` (+ `service cp` a warm-start checkpoint) →
`service exec --no-stream -- env TASK=… RUN_NAME=… TOTAL_ITERS=… bash /workspace/project/hope_training/antioch/train.sh`
→ watch `logs/antioch/<RUN_NAME>.log` → `service cp` the checkpoints home → `session release`.
Session files disappear on release. The Slurm sections above are kept for reference.
