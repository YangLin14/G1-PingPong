# G1 training on Antioch

Antioch project for training the G1 forehand policy on cloud GPUs (RTX PRO 6000) instead of the SDSC
Slurm cluster. Engine `base` + our own Dockerfile: Isaac Sim 5.1 + Isaac Lab v2.3.2 with the cluster's
exact package versions (`cluster-constraints.txt`), so cluster checkpoints resume unchanged.

Run every `antioch` command from this folder with `.venv/bin/antioch` (SDK 0.5.338).

```bash
cd antioch_sim
A=.venv/bin/antioch

# 1. start a session (builds the image the first time; later starts reuse it).
#    Pick the idle timeout to cover the run: the session is released after that much idle time.
$A session new --idle-timeout 1d

# 2. copy the committed-but-git-ignored inputs (question banks); sync skips ignored files
./push_ignored.sh
#    and a warm-start checkpoint, if any
$A service cp ../artifacts/evals/g1_fh_stable_swing_model_40299/model_40299.pt \
  sim:/workspace/project/hope_training/whole_body_tracking/checkpoints_in/model_40299.pt

# 3. start training (detached; survives closing the terminal)
#    (use absolute paths: exec does not always start in /workspace/project)
$A service exec --no-stream -- env TASK=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING RUN_NAME=my_run TOTAL_ITERS=48300 \
  INIT_CHECKPOINT=/workspace/project/hope_training/whole_body_tracking/checkpoints_in/model_40299.pt \
  bash /workspace/project/hope_training/antioch/train.sh

# 4. watch it
$A service exec --no-stream -- tail -n 40 /workspace/project/hope_training/whole_body_tracking/logs/antioch/my_run.log
$A service exec --no-stream -- bash -c 'ls /workspace/project/hope_training/whole_body_tracking/logs/rsl_rl/*/*_my_run | tail'

# 5. copy checkpoints home BEFORE releasing the session (session files are deleted with it)
$A service cp sim:/workspace/project/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong/<dir>/model_N.pt ./
$A session release
```

Notes
* Source edits sync automatically on every `service exec`; files created on the session (logs,
  checkpoints) are never touched by sync. Only an image change (Dockerfile, constraints) needs a new
  session.
* `NUM_ENVS` stays 4096 by default so the PPO batch matches the cluster runs. Speed (2026-10-19,
  `_FH_FIST_STABLE_SWING`): 4096 envs ~16k steps/s, 6.0 s/iteration, the same as the 2080 Ti: the step
  is bound by per-step overhead, not GPU compute. 16384 envs fit and give 45-61k steps/s at 6.4-8.7
  s/iteration (3-4x throughput; measured while a hung run still shared the GPU, so likely
  conservative) but change the PPO batch.
* `train.sh` has a watchdog: Isaac can hang in shutdown after an error or after the final save; the
  launcher kills it `HANG_GRACE_S` (300) s later. Check `pgrep -fa scripts/train.py` before releasing.
* Image: Ubuntu 24.04 base + `libgomp1`/`libatomic1` (Isaac's USD libraries need them); every `uv`
  install names `--python /opt/isaac-venv/bin/python` (the base image points uv at its system Python).
* The session keeps billing while it is up; `session status` shows what keeps it alive.
