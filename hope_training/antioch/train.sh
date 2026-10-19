#!/bin/bash
# G1 training on an Antioch session (replaces slurm/pip_env/train.sbatch). Same knobs and resume rules:
# resumes from the newest checkpoint of RUN_NAME, else warm-starts from INIT_CHECKPOINT, and keeps
# TOTAL_ITERS absolute. Runs detached; the log is logs/antioch/<RUN_NAME>.log under whole_body_tracking.
#   antioch service exec -- env TASK=... RUN_NAME=... TOTAL_ITERS=... [INIT_CHECKPOINT=...] \
#     bash hope_training/antioch/train.sh            (add FOREGROUND=1 to run attached, e.g. smoke tests)
# Knobs: TASK, RUN_NAME, TOTAL_ITERS (20000), NUM_ENVS (4096, as on the cluster: PPO batch unchanged),
#        EXPERIMENT (unitree_g1_hitter_pingpong), INIT_CHECKPOINT, EXTRA (more train.py overrides).
set -uo pipefail
TASK="${TASK:?set TASK}"; RUN_NAME="${RUN_NAME:?set RUN_NAME}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"; NUM_ENVS="${NUM_ENVS:-4096}"; EXPERIMENT="${EXPERIMENT:-unitree_g1_hitter_pingpong}"
WBT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../whole_body_tracking" && pwd)"
export ISAAC_PYTHON="${ISAAC_PYTHON:-/opt/isaac-venv/bin/python}" ISAACLAB_ROOT="${ISAACLAB_ROOT:-/opt/IsaacLab}"
export ISAAC_VENV_SITE="${ISAAC_VENV_SITE:-/opt/isaac-venv/lib/python3.11/site-packages}"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$WBT"
set +u; source setup_train_env.sh >/dev/null; set -u
mkdir -p logs/antioch; LOG="logs/antioch/$RUN_NAME.log"

run() {
  echo "[train] host=$(hostname) $(date) task=$TASK run=$RUN_NAME total=$TOTAL_ITERS envs=$NUM_ENVS"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  latest=$(ls -1 logs/rsl_rl/"$EXPERIMENT"/*_"$RUN_NAME"/model_*.pt 2>/dev/null \
    | awk -F"model_" '{split($2,a,"."); print a[1]" "$0}' | sort -n | tail -1)
  start=0; resume=""
  if [ -n "$latest" ]; then start=${latest%% *}; resume="checkpoint_path=${latest#* } restore_step_counter=true"; echo "[train] resuming from ${latest#* }"
  elif [ -n "${INIT_CHECKPOINT:-}" ]; then
    [ -f "$INIT_CHECKPOINT" ] || { echo "[train] INIT_CHECKPOINT not found: $INIT_CHECKPOINT" >&2; return 1; }
    init=$(readlink -f "$INIT_CHECKPOINT"); start=$(basename "$init" .pt); start=${start#model_}
    resume="checkpoint_path=$init"; echo "[train] warm start from $init (iteration $start)"
  fi
  remaining=$(( TOTAL_ITERS - start ))
  if (( remaining <= 0 )); then echo "[train] done ($start >= $TOTAL_ITERS)"; return 0; fi
  off=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  hope_isaac_py -u scripts/train.py task="$TASK" headless=true num_envs="$NUM_ENVS" \
    run_name="$RUN_NAME" max_iterations="$remaining" $resume ${EXTRA:-} &
  pid=$!
  # Watchdog: Isaac can hang in shutdown after an error or after the final save (seen on the cluster,
  # job 5719, and here). Kill it HANG_GRACE_S (300) s after either, instead of billing an idle GPU.
  last=$(( TOTAL_ITERS - 1 )); since=""
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    if tail -c +$((off + 1)) "$LOG" 2>/dev/null | grep -q "\[train.py\] ERROR during run" \
       || ls logs/rsl_rl/"$EXPERIMENT"/*_"$RUN_NAME"/model_"$last".pt >/dev/null 2>&1; then
      since=${since:-$SECONDS}
      if (( SECONDS - since > ${HANG_GRACE_S:-300} )); then
        echo "[train] watchdog: still alive ${HANG_GRACE_S:-300} s after an error/final save -> kill"
        pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid; sleep 15; kill -KILL $pid 2>/dev/null
      fi
    fi
  done
  wait $pid; rc=$?; echo "[train] end $(date) exit=$rc"; return $rc
}

if [ -n "${FOREGROUND:-}" ]; then run 2>&1 | tee -a "$LOG"; exit "${PIPESTATUS[0]}"; fi
FOREGROUND=1 setsid nohup bash "$(readlink -f "${BASH_SOURCE[0]}")" > /dev/null 2>&1 < /dev/null &
echo "[train] started pid $! -> $WBT/$LOG"
