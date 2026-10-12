#!/bin/bash
# Per-zone success rate of a G1 forehand checkpoint under the club's success definition, in Isaac
# (scripts/evaluate_forehand.py). Run from the repository root, in one of two ways:
#   bash  hope_training/slurm/pip_env/eval_forehand.sh --in-job auto g1_fh_diag   # inside the run's job
#         (or --in-job 4382); no extra quota, shares the training GPU (~3 GB, ~20 min at 512 envs)
#   sbatch hope_training/slurm/pip_env/eval_forehand.sh g1_fh_diag                 # as its own small job
# Optional 2nd argument: a checkpoint path (default: the newest model_*.pt of the run).
# Knobs: TASK (default HOPEPingPongUnitreeG1_FH_XC), NUM_ENVS (512), NUM_STEPS (3000 = 60 s per env).
# Output: $HOPE_ROOT/evals/<run>/<model_N>/forehand_zones.{json,txt}
#SBATCH --job-name=hope-eval-fh
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:45:00
#SBATCH --output=%x-%j.out
set -euo pipefail
if [ "${1:-}" = "--in-job" ]; then
  job=$2; shift 2
  if [ "$job" = "auto" ]; then  # the running job whose training log is for this run name
    for j in $(squeue -h -u "$USER" -t RUNNING -o %i); do
      grep -qsE "^\[(fh-)?train\] job=$j .* run=${1:-}( |$)" "${HOPE_LOGS:-$HOME/hope/slurm_logs}"/*-"$j".out && { job=$j; break; }
    done
    [ "$job" = "auto" ] && { echo "no running training job for run '${1:-}' (squeue -u $USER)" >&2; exit 1; }
    echo "[eval-fh] using job $job"
  fi
  exec srun --overlap --jobid "$job" --ntasks=1 --cpus-per-task=1 bash "$(readlink -f "$0")" "$@"
fi
source "${SLURM_SUBMIT_DIR:-$PWD}/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
RUN_NAME="${1:?usage: eval_forehand.sh [--in-job JOBID|auto] RUN_NAME [CHECKPOINT]}"
EXPERIMENT="${EXPERIMENT:-unitree_g1_hitter_pingpong}"
TASK="${TASK:-HOPEPingPongUnitreeG1_FH_XC}"
cd "$WBT"
ckpt="${2:-$(ls -1 logs/rsl_rl/"$EXPERIMENT"/*_"$RUN_NAME"/model_*.pt 2>/dev/null \
  | awk -F"model_" '{split($2,a,"."); print a[1]" "$0}' | sort -n | tail -1 | cut -d" " -f2-)}"
[ -n "$ckpt" ] || { echo "no checkpoint for run $RUN_NAME" >&2; exit 1; }
ckpt=$(readlink -f "$ckpt")
out="$HOPE_ROOT/evals/$RUN_NAME/$(basename "$ckpt" .pt)"; mkdir -p "$out"
echo "[eval-fh] $ckpt (task $TASK) -> $out"
set +u; source setup_train_env.sh >/dev/null; set -u
hope_isaac_py scripts/evaluate_forehand.py task="$TASK" checkpoint="$ckpt" \
  num_envs="${NUM_ENVS:-512}" num_steps="${NUM_STEPS:-3000}" json_out="$out/forehand_zones.json" 2>&1 \
  | grep --line-buffered -E "^\[evaluate_forehand\]|^zone |^(opp_|center_|OVERALL)|Error|Traceback" | tee "$out/forehand_zones.txt"
[ -f "$out/forehand_zones.json" ] || { echo "[eval-fh] failed (see the output above)" >&2; exit 1; }
echo "[eval-fh] done: $out/forehand_zones.json"
