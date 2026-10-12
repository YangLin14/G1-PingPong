#!/bin/bash
# Export a training checkpoint to ONNX and watch it in MuJoCo (stats JSON + MP4).
# Run from the repository root, in one of two ways:
#   bash  hope_training/slurm/pip_env/eval_checkpoint.sh --in-job auto g1_fh_diag   # inside the run's job
#         (or --in-job 4382 to name the job yourself)
#         allocation (srun --overlap): no extra quota, starts immediately; uses a few GB of its GPU
#   sbatch hope_training/slurm/pip_env/eval_checkpoint.sh g1_fh_diag                 # as its own small job
# Optional 2nd argument: a checkpoint path (default: the newest model_*.pt of the run).
# Output: $HOPE_ROOT/evals/<run>/<model_N>/
#   eval.{json,mp4}        MuJoCo with mujoco_eval_onnx.py's own example serves (FH/BH, fixed landing)
#   eval_bank.{json,mp4}   MuJoCo with OUR bank: diagonal feed with spin, apex strike, the row's target
#                          zone (drawn on the table, the active one bright), club success per zone
# Knobs: SERVES (6, example serves), BANK_SERVES (12), ZONES (e.g. "opp_right_mid opp_right_corner"),
#        FEED_LEAD_S (0.9; 0 = the question is only known once the ball is launched).
#SBATCH --job-name=hope-eval
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
if [ "${1:-}" = "--in-job" ]; then
  job=$2; shift 2
  if [ "$job" = "auto" ]; then  # the running job whose training log is for this run name
    for j in $(squeue -h -u "$USER" -t RUNNING -o %i); do
      grep -qsE "^\[(fh-)?train\] job=$j .* run=${1:-}( |$)" "${HOPE_LOGS:-$HOME/hope/slurm_logs}"/*-"$j".out && { job=$j; break; }
    done
    [ "$job" = "auto" ] && { echo "no running training job for run '${1:-}' (squeue -u $USER)" >&2; exit 1; }
    echo "[eval] using job $job"
  fi
  exec srun --overlap --jobid "$job" --ntasks=1 --cpus-per-task=1 bash "$(readlink -f "$0")" "$@"
fi
source "${SLURM_SUBMIT_DIR:-$PWD}/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
RUN_NAME="${1:?usage: eval_checkpoint.sh [--in-job JOBID] RUN_NAME [CHECKPOINT]}"
EXPERIMENT="${EXPERIMENT:-unitree_g1_hitter_pingpong}"
SERVES="${SERVES:-6}"
cd "$WBT"
ckpt="${2:-$(ls -1 logs/rsl_rl/"$EXPERIMENT"/*_"$RUN_NAME"/model_*.pt 2>/dev/null \
  | awk -F"model_" '{split($2,a,"."); print a[1]" "$0}' | sort -n | tail -1 | cut -d" " -f2-)}"
[ -n "$ckpt" ] || { echo "no checkpoint for run $RUN_NAME" >&2; exit 1; }
out="$HOPE_ROOT/evals/$RUN_NAME/$(basename "$ckpt" .pt)"; mkdir -p "$out"
echo "[eval] $ckpt -> $out"
set +u; source setup_train_env.sh >/dev/null; set -u
hope_isaac_py scripts/export_onnx.py --robot unitree_g1 --task HOPE-HitterPingPong-UnitreeG1-v0 \
  --experiment-name "$EXPERIMENT" --checkpoint "$(readlink -f "$ckpt")" --output-dir "$out" \
  --motion-file "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_forehand.npz" \
  --motion-file-2 "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_backhand.npz" 2>&1 | grep -E "^\[export|ONNX|parity|Error|Traceback" || true
[ -f "$out/policy.onnx" ] || { echo "[eval] export failed" >&2; exit 1; }
cd "$HOPE_REPO"
egl=$(MUJOCO_GL=egl "$ISAAC_PYTHON" hope_training/slurm/pick_egl_device.py)
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$egl" "$ISAAC_PYTHON" hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  --robot unitree_g1 --onnx "$out/policy.onnx" --num-serves "$SERVES" --seed 0 \
  --video-out "$out/eval.mp4" --json-out "$out/eval.json" 2>&1 | grep -E "mujoco_eval\]" | tail -3
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$egl" "$ISAAC_PYTHON" hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  --robot unitree_g1 --onnx "$out/policy.onnx" --serve-bank default ${ZONES:+--zones $ZONES} \
  --feed-lead-s "${FEED_LEAD_S:-0.9}" --num-serves "${BANK_SERVES:-12}" --seed 0 \
  --video-out "$out/eval_bank.mp4" --json-out "$out/eval_bank.json" 2>&1 | grep -E "mujoco_eval\]" | tail -3
echo "[eval] done: $out (copy to the laptop: scp -r sdsc-scc:${out#"$HOME"/} .)"
