#!/bin/bash
#SBATCH -J hope-g1-sidecar
#SBATCH -A tritondroids
#SBATCH -p normal
#SBATCH --gres=rtx2080ti:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH -o /home/%u/.cache/hope-slurm/sidecar/%j-sidecar.out
# MuJoCo rollout sidecar for a G1 training run: every HOPE_G1_SIDECAR_PERIOD it takes the newest local
# checkpoint of HOPE_G1_RUN_NAME, exports it to ONNX inside the Isaac image (GPU, headless), runs a few
# serves in MuJoCo on the host venv (EGL offscreen, mp4 + per-serve diagnostics) and logs them to the
# companion W&B run <run_name>-rollouts. In-Isaac RecordVideo is unavailable here (Isaac Sim 5.1 RTX
# crashes on driver 610.43). The job re-queues itself while a hope-g1-train job for the same run name
# is running or pending; stop it with: scancel -n hope-g1-sidecar
#
# Knobs: HOPE_REPO, HOPE_SIF, HOPE_VENV (default $HOME/.cache/hope-venv), HOPE_G1_RUN_NAME, HOPE_G1_PROJECT,
#        HOPE_G1_SIDECAR_PERIOD (default 2hours), HOPE_G1_SIDECAR_SERVES (default 8),
#        HOPE_G1_MOTION_FH / HOPE_G1_MOTION_BH (clips the export env is built with; variant B uses the IK pair).
set -uo pipefail
HOPE_REPO="${HOPE_REPO:-$HOME/HOPE}"
SIF="${HOPE_SIF:-$HOME/containers/hope_grasping.sif}"
VENV="${HOPE_VENV:-$HOME/.cache/hope-venv}"
ICD="$HOPE_REPO/hope_training/slurm/nvidia_icd.json"
export HOPE_G1_RUN_NAME="${HOPE_G1_RUN_NAME:-g1_v2a}"
export HOPE_G1_PROJECT="${HOPE_G1_PROJECT:-hope-g1}"
export HOPE_G1_SIDECAR_PERIOD="${HOPE_G1_SIDECAR_PERIOD:-2hours}"
export HOPE_G1_SIDECAR_SERVES="${HOPE_G1_SIDECAR_SERVES:-8}"
export HOPE_G1_MOTION_FH="${HOPE_G1_MOTION_FH:-hope_training/motions/preprocessed/g1_hope_forehand.npz}"
export HOPE_G1_MOTION_BH="${HOPE_G1_MOTION_BH:-hope_training/motions/preprocessed/g1_hope_backhand.npz}"
WBT="$HOPE_REPO/hope_training/whole_body_tracking"
STATE="$HOME/.cache/hope-slurm/sidecar/${HOPE_G1_RUN_NAME}.last_iter"
mkdir -p "$(dirname "$STATE")"
echo "[sidecar] host $(hostname) job=$SLURM_JOB_ID start $(date) run=$HOPE_G1_RUN_NAME period=$HOPE_G1_SIDECAR_PERIOD"

trainer_alive () {
  # The trainer for THIS run name: the job name is shared, so read the "run_name=<x>" the trainer
  # prints on its first log line (Slurm does not expose a job's environment). A trainer that is
  # still PENDING has no log yet; keep re-queuing for it as long as any hope-g1-train job pends.
  local j log
  for j in $(squeue -h -u "$USER" -n hope-g1-train -t RUNNING -o %i); do
    log="$HOME/.cache/hope-slurm/$j-g1-train.out"
    [ -f "$log" ] && grep -m1 -qE "^\[train\] .*run_name=${HOPE_G1_RUN_NAME}( |$)" "$log" && return 0
  done
  squeue -h -u "$USER" -n hope-g1-train -t PENDING | grep -q .
}

resubmit () {
  if trainer_alive; then
    next=$(sbatch --parsable --dependency=singleton --begin="now+${HOPE_G1_SIDECAR_PERIOD}" \
      --export=ALL,HOPE_REPO,HOPE_SIF,HOPE_VENV,HOPE_G1_RUN_NAME,HOPE_G1_PROJECT,HOPE_G1_SIDECAR_PERIOD,HOPE_G1_SIDECAR_SERVES,HOPE_G1_MOTION_FH,HOPE_G1_MOTION_BH \
      "$HOPE_REPO/hope_training/slurm/g1_rollout_sidecar.sbatch")
    echo "[sidecar] resubmitted as job $next (begin now+${HOPE_G1_SIDECAR_PERIOD})"
  else
    echo "[sidecar] no hope-g1-train job for run $HOPE_G1_RUN_NAME running or pending -> not resubmitting"
  fi
}

ckpt=$(ls -1 "$WBT"/logs/rsl_rl/unitree_g1_hitter_pingpong/*_"${HOPE_G1_RUN_NAME}"/model_*.pt 2>/dev/null \
  | awk -F"model_" '{split($2,a,"."); print a[1]" "$0}' | sort -n | tail -1 | cut -d" " -f2-)
if [ -z "$ckpt" ]; then echo "[sidecar] no checkpoint yet"; resubmit; exit 0; fi
iter=$(basename "$ckpt" .pt); iter=${iter#model_}
run_dir=$(dirname "$ckpt")
last=$(cat "$STATE" 2>/dev/null || echo -1)
if [ "$iter" -le "$last" ]; then echo "[sidecar] newest checkpoint iter=$iter already logged (last=$last)"; resubmit; exit 0; fi
echo "[sidecar] checkpoint $ckpt (iter=$iter, last logged=$last)"

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N OMNI_KIT_ALLOW_ROOT=1
exported="$run_dir/exported_iter${iter}"
apptainer exec --nv --writable-tmpfs --bind "$HOME" --bind "$ICD:/usr/share/vulkan/icd.d/nvidia_icd.json" \
  --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  --env HOPE_REPO="$HOPE_REPO" "$SIF" bash -c '
set -uo pipefail
cd "$HOPE_REPO/hope_training/whole_body_tracking" && source setup_train_env.sh >/dev/null
timeout -k 30 15m env PYTHONPATH="${HOPE_PYTHONPATH}" "${ISAAC_PYTHON}" -u scripts/export_onnx.py --robot unitree_g1 \
  --task HOPE-HitterPingPong-UnitreeG1-v0 --experiment-name unitree_g1_hitter_pingpong \
  --checkpoint '"$ckpt"' --output-dir '"$exported"' \
  --motion-file "$HOPE_REPO/'"$HOPE_G1_MOTION_FH"'" --motion-file-2 "$HOPE_REPO/'"$HOPE_G1_MOTION_BH"'"'
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$exported/policy.onnx" ]; then echo "[sidecar] export FAILED rc=$rc"; resubmit; exit 1; fi

mkdir -p "$run_dir/videos"
mp4="$run_dir/videos/mujoco_iter${iter}.mp4"; js="$run_dir/videos/mujoco_iter${iter}.json"
cd "$HOPE_REPO"
MUJOCO_GL=egl timeout -k 30 15m "$VENV/bin/python" "$WBT/scripts/mujoco_eval_onnx.py" --robot unitree_g1 \
  --onnx "$exported/policy.onnx" --num-serves "$HOPE_G1_SIDECAR_SERVES" --seed "$iter" \
  --video-out "$mp4" --json-out "$js"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$mp4" ]; then echo "[sidecar] mujoco rollout FAILED rc=$rc"; resubmit; exit 1; fi

"$VENV/bin/python" "$WBT/scripts/wandb_log_rollout.py" --project "$HOPE_G1_PROJECT" --run-name "$HOPE_G1_RUN_NAME" \
  --iteration "$iter" --video "$mp4" --json "$js" --checkpoint "$ckpt"
rc=$?
if [ $rc -eq 0 ]; then echo "$iter" > "$STATE"; echo "[sidecar] logged iter=$iter"; else echo "[sidecar] wandb upload FAILED rc=$rc"; fi
resubmit
echo "[sidecar] end $(date) rc=$rc"; exit $rc
