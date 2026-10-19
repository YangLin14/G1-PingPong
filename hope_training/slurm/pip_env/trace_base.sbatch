#!/bin/bash
# Isaac trace of base motion between balls (scripts/trace_base_motion.py), with and without shoves.
#   sbatch --gres=rtx2080ti:2 --export=ALL,CKPT=<abs model_N.pt>,TASK=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING \
#     hope_training/slurm/pip_env/trace_base.sbatch      -> $HOPE_ROOT/trace_base/<jobid>/{push,nopush}.npz
#SBATCH --job-name=hope-g1-trace
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
TASK="${TASK:-HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING}"; CKPT="${CKPT:?set CKPT}"
# Never run on a known-broken GPU. AVOID_GPU_IDX lists Slurm GPU indices on zixian (default 0,5:
# 5 = 0000:B2:00.0 fell off the bus on 2026-10-12; 0 = 0000:3D:00.0 failed at the end of job 5719
# on 2026-10-16 and now returns CUDA error 999). The check reads only the job's GPU indices from Slurm
# (this cluster sets neither SLURM_JOB_GPUS nor CUDA_VISIBLE_DEVICES; ConstrainDevices=yes) and never
# touches a listed device. Slurm hands out the lowest free index, so a broken idx 0 would be picked
# every time: submit with --gres=rtx2080ti:2 and the job hides the listed GPU from CUDA
# (CUDA_VISIBLE_DEVICES = the first good one, PCI order = Slurm index order on zixian). With no good
# GPU in the allocation the job stops before Isaac starts.
AVOID_GPU_IDX="${AVOID_GPU_IDX-0,5}"
job_gpus=$(scontrol show job -d "$SLURM_JOB_ID" | grep -oE 'IDX:[0-9,-]+' | head -1 | cut -d: -f2)
job_gpus=$(for r in ${job_gpus//,/ }; do seq "${r%-*}" "${r#*-}"; done | sort -n)
echo "[trace] GPU index(es): ${job_gpus//$'\n'/ }"
ordinal=0; use_ordinal=""; use_idx=""
for g in $job_gpus; do
  bad_gpu=0
  for bad in ${AVOID_GPU_IDX//,/ }; do [ "$g" = "$bad" ] && bad_gpu=1; done
  if [ $bad_gpu = 0 ] && [ -z "$use_ordinal" ]; then use_ordinal=$ordinal; use_idx=$g; fi
  ordinal=$((ordinal + 1))
done
[ -n "$use_ordinal" ] || { echo "[trace] every allocated GPU is in AVOID_GPU_IDX=$AVOID_GPU_IDX; not starting (resubmit with --gres=rtx2080ti:2)" >&2; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$use_ordinal
echo "[trace] using GPU index $use_idx (CUDA_VISIBLE_DEVICES=$use_ordinal)"
OUT="$HOPE_ROOT/trace_base/$SLURM_JOB_ID"; mkdir -p "$OUT"
cd "$WBT"
set +u; source setup_train_env.sh; set -u
py() { PYTHONPATH="${HOPE_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" timeout -k 30 "$1" "$ISAAC_PYTHON" "${@:2}"; }
for mode in push nopush; do
  extra=""; [ $mode = nopush ] && extra="no_push=true"
  py 25m -u scripts/trace_base_motion.py task="$TASK" checkpoint="$CKPT" num_envs=256 num_steps=1500 \
    json_out="$OUT/$mode.npz" $extra
  echo "[trace] $mode exit=$?"
done
