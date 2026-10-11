#!/bin/bash
# Requeue-safe G1 training on the pip env. Resumes from the newest checkpoint of RUN_NAME and keeps
# TOTAL_ITERS fixed (rsl_rl runs max_iterations MORE iterations after a resume).
#   sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_xc hope_training/slurm/pip_env/train.sbatch
# Knobs: TASK (cfg/task/*.yaml), RUN_NAME, TOTAL_ITERS (20000), NUM_ENVS (4096 fits the 11 GB 2080 Ti),
#        EXPERIMENT (rsl_rl log dir, default unitree_g1_hitter_pingpong).
#SBATCH --job-name=hope-g1-train
#SBATCH --account=tritondroids
#SBATCH --partition=long
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
TASK="${TASK:-HOPEPingPongUnitreeG1_FH_XC}"; RUN_NAME="${RUN_NAME:-g1_fh_xc}"
TOTAL_ITERS="${TOTAL_ITERS:-20000}"; NUM_ENVS="${NUM_ENVS:-4096}"; EXPERIMENT="${EXPERIMENT:-unitree_g1_hitter_pingpong}"
cd "$WBT"
set +u; source setup_train_env.sh; set -u
echo "[train] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) $(date) task=$TASK run=$RUN_NAME total=$TOTAL_ITERS envs=$NUM_ENVS"
latest=$(ls -1 logs/rsl_rl/"$EXPERIMENT"/*_"$RUN_NAME"/model_*.pt 2>/dev/null \
  | awk -F"model_" '{split($2,a,"."); print a[1]" "$0}' | sort -n | tail -1)
start=0; resume=""
if [ -n "$latest" ]; then start=${latest%% *}; resume="checkpoint_path=${latest#* }"; echo "[train] resuming from ${latest#* }"; fi
remaining=$(( TOTAL_ITERS - start ))
if (( remaining <= 0 )); then echo "[train] done ($start >= $TOTAL_ITERS)"; exit 0; fi
hope_isaac_py -u scripts/train.py task="$TASK" headless=true num_envs="$NUM_ENVS" \
  run_name="$RUN_NAME" max_iterations="$remaining" $resume
rc=$?; echo "[train] end $(date) exit=$rc"; exit $rc
