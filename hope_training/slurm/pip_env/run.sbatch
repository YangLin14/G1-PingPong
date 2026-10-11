#!/bin/bash
# Run any HOPE Isaac command on one RTX 2080 Ti from hope_training/whole_body_tracking.
#   sbatch hope_training/slurm/pip_env/run.sbatch scripts/evaluate.py --checkpoint ...
#   sbatch --job-name=x hope_training/slurm/pip_env/run.sbatch scripts/train.py task=... max_iterations=20
#SBATCH --job-name=hope
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
(( $# )) || { echo "usage: sbatch run.sbatch <script.py> [args...]" >&2; exit 2; }
cd "$WBT"
set +u; source setup_train_env.sh; set -u
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
hope_isaac_py "$@"
