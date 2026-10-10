#!/bin/bash
#SBATCH -J hope-g1-train
#SBATCH -A tritondroids
#SBATCH -p long
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --gres=rtx2080ti:1
#SBATCH -c 12
#SBATCH --mem=24G
#SBATCH -o /home/%u/.cache/hope-slurm/%j-g1-train.out
# Unitree G1 HitterPingPong training inside the pinned HOPE Isaac image, logged to W&B.
# Preemption on `long` requeues the job (--requeue); on restart it resumes from the newest local
# checkpoint of THIS run name and re-attaches to the same W&B run (id derived from the Slurm job id).
# Log dir: ~/.cache/hope-slurm (create it once: mkdir -p ~/.cache/hope-slurm/sidecar). See README.md.
#
# Knobs (env vars, e.g. sbatch --export=ALL,HOPE_G1_RUN_NAME=g1_v2a hope_training/slurm/g1_train.sbatch):
#   HOPE_REPO          repository checkout                   (default $HOME/HOPE)
#   HOPE_SIF           Apptainer image                        (default $HOME/containers/hope_grasping.sif)
#   HOPE_G1_RUN_NAME   local run-name suffix + W&B name       (default g1_v2a)
#   HOPE_G1_TASK       Hydra task recipe (cfg/task/*.yaml)    (default HOPEPingPongUnitreeG1; variant B: HOPEPingPongUnitreeG1_IK)
#   HOPE_G1_NUM_ENVS   parallel envs                          (default 4096; the A3 recipe's 6144 needs >11 GB VRAM)
#   HOPE_G1_PROJECT    W&B project                            (default hope-g1)
#   HOPE_G1_EXTRA      extra Hydra overrides appended verbatim (e.g. "max_iterations=300")
#   HOPE_G1_VIDEO      1 = in-Isaac RecordVideo clips (default 0: Isaac Sim 5.1 RTX crashes on driver 610.43,
#                      github.com/isaac-sim/IsaacSim/issues/650; rollouts come from the MuJoCo sidecar instead)
set -uo pipefail
HOPE_REPO="${HOPE_REPO:-$HOME/HOPE}"
SIF="${HOPE_SIF:-$HOME/containers/hope_grasping.sif}"
ICD="$HOPE_REPO/hope_training/slurm/nvidia_icd.json"
export HOPE_G1_RUN_NAME="${HOPE_G1_RUN_NAME:-g1_v2a}"
export HOPE_G1_TASK="${HOPE_G1_TASK:-HOPEPingPongUnitreeG1}"
export HOPE_G1_NUM_ENVS="${HOPE_G1_NUM_ENVS:-4096}"
export HOPE_G1_PROJECT="${HOPE_G1_PROJECT:-hope-g1}"
export HOPE_G1_EXTRA="${HOPE_G1_EXTRA:-}"
export HOPE_G1_VIDEO="${HOPE_G1_VIDEO:-0}"
# Same W&B run across requeues of this job; ~/.netrc (bound via $HOME) supplies the API key.
export WANDB_RUN_ID="hope-g1-${HOPE_G1_RUN_NAME}-${SLURM_JOB_ID}"
export WANDB_RESUME=allow
echo "[train] host $(hostname) job=$SLURM_JOB_ID start $(date) repo=$HOPE_REPO run_name=$HOPE_G1_RUN_NAME task=$HOPE_G1_TASK num_envs=$HOPE_G1_NUM_ENVS project=$HOPE_G1_PROJECT extra='$HOPE_G1_EXTRA'"; nvidia-smi -L
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N OMNI_KIT_ALLOW_ROOT=1
# Apptainer --nv binds the NVIDIA libs but not a Vulkan ICD manifest; Kit needs one to create a GPU device.
apptainer exec --nv --writable-tmpfs --bind "$HOME" --bind "$ICD:/usr/share/vulkan/icd.d/nvidia_icd.json" \
  --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  --env HOPE_REPO="$HOPE_REPO" "$SIF" bash -c '
set -uo pipefail
cd "$HOPE_REPO/hope_training/whole_body_tracking"
source setup_train_env.sh
# Resume from the newest checkpoint of this run name, if any (requeue after preemption).
latest=$(ls -1 logs/rsl_rl/unitree_g1_hitter_pingpong/*_${HOPE_G1_RUN_NAME}/model_*.pt 2>/dev/null \
  | awk -F"model_" "{split(\$2,a,\".\"); print a[1]\" \"\$0}" | sort -n | tail -1 | cut -d" " -f2-)
resume=""
if [ -n "$latest" ]; then echo "[train] resuming from $latest"; resume="checkpoint_path=$latest"; fi
# video_interval is in env steps: 12000 = 500 iterations x 24 steps, i.e. one 300-step (6 s) clip per checkpoint.
video=""
if [ "${HOPE_G1_VIDEO}" = "1" ]; then video="video=true video_length=300 video_interval=12000"; fi
hope_isaac_py -u scripts/train.py task=$HOPE_G1_TASK headless=true num_envs=$HOPE_G1_NUM_ENVS \
  run_name=$HOPE_G1_RUN_NAME logger=wandb wandb_project=$HOPE_G1_PROJECT $video $resume $HOPE_G1_EXTRA
'
rc=$?
echo "[train] end $(date) exit=$rc"; exit $rc
