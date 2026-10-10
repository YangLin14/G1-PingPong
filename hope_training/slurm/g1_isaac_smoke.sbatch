#!/bin/bash
#SBATCH -J hope-g1-smoke
#SBATCH -A tritondroids
#SBATCH -p normal
#SBATCH --gres=rtx2080ti:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH -o /home/ogershony/.cache/hope-slurm/%j-g1-smoke.out
# Unitree G1 Isaac pre-flight + 2-iteration training smoke inside the pinned HOPE image.
set -uo pipefail
SIF=/home/ogershony/containers/hope_grasping.sif
echo "[smoke] host $(hostname) start $(date)"; nvidia-smi -L
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N OMNI_KIT_ALLOW_ROOT=1
# Apptainer --nv binds the NVIDIA libs but not a Vulkan ICD manifest; Kit needs one to create a GPU device.
apptainer exec --nv --writable-tmpfs --bind /home/ogershony --bind /home/ogershony/.cache/hope-slurm/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json "$SIF" bash -c '
set -uo pipefail
cd /home/ogershony/HOPE/hope_training/whole_body_tracking
source setup_train_env.sh
# `timeout` cannot exec a shell function, so expand hope_isaac_py inline with a wall-clock cap.
hope_isaac_py_t () { local t=$1; shift; timeout -k 30 "$t" env PYTHONPATH="${HOPE_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" "${ISAAC_PYTHON}" "$@"; }
echo "=== 1. version probe ==="
hope_isaac_py - <<PY
from importlib import metadata, util
import sys, torch
for name in ("isaaclab","torch","hydra-core","omegaconf","rsl-rl-lib","gymnasium","numpy"):
    try: print(f"{name}={metadata.version(name)}")
    except Exception as e: print(f"{name}=? ({e})")
print("python", sys.version.split()[0], "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("whole_body_tracking", util.find_spec("whole_body_tracking").origin)
assert torch.cuda.is_available()
PY
echo "=== 2. prepare G1 asset ==="
hope_isaac_py scripts/prepare_g1_isaac_asset.py --force && hope_isaac_py scripts/prepare_g1_isaac_asset.py --check || exit 2
echo "=== 3. pin joint order ==="
hope_isaac_py_t 20m -u scripts/check_g1_asset_isaac.py; rc_pin=$?
echo "[smoke] check_g1_asset_isaac exit=$rc_pin"
echo "=== 4. env cfg name audit ==="
hope_isaac_py_t 15m -m pytest -q -p no:cacheprovider tests/test_g1_hope_env_cfg_names.py; rc_test=$?
echo "[smoke] test_g1_hope_env_cfg_names exit=$rc_test"
if [ $rc_pin -ne 0 ]; then echo "[smoke] joint order NOT pinned -> skipping the training smoke (clips/YAML must be fixed first)"; exit 3; fi
echo "=== 5. training smoke (16 envs, 2 iterations) ==="
hope_isaac_py_t 60m -u scripts/train.py task=HOPEPingPongUnitreeG1 num_envs=16 max_iterations=2 headless=true run_name=g1_smoke; rc_train=$?
echo "[smoke] train exit=$rc_train"
exit $(( rc_test != 0 || rc_train != 0 ))
'
rc=$?
echo "[smoke] end $(date) exit=$rc"; exit $rc
