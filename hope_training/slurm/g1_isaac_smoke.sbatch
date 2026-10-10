#!/bin/bash
#SBATCH -J hope-g1-smoke
#SBATCH -A tritondroids
#SBATCH -p normal
#SBATCH --gres=rtx2080ti:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o /home/%u/.cache/hope-slurm/%j-g1-smoke.out
# Unitree G1 Isaac pre-flight inside the pinned HOPE image: asset prep, joint-order pin, env-cfg name
# audit, Isaac-vs-MuJoCo plant replay, 2-iteration training smoke of the G1 recipes, export + a
# 2-serve MuJoCo eval of the iteration-2 checkpoint. Knobs: HOPE_REPO, HOPE_SIF, HOPE_VENV,
# HOPE_G1_SMOKE_TASKS (space-separated task recipes; default "HOPEPingPongUnitreeG1 HOPEPingPongUnitreeG1_IK").
set -uo pipefail
HOPE_REPO="${HOPE_REPO:-$HOME/HOPE}"
SIF="${HOPE_SIF:-$HOME/containers/hope_grasping.sif}"
VENV="${HOPE_VENV:-$HOME/.cache/hope-venv}"
ICD="$HOPE_REPO/hope_training/slurm/nvidia_icd.json"
export HOPE_G1_SMOKE_TASKS="${HOPE_G1_SMOKE_TASKS:-HOPEPingPongUnitreeG1 HOPEPingPongUnitreeG1_IK}"
SMOKE_OUT="$HOME/.cache/hope-slurm/smoke-$SLURM_JOB_ID"; mkdir -p "$SMOKE_OUT"
echo "[smoke] host $(hostname) job=$SLURM_JOB_ID start $(date) repo=$HOPE_REPO tasks='$HOPE_G1_SMOKE_TASKS' out=$SMOKE_OUT"; nvidia-smi -L
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=N OMNI_KIT_ALLOW_ROOT=1
apptainer exec --nv --writable-tmpfs --bind "$HOME" --bind "$ICD:/usr/share/vulkan/icd.d/nvidia_icd.json" \
  --env VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json --env VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  --env HOPE_REPO="$HOPE_REPO" --env SMOKE_OUT="$SMOKE_OUT" "$SIF" bash -c '
set -uo pipefail
cd "$HOPE_REPO/hope_training/whole_body_tracking"
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
if [ $rc_pin -ne 0 ]; then echo "[smoke] joint order NOT pinned -> stopping (clips/YAML must be fixed first)"; exit 3; fi
echo "=== 5. Isaac-vs-MuJoCo plant replay (Isaac half) ==="
hope_isaac_py_t 20m -u scripts/sim2sim_replay_g1.py --isaac --out "$SMOKE_OUT/replay_isaac.npz"; rc_replay=$?
echo "[smoke] sim2sim isaac half exit=$rc_replay"
rc_train=0; rc_export=0
for task in $HOPE_G1_SMOKE_TASKS; do
  echo "=== 6. training smoke $task (16 envs, 2 iterations) ==="
  hope_isaac_py_t 60m -u scripts/train.py task=$task num_envs=16 max_iterations=2 headless=true run_name=smoke_${task}_$SLURM_JOB_ID; rc=$?
  echo "[smoke] train $task exit=$rc"; rc_train=$(( rc_train || rc ))
done
ckpt=$(ls -1t logs/rsl_rl/unitree_g1_hitter_pingpong/*_smoke_HOPEPingPongUnitreeG1_$SLURM_JOB_ID/model_*.pt 2>/dev/null | head -1)
if [ -n "$ckpt" ]; then
  echo "=== 7. export iteration-2 checkpoint $ckpt ==="
  hope_isaac_py_t 15m -u scripts/export_onnx.py --robot unitree_g1 --task HOPE-HitterPingPong-UnitreeG1-v0 \
    --experiment-name unitree_g1_hitter_pingpong --checkpoint "$ckpt" --output-dir "$SMOKE_OUT/exported" \
    --motion-file "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_forehand.npz" \
    --motion-file-2 "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_backhand.npz"; rc_export=$?
  echo "[smoke] export exit=$rc_export"
fi
exit $(( rc_test != 0 || rc_train != 0 || rc_export != 0 || rc_replay != 0 ))
'
rc=$?
echo "=== 5b. Isaac-vs-MuJoCo plant replay (MuJoCo half + comparison, host venv) ==="
if [ -f "$SMOKE_OUT/replay_isaac.npz" ]; then
  (cd "$HOPE_REPO" && MUJOCO_GL=egl "$VENV/bin/python" hope_training/whole_body_tracking/scripts/sim2sim_replay_g1.py --mujoco --compare "$SMOKE_OUT/replay_isaac.npz" --out "$SMOKE_OUT/replay_mujoco.npz"); rc_cmp=$?
  echo "[smoke] sim2sim compare exit=$rc_cmp"; rc=$(( rc || rc_cmp ))
fi
echo "=== 8. MuJoCo eval of the iteration-2 export (must load and stand, not score) ==="
if [ -f "$SMOKE_OUT/exported/policy.onnx" ]; then
  (cd "$HOPE_REPO" && MUJOCO_GL=egl timeout -k 30 10m "$VENV/bin/python" hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py --robot unitree_g1 \
    --onnx "$SMOKE_OUT/exported/policy.onnx" --num-serves 2 --seed 0 --video-out "$SMOKE_OUT/eval.mp4" --json-out "$SMOKE_OUT/eval.json"); rc_eval=$?
  echo "[smoke] mujoco eval exit=$rc_eval"; rc=$(( rc || rc_eval ))
fi
echo "[smoke] end $(date) exit=$rc"; exit $rc
