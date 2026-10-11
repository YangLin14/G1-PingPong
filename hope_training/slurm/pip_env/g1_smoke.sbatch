#!/bin/bash
# G1 pre-flight on the pip env (mirrors ../g1_isaac_smoke.sbatch, which needs the container):
# asset prep -> joint-order pin -> Isaac/MuJoCo plant replay -> 2-iteration train of TASK -> export ->
# 2-serve MuJoCo eval with an mp4. Outputs in $HOPE_ROOT/g1_smoke/<jobid>/.
#   sbatch [--export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC] hope_training/slurm/pip_env/g1_smoke.sbatch
#SBATCH --job-name=hope-g1-smoke
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
TASK="${TASK:-HOPEPingPongUnitreeG1}"
OUT="$HOPE_ROOT/g1_smoke/$SLURM_JOB_ID"; mkdir -p "$OUT"
cd "$WBT"
set +u; source setup_train_env.sh; set -u
py() { PYTHONPATH="${HOPE_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}" timeout -k 30 "$1" "$ISAAC_PYTHON" "${@:2}"; }
step() { echo; echo "### $*"; }
step "1. prepare G1 asset"
py 10m scripts/prepare_g1_isaac_asset.py --force && py 10m scripts/prepare_g1_isaac_asset.py --check || exit 2
step "2. pin joint order";      py 20m -u scripts/check_g1_asset_isaac.py; echo "[smoke] pin exit=$?"
step "3. plant replay (Isaac)"; py 20m -u scripts/sim2sim_replay_g1.py --isaac --out "$OUT/replay_isaac.npz"; echo "[smoke] replay isaac exit=$?"
step "4. train $TASK for 2 iterations"
py 60m -u scripts/train.py task="$TASK" num_envs=16 max_iterations=2 headless=true run_name="smoke_$SLURM_JOB_ID"
echo "[smoke] train exit=$?"
ckpt=$(ls -1t logs/rsl_rl/unitree_g1_hitter_pingpong/*_smoke_"$SLURM_JOB_ID"/model_*.pt 2>/dev/null | head -1)
step "5. export $ckpt"
py 15m -u scripts/export_onnx.py --robot unitree_g1 --task HOPE-HitterPingPong-UnitreeG1-v0 \
  --experiment-name unitree_g1_hitter_pingpong --checkpoint "$ckpt" --output-dir "$OUT/exported" \
  --motion-file "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_forehand.npz" \
  --motion-file-2 "$HOPE_REPO/hope_training/motions/preprocessed/g1_hope_backhand.npz"
echo "[smoke] export exit=$?"
cd "$HOPE_REPO"
step "6. plant replay (MuJoCo + compare)"
MUJOCO_GL=egl "$ISAAC_PYTHON" hope_training/whole_body_tracking/scripts/sim2sim_replay_g1.py --mujoco \
  --compare "$OUT/replay_isaac.npz" --out "$OUT/replay_mujoco.npz"; echo "[smoke] replay compare exit=$?"
step "7. MuJoCo eval of the 2-iteration export (must load and stand, not score)"
egl=$(MUJOCO_GL=egl "$ISAAC_PYTHON" hope_training/slurm/pick_egl_device.py)
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$egl" timeout 10m "$ISAAC_PYTHON" hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py \
  --robot unitree_g1 --onnx "$OUT/exported/policy.onnx" --num-serves 2 --seed 0 --video-out "$OUT/eval.mp4" --json-out "$OUT/eval.json"
echo "[smoke] mujoco eval exit=$?"
echo SMOKE_END
