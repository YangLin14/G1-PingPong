#!/bin/bash
# Human swing video -> GVHMR (SMPL-X) -> GMR (Unitree G1) -> HOPE G1 reference clip + MP4 preview.
# Submit from the repository root:
#   sbatch --export=ALL,VIDEO=$HOME/hope/mocap/videos/forehand.mp4,NAME=g1vid_forehand \
#          hope_training/slurm/video_to_g1/video_to_g1.sbatch
# Then trim with START/END (source frames, 30 Hz) and fix the strike with STRIKE (output frame, 50 Hz)
# if the preview needs it, and resubmit (GVHMR is cached per video).
#SBATCH --job-name=video-to-g1
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/video_to_g1/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/video_to_g1/env.sh"
: "${VIDEO:?set VIDEO=/path/to/video.mp4}"; NAME="${NAME:-g1vid_forehand}"
START="${START:-0}"; END="${END:-}"; STRIKE="${STRIKE:-}"; BLEND="${BLEND:-0.3}"
M="$MOCAP_ROOT"; OUT="$M/clips/$NAME"; mkdir -p "$OUT"
stem=$(basename "${VIDEO%.*}")
pred="$M/GVHMR/outputs/demo/$stem/hmr4d_results.pt"
if [ ! -f "$pred" ]; then
  echo "### GVHMR on $VIDEO (static camera)"
  (cd "$M/GVHMR" && "$M/gvhmr-venv/bin/python" tools/demo/demo.py --video="$VIDEO" -s)
fi
echo "### GMR retarget -> G1"
"$M/gmr-venv/bin/python" "$WBT/scripts/gvhmr_to_g1_headless.py" --gmr-root "$M/GMR" --gvhmr-pred "$pred" --save-path "$OUT/$NAME.gmr.pkl"
echo "### HOPE clip"
egl=$(MUJOCO_GL=egl "$M/gmr-venv/bin/python" "$HOPE_REPO/hope_training/slurm/pick_egl_device.py")
args=(--gmr-pkl "$OUT/$NAME.gmr.pkl" --gmr-xml "$M/GMR/assets/unitree_g1/g1_mocap_29dof.xml" --name "$NAME"
      --start "$START" --blend-end-s "$BLEND" --out-dir "$OUT" --render "$OUT/$NAME.mp4")
[ -n "$END" ] && args+=(--end "$END"); [ -n "$STRIKE" ] && args+=(--strike-frame "$STRIKE")
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$egl" "$M/gmr-venv/bin/python" "$WBT/scripts/gmr_video_clip_to_g1.py" "${args[@]}"
echo "### done: $OUT ($NAME.npz/.yaml/.mp4); GVHMR overlays in $M/GVHMR/outputs/demo/$stem/"
