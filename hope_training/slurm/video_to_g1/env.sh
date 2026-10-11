# Shared paths for the video -> G1 pipeline.
#   HOPE_ROOT  working root on the cluster                     (default $HOME/hope)
#   HOPE_REPO  this repository's checkout                       (default $HOPE_ROOT/HOPE)
#   MOCAP_ROOT GVHMR + GMR checkouts, venvs, videos/, clips/   (default $HOPE_ROOT/mocap)
export HOPE_ROOT="${HOPE_ROOT:-$HOME/hope}"
export HOPE_REPO="${HOPE_REPO:-$HOPE_ROOT/HOPE}"
export MOCAP_ROOT="${MOCAP_ROOT:-$HOPE_ROOT/mocap}"
export WBT="$HOPE_REPO/hope_training/whole_body_tracking"
export PATH="$MOCAP_ROOT/gvhmr-venv/bin:$HOME/.local/bin:$PATH"  # gvhmr-venv/bin carries ffmpeg
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOPE_ROOT/.uv-cache}"
