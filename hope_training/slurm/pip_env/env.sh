# Shared paths for the pip_env jobs. Sourced by every script in this folder.
#   HOPE_ROOT  working root on the cluster        (default $HOME/hope)
#   HOPE_REPO  this repository's checkout          (default $HOPE_ROOT/HOPE)
#   ISAAC_VENV Isaac Sim 5.1 + Isaac Lab venv      (default $HOPE_ROOT/isaac-venv)
#   ISAACLAB   Isaac Lab v2.3.2 source checkout    (default $HOPE_ROOT/IsaacLab)
#   SYSLIBS    user-space system libraries (libGLU) (default $HOPE_ROOT/syslibs)
export HOPE_ROOT="${HOPE_ROOT:-$HOME/hope}"
export HOPE_REPO="${HOPE_REPO:-$HOPE_ROOT/HOPE}"
export ISAAC_VENV="${ISAAC_VENV:-$HOPE_ROOT/isaac-venv}"
export ISAACLAB="${ISAACLAB:-$HOPE_ROOT/IsaacLab}"
export SYSLIBS="${SYSLIBS:-$HOPE_ROOT/syslibs}"
export WBT="$HOPE_REPO/hope_training/whole_body_tracking"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOPE_ROOT/.uv-cache}"
