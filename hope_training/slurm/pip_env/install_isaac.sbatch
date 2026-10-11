#!/bin/bash
# One-time install of the HOPE training stack WITHOUT the 64 GB container: Isaac Sim 5.1 (pip) +
# Isaac Lab v2.3.2 (isaaclab 0.54.2) + the docs/DISTROBOX_SETUP.md pins, then a headless Kit boot.
# Idempotent; rerun after a failure. Needs no sudo (see docs/TRITON_G1_FOREHAND.md, "Cluster").
#   sbatch hope_training/slurm/pip_env/install_isaac.sbatch
#SBATCH --job-name=hope-install
#SBATCH --account=tritondroids
#SBATCH --partition=normal
#SBATCH --gres=rtx2080ti:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "$SLURM_SUBMIT_DIR/hope_training/slurm/pip_env/env.sh" 2>/dev/null || source "$HOME/hope/HOPE/hope_training/slurm/pip_env/env.sh"
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
[ -x "$ISAAC_VENV/bin/python" ] || uv venv --python 3.11 --seed "$ISAAC_VENV"
source "$ISAAC_VENV/bin/activate"
uv pip install "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab at the tag whose isaaclab extension is 0.54.2. Skip LFS: its test images block checkout.
[ -d "$ISAACLAB" ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --branch v2.3.2 \
  https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
cd "$ISAACLAB"
# Shared nodes: /tmp/IsaacLab belongs to whoever made it first; make the URDF converter honor TMPDIR.
f=source/isaaclab/isaaclab/sim/converters/asset_converter_base.py
grep -q tempfile.gettempdir "$f" || sed -i \
  's|self._usd_dir = f"/tmp/IsaacLab/usd_{time_tag}|self._usd_dir = f"{tempfile.gettempdir()}/IsaacLab/usd_{time_tag}|; s|^import random$|import random\nimport tempfile|' "$f"
# isaaclab.sh --install wants sudo for apt; install the source packages directly.
# flatdict 4.0.1 needs pkg_resources at build time (removed in setuptools>=81).
uv pip install "setuptools<70" && uv pip install --no-build-isolation "flatdict==4.0.1"
uv pip install -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_rl -e source/isaaclab_tasks
uv pip install "rsl-rl-lib==3.1.2" "hydra-core==1.3.2" "omegaconf==2.3.0" "numpy==1.26.4" \
  onnx onnxruntime mujoco imageio imageio-ffmpeg

# libGLU.so.1 from the distro RPM, extracted into user space.
if [ ! -e "$SYSLIBS/usr/lib64/libGLU.so.1" ]; then
  mkdir -p "$SYSLIBS/rpm" && (cd "$SYSLIBS/rpm" && dnf download -q mesa-libGLU)
  (cd "$SYSLIBS" && rpm2cpio rpm/mesa-libGLU-*.rpm | cpio -idm 2>/dev/null)
fi

sed -e "s|@ISAAC_VENV@|$ISAAC_VENV|" -e "s|@ISAACLAB@|$ISAACLAB|" -e "s|@SYSLIBS@|$SYSLIBS|" \
  "$HOPE_REPO/hope_training/slurm/pip_env/setup_train_env.local.sh.template" > "$WBT/setup_train_env.local.sh"

cd "$WBT"
set +u; source setup_train_env.sh; set -u
hope_isaac_py - <<'PY'
from importlib import metadata
import sys, torch
for n in ("isaacsim", "isaaclab", "torch", "rsl-rl-lib", "hydra-core", "gymnasium", "numpy"):
    print(f"{n}={metadata.version(n)}")
print("python", sys.version.split()[0], "cuda", torch.version.cuda, "available", torch.cuda.is_available())
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
print("ISAAC_KIT_BOOT=OK", torch.cuda.get_device_name(0), flush=True)
app.close()
PY
echo INSTALL_DONE
