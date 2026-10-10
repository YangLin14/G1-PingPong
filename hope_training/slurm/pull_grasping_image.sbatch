#!/bin/bash
#SBATCH -J hope-pull-grasping
#SBATCH -A tritondroids
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o /home/%u/.cache/hope-slurm/%j-pull.out
# Stage the pinned HOPE Isaac image (docs/DISTROBOX_SETUP.md digest) as an Apptainer SIF.
set -euo pipefail
export APPTAINER_CACHEDIR=$HOME/.cache/apptainer
export APPTAINER_TMPDIR=$HOME/.cache/apptainer/tmp
IMAGE='docker://docker.io/danielmunicio/omnidrones@sha256:825c2da0a5ff581a7009c4f2e0ac44b15e83490242de17b13093f872fe569db2'
SIF="${HOPE_SIF:-$HOME/containers/hope_grasping.sif}"
echo "[pull] host $(hostname) start $(date)"
if [ -s "$SIF" ]; then echo "[pull] $SIF already exists ($(du -h "$SIF" | cut -f1)); nothing to do"; exit 0; fi
apptainer pull --disable-cache=false "$SIF.part" "$IMAGE"
mv "$SIF.part" "$SIF"
echo "[pull] done $(date): $(du -h "$SIF" | cut -f1)"
apptainer exec "$SIF" bash -c 'cat /workspace/isaacsim/VERSION 2>/dev/null; ls /workspace/omni_drones/third_party/IsaacLab/source | head -3; ls /opt/drone_venv/lib | head -2'
