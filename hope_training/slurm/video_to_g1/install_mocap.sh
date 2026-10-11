#!/bin/bash
# One-time install of the video -> G1 pipeline (HITTER: GVHMR + GMR), Python 3.10 venvs, no conda/sudo.
# Run on the login node from the repository root:  bash hope_training/slurm/video_to_g1/install_mocap.sh
# Afterwards place the SMPL / SMPL-X body models (licensed; see README.md).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
mkdir -p "$MOCAP_ROOT" && cd "$MOCAP_ROOT"
for repo in zju3dv/GVHMR YanjieZe/GMR; do
  [ -d "${repo#*/}" ] || git clone -q --depth 1 "https://github.com/$repo.git"
done

[ -x gvhmr-venv/bin/python ] || uv venv -q --python 3.10 --seed gvhmr-venv
VIRTUAL_ENV=$PWD/gvhmr-venv uv pip install -q "numpy==1.23.5" cython "setuptools<70" wheel
VIRTUAL_ENV=$PWD/gvhmr-venv uv pip install -q --index-strategy unsafe-best-match -r GVHMR/requirements.txt --no-build-isolation
# hydra-zen needs omegaconf < 2.4 (get_yaml_loader).
VIRTUAL_ENV=$PWD/gvhmr-venv uv pip install -q -e GVHMR "omegaconf==2.3.0"

[ -x gmr-venv/bin/python ] || uv venv -q --python 3.10 --seed gmr-venv
VIRTUAL_ENV=$PWD/gmr-venv uv pip install -q -e GMR
VIRTUAL_ENV=$PWD/gmr-venv uv pip install -q pyyaml imageio imageio-ffmpeg PyOpenGL \
  torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# GVHMR shells out to ffmpeg; the nodes have none -> the imageio-ffmpeg static build.
ln -sf "$(ls "$PWD"/gmr-venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-* | head -1)" gvhmr-venv/bin/ffmpeg

# GVHMR pretrained models. The official Google Drive folder is often rate-limited, so use the
# HuggingFace mirror camenduru/GVHMR (third-party; same file names and sizes as the Drive release).
ck=GVHMR/inputs/checkpoints
for f in gvhmr/gvhmr_siga24_release.ckpt "hmr2/epoch=10-step=25000.ckpt" vitpose/vitpose-h-multi-coco.pth yolo/yolov8x.pt dpvo/dpvo.pth; do
  mkdir -p "$ck/$(dirname "$f")"
  [ -s "$ck/$f" ] || curl -sfL -o "$ck/$f" "https://huggingface.co/camenduru/GVHMR/resolve/main/$f"
done
mkdir -p "$ck/body_models/smpl" "$ck/body_models/smplx" GMR/assets/body_models/smplx videos clips
echo "MOCAP_INSTALL_DONE: now copy SMPL/SMPL-X (README.md)"
