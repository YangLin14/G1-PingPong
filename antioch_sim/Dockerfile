FROM antioch-engine/base:0.5.338

# Isaac Sim 5.1 (pip) + Isaac Lab v2.3.2 + HOPE pins: the same stack as the SDSC cluster
# (~/hope/install_isaac.sbatch); cluster-constraints.txt is that venv's pip freeze.
ENV ANTIOCH_PROJECT_DIR=/workspace/project \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    NVIDIA_DRIVER_CAPABILITIES=all \
    ISAAC_VENV=/opt/isaac-venv \
    ISAACLAB_ROOT=/opt/IsaacLab \
    ISAAC_PYTHON=/opt/isaac-venv/bin/python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=8
USER root
RUN if command -v apt-get >/dev/null; then \
      apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git libgomp1 libatomic1 libglu1-mesa libgl1 libegl1 libvulkan1 libxt6 libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libsm6 libice6 && rm -rf /var/lib/apt/lists/*; fi
RUN command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh)
COPY cluster-constraints.txt /opt/cluster-constraints.txt
RUN uv venv --python 3.11 --seed $ISAAC_VENV && $ISAAC_VENV/bin/python --version
ENV VIRTUAL_ENV=/opt/isaac-venv UV_SYSTEM_PYTHON=0
RUN uv pip install --python /opt/isaac-venv/bin/python "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python /opt/isaac-venv/bin/python "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git $ISAACLAB_ROOT \
 && cd $ISAACLAB_ROOT \
 && uv pip install --python /opt/isaac-venv/bin/python "setuptools<70" && uv pip install --python /opt/isaac-venv/bin/python --no-build-isolation "flatdict==4.0.1" \
 && uv pip install --python /opt/isaac-venv/bin/python -c /opt/cluster-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
      -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_rl -e source/isaaclab_tasks
RUN uv pip install --python /opt/isaac-venv/bin/python -c /opt/cluster-constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match "rsl-rl-lib==3.1.2" "hydra-core==1.3.2" "omegaconf==2.3.0" \
      "numpy==1.26.4" onnx onnxruntime mujoco

WORKDIR /workspace/project
