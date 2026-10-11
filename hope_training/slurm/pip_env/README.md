# pip_env: HOPE on SDSC Atlantis without the container

The sibling jobs in `../` run Isaac inside the 64 GB `grasping` image. These run the same stack
from a pip-installed Isaac Sim 5.1 + Isaac Lab v2.3.2 venv instead, which needs no container and
no sudo. Submit every job **from the repository root** (`$SLURM_SUBMIT_DIR` locates `env.sh`).
Paths are set in [`env.sh`](env.sh) (`HOPE_ROOT`, default `~/hope`; `HOPE_REPO`, default
`~/hope/HOPE`). Full background: [`docs/TRITON_G1_FOREHAND.md`](../../../docs/TRITON_G1_FOREHAND.md).

```bash
sbatch hope_training/slurm/pip_env/install_isaac.sbatch                     # once, ~30 min
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_xc hope_training/slurm/pip_env/train.sbatch
sbatch hope_training/slurm/pip_env/run.sbatch scripts/evaluate.py --task HOPE-HitterPingPong-UnitreeG1-v0 \
       --experiment-name unitree_g1_hitter_pingpong --checkpoint logs/rsl_rl/unitree_g1_hitter_pingpong/<run>/model_<i>.pt
```

| file | what |
|---|---|
| `env.sh` | shared paths |
| `install_isaac.sbatch` | venv (Python 3.11, torch 2.7.0+cu128, `isaacsim[all,extscache]==5.1.0`), Isaac Lab v2.3.2 source install, node workarounds below, renders `setup_train_env.local.sh`, Kit boot check |
| `setup_train_env.local.sh.template` | the machine-local overrides `setup_train_env.sh` auto-sources |
| `run.sbatch` | run any `hope_isaac_py` command on one GPU |
| `train.sbatch` | requeue-safe training with resume and a fixed iteration total |
| `g1_smoke.sbatch` | G1 pre-flight: asset, joint order, plant replay, 2-iteration train, export, MuJoCo eval |
| `status.sh` | one-screen status: jobs, club quota, iteration and key metrics per run |
| `eval_checkpoint.sh` | newest (or given) checkpoint -> ONNX -> MuJoCo stats + MP4; `--in-job <id>` runs inside a running job, or `sbatch` it |

Day-to-day commands: [`docs/RUNBOOK_G1_FOREHAND.md`](../../../docs/RUNBOOK_G1_FOREHAND.md).

Node workarounds (all applied by `install_isaac.sbatch`; each was a real failure):

| symptom | cause | fix |
|---|---|---|
| Kit: `No device could be created`, GPU table empty | Kit's bundled Vulkan loader misses `nvidia_icd.x86_64.json` | `VK_ICD_FILENAMES` |
| Kit hangs forever after URDF import | `libneuray.so` needs `libGLU.so.1`, absent on nodes | extract `mesa-libGLU` RPM, `LD_LIBRARY_PATH` |
| `PermissionError: /tmp/isaaclab/...` | another user created the shared `/tmp` dir first | per-user `TMPDIR` |
| `PermissionError: /tmp/IsaacLab/usd_...` | Isaac Lab hard-codes that path | one-line patch to honor `TMPDIR` |
| `isaaclab.sh --install` asks for sudo | it apt-installs build tools | `pip install -e source/*` |
| `flatdict` build fails (`pkg_resources`) | setuptools >= 81 | prebuild with `setuptools<70` |

Quota: the `tritondroids` association caps the whole club at 32 CPUs / 96 GB / 8 GPUs; these jobs
ask for 3–4 CPUs and 24 GB so they fit next to the container jobs.
