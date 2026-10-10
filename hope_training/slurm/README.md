# Slurm jobs for the Unitree G1 training line

Everything Isaac-side runs inside the pinned HOPE image as an Apptainer SIF
(`pull_grasping_image.sbatch` stages it to `~/containers/hope_grasping.sif`); the MuJoCo
sim-to-sim side runs on the host venv (`~/.cache/hope-venv`: numpy, mujoco, onnxruntime,
pyyaml, pytest, imageio-ffmpeg, wandb). Logs go to `~/.cache/hope-slurm/` (create it once with
`mkdir -p ~/.cache/hope-slurm/sidecar`). All scripts take `HOPE_REPO` (default `$HOME/HOPE`),
`HOPE_SIF` and `HOPE_VENV` as environment knobs, so they can be submitted from any checkout:

```bash
sbatch hope_training/slurm/pull_grasping_image.sbatch                      # once per machine
sbatch hope_training/slurm/g1_isaac_smoke.sbatch                           # pre-flight, ~10 min
sbatch --export=ALL,HOPE_G1_RUN_NAME=g1_v2a hope_training/slurm/g1_train.sbatch
sbatch --export=ALL,HOPE_G1_RUN_NAME=g1_v2b,HOPE_G1_TASK=HOPEPingPongUnitreeG1_IK \
       hope_training/slurm/g1_train.sbatch
sbatch --export=ALL,HOPE_G1_RUN_NAME=g1_v2a hope_training/slurm/g1_rollout_sidecar.sbatch
scancel -n hope-g1-sidecar                                                 # stop the sidecars
```

| script | what |
|---|---|
| `g1_isaac_smoke.sbatch` | asset prep, joint-order pin (`check_g1_asset_isaac.py`), env-cfg name audit, Isaac half of the plant replay (`sim2sim_replay_g1.py`), 2-iteration train of each recipe in `HOPE_G1_SMOKE_TASKS`, export of the iteration-2 checkpoint, then on the host: MuJoCo half of the replay + comparison and a 2-serve MuJoCo eval |
| `g1_train.sbatch` | the full run on `long` (requeue + resume from the newest `*_<run_name>` checkpoint), `logger=wandb` project `hope-g1`, W&B run id `hope-g1-<run_name>-<jobid>`; knobs `HOPE_G1_RUN_NAME`, `HOPE_G1_TASK`, `HOPE_G1_NUM_ENVS` (4096 on the 11 GB 2080 Ti), `HOPE_G1_EXTRA`, `HOPE_G1_VIDEO` |
| `g1_rollout_sidecar.sbatch` | self-rescheduling (default every 2 h) while a trainer for the same run name is queued: newest checkpoint -> `export_onnx.py --robot unitree_g1` -> `mujoco_eval_onnx.py --video-out --json-out` -> `wandb_log_rollout.py` into the companion run `<run_name>-rollouts` (same project), x-axis = iteration |
| `nvidia_icd.json` | Vulkan ICD manifest bound into the container (`apptainer --nv` binds the driver libraries but no ICD, and Kit needs one to create a GPU device) |

Resource notes (2026-10): the group association allows 8 RTX 2080 Ti, 96 GB and 32 CPUs in
total; a 4096-env trainer uses about 12 GB of host RAM (`--mem=24G`) and 8 s per iteration, so
two variants plus their sidecars fit side by side. `HOPE_G1_VIDEO=1` (in-Isaac `RecordVideo`)
is off by default because Isaac Sim 5.1's RTX renderer crashes on NVIDIA driver 610.43
(isaac-sim/IsaacSim#650); the sidecar's MuJoCo-rendered rollouts are the mp4 source.
