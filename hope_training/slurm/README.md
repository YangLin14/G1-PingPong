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

# The original A3 recipe (QUICKSTART_A3_ISAAC.md step 5) through the same jobs, as the reference line
# in the same W&B project; the sidecar needs the A3 export/MuJoCo profile:
A3="HOPE_G1_RUN_NAME=a3_ref,HOPE_G1_TASK=HOPEPingPong,HOPE_G1_EXPERIMENT=agibot_a3_hitter_pingpong,\
HOPE_G1_MOTION_FH=hope_training/motions/preprocessed/hope_forehand.npz,HOPE_G1_MOTION_BH=hope_training/motions/preprocessed/hope_backhand.npz"
sbatch --export=ALL,$A3 hope_training/slurm/g1_train.sbatch
sbatch --export=ALL,$A3,HOPE_G1_ROBOT=agibot_a3,HOPE_G1_GYM_TASK=HOPE-HitterPingPong-AgibotA3-v0 \
       hope_training/slurm/g1_rollout_sidecar.sbatch
```

The A3 asset must be prepared on the checkout first (`scripts/prepare_a3_isaac_asset.py --force`,
done by the smoke job's A3 baseline step or by hand inside the image).

| script | what |
|---|---|
| `g1_isaac_smoke.sbatch` | asset prep, joint-order pin (`check_g1_asset_isaac.py`), env-cfg name audit, Isaac half of the plant replay (`sim2sim_replay_g1.py`), 2-iteration train of each recipe in `HOPE_G1_SMOKE_TASKS`, export of the iteration-2 checkpoint, then on the host: MuJoCo half of the replay + comparison and a 2-serve MuJoCo eval |
| `g1_train.sbatch` | the full run on `long` (requeue + resume from the newest `*_<run_name>` checkpoint), `logger=wandb` project `hope-g1`, W&B run id `hope-g1-<run_name>-<jobid>`; knobs `HOPE_G1_RUN_NAME`, `HOPE_G1_TASK`, `HOPE_G1_EXPERIMENT` (log dir), `HOPE_G1_MOTION_FH`/`_BH`, `HOPE_G1_NUM_ENVS` (4096 on the 11 GB 2080 Ti), `HOPE_G1_EXTRA`, `HOPE_G1_VIDEO` |
| `g1_rollout_sidecar.sbatch` | self-rescheduling (default every 2 h) while a trainer for the same run name is queued: newest checkpoint -> `export_onnx.py --robot unitree_g1` -> `mujoco_eval_onnx.py --video-out --json-out` -> `wandb_log_rollout.py` into the companion run `<run_name>-rollouts` (same project), x-axis = iteration; `HOPE_G1_ROBOT`/`HOPE_G1_GYM_TASK`/`HOPE_G1_EXPERIMENT` select the G1 (default) or A3 profile |
| `pick_egl_device.py` | prints the first `MUJOCO_EGL_DEVICE_ID` that renders; the sidecar pins it because MuJoCo's own device walk aborts on a node with a faulty GPU |
| `nvidia_icd.json` | Vulkan ICD manifest bound into the container (`apptainer --nv` binds the driver libraries but no ICD, and Kit needs one to create a GPU device) |

Resource notes (2026-10): the group association allows 8 RTX 2080 Ti, 96 GB and 32 CPUs in
total; a 4096-env trainer uses about 12 GB of host RAM (`--mem=24G`) and 8 s per iteration, so
two variants plus their sidecars fit side by side. `HOPE_G1_VIDEO=1` (in-Isaac `RecordVideo`)
is off by default because Isaac Sim 5.1's RTX renderer crashes on NVIDIA driver 610.43
(isaac-sim/IsaacSim#650); the sidecar's MuJoCo-rendered rollouts are the mp4 source.

## Without the container: `pip_env/` and `video_to_g1/`

[`pip_env/`](pip_env/README.md) runs the same Isaac stack from a pip-installed Isaac Sim 5.1 venv
(no 64 GB image, no sudo) and documents the node workarounds it needs; [`video_to_g1/`](video_to_g1/README.md)
turns a phone video of a human swing into a G1 reference clip (GVHMR + GMR, the HITTER pipeline).
Both are part of the forehand cross-court line described in
[`docs/g1_forehand/README.md`](../../docs/g1_forehand/README.md).
