# video_to_g1: phone video -> G1 reference clip

The HITTER reference pipeline (arXiv 2508.21043): a monocular video of a human swing is lifted to
SMPL-X with [GVHMR](https://github.com/zju3dv/GVHMR), retargeted to the Unitree G1 with
[GMR](https://github.com/YanjieZe/GMR), and converted to the HOPE G1 motion schema by
[`gmr_video_clip_to_g1.py`](../../whole_body_tracking/scripts/gmr_video_clip_to_g1.py).
Background and design notes: [`docs/g1_forehand/README.md`](../../../docs/g1_forehand/README.md).

## One-time setup (cluster login node, from the repository root)

```bash
bash hope_training/slurm/video_to_g1/install_mocap.sh
```

Then the body models, which are licensed per user and must **never be committed**: register at
[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de)
and place (`M=~/hope/mocap`):

| file | destination |
|---|---|
| `basicmodel_{neutral,m,f}_lbs_10_207_0_v1.1.0.pkl` (SMPL) | `$M/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` |
| `SMPLX_{NEUTRAL,MALE,FEMALE}.npz` (SMPL-X) | `$M/GVHMR/inputs/checkpoints/body_models/smplx/` **and** `$M/GMR/assets/body_models/smplx/` |

## Per video

Record: static camera, about 45° front-side, whole body and racket in frame, 30–60 fps, a full
swing from the ready pose back to the ready pose.

```bash
scp forehand.mp4 sdsc-scc:hope/mocap/videos/
sbatch --export=ALL,VIDEO=$HOME/hope/mocap/videos/forehand.mp4,NAME=g1vid_forehand \
       hope_training/slurm/video_to_g1/video_to_g1.sbatch
# -> ~/hope/mocap/clips/g1vid_forehand/{g1vid_forehand.npz, .yaml, .mp4}
```

Check the MP4: the red border marks the strike frame (default: peak racket-centre speed; the top
five candidates are printed in the log). Resubmit with `START`/`END` (source frames at 30 Hz) to
trim to ready -> swing -> ready, and `STRIKE` (output frame at 50 Hz) to override the strike. The
last 0.3 s are blended back to frame 0 so the clip loops in a rally (`BLEND=0` disables).

| file | what |
|---|---|
| `env.sh` | shared paths (`HOPE_ROOT`, `HOPE_REPO`, `MOCAP_ROOT`) |
| `install_mocap.sh` | GVHMR/GMR checkouts, two Python 3.10 venvs, ffmpeg, GVHMR checkpoints |
| `video_to_g1.sbatch` | GVHMR -> `gvhmr_to_g1_headless.py` -> `gmr_video_clip_to_g1.py --render` |
