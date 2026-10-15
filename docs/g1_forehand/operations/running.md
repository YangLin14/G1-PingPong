# Running it (laptop and cluster)

[繁體中文](running.zh-TW.md) · [Docs index](../README.md)

### Local (macOS)

```bash
cd G1-PingPong
python3 -m venv .venv && .venv/bin/pip install -r a3_deploy/a3_deploy_example/reference/requirements.txt torch pytest imageio imageio-ffmpeg
.venv/bin/python -m pytest -q unitree/tests hope_training/whole_body_tracking/tests/test_forehand_success.py \
  hope_training/whole_body_tracking/tests/test_make_g1_forehand_bank.py hope_training/whole_body_tracking/tests/test_gmr_video_clip_to_g1.py \
  hope_training/whole_body_tracking/tests/test_mujoco_forehand_bank.py

# Watch the ball feeding. MuJoCo windows on macOS need mjpython. Because this path has a space,
# call mjpython through the venv python (its shebang breaks on the space).
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 30 --speed 0.5
.venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py --camera top
.venv/bin/python hope_training/whole_body_tracking/scripts/view_feed_bank.py --rows 8 --record artifacts/feed.mp4
```

In the viewer: purple is the feed origin (opponent's right, at the far end line), orange the feed
bounce zone, faint green the four target zones and bright green the current question's target. The
yellow trail is the ball; the red disc and arrow are the racket target (face and velocity).

### Cluster (SDSC Atlantis, `ssh sdsc-scc`)

* Account `tritondroids`. The whole club shares **32 CPUs / 96 GB / 8 GPUs**. Isaac needs an RTX
  GPU, so use `--gres=rtx2080ti:1` (node `zixian`, 11 GB, driver 610.43). One training run uses one GPU.
* `zixian`'s 8 GPUs are also open to other accounts (on 10-11 an `scc` job held 4), so the club quota
  of 8 does not mean 8 are free. Jobs in QOS `normal` (4 h limit) can preempt QOS `long` (training);
  `train.sbatch` requeues and resumes from the newest checkpoint (up to 100 iterations are redone).
* After the 10-11 outage GPU `0000:61:00.0` stopped answering the driver for a day; it was fixed on
  10-12. `train.sbatch` still falls back to a working device if CUDA does not come up.
* Layout: `~/hope/{HOPE (this repo), isaac-venv, IsaacLab, syslibs, mocap, slurm_logs}`.
* Submit every job **from the repository root**.

```bash
sbatch hope_training/slurm/pip_env/install_isaac.sbatch                    # once
bash   hope_training/slurm/video_to_g1/install_mocap.sh                    # once (+ SMPL/SMPL-X, see its README)
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC hope_training/slurm/pip_env/g1_smoke.sbatch
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC,RUN_NAME=g1_fh_diag hope_training/slurm/pip_env/train.sbatch
# warm start a new run from another run's checkpoint (TOTAL_ITERS is absolute):
sbatch --export=ALL,TASK=HOPEPingPongUnitreeG1_FH_XC_RALLY,RUN_NAME=g1_fh_rally,TOTAL_ITERS=14600,INIT_CHECKPOINT=<.../model_2600.pt> \
  hope_training/slurm/pip_env/train.sbatch
bash hope_training/slurm/pip_env/status.sh            # every run: iteration, key metrics, newest checkpoint
# TensorBoard on the laptop:
ssh -L 6006:localhost:6006 sdsc-scc '~/hope/isaac-venv/bin/tensorboard --port 6006 \
  --logdir ~/hope/HOPE/hope_training/whole_body_tracking/logs/rsl_rl/unitree_g1_hitter_pingpong'
```

Watch `strike_composite_success_exact`, `exact_strike_pos_success_10cm`, `virtual_legal_rate_forehand`,
`virtual_landing_error_m_forehand`, `virtual_contact_rate_forehand` and `Episode_Termination/base_fell_tilt`. Do not watch `success_rate`.

Node workarounds (each was a real failure; all are handled by `install_isaac.sbatch` and the
rendered `setup_train_env.local.sh`):

| Symptom | Fix |
|---|---|
| Kit `No device could be created` | `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` |
| Kit hangs after URDF import (`Neuray::LoadModule`) | `libGLU.so.1` extracted from the `mesa-libGLU` RPM onto `LD_LIBRARY_PATH` |
| `PermissionError` on `/tmp/isaaclab`, `/tmp/IsaacLab` | per-user `TMPDIR` + a one-line Isaac Lab patch |
| `isaaclab.sh` needs sudo; `flatdict` build fails | `pip install -e source/*`; prebuild `flatdict` with `setuptools<70` |
| GVHMR: `get_yaml_loader` error; no `ffmpeg` | `omegaconf==2.3.0`; imageio-ffmpeg static binary |
| GMR viewer needs a display | `gvhmr_to_g1_headless.py` |
| MuJoCo EGL picks a broken device | `hope_training/slurm/pick_egl_device.py` |

### Recording the reference video

Static camera at about 45° front-side, whole body and racket in frame, 30–60 fps, a full forehand from
the ready pose back to the ready pose, 5–10 takes. Then follow
[`slurm/video_to_g1/README.md`](../../../hope_training/slurm/video_to_g1/README.md). SMPL/SMPL-X are
licensed per person and must never be committed (they are in `.git/info/exclude` locally).

