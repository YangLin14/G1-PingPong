"""``RecordVideo`` that also pushes each finished rollout mp4 to the active Weights & Biases run.

The local mp4 under ``<log_dir>/videos/train`` is always written first (gymnasium + moviepy); the
upload is best-effort and silently skipped when no W&B run is active, so ``video=true`` keeps
working with the plain tensorboard logger.
"""

from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym


class WandbRecordVideo(gym.wrappers.RecordVideo):
    """``gym.wrappers.RecordVideo`` plus a W&B upload of every saved clip.

    Args:
        step_provider: returns the x-axis value to log the clip at (the PPO iteration, so the clip
            lines up with the scalars rsl_rl logs with ``step=it``). ``None`` lets W&B pick.
        log_key: W&B key for the clips.
    """

    def __init__(
        self,
        env,
        *,
        step_provider: Callable[[], int | None] | None = None,
        log_key: str = "Video/train",
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.step_provider = step_provider
        self.log_key = log_key

    def stop_recording(self) -> None:
        name = self._video_name
        super().stop_recording()
        if name is None:
            return
        path = os.path.join(self.video_folder, f"{name}.mp4")
        if not os.path.isfile(path):
            return
        try:
            import wandb

            if wandb.run is None:
                return
            step = self.step_provider() if self.step_provider is not None else None
            wandb.log({self.log_key: wandb.Video(path, format="mp4")}, step=step)
            print(f"[wandb_video] uploaded {os.path.basename(path)} at step={step}", flush=True)
        except Exception as exc:  # never let logging take the training run down
            print(f"[wandb_video] WARNING: could not upload {path}: {exc!r}", flush=True)
