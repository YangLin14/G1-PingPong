"""Offscreen mp4 capture for the MuJoCo ping-pong harness (``mujoco_eval_onnx.py --video-out``).

Renders the live ``MjData`` every ``every``-th control tick with ``mujoco.Renderer`` (needs an
offscreen GL backend: ``MUJOCO_GL=egl`` on a machine with a GPU, ``osmesa`` otherwise) and writes
the frames with ``imageio`` + ``imageio-ffmpeg`` (both optional dependencies: only imported here).
"""

from __future__ import annotations

import os

import numpy as np


class RolloutVideo:
    """Accumulate rendered frames of a MuJoCo rollout and write one H.264 mp4."""

    def __init__(
        self,
        model,
        data,
        out_path: str,
        *,
        control_dt: float,
        every: int = 2,
        width: int = 640,
        height: int = 480,
        lookat=None,
        distance: float = 3.5,
        azimuth: float = 140.0,
        elevation: float = -18.0,
        max_frames: int = 6000,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = data
        self.out_path = str(out_path)
        self.every = max(1, int(every))
        self.fps = 1.0 / (float(control_dt) * self.every)
        self.max_frames = int(max_frames)
        self._tick = 0
        self._frames: list[np.ndarray] = []
        self._renderer = mujoco.Renderer(model, height=int(height), width=int(width))
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = float(distance)
        self._camera.azimuth = float(azimuth)
        self._camera.elevation = float(elevation)
        self._camera.lookat[:] = np.asarray(
            lookat if lookat is not None else [0.0, 0.0, 0.9], dtype=np.float64
        )
        self._option = mujoco.MjvOption()

    def tick(self) -> None:
        """Call once per control step (after physics stepped); renders on every ``every``-th call."""
        self._tick += 1
        if self._tick % self.every != 0 or len(self._frames) >= self.max_frames:
            return
        self._renderer.update_scene(self.data, camera=self._camera, scene_option=self._option)
        self._frames.append(self._renderer.render().copy())

    def mark(self) -> None:
        """Drop a short visual separator (black frames) between serves."""
        if self._frames:
            blank = np.zeros_like(self._frames[-1])
            self._frames.extend([blank] * max(1, int(round(self.fps * 0.2))))

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def close(self) -> str | None:
        """Write the mp4 (if any frames) and release the renderer. Returns the path written."""
        path = None
        try:
            if self._frames:
                import imageio

                os.makedirs(os.path.dirname(os.path.abspath(self.out_path)) or ".", exist_ok=True)
                with imageio.get_writer(
                    self.out_path,
                    fps=self.fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=16,
                ) as writer:
                    for frame in self._frames:
                        writer.append_data(frame)
                path = self.out_path
        finally:
            self._frames = []
            try:
                self._renderer.close()
            except Exception:
                pass
        return path
