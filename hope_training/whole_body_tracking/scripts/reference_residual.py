"""Deploy side of the D26 reference-residual action (training side: ``mdp/hope_actions.py``).

For the joints in ``actions.joint_pos.reference_residual_bounds`` the policy output is a bounded correction
around the motion clip, not an offset from the default pose:

    q_des[j] = q_clip(frame)[j] + bound[j] * tanh(scale[j] * a[j] / bound[j])

Training reads ``q_clip`` at the motion command's time step, which is locked to the observed time to
strike: ``tts = (strike_frame - frame) * dt``, and a hold waits on the swing's first frame. The runner
has only ``tts``, so it inverts that: ``frame = strike_frame - tts / dt``, clamped to the clip and
interpolated between frames. A ready / hold tts (>= the clip's strike time) gives frame 0.
"""

from __future__ import annotations

import pathlib

import numpy as np
import yaml

WBT_DIR = pathlib.Path(__file__).resolve().parents[1]
BOUNDS_KEY = "actions.joint_pos.reference_residual_bounds"


class ReferenceResidual:
    def __init__(self, clip_npz: str | pathlib.Path, bounds: dict[str, float], joint_order: list[str],
                 action_scale: np.ndarray, clamp_lower: np.ndarray, clamp_upper: np.ndarray):
        clip_npz = pathlib.Path(clip_npz)
        with open(clip_npz.with_suffix(".yaml"), "r", encoding="utf-8") as fh:
            side = yaml.safe_load(fh)
        if [str(n) for n in side["joint_order"]] != list(joint_order):
            raise ValueError(f"{clip_npz.name}: joint order differs from the runtime order")
        self.q_clip = np.asarray(np.load(clip_npz)["joint_pos"], dtype=np.float64)
        self.dt = float(side["frame_time_s"])
        self.strike_frame = int(side["strike_frame"])
        self.cols = np.array([list(joint_order).index(n) for n in bounds], dtype=np.int64)
        self.bounds = np.array([float(bounds[n]) for n in bounds], dtype=np.float64)
        if (self.bounds <= 0).any():
            raise ValueError(f"reference residual bounds must be positive, got {bounds}")
        self.scale = np.asarray(action_scale, dtype=np.float64)[self.cols]
        self.lower = np.asarray(clamp_lower, dtype=np.float64)[self.cols]
        self.upper = np.asarray(clamp_upper, dtype=np.float64)[self.cols]
        self.clip_name = clip_npz.name

    @classmethod
    def from_task(cls, task_yaml: str | pathlib.Path, joint_order: list[str], action_scale, clamp_lower,
                  clamp_upper) -> "ReferenceResidual":
        """Bounds from the task's overrides; clip = the task's only evaluated clip (eval_clip_sequence)."""
        with open(task_yaml, "r", encoding="utf-8") as fh:
            task = yaml.safe_load(fh)
        over = task.get("overrides", {}) or {}
        bounds = over.get(BOUNDS_KEY)
        if not bounds:
            raise ValueError(f"{task_yaml}: no {BOUNDS_KEY} override")
        clips = over.get("commands.motion.eval_clip_sequence")
        if not clips or len(set(clips)) != 1:
            raise ValueError(f"{task_yaml}: needs a single commands.motion.eval_clip_sequence clip")
        key = "motion_file" if int(clips[0]) == 0 else f"motion_file_{int(clips[0]) + 1}"
        clip = (WBT_DIR / str(task[key])).resolve()
        return cls(clip, dict(bounds), joint_order, action_scale, clamp_lower, clamp_upper)

    def reference(self, time_to_strike: float) -> np.ndarray:
        last = self.q_clip.shape[0] - 1
        f = float(np.clip(self.strike_frame - float(time_to_strike) / self.dt, 0.0, last))
        lo = int(np.floor(f))
        hi = min(lo + 1, last)
        w = f - lo
        return (1.0 - w) * self.q_clip[lo, self.cols] + w * self.q_clip[hi, self.cols]

    def apply(self, q_des: np.ndarray, action: np.ndarray, time_to_strike: float) -> np.ndarray:
        """Overwrite the reference-residual joints of ``q_des`` (decoded as default + scale * a)."""
        q = np.asarray(q_des, dtype=np.float64).copy()
        a = np.asarray(action, dtype=np.float64)[self.cols]
        residual = self.bounds * np.tanh(self.scale * a / self.bounds)
        q[self.cols] = np.clip(self.reference(time_to_strike) + residual, self.lower, self.upper)
        return q
