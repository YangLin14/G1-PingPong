# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Shared ActionAdapter: raw policy output -> joint-position targets.

The adapter is a pure, deterministic numeric transform (a joint-position residual
plus a clamp). It is NOT a rejection filter and emits no failure status:

    q_des = default_q + raw_action * action_scale
    q_des = clip(q_des, clamp_lower, clamp_upper)

The runtime may load either the neutral public example
(``config/action_adapter.yaml``) or the exact Unitree-style ``deploy.yaml`` shipped
with a published model bundle.  In both cases the transform is resolved into the
same SDK/MuJoCo joint order.

Vendor hard limits, motor protection, and e-stop remain the responsibility of the
robot backend; this transform neither probes nor bypasses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .joint_order import JOINT_NAMES, NUM_JOINTS


@dataclass
class ActionAdapter:
    default_q: np.ndarray     # (N,) neutral/upright joint positions, rad
    action_scale: np.ndarray  # (N,) per-column residual scale (uniform in the A3 example)
    clamp_lower: np.ndarray   # (N,) lower joint-position clamp, rad
    clamp_upper: np.ndarray   # (N,) upper joint-position clamp, rad
    # Canonical joint order the arrays follow (N = 31 for the A3 default, 29 for the Unitree G1).
    joint_names: tuple[str, ...] = JOINT_NAMES

    def __post_init__(self) -> None:
        self.joint_names = tuple(self.joint_names)
        n = len(self.joint_names)
        for field in ("default_q", "action_scale", "clamp_lower", "clamp_upper"):
            v = np.asarray(getattr(self, field), dtype=np.float64).reshape(-1)
            if v.shape[0] != n:
                raise ValueError(f"{field} must be length {n}, got {v.shape[0]}")
            setattr(self, field, v)
        if np.any(self.clamp_lower > self.clamp_upper):
            raise ValueError("action_adapter clamp_lower must be <= clamp_upper for every joint")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def decode(self, raw_action: np.ndarray) -> np.ndarray:
        """Map ``raw_action[N]`` to N clamped joint-position targets."""
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        n = self.num_joints
        if raw.shape[0] != n:
            raise ValueError(f"raw_action must be length {n}, got {raw.shape[0]}")
        q_des = self.default_q + raw * self.action_scale
        return np.clip(q_des, self.clamp_lower, self.clamp_upper)

    @classmethod
    def from_yaml(cls, path: str | Path, joint_names: tuple[str, ...] = JOINT_NAMES) -> "ActionAdapter":
        """Load the adapter for the robot whose canonical order is ``joint_names`` (default: A3)."""
        joint_names = tuple(joint_names)
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        # A published model bundle carries the authoritative Unitree-style
        # deploy.yaml used by the native runner. Reuse its resolved action arrays
        # directly so the lightweight MuJoCo runner does not need a second copy of
        # model-specific defaults, scales, or clamps.
        if doc.get("format") == "unitree_rl_lab.deploy":
            return cls._from_deploy_yaml(doc, joint_names)

        default_q = _resolve_per_joint(doc["default_q"], "default_q", joint_names)

        scale_spec = doc["action_scale"]
        if isinstance(scale_spec, (int, float)):
            action_scale = np.full(len(joint_names), float(scale_spec), dtype=np.float64)
        else:
            action_scale = _resolve_per_joint(scale_spec, "action_scale", joint_names)

        clamp = doc["joint_position_clamp"]
        clamp_lower = _resolve_per_joint(clamp["lower"], "joint_position_clamp.lower", joint_names)
        clamp_upper = _resolve_per_joint(clamp["upper"], "joint_position_clamp.upper", joint_names)
        return cls(default_q, action_scale, clamp_lower, clamp_upper, joint_names)

    @classmethod
    def _from_deploy_yaml(cls, doc: dict, joint_names: tuple[str, ...] = JOINT_NAMES) -> "ActionAdapter":
        joint_names = tuple(joint_names)
        n = len(joint_names)
        action = doc["actions"]["JointPositionAction"]
        names = tuple(str(name) for name in action["joint_names"])
        if len(names) != n or set(names) != set(joint_names):
            raise ValueError(f"deploy.yaml JointPositionAction must name all {n} joints once")

        def reorder(values, field_name: str) -> np.ndarray:
            if len(values) != n:
                raise ValueError(f"deploy.yaml {field_name} must be length {n}")
            by_name = {name: value for name, value in zip(names, values)}
            return np.asarray([by_name[name] for name in joint_names], dtype=np.float64)

        clips = action["clip"]
        if len(clips) != n or any(len(pair) != 2 for pair in clips):
            raise ValueError(f"deploy.yaml JointPositionAction.clip must contain {n} [lo, hi] pairs")
        lower = reorder([pair[0] for pair in clips], "JointPositionAction.clip.lower")
        upper = reorder([pair[1] for pair in clips], "JointPositionAction.clip.upper")
        return cls(
            default_q=reorder(action["offset"], "JointPositionAction.offset"),
            action_scale=reorder(action["scale"], "JointPositionAction.scale"),
            clamp_lower=lower,
            clamp_upper=upper,
            joint_names=joint_names,
        )


def _resolve_per_joint(spec, field_name: str, joint_names: tuple[str, ...] = JOINT_NAMES) -> np.ndarray:
    """Accept either an ordered length-N list or a ``{joint_name: value}`` map."""
    n = len(joint_names)
    if isinstance(spec, dict):
        missing = [name for name in joint_names if name not in spec]
        if missing:
            raise ValueError(f"{field_name} is missing joints: {missing[:3]}...")
        return np.array([float(spec[name]) for name in joint_names], dtype=np.float64)
    arr = np.asarray(spec, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{field_name} must be length {n}, got {arr.shape[0]}")
    return arr
