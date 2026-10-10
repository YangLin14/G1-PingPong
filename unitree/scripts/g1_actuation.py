"""Unitree G1 actuation tables shared by the host-side tooling (pure Python, no Isaac imports).

The numbers mirror ``hope_training/.../robots/g1.py`` ``G1_CYLINDER_CFG`` (BeyondMimic G1 recipe:
per-motor armature, 10 Hz natural frequency, damping ratio 2, the recipe's effort limits) and its
``init_state``. They are duplicated here so the MJCF generator, the runtime/action-adapter YAML
generator and the host tests can use them without importing Isaac Lab; the Isaac-side test
``tests/test_g1_hope_env_cfg_names.py`` pins the two copies equal.
"""

from __future__ import annotations

import math
import re

ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

#: (joint-name regex, motor, effort limit [N·m], multiplier on armature/stiffness/damping)
MOTOR_RULES: tuple[tuple[str, str, float, float], ...] = (
    (r".*_hip_pitch_joint", "7520_14", 88.0, 1.0),
    (r".*_hip_roll_joint", "7520_22", 139.0, 1.0),
    (r".*_hip_yaw_joint", "7520_14", 88.0, 1.0),
    (r".*_knee_joint", "7520_22", 139.0, 1.0),
    (r".*_ankle_pitch_joint", "5020", 50.0, 2.0),
    (r".*_ankle_roll_joint", "5020", 50.0, 2.0),
    (r"waist_roll_joint", "5020", 50.0, 2.0),
    (r"waist_pitch_joint", "5020", 50.0, 2.0),
    (r"waist_yaw_joint", "7520_14", 88.0, 1.0),
    (r".*_shoulder_pitch_joint", "5020", 25.0, 1.0),
    (r".*_shoulder_roll_joint", "5020", 25.0, 1.0),
    (r".*_shoulder_yaw_joint", "5020", 25.0, 1.0),
    (r".*_elbow_joint", "5020", 25.0, 1.0),
    (r".*_wrist_roll_joint", "5020", 25.0, 1.0),
    (r".*_wrist_pitch_joint", "4010", 5.0, 1.0),
    (r".*_wrist_yaw_joint", "4010", 5.0, 1.0),
)

#: ``G1_CYLINDER_CFG.init_state.joint_pos`` (unlisted joints are 0) and pelvis height.
INIT_JOINT_POS: tuple[tuple[str, float], ...] = (
    (r".*_hip_pitch_joint", -0.312),
    (r".*_knee_joint", 0.669),
    (r".*_ankle_pitch_joint", -0.363),
    (r".*_elbow_joint", 0.6),
    (r"left_shoulder_roll_joint", 0.2),
    (r"left_shoulder_pitch_joint", 0.2),
    (r"right_shoulder_roll_joint", -0.2),
    (r"right_shoulder_pitch_joint", 0.2),
)
INIT_PELVIS_Z = 0.76


def _rule(joint: str) -> tuple[str, str, float, float]:
    for rule in MOTOR_RULES:
        if re.fullmatch(rule[0], joint):
            return rule
    raise KeyError(f"no motor rule for joint {joint!r}")


def armature(joint: str) -> float:
    _, motor, _, mult = _rule(joint)
    return ARMATURE[motor] * mult


def stiffness(joint: str) -> float:
    _, motor, _, mult = _rule(joint)
    return ARMATURE[motor] * NATURAL_FREQ**2 * mult


def damping(joint: str) -> float:
    _, motor, _, mult = _rule(joint)
    return 2.0 * DAMPING_RATIO * ARMATURE[motor] * NATURAL_FREQ * mult


def effort_limit(joint: str) -> float:
    return _rule(joint)[2]


def action_scale(joint: str) -> float:
    """``0.25 * effort / stiffness`` as in ``G1_ACTION_SCALE``."""
    return 0.25 * effort_limit(joint) / stiffness(joint)


def init_joint_pos(joint: str) -> float:
    for pattern, value in INIT_JOINT_POS:
        if re.fullmatch(pattern, joint):
            return value
    return 0.0


def table(joint_names) -> dict[str, dict[str, float]]:
    return {
        n: {
            "armature": armature(n), "stiffness": stiffness(n), "damping": damping(n),
            "effort": effort_limit(n), "action_scale": action_scale(n), "init": init_joint_pos(n),
        }
        for n in joint_names
    }


__all__ = [
    "ARMATURE", "NATURAL_FREQ", "DAMPING_RATIO", "MOTOR_RULES", "INIT_JOINT_POS", "INIT_PELVIS_Z",
    "armature", "stiffness", "damping", "effort_limit", "action_scale", "init_joint_pos", "table",
]
if not math.isclose(stiffness("left_knee_joint"), 0.025101925 * NATURAL_FREQ**2):
    raise AssertionError("g1_actuation self-check failed")
