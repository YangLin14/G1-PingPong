# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Canonical Unitree G1 joint order (29 controllable DOF, ``g1_29dof_rev_1_0``).

Shared by the G1 training path (``hope_training/config/joint_order_unitree_g1.yaml``, pinned
equal by a test), the exported G1 ONNX policy and the MuJoCo sim-to-sim evaluation through
:data:`robot_profile.UNITREE_G1`. The racket is mounted on the right wrist. The G1 has no neck,
so there are NO passive columns: every action column is actuated.

This is the Isaac Lab / PhysX breadth-first articulation enumeration of the racket-equipped G1
URDF, not the Unitree SDK / MJCF actuator order; MuJoCo resolves joints by name, so the order only
fixes the policy's observation/action columns there.
"""

from __future__ import annotations

JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",        # 0
    "right_hip_pitch_joint",       # 1
    "waist_yaw_joint",             # 2
    "left_hip_roll_joint",         # 3
    "right_hip_roll_joint",        # 4
    "waist_roll_joint",            # 5
    "left_hip_yaw_joint",          # 6
    "right_hip_yaw_joint",         # 7
    "waist_pitch_joint",           # 8
    "left_knee_joint",             # 9
    "right_knee_joint",            # 10
    "left_shoulder_pitch_joint",   # 11
    "right_shoulder_pitch_joint",  # 12
    "left_ankle_pitch_joint",      # 13
    "right_ankle_pitch_joint",     # 14
    "left_shoulder_roll_joint",    # 15
    "right_shoulder_roll_joint",   # 16
    "left_ankle_roll_joint",       # 17
    "right_ankle_roll_joint",      # 18
    "left_shoulder_yaw_joint",     # 19
    "right_shoulder_yaw_joint",    # 20
    "left_elbow_joint",            # 21
    "right_elbow_joint",           # 22
    "left_wrist_roll_joint",       # 23
    "right_wrist_roll_joint",      # 24
    "left_wrist_pitch_joint",      # 25
    "right_wrist_pitch_joint",     # 26
    "left_wrist_yaw_joint",        # 27
    "right_wrist_yaw_joint",       # 28
)

NUM_JOINTS: int = 29

# No passive-at-deploy columns on the G1.
HEAD_INDICES: tuple[int, ...] = ()

NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

assert len(JOINT_NAMES) == NUM_JOINTS, "joint order must contain exactly 29 names"
