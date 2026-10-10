# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Assemble the 110-D ``hitter_pure`` actor observation.

The layout is fixed and must byte-for-byte match the training/export contract
(``hitter_pure`` in the training-side actor observation contract registry).
There is exactly one layout; no normalization is applied (raw observation).

| slice      | term                    | dim | frame / units          |
|------------|-------------------------|-----|------------------------|
| [0:3]      | base_ang_vel            | 3   | pelvis body, rad/s     |
| [3:34]     | joint_pos               | 31  | rad, q - default_q     |
| [34:65]    | joint_vel               | 31  | rad/s                  |
| [65:96]    | actions (last_action)   | 31  | applied (prev tick)(*) |
| [96:99]    | projected_gravity       | 3   | base frame, unit       |
| [99:101]   | base_forward_xy         | 2   | world xy, unit         |
| [101:103]  | base_target_delta_xy    | 2   | world xy, m            |
| [103:106]  | racket_target_rel_base  | 3   | world, m               |
| [106:109]  | racket_target_vel_w     | 3   | world, m/s             |
| [109:110]  | time_to_strike          | 1   | s                      |

``base_target_delta_xy`` = target base position - current base position (world
xy). In this sim harness the base target is the fixed station captured at spawn
(the historical name of this term), so the delta is the in-place recentring
feedback the policy uses to hold its station; it is NOT constant zero because the
base drifts across rallies. On hardware the delta is 0 on mocap dropout.

There is NO swing-side observation slot: deploy infers forehand/backhand outside
the policy, so the side never enters the observation.

(*) the actions term is the previous tick's APPLIED action: the actor's raw output
with the passive head columns (idx 3, 4) zeroed, exactly as training zeroes them
in its applied-action feedback. Those two columns are therefore always 0 here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quaternion as quat
from .joint_order import NUM_JOINTS

CONTRACT_NAME: str = "hitter_pure"
OBS_DIM: int = 110
_ACTION_DIM: int = NUM_JOINTS


def obs_dim_for(num_joints: int) -> int:
    """``hitter_pure`` observation size for a robot with ``num_joints`` policy joints (17 + 3 N)."""
    return 17 + 3 * int(num_joints)


assert obs_dim_for(NUM_JOINTS) == OBS_DIM, "A3 hitter_pure layout must stay 110-D"


@dataclass
class RobotState:
    """Proprioceptive robot state consumed by the observation builder.

    All world poses are expressed in the same frame the planner uses for the
    racket target. On hardware this is the operator/table frame; in the reference
    MuJoCo sim it is the simulator world frame.
    """

    base_pos_w: np.ndarray      # (3,) pelvis position, world m
    base_quat_w: np.ndarray     # (4,) pelvis orientation, (w, x, y, z)
    base_ang_vel_b: np.ndarray  # (3,) pelvis gyro, body frame rad/s
    q: np.ndarray               # (31,) joint positions, joint order
    qd: np.ndarray              # (31,) joint velocities, joint order


@dataclass
class ObsTarget:
    """The strike goal fed into the observation for the current tick."""

    pos_w: np.ndarray   # (3,) racket target position, world m
    vel_w: np.ndarray   # (3,) racket target velocity, world m/s
    time_to_strike: float


def build_observation(
    state: RobotState,
    target: ObsTarget,
    last_action: np.ndarray,
    default_q: np.ndarray,
    base_target_xy: np.ndarray,
    num_joints: int = NUM_JOINTS,
) -> np.ndarray:
    """Return the ``hitter_pure`` observation (110-D for the A3) as a contiguous ``float32`` vector.

    ``num_joints`` sizes the three joint blocks (31 -> 110-D A3 default; 29 -> 104-D Unitree G1).
    """
    n = int(num_joints)
    obs_dim = obs_dim_for(n)
    q = np.asarray(state.q, dtype=np.float64)
    qd = np.asarray(state.qd, dtype=np.float64)
    default_q = np.asarray(default_q, dtype=np.float64)
    last_action = np.asarray(last_action, dtype=np.float64)
    base_pos = np.asarray(state.base_pos_w, dtype=np.float64)
    base_quat = quat.normalize(state.base_quat_w)

    if q.shape[0] != n or qd.shape[0] != n:
        raise ValueError(f"joint vectors must be length {n}")
    if last_action.shape[0] != n:
        raise ValueError(f"last_action must be length {n}")
    if default_q.shape[0] != n:
        raise ValueError(f"default_q must be length {n}")

    obs = np.empty(obs_dim, dtype=np.float64)
    o = 0
    obs[o:o + 3] = state.base_ang_vel_b;                       o += 3          # base_ang_vel
    obs[o:o + n] = q - default_q;                              o += n          # joint_pos
    obs[o:o + n] = qd;                                         o += n          # joint_vel
    obs[o:o + n] = last_action;                                o += n          # actions (applied)
    obs[o:o + 3] = quat.projected_gravity_body(base_quat);     o += 3          # projected_gravity
    obs[o:o + 2] = quat.base_forward_xy(base_quat);            o += 2          # base_forward_xy
    obs[o:o + 2] = np.asarray(base_target_xy)[:2] - base_pos[:2]; o += 2       # base_target_delta_xy
    obs[o:o + 3] = np.asarray(target.pos_w) - base_pos;        o += 3          # racket_target_rel_base
    obs[o:o + 3] = np.asarray(target.vel_w);                   o += 3          # racket_target_vel_w
    obs[o] = float(target.time_to_strike);                     o += 1          # time_to_strike
    assert o == obs_dim, o
    return obs.astype(np.float32)
