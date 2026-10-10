# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Robot profiles: everything in the reference runner that depends on WHICH robot is driven.

The runner was written for the 31-joint Agibot A3 and that remains the default everywhere
(``AGIBOT_A3``), so default behaviour is unchanged. A profile bundles the canonical joint order,
the passive (held-at-default) action columns, the actor observation contract name and dims, the
PD-gain group index sets, and the MJCF element names the MuJoCo bridges resolve. The robot is
NEVER inferred from a joint count: it is selected explicitly (``RuntimeConfig.load(...,
profile=...)``, ``--robot``, or ``policy.joint_order`` in the runtime YAML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from . import joint_order as _a3
from . import joint_order_unitree_g1 as _g1
from .observation import CONTRACT_NAME as _A3_CONTRACT_NAME
from .observation import obs_dim_for


@dataclass(frozen=True)
class RobotProfile:
    name: str
    joint_names: tuple[str, ...]
    #: Action columns held at their default angle and zeroed in the applied-action feedback.
    passive_indices: tuple[int, ...] = ()
    contract_name: str = _A3_CONTRACT_NAME
    #: Joint-group name -> joint indices (for the example group PD gains in the runtime YAML).
    group_ranges: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    # MJCF names resolved by the MuJoCo bridges / the ping-pong evaluation scene.
    base_free_joint: str = "pelvis_free_joint"
    gyro_sensor: str = "pelvis_imu_gyro"
    racket_site: str = "right_racket"
    racket_collision_geom: str = "right_racket_collision"
    floor_geom: str = "floor"

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError(f"profile {self.name!r}: joint names must be unique")
        bad = [i for i in self.passive_indices if not 0 <= i < len(self.joint_names)]
        if bad:
            raise ValueError(f"profile {self.name!r}: passive indices out of range: {bad}")

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def obs_dim(self) -> int:
        return obs_dim_for(self.num_joints)

    # hitter_pure layout: [0:3] base_ang_vel, then joint_pos / joint_vel / actions blocks.
    @property
    def joint_pos_slice(self) -> slice:
        return slice(3, 3 + self.num_joints)

    @property
    def joint_vel_slice(self) -> slice:
        return slice(3 + self.num_joints, 3 + 2 * self.num_joints)

    @property
    def actions_slice(self) -> slice:
        return slice(3 + 2 * self.num_joints, 3 + 3 * self.num_joints)

    def index_of(self, joint_name: str) -> int:
        return self.joint_names.index(joint_name)


def _indices_matching(joint_names: tuple[str, ...], *fragments: str) -> tuple[int, ...]:
    return tuple(i for i, name in enumerate(joint_names) if any(f in name for f in fragments))


# Index ranges of the four A3 joint groups (waist / neck / arm / leg) in the canonical A3 order.
A3_GROUP_RANGES: dict[str, tuple[int, ...]] = {
    "waist": tuple(range(0, 3)),
    "neck": tuple(range(3, 5)),
    "arm": tuple(range(5, 19)),
    "leg": tuple(range(19, 31)),
}

AGIBOT_A3 = RobotProfile(
    name="agibot_a3",
    joint_names=tuple(_a3.JOINT_NAMES),
    passive_indices=tuple(_a3.HEAD_INDICES),
    contract_name=_A3_CONTRACT_NAME,
    group_ranges=A3_GROUP_RANGES,
)

UNITREE_G1 = RobotProfile(
    name="unitree_g1",
    joint_names=tuple(_g1.JOINT_NAMES),
    passive_indices=tuple(_g1.HEAD_INDICES),  # () — the G1 has no neck
    contract_name="hitter_pure_unitree_g1",
    group_ranges={
        "waist": _indices_matching(tuple(_g1.JOINT_NAMES), "waist_"),
        "arm": _indices_matching(tuple(_g1.JOINT_NAMES), "shoulder_", "elbow_", "wrist_"),
        "leg": _indices_matching(tuple(_g1.JOINT_NAMES), "hip_", "knee_", "ankle_"),
    },
)

PROFILES: dict[str, RobotProfile] = {AGIBOT_A3.name: AGIBOT_A3, UNITREE_G1.name: UNITREE_G1}

assert AGIBOT_A3.num_joints == 31 and AGIBOT_A3.obs_dim == 110
assert UNITREE_G1.num_joints == 29 and UNITREE_G1.obs_dim == 104
assert sum(len(v) for v in UNITREE_G1.group_ranges.values()) == 29


def resolve_profile(profile: str | RobotProfile | None) -> RobotProfile:
    """``None`` -> the A3 (historical default); a name -> the registered profile; unknown -> error."""
    if profile is None:
        return AGIBOT_A3
    if isinstance(profile, RobotProfile):
        return profile
    key = str(profile).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown robot profile {profile!r}; known: {sorted(PROFILES)}")
    return PROFILES[key]
