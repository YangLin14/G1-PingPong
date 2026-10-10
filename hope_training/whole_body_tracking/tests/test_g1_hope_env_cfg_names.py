"""Unitree G1 HOPE env configs reference only G1 names (configclass instantiation, no simulation).

Needs Isaac Lab importable (the configs import ``isaaclab.utils.configclass`` and the MDP
modules); skipped on a plain host. Run inside the Isaac environment:

    hope_isaac_py -m pytest -q tests/test_g1_hope_env_cfg_names.py

Importing ``whole_body_tracking`` pulls in ``isaaclab.envs``/``isaaclab.managers``, which import
``omni.timeline`` and friends: those modules only exist once the Kit app is up. So, as in Isaac
Lab's own test suite, a headless app is launched at module import time (no stage, no sim).
"""

from __future__ import annotations

import atexit
import sys

import pytest

isaaclab = pytest.importorskip("isaaclab")  # noqa: F841

from isaaclab.app import AppLauncher  # noqa: E402

_saved_argv = sys.argv
sys.argv = sys.argv[:1]  # keep pytest's own CLI flags away from Kit's argument parser
try:
    _simulation_app = AppLauncher(headless=True).app
finally:
    sys.argv = _saved_argv
atexit.register(_simulation_app.close)

from whole_body_tracking.robots.g1 import (  # noqa: E402
    G1_ANCHOR_BODY,
    G1_MOUNT_OFFSET,
    G1_TRACKED_BODIES,
    G1_WRIST_BODY,
)
from whole_body_tracking.tasks.tracking.actor_observation_contract import HITTER_PURE_UNITREE_G1  # noqa: E402
from whole_body_tracking.tasks.tracking.config.g1 import hope_env_cfg  # noqa: E402
from whole_body_tracking.utils.g1_retarget import find_residual_a3_names  # noqa: E402


@pytest.fixture(params=[hope_env_cfg.HOPEPingPongUnitreeG1EnvCfg, hope_env_cfg.HitterPingPongUnitreeG1EnvCfg])
def cfg(request):
    return request.param()


def test_no_a3_names_anywhere_in_the_tree(cfg):
    assert find_residual_a3_names(cfg, skip_attrs=("scene",)) == []


def test_robot_and_motion_wiring(cfg):
    assert "unitree_g1" in str(cfg.scene.robot.spawn.asset_path)
    assert cfg.commands.motion.anchor_body_name == G1_ANCHOR_BODY
    assert list(cfg.commands.motion.body_names) == list(G1_TRACKED_BODIES)
    assert cfg.commands.motion.wrap_teleport is False


def test_racket_mount_wiring(cfg):
    racket = cfg.commands.racket_target
    assert racket.wrist_body_name == G1_WRIST_BODY == "right_wrist_yaw_link"
    assert racket.racket_body_name == "pingpang_red_Link"
    assert tuple(racket.mount_offset) == tuple(G1_MOUNT_OFFSET)
    assert tuple(racket.mount_quat) == (1.0, 0.0, 0.0, 0.0)
    assert racket.mount_normal_axis == 1
    assert racket.torso_body_name == "torso_link"
    assert racket.left_hand_body_name == "left_wrist_yaw_link"


def test_no_passive_joints_and_no_head_terms(cfg):
    assert tuple(cfg.actions.joint_pos.passive_joint_names) == ()
    assert getattr(cfg.rewards, "passive_head_raw_action", None) is None


def test_pd_randomization_is_the_generic_event(cfg):
    import whole_body_tracking.tasks.tracking.mdp as mdp

    term = cfg.events.randomize_pd_gains
    assert term is not None and term.func is mdp.randomize_actuator_gains
    assert "stiffness_distribution_params" in term.params


def test_hitter_contract_is_registered_for_29_joints():
    assert HITTER_PURE_UNITREE_G1.total_dim == 104
    assert dict(HITTER_PURE_UNITREE_G1.layout)["joint_pos"] == 29
