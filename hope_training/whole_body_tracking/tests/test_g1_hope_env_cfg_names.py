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
from whole_body_tracking.tasks.tracking.config.g1 import g1_recipe, hope_env_cfg  # noqa: E402
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


def test_hitter_cfg_carries_the_g1_recipe_v2():
    cfg = hope_env_cfg.HitterPingPongUnitreeG1EnvCfg()
    racket = cfg.commands.racket_target
    assert tuple(tuple(tuple(r) for r in clip) for clip in racket.racket_pos_range_per_clip) == g1_recipe.POS_RANGE_PER_CLIP
    assert tuple(racket.venue_tuple_contact_z_range) == g1_recipe.VENUE_CONTACT_Z_RANGE
    assert cfg.rewards.rally_ready_root_height_debt.params["min_height"] == g1_recipe.ROOT_HEIGHT_MIN
    assert cfg.rewards.hold_ready.params["reach"] == g1_recipe.HOLD_READY_REACH
    assert (cfg.rewards.ready_stance_width.params["lo"], cfg.rewards.ready_stance_width.params["hi"]) == g1_recipe.STANCE_WIDTH
    assert cfg.terminations.base_too_low.params["minimum_height"] == g1_recipe.BASE_TOO_LOW
    # the research env keeps the A3 numbers (only the deploy-grade hitter is retuned)
    full = hope_env_cfg.HOPEPingPongUnitreeG1EnvCfg()
    assert tuple(full.commands.racket_target.venue_tuple_contact_z_range) == g1_recipe.A3_VENUE_CONTACT_Z_RANGE


def test_host_actuation_table_matches_the_isaac_articulation():
    """unitree/scripts/g1_actuation.py (used by the MJCF/YAML generators) == robots/g1.py G1_CYLINDER_CFG."""
    import importlib.util
    import pathlib

    from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG, UNITREE_G1_JOINT_NAMES

    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("g1_actuation", root / "unitree" / "scripts" / "g1_actuation.py")
    act = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(act)
    import re

    def resolve(mapping, joint):
        if not isinstance(mapping, dict):
            return float(mapping)
        for pattern, value in mapping.items():
            if re.fullmatch(pattern, joint):
                return float(value)
        raise KeyError(joint)

    for joint in UNITREE_G1_JOINT_NAMES:
        actuator = next(a for a in G1_CYLINDER_CFG.actuators.values() if any(re.fullmatch(e, joint) for e in a.joint_names_expr))
        assert resolve(actuator.stiffness, joint) == pytest.approx(act.stiffness(joint), rel=1e-6), joint
        assert resolve(actuator.damping, joint) == pytest.approx(act.damping(joint), rel=1e-6), joint
        assert resolve(actuator.armature, joint) == pytest.approx(act.armature(joint), rel=1e-6), joint
        assert resolve(actuator.effort_limit_sim, joint) == pytest.approx(act.effort_limit(joint)), joint
        scale = next(v for k, v in G1_ACTION_SCALE.items() if re.fullmatch(k, joint))
        assert scale == pytest.approx(act.action_scale(joint), rel=1e-6), joint
    init = G1_CYLINDER_CFG.init_state
    assert init.pos[2] == pytest.approx(act.INIT_PELVIS_Z)
    for joint in UNITREE_G1_JOINT_NAMES:
        expected = next((v for k, v in init.joint_pos.items() if re.fullmatch(k, joint)), 0.0)
        assert expected == pytest.approx(act.init_joint_pos(joint)), joint
