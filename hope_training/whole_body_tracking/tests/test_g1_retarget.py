"""Host-only tests for the A3 -> Unitree G1 config-name retarget (utils/g1_retarget.py).

Pure Python: the module is loaded by file path so the Isaac Lab package ``__init__`` is never
imported. The Isaac-side consumer (``config/g1/hope_env_cfg.py``) is covered by
``test_g1_hope_env_cfg_names.py`` inside the Isaac environment.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking")
_RETARGET_PY = os.path.join(_PKG, "utils", "g1_retarget.py")
_A3_FLAT_PY = os.path.join(_PKG, "tasks", "tracking", "config", "agibot_a3", "flat_env_cfg.py")
_G1_ROBOT_PY = os.path.join(_PKG, "robots", "g1.py")
_A3_ROBOT_PY = os.path.join(_PKG, "robots", "agibot_a3.py")


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


retarget = _load_by_path("hope_g1_retarget", _RETARGET_PY)


def _literal(path: str, name: str):
    """Read a module-level ``NAME = <literal>`` from a file that imports isaaclab (ast only)."""
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def test_plain_names():
    assert retarget.retarget_name("torso_Link") == "torso_link"
    assert retarget.retarget_name("pelvis_link") == "pelvis"
    assert retarget.retarget_name("left_wrist_yaw_Link") == "left_wrist_yaw_link"
    assert retarget.retarget_name("imu_in_pelvis_Link") == "imu_in_pelvis"
    assert retarget.retarget_name("left_hand_Link") == "left_rubber_hand"
    # Racket links keep the A3 spelling on the G1 asset.
    for name in ("right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link"):
        assert retarget.retarget_name(name) == name
    # Unrelated strings and joint names pass through untouched.
    assert retarget.retarget_name("racket_target") == "racket_target"
    assert retarget.retarget_name("right_elbow_joint") == "right_elbow_joint"


def test_regex_strings_are_retargeted_in_place():
    regex = _literal(_A3_FLAT_PY, "A3_CONTACT_EXCLUDE_REGEX")
    g1_regex = _literal(_G1_ROBOT_PY, "G1_CONTACT_EXCLUDE_REGEX")
    mapped = retarget.retarget_name(regex)
    assert "_Link$)" not in mapped.replace("pingpang_Link$)", "").replace("red_Link$)", "").replace(
        "black_Link$)", ""
    ).replace("ball_Link$)", "")
    # The hand-written G1 regex equals the A3 regex retargeted, minus the (absent) calibration ball.
    assert mapped.replace("(?!pingbang_ball_Link$)", "") == g1_regex


def test_a3_tracked_bodies_map_onto_the_g1_lists():
    a3_tracked = _literal(_A3_ROBOT_PY, "A3_TRACKED_BODIES")
    g1_tracked = _literal(_G1_ROBOT_PY, "G1_TRACKED_BODIES")
    assert [retarget.retarget_name(n) for n in a3_tracked] == g1_tracked
    assert [retarget.retarget_name(n) for n in _literal(_A3_ROBOT_PY, "A3_FEET_BODIES")] == _literal(
        _G1_ROBOT_PY, "G1_FEET_BODIES"
    )
    assert [retarget.retarget_name(n) for n in _literal(_A3_ROBOT_PY, "A3_UPPER_TRACKED")] == _literal(
        _G1_ROBOT_PY, "G1_UPPER_TRACKED"
    )
    assert retarget.retarget_name(_literal(_A3_ROBOT_PY, "A3_WRIST_BODY")) == _literal(_G1_ROBOT_PY, "G1_WRIST_BODY")
    assert retarget.retarget_name(_literal(_A3_ROBOT_PY, "A3_ROOT_BODY")) == _literal(_G1_ROBOT_PY, "G1_ROOT_BODY")


def test_unmappable_a3_names_raise():
    for text in ("head_yaw_Link", "torso_shell_Link", "head_yaw_joint", "x head_pitch_joint y"):
        with pytest.raises(retarget.UnmappableA3NameError):
            retarget.retarget_name(text)


def _fake_reward(env, command_name: str, std: float = 0.3, torso_body_name: str = "torso_Link"):
    return None


class _FakeClassTerm:
    def __init__(self, cfg, env):
        pass

    def __call__(self, env, left_body_name: str = "left_wrist_yaw_Link", margin: float = 0.1):
        return None


@dataclasses.dataclass
class _SceneEntity:
    name: str
    body_names: object = None
    joint_names: object = None


@dataclasses.dataclass
class _Term:
    func: object
    params: dict
    weight: float = 1.0


@dataclasses.dataclass
class _Rewards:
    front: _Term
    clearance: _Term
    contacts: _Term
    passive: object = None


@dataclasses.dataclass
class _Racket:
    wrist_body_name: str = "right_wrist_yaw_Link"
    racket_body_name: str = "pingpang_red_Link"
    torso_body_name: str = "torso_Link"
    mount_offset: tuple = (0.1, 0.2, 0.3)


@dataclasses.dataclass
class _Commands:
    racket_target: _Racket
    body_names: list


@dataclasses.dataclass
class _Env:
    scene: _SceneEntity
    rewards: _Rewards
    commands: _Commands
    obs_mode: str = "hitter_pure"


def _build_tree() -> _Env:
    return _Env(
        scene=_SceneEntity("robot", body_names="torso_Link"),  # skipped (asset cfg)
        rewards=_Rewards(
            front=_Term(_fake_reward, {"command_name": "racket_target", "std": 0.2}),
            clearance=_Term(_FakeClassTerm, {"margin": 0.15}),
            contacts=_Term(
                _fake_reward,
                {"sensor_cfg": _SceneEntity("contact_forces", body_names=[r"^(?!left_ankle_roll_Link$).+$"])},
            ),
        ),
        commands=_Commands(_Racket(), ["pelvis_link", "torso_Link", "right_wrist_yaw_Link"]),
    )


def test_tree_walk_rewrites_strings_lists_tuples_and_injects_defaults():
    tree = _build_tree()
    log = retarget.retarget_cfg_tree(tree)
    assert log  # something changed
    # Strings in lists / dataclass fields / regexes.
    assert tree.commands.body_names == ["pelvis", "torso_link", "right_wrist_yaw_link"]
    assert tree.commands.racket_target.wrist_body_name == "right_wrist_yaw_link"
    assert tree.commands.racket_target.racket_body_name == "pingpang_red_Link"
    assert tree.commands.racket_target.torso_body_name == "torso_link"
    assert tree.commands.racket_target.mount_offset == (0.1, 0.2, 0.3)
    assert tree.rewards.contacts.params["sensor_cfg"].body_names == [r"^(?!left_ankle_roll_link$).+$"]
    # A3 default arguments become explicit, retargeted params (function and class terms).
    assert tree.rewards.front.params["torso_body_name"] == "torso_link"
    assert tree.rewards.front.params["std"] == 0.2  # existing params untouched
    assert tree.rewards.clearance.params["left_body_name"] == "left_wrist_yaw_link"
    # Callables and the scene are never touched.
    assert tree.rewards.front.func is _fake_reward
    assert tree.scene.body_names == "torso_Link"
    assert tree.obs_mode == "hitter_pure"
    assert retarget.find_residual_a3_names(tree) == []


def test_residual_scan_reports_leftovers_but_tolerates_racket_links():
    tree = _build_tree()
    tree.rewards.passive = _Term(_fake_reward, {"joint_names": ["head_yaw_joint"], "body": "pingpang_red_Link"})
    residual = retarget.find_residual_a3_names(tree)
    paths = {path for path, _ in residual}
    assert "cfg.rewards.passive.params['joint_names'][0]" in paths
    assert "cfg.rewards.passive.params['body']" not in paths
    assert all("scene" not in path for path in paths)
