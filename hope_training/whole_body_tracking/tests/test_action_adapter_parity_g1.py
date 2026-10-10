"""Training <-> deployment action-adapter parity for the Unitree G1 (mirror of test_action_adapter_parity.py).

Both sides read ``a3_deploy/a3_deploy_example/config/action_adapter_unitree_g1.yaml`` against the
29-joint G1 order and must map identical raw actions to identical joint targets. The YAML's values
are also checked against the ``robots/g1.py`` recipe they were generated from (action scale =
0.25 * effort / stiffness; default_q = the articulation init pose; clamps = the upstream URDF limits).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_UTILS = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "utils")
_G1_ROBOT_PY = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "robots", "g1.py")
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_SHARED_YAML = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "config", "action_adapter_unitree_g1.yaml")
_UPSTREAM_URDF = os.path.join(_REPO, "unitree", "URDF", "g1_description", "urdf", "g1_29dof_rev_1_0.urdf")


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train_side = _load_by_path("hope_action_adapter_config_g1p", os.path.join(_UTILS, "action_adapter_config.py"))

sys.path.insert(0, _REFERENCE_DIR)
from a3_deploy_onnx_ref_pingpong.action_adapter import ActionAdapter  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order_unitree_g1 import JOINT_NAMES, NUM_JOINTS  # noqa: E402


def _both():
    deploy = ActionAdapter.from_yaml(_SHARED_YAML, joint_names=JOINT_NAMES)
    training = train_side.load_action_adapter_config(path=_SHARED_YAML, robot="unitree_g1")
    return deploy, training


def test_shared_config_exists_and_is_discovered():
    assert os.path.isfile(_SHARED_YAML), _SHARED_YAML
    assert os.path.samefile(train_side.find_shared_action_adapter_config("unitree_g1"), _SHARED_YAML)


def test_joint_order_matches_deploy():
    _deploy, training = _both()
    assert tuple(training.joint_names) == tuple(JOINT_NAMES) and len(JOINT_NAMES) == NUM_JOINTS == 29


def test_constants_identical():
    deploy, training = _both()
    np.testing.assert_array_equal(training.default_q, deploy.default_q)
    np.testing.assert_array_equal(training.action_scale, deploy.action_scale)
    np.testing.assert_array_equal(training.clamp_lower, deploy.clamp_lower)
    np.testing.assert_array_equal(training.clamp_upper, deploy.clamp_upper)


def test_decode_parity_on_representative_actions():
    deploy, training = _both()
    rng = np.random.default_rng(0)
    cases = [
        np.zeros(NUM_JOINTS), np.ones(NUM_JOINTS), -np.ones(NUM_JOINTS),
        rng.uniform(-3.0, 3.0, NUM_JOINTS), np.full(NUM_JOINTS, 50.0), np.full(NUM_JOINTS, -50.0),
    ]
    for raw in cases:
        np.testing.assert_allclose(training.decode(raw), deploy.decode(raw), rtol=0, atol=0)
    with pytest.raises(ValueError):
        deploy.decode(np.zeros(31))  # an A3-sized action can never drive the G1 adapter


def test_zero_action_returns_default_pose_within_clamp():
    deploy, training = _both()
    np.testing.assert_array_equal(deploy.decode(np.zeros(NUM_JOINTS)), deploy.default_q)
    assert np.all(training.default_q >= training.clamp_lower) and np.all(training.default_q <= training.clamp_upper)


def _g1_recipe():
    """Re-derive G1_ACTION_SCALE / init pose from robots/g1.py as source (it imports isaaclab)."""
    with open(_G1_ROBOT_PY, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    armature = {k: v for k, v in consts.items() if k.startswith("ARMATURE_")}
    natural_freq = 10 * 2.0 * 3.1415926535
    damping_ratio = 2.0
    stiffness = {k.replace("ARMATURE_", ""): v * natural_freq**2 for k, v in armature.items()}
    init_pose = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "InitialStateCfg":
            for kw in node.keywords:
                if kw.arg == "joint_pos":
                    init_pose = ast.literal_eval(kw.value)
    assert init_pose is not None
    # (pattern, motor, effort_limit_sim, stiffness multiplier) as declared in G1_CYLINDER_CFG.actuators.
    groups = [
        (r".*_hip_pitch_joint", "7520_14", 88.0, 1.0), (r".*_hip_roll_joint", "7520_22", 139.0, 1.0),
        (r".*_hip_yaw_joint", "7520_14", 88.0, 1.0), (r".*_knee_joint", "7520_22", 139.0, 1.0),
        (r".*_ankle_(pitch|roll)_joint", "5020", 50.0, 2.0), (r"waist_(roll|pitch)_joint", "5020", 50.0, 2.0),
        (r"waist_yaw_joint", "7520_14", 88.0, 1.0), (r".*_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint", "5020", 25.0, 1.0),
        (r".*_wrist_(pitch|yaw)_joint", "4010", 5.0, 1.0),
    ]
    scale, q0 = {}, {}
    for name in JOINT_NAMES:
        pattern, motor, effort, mult = next(g for g in groups if re.fullmatch(g[0], name))
        scale[name] = 0.25 * effort / (stiffness[motor] * mult)
        q0[name] = 0.0
        for pat, value in init_pose.items():
            if re.fullmatch(pat, name):
                q0[name] = value
    _ = damping_ratio
    return scale, q0


def test_yaml_values_match_the_robots_g1_recipe_and_urdf_limits():
    deploy, _training = _both()
    scale, q0 = _g1_recipe()
    np.testing.assert_allclose(deploy.action_scale, [scale[n] for n in JOINT_NAMES], rtol=1e-4)
    np.testing.assert_allclose(deploy.default_q, [q0[n] for n in JOINT_NAMES], atol=1e-9)
    limits = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in ET.parse(_UPSTREAM_URDF).getroot().findall("joint")
        if j.get("type") == "revolute"
    }
    np.testing.assert_allclose(deploy.clamp_lower, [limits[n][0] for n in JOINT_NAMES], rtol=1e-5)
    np.testing.assert_allclose(deploy.clamp_upper, [limits[n][1] for n in JOINT_NAMES], rtol=1e-5)


def test_a3_adapter_defaults_unchanged():
    a3_yaml = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "config", "action_adapter.yaml")
    a3 = ActionAdapter.from_yaml(a3_yaml)
    assert a3.num_joints == 31 and a3.joint_names[3] == "head_yaw_joint"
    with pytest.raises(ValueError):
        ActionAdapter.from_yaml(_SHARED_YAML)  # the G1 YAML lacks the A3 head joints -> rejected under A3 defaults
