# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Vendored Unitree G1 description + racket-equipped variant: provenance, regeneration, structure.

Plain-Python tests (xml.etree + hashlib). The MuJoCo load tests are skipped when ``mujoco`` is
not installed; the Isaac-side consumers are exercised by ``hope_training/whole_body_tracking/tests``.

Run (from the repo root):  python3 -m pytest -q unitree/tests
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

_UNITREE = pathlib.Path(__file__).resolve().parents[1]
_REPO = _UNITREE.parent
_UPSTREAM = _UNITREE / "URDF" / "g1_description"
_VARIANT = _UNITREE / "URDF" / "g1_29dof_pingpang"
_URDF = _VARIANT / "urdf" / "g1_29dof_pingpang.urdf"
_MJCF = _VARIANT / "mjcf" / "g1_29dof_pingpang.xml"
_G1_ROBOT_PY = (
    _REPO / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
    / "whole_body_tracking" / "robots" / "g1.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "hope_build_g1_pingpang_models", _UNITREE / "scripts" / "build_g1_pingpang_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()

RACKET_LINKS = ("right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link")
RACKET_JOINTS = ("right_hand_pingpang_joint", "pingpang_red_joint", "pingpang_black_joint")
G1_HINGE_JOINT_COUNT = 29


# --------------------------------------------------------------------------------------------------
# Upstream provenance
# --------------------------------------------------------------------------------------------------
def test_upstream_files_match_recorded_checksums():
    sums = (_UPSTREAM / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(sums) == 38  # LICENSE + URDF + MJCF + 35 meshes
    for line in sums:
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        actual = hashlib.sha256((_UPSTREAM / rel).read_bytes()).hexdigest()
        assert actual == digest, f"vendored upstream file modified: {rel}"


def test_upstream_license_is_bsd_3_clause():
    text = (_UPSTREAM / "LICENSE").read_text(encoding="utf-8")
    assert "BSD 3-Clause License" in text
    assert "Unitree" in text


def test_every_upstream_mesh_reference_is_vendored():
    urdf_refs = {m.get("filename") for m in ET.parse(_UPSTREAM / "urdf" / "g1_29dof_rev_1_0.urdf").iter("mesh")}
    mjcf_refs = {m.get("file") for m in ET.parse(_UPSTREAM / "mjcf" / "g1_29dof_rev_1_0.xml").iter("mesh")}
    assert len(urdf_refs) == 35 and len(mjcf_refs) == 35
    for ref in urdf_refs:
        assert (_UPSTREAM / ref).is_file(), ref
    for ref in mjcf_refs:
        assert (_UPSTREAM / "meshes" / ref).is_file(), ref


# --------------------------------------------------------------------------------------------------
# Regeneration
# --------------------------------------------------------------------------------------------------
def test_committed_variant_equals_fresh_generation():
    with tempfile.TemporaryDirectory() as tmp:
        urdf, mjcf = builder.generate(pathlib.Path(tmp))
        assert urdf.read_bytes() == _URDF.read_bytes(), "regenerate: python3 unitree/scripts/build_g1_pingpang_models.py"
        assert mjcf.read_bytes() == _MJCF.read_bytes(), "regenerate: python3 unitree/scripts/build_g1_pingpang_models.py"
    assert builder.check() == []


def test_generation_is_deterministic():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        ua, ma = builder.generate(pathlib.Path(a))
        ub, mb = builder.generate(pathlib.Path(b))
        assert ua.read_bytes() == ub.read_bytes()
        assert ma.read_bytes() == mb.read_bytes()


# --------------------------------------------------------------------------------------------------
# URDF structure
# --------------------------------------------------------------------------------------------------
def _urdf_root() -> ET.Element:
    return ET.parse(_URDF).getroot()


def test_urdf_has_racket_links_and_fixed_joints_without_axes():
    root = _urdf_root()
    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    assert root.get("name") == "g1_29dof_pingpang"
    for name in RACKET_LINKS:
        assert name in links, name
    assert "pingbang_ball_Link" not in links
    assert "right_rubber_hand" in links  # upstream hand kept
    for name in RACKET_JOINTS:
        assert joints[name].get("type") == "fixed"
        assert joints[name].find("axis") is None
    assert joints["right_hand_pingpang_joint"].find("parent").get("link") == "right_wrist_yaw_link"
    assert joints["right_hand_pingpang_joint"].find("child").get("link") == "right_hand_pingpang_Link"
    for name in ("pingpang_red_joint", "pingpang_black_joint"):
        assert joints[name].find("parent").get("link") == "right_hand_pingpang_Link"
        xyz = tuple(float(v) for v in joints[name].find("origin").get("xyz").split())
        rpy = tuple(float(v) for v in joints[name].find("origin").get("rpy").split())
        assert xyz == pytest.approx(builder.G1_RACKET_MOUNT_XYZ)
        assert rpy == pytest.approx(builder.G1_RACKET_MOUNT_RPY)


def test_urdf_keeps_29_movable_joints_with_valid_axes():
    root = _urdf_root()
    movable = [j for j in root.findall("joint") if j.get("type") in {"revolute", "continuous", "prismatic"}]
    assert len(movable) == G1_HINGE_JOINT_COUNT
    for joint in movable:
        xyz = [float(v) for v in joint.find("axis").get("xyz").split()]
        assert len(xyz) == 3 and sum(v * v for v in xyz) > 0.0, joint.get("name")


def test_urdf_mesh_references_resolve_from_the_urdf_directory():
    root = _urdf_root()
    refs = [m.get("filename") for m in root.iter("mesh")]
    assert refs and all(not r.startswith("package://") for r in refs)
    for ref in refs:
        assert (_URDF.parent / ref).resolve().is_file(), ref
    blade_refs = {r for r in refs if "pingpang" in r}
    assert blade_refs == {"../meshes/pingpang_red_Link.STL", "../meshes/pingpang_black_Link.STL"}


def test_urdf_racket_inertial_is_on_the_hand_link():
    root = _urdf_root()
    hand = next(link for link in root.findall("link") if link.get("name") == "right_hand_pingpang_Link")
    mass = float(hand.find("inertial/mass").get("value"))
    assert mass == pytest.approx(builder.RACKET_MASS)
    origin = tuple(float(v) for v in hand.find("inertial/origin").get("xyz").split())
    assert origin == pytest.approx(builder.G1_RACKET_MOUNT_XYZ)
    for name in ("pingpang_red_Link", "pingpang_black_Link"):
        blade = next(link for link in root.findall("link") if link.get("name") == name)
        assert float(blade.find("inertial/mass").get("value")) == 0.0


# --------------------------------------------------------------------------------------------------
# MJCF structure (no MuJoCo needed)
# --------------------------------------------------------------------------------------------------
def _mjcf_root() -> ET.Element:
    return ET.parse(_MJCF).getroot()


def test_mjcf_names_the_evaluator_depends_on():
    root = _mjcf_root()
    assert root.get("model") == "g1_29dof_pingpang"
    joints = {j.get("name"): j for j in root.iter("joint")}
    assert joints["pelvis_free_joint"].get("type") == "free"
    assert "floating_base_joint" not in joints
    geoms = {g.get("name") for g in root.iter("geom") if g.get("name")}
    for name in ("right_racket_collision", "right_racket_handle_collision", "right_racket_clearance_proxy", "floor"):
        assert name in geoms, name
    sites = {s.get("name") for s in root.iter("site")}
    assert {"right_racket", "pelvis_site", "imu_in_pelvis"} <= sites
    sensors = {s.get("name") for s in root.find("sensor")}
    assert {"pelvis_imu_gyro", "pelvis_framepos", "right_racket_framepos", "right_racket_framequat"} <= sensors
    # Names the ping-pong scene builder adds itself must be free.
    reserved = {"ball", "ball_geom", "ball_free_joint", "table_top", "table_geom", "net_body", "net_geom"}
    all_names = geoms | sites | set(joints) | {b.get("name") for b in root.iter("body")}
    assert not (reserved & all_names)


def test_mjcf_motors_cover_every_hinge_with_training_recipe_ctrlrange():
    """ctrlrange / actuatorfrcrange = the Isaac training recipe's effort limits (robots/g1.py via
    g1_actuation), NOT the vendor URDF: the recipe uses 50 N.m where the URDF says 35 N.m for the
    ankle and waist roll/pitch motors, and the MuJoCo plant must clip like the plant the policy was
    trained on. Armature matches the Isaac drive; no implicit damping/friction (the harness PD is
    the whole actuation)."""
    root = _mjcf_root()
    urdf_efforts = builder.joint_effort_limits(ET.parse(_UPSTREAM / "urdf" / "g1_29dof_rev_1_0.urdf").getroot())
    motors = root.find("actuator").findall("motor")
    assert len(motors) == G1_HINGE_JOINT_COUNT
    hinges = {j.get("name"): j for j in root.iter("joint") if j.get("type") != "free"}
    assert sorted(m.get("joint") for m in motors) == sorted(hinges)
    differing = set()
    for motor in motors:
        name = motor.get("joint")
        effort = builder.g1_actuation.effort_limit(name)
        lo, hi = (float(v) for v in motor.get("ctrlrange").split())
        assert motor.get("ctrllimited") == "true"
        assert -lo == hi == pytest.approx(effort)
        jlo, jhi = (float(v) for v in hinges[name].get("actuatorfrcrange").split())
        assert -jlo == jhi == pytest.approx(effort)
        assert float(hinges[name].get("armature")) == pytest.approx(builder.g1_actuation.armature(name), rel=1e-4)
        assert hinges[name].get("damping") == "0" and hinges[name].get("frictionloss") == "0"
        if effort != urdf_efforts[name]:
            differing.add(name)
    assert differing == {
        "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
        "waist_roll_joint", "waist_pitch_joint",
    }


def test_mjcf_stand_keyframe_matches_the_builder_pose():
    root = _mjcf_root()
    key = root.find("keyframe").find("key")
    assert key.get("name") == "stand"
    qpos = [float(v) for v in key.get("qpos").split()]
    hinges = [j.get("name") for j in root.iter("joint") if j.get("type") != "free"]
    assert len(qpos) == 7 + len(hinges) == 36
    assert qpos[:7] == pytest.approx([0.0, 0.0, builder.STAND_PELVIS_Z, 1.0, 0.0, 0.0, 0.0])
    for name, value in zip(hinges, qpos[7:]):
        assert value == pytest.approx(builder.STAND_JOINT_POS.get(name, 0.0)), name


def test_mjcf_option_and_collision_class():
    root = _mjcf_root()
    assert float(root.find("option").get("timestep")) == pytest.approx(0.001)
    cls = next(d for d in root.find("default").findall("default") if d.get("class") == "collision")
    geom = cls.find("geom")
    assert geom.get("group") == "3" and geom.get("contype") == "1" and geom.get("conaffinity") == "7"


# --------------------------------------------------------------------------------------------------
# Parity with the Isaac Lab robot config (robots/g1.py is read as source; it imports isaaclab)
# --------------------------------------------------------------------------------------------------
def _g1_init_state_joint_pos() -> dict[str, float]:
    tree = ast.parse(_G1_ROBOT_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "InitialStateCfg":
            for kw in node.keywords:
                if kw.arg == "joint_pos":
                    return {str(k): float(v) for k, v in ast.literal_eval(kw.value).items()}
    raise AssertionError("InitialStateCfg(joint_pos=...) not found in robots/g1.py")


def test_stand_pose_mirrors_robots_g1_init_state():
    import re

    patterns = _g1_init_state_joint_pos()
    hinges = [j.get("name") for j in _mjcf_root().iter("joint") if j.get("type") != "free"]
    for name in hinges:
        expected = 0.0
        for pattern, value in patterns.items():
            if re.fullmatch(pattern, name):
                expected = value
        assert builder.STAND_JOINT_POS.get(name, 0.0) == pytest.approx(expected), name


def test_mount_offset_mirrors_robots_g1_constant():
    tree = ast.parse(_G1_ROBOT_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "G1_MOUNT_OFFSET" for t in node.targets):
            assert tuple(ast.literal_eval(node.value)) == pytest.approx(builder.G1_RACKET_MOUNT_XYZ)
            return
    raise AssertionError("G1_MOUNT_OFFSET not found in robots/g1.py")


# --------------------------------------------------------------------------------------------------
# MuJoCo load (optional)
# --------------------------------------------------------------------------------------------------
def _load_mujoco_model():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_mujoco_model_dimensions():
    mujoco, model, _data = _load_mujoco_model()
    assert model.nq == 36 and model.nv == 35 and model.nu == 29 and model.nkey == 1
    assert model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_free_joint")] == 0
    assert model.opt.timestep == pytest.approx(0.001)


def test_mujoco_racket_site_sits_at_the_mount_offset():
    mujoco, model, data = _load_mujoco_model()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_racket")
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    rel_world = data.site_xpos[sid] - data.xpos[wid]
    rot = data.xmat[wid].reshape(3, 3)
    rel_wrist = [float(sum(rot[r][c] * rel_world[r] for r in range(3))) for c in range(3)]
    assert rel_wrist == pytest.approx(builder.G1_RACKET_MOUNT_XYZ, abs=1e-6)


def test_mujoco_stand_keyframe_is_grounded():
    mujoco, model, data = _load_mujoco_model()
    lowest = math.inf
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        if body and "ankle_roll" in body and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            lowest = min(lowest, float(data.geom_xpos[gid][2] - model.geom_size[gid][0]))
    assert 0.0 <= lowest < 0.005, lowest


def test_mujoco_racket_collision_geom_is_in_the_contact_class():
    mujoco, model, _data = _load_mujoco_model()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_racket_collision")
    assert model.geom_contype[gid] == 1 and model.geom_conaffinity[gid] == 7 and model.geom_group[gid] == 3
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_imu_gyro") >= 0
