"""The derived Unitree G1 reference clips: schema, joint order, limits, floor contact, regeneration.

Mirror of test_published_motion_assets.py for ``g1_hope_forehand.npz`` / ``g1_hope_backhand.npz``,
which are joint-space transfers of the A3 pair (``scripts/retarget_a3_clip_to_g1.py``). Host-only
(NumPy + PyYAML); the regeneration check needs ``mujoco`` and is skipped without it.
"""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
_MOTIONS = _REPO / "hope_training" / "motions" / "preprocessed"
_JOINT_ORDER_YAML = _REPO / "hope_training" / "config" / "joint_order_unitree_g1.yaml"
_G1_URDF = _REPO / "unitree" / "URDF" / "g1_description" / "urdf" / "g1_29dof_rev_1_0.urdf"
_SCRIPT = _ROOT / "scripts" / "retarget_a3_clip_to_g1.py"
_TASK_YAML = _ROOT / "cfg" / "task" / "HOPEPingPongUnitreeG1.yaml"

_EXPECTED = {
    "g1_hope_forehand": {"source": "hope_forehand", "frames": 107, "strike_frame": 41, "swing_side": 1},
    "g1_hope_backhand": {"source": "hope_backhand", "frames": 109, "strike_frame": 48, "swing_side": -1},
}
_TRACKED = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def _joint_order() -> list[str]:
    return [str(n) for n in yaml.safe_load(_JOINT_ORDER_YAML.read_text())["joint_order"]]


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_schema_and_sidecar(name: str) -> None:
    exp = _EXPECTED[name]
    data = np.load(_MOTIONS / f"{name}.npz")
    side = yaml.safe_load((_MOTIONS / f"{name}.yaml").read_text())
    frames = exp["frames"]
    assert int(np.asarray(data["fps"]).reshape(-1)[0]) == 50
    assert data["joint_pos"].shape == (frames, 29) and data["joint_vel"].shape == (frames, 29)
    for key, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        assert data[key].shape == (frames, 14, width), key
        assert data[key].dtype == np.float32
        assert np.all(np.isfinite(data[key]))
    assert side["robot"] == "unitree_g1" and side["body_schema"] == "compact_tracked_bodies_v1"
    assert side["joint_order"] == _joint_order()
    assert list(side["tracked_bodies"]) == list(_TRACKED)
    assert side["anchor_body"] == "torso_link" and side["root_body"] == "pelvis"
    assert side["racket_link"] == "right_wrist_yaw_link" and side["racket_body"] == "pingpang_red_Link"
    assert side["mount_offset_xyz"] == [0.21, 0.015, 0.01]
    assert side["source_clip"] == f"{exp['source']}.npz"
    # Phase structure carried over unchanged from the A3 clip (same frames, same strike).
    assert side["frame_count"] == frames and side["strike_frame"] == exp["strike_frame"]
    assert side["swing_side"] == exp["swing_side"]
    src = yaml.safe_load((_MOTIONS / f"{exp['source']}.yaml").read_text())
    assert side["strike_phase"] == src["strike_phase"] and side["blade_normal_sign"] == src["blade_normal_sign"]


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_joints_within_g1_limits_and_match_the_a3_source_by_name(name: str) -> None:
    exp = _EXPECTED[name]
    data = np.load(_MOTIONS / f"{name}.npz")
    order = _joint_order()
    limits = {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in ET.parse(_G1_URDF).getroot().findall("joint")
        if j.get("type") == "revolute"
    }
    lo = np.array([limits[n][0] for n in order])
    hi = np.array([limits[n][1] for n in order])
    q = data["joint_pos"]
    assert np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6)
    # Same-named A3 column, up to the limit clamp.
    src = np.load(_MOTIONS / f"{exp['source']}.npz")
    src_order = yaml.safe_load((_MOTIONS / f"{exp['source']}.yaml").read_text())["joint_order"]
    for j, joint in enumerate(order):
        a3 = src["joint_pos"][:, src_order.index(joint)]
        np.testing.assert_allclose(q[:, j], np.clip(a3, lo[j], hi[j]), atol=1e-6)


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_feet_touch_the_floor_and_the_robot_is_g1_sized(name: str) -> None:
    data = np.load(_MOTIONS / f"{name}.npz")
    pos = data["body_pos_w"]
    feet_z = pos[:, [3, 6], 2]
    # ankle_roll body origin sits ~3.5 cm above the sole; the lowest sole over the clip is on the floor.
    assert 0.02 < feet_z.min() < 0.06
    root_z = pos[:, 0, 2]
    assert 0.55 < root_z.min() and root_z.max() < 0.85  # G1 pelvis, not the A3's ~0.93 m
    quat = data["body_quat_w"]
    np.testing.assert_allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-4)
    # Velocities are finite differences of the stored tracks (sanity on the first body).
    dt = 1.0 / 50.0
    v_mid = (pos[2:, 0] - pos[:-2, 0]) / (2 * dt)
    np.testing.assert_allclose(data["body_lin_vel_w"][1:-1, 0], v_mid, atol=1e-3)


def test_task_yaml_points_at_the_g1_clips() -> None:
    task = yaml.safe_load(_TASK_YAML.read_text())
    assert task["motion_file"] == "../motions/preprocessed/g1_hope_forehand.npz"
    assert task["motion_file_2"] == "../motions/preprocessed/g1_hope_backhand.npz"
    for rel in (task["motion_file"], task["motion_file_2"]):
        assert (_ROOT / rel).resolve().is_file(), rel


def test_committed_clips_match_a_fresh_transfer() -> None:
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location("hope_retarget_a3_to_g1", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.check() == []
