"""The keypoint/IK-retargeted Unitree G1 clips (variant B): schema, limits, tracking error, regeneration.

Host-only (NumPy + PyYAML + mujoco). The IK clips share legs/waist/root with the joint-space pair
(``g1_hope_*``) and re-solve both arms toward the A3 elbow / wrist / racket keypoints scaled to the
G1; the sidecar records the per-frame tracking errors the script measured.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
_MOTIONS = _REPO / "hope_training" / "motions" / "preprocessed"
_SCRIPTS = _ROOT / "scripts"
_G1_MJCF = _REPO / "unitree" / "URDF" / "g1_29dof_pingpang" / "mjcf" / "g1_29dof_pingpang.xml"
_TASK_YAML = _ROOT / "cfg" / "task" / "HOPEPingPongUnitreeG1.yaml"
_TASK_YAML_IK = _ROOT / "cfg" / "task" / "HOPEPingPongUnitreeG1_IK.yaml"

_EXPECTED = {
    "g1ik_hope_forehand": {"base": "g1_hope_forehand", "frames": 107, "strike_frame": 41, "swing_side": 1},
    "g1ik_hope_backhand": {"base": "g1_hope_backhand", "frames": 109, "strike_frame": 48, "swing_side": -1},
}
# Tracking tolerances (metres / degrees): the two robots' blade-in-wrist geometry differs, so the
# racket centre cannot be matched exactly while the arm shape is kept.
MAX_MEAN_RACKET_ERR = 0.07
MAX_STRIKE_RACKET_ERR = 0.10
MAX_STRIKE_PLANE_DEG = 15.0
ARM_JOINTS = tuple(f"{side}_{j}_joint" for side in ("left", "right")
                   for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))


def _load_script():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("retarget_a3_clip_to_g1_ik", _SCRIPTS / "retarget_a3_clip_to_g1_ik.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_schema_limits_and_shared_lower_body(name: str) -> None:
    exp = _EXPECTED[name]
    data = np.load(_MOTIONS / f"{name}.npz")
    base = np.load(_MOTIONS / f"{exp['base']}.npz")
    side = yaml.safe_load((_MOTIONS / f"{name}.yaml").read_text())
    base_side = yaml.safe_load((_MOTIONS / f"{exp['base']}.yaml").read_text())
    frames = exp["frames"]
    assert data["joint_pos"].shape == (frames, 29) and data["body_pos_w"].shape == (frames, 14, 3)
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        assert np.all(np.isfinite(data[key])) and data[key].dtype == np.float32, key
    assert side["name"] == name and side["robot"] == "unitree_g1" and side["strike_frame"] == exp["strike_frame"]
    assert side["swing_side"] == exp["swing_side"] and side["joint_order"] == base_side["joint_order"]
    assert "keypoint/IK retarget" in side["provenance"]
    order = list(side["joint_order"])
    mod = _load_script()
    limits = mod.g1_joint_limits()
    lo = np.array([limits[n][0] for n in order]); hi = np.array([limits[n][1] for n in order])
    assert np.all(data["joint_pos"] >= lo - 1e-6) and np.all(data["joint_pos"] <= hi + 1e-6)
    # legs / waist / root identical to the joint-space transfer; arms re-solved
    non_arm = [i for i, n in enumerate(order) if n not in ARM_JOINTS]
    np.testing.assert_allclose(data["joint_pos"][:, non_arm], base["joint_pos"][:, non_arm], atol=1e-6)
    np.testing.assert_allclose(data["body_pos_w"][:, 0], base["body_pos_w"][:, 0], atol=1e-5)
    arm = [i for i, n in enumerate(order) if n in ARM_JOINTS]
    assert np.abs(data["joint_pos"][:, arm] - base["joint_pos"][:, arm]).max() > 0.05


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_tracking_errors_recorded_and_reproduced_by_fk(name: str) -> None:
    exp = _EXPECTED[name]
    side = yaml.safe_load((_MOTIONS / f"{name}.yaml").read_text())
    mean, strike = side["ik_mean_errors_m"], side["ik_strike_frame_errors"]
    assert mean["right"]["racket"] <= MAX_MEAN_RACKET_ERR
    assert strike["racket"] <= MAX_STRIKE_RACKET_ERR and strike["normal_deg"] <= MAX_STRIKE_PLANE_DEG
    # Reproduce the strike-frame racket error from the committed arrays: FK the clip through the
    # G1 MJCF and compare the racket site with the scaled A3 target the script builds.
    mod = _load_script()
    kin = mod.G1Kinematics()
    ik = mod.G1ArmIK(kin)
    src = exp["base"].replace("g1_", "")
    a3 = mod.A3Keypoints(np.load(_MOTIONS / f"{src}.npz"), yaml.safe_load((_MOTIONS / f"{src}.yaml").read_text()))
    data = np.load(_MOTIONS / f"{name}.npz")
    order = list(side["joint_order"])
    f = exp["strike_frame"]
    kin.pose(data["body_pos_w"][f, 0].astype(float), data["body_quat_w"][f, 0].astype(float),
             {n: float(data["joint_pos"][f, i]) for i, n in enumerate(order)})
    t_pos, t_rot = ik.torso_frame()
    kp0 = a3.arm(0, "right"); kp = a3.arm(f, "right")
    kin_stand = mod.G1Kinematics(); ik0 = mod.G1ArmIK(kin_stand)
    base = np.load(_MOTIONS / f"{exp['base']}.npz")
    kin_stand.pose(base["body_pos_w"][0, 0].astype(float), base["body_quat_w"][0, 0].astype(float),
                   {n: float(base["joint_pos"][0, i]) for i, n in enumerate(order)})
    g1u, g1f = ik0.segment_lengths("right")
    up = g1u / np.linalg.norm(kp0["elbow"] - kp0["shoulder"]); fo = g1f / np.linalg.norm(kp0["wrist"] - kp0["elbow"])
    g1_sh = t_rot.T @ (ik.d.xpos[ik.body["right_shoulder_pitch_link"]] - t_pos)
    wr = g1_sh + (kp["elbow"] - kp["shoulder"]) * up + (kp["wrist"] - kp["elbow"]) * fo
    racket_target = t_pos + t_rot @ (wr + (kp["racket"] - kp["wrist"]))
    err = float(np.linalg.norm(ik.d.site_xpos[ik.site] - racket_target))
    assert err == pytest.approx(strike["racket"], abs=0.02)  # the 5-tap smoothing moves it slightly
    assert err <= MAX_STRIKE_RACKET_ERR + 0.02


def test_variant_b_task_yaml_only_swaps_the_clips() -> None:
    proc = subprocess.run([sys.executable, str(_REPO / "unitree" / "scripts" / "sync_g1_variant_yaml.py"), "--check"],
                          capture_output=True, text=True, cwd=str(_REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    a = yaml.safe_load(_TASK_YAML.read_text()); b = yaml.safe_load(_TASK_YAML_IK.read_text())
    assert b["motion_file"].endswith("g1ik_hope_forehand.npz") and b["motion_file_2"].endswith("g1ik_hope_backhand.npz")
    assert b["name"] == "HitterPingPongUnitreeG1_IK"
    for key in ("name", "motion_file", "motion_file_2"):
        a.pop(key); b.pop(key)
    assert a == b


def test_committed_ik_clips_match_a_fresh_retarget() -> None:
    proc = subprocess.run([sys.executable, str(_SCRIPTS / "retarget_a3_clip_to_g1_ik.py"), "--check"],
                          capture_output=True, text=True, cwd=str(_REPO), env={"MUJOCO_GL": "egl", "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
