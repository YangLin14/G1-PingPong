"""Round-trip test of scripts/gmr_video_clip_to_g1.py (host: mujoco, numpy, pyyaml).

A published G1 clip is written out as a fake 30 Hz GMR pickle (GMR column order = the vendor MJCF's
qpos order, xyzw root quaternion), converted back, and compared: joints, tracked bodies and the
auto-detected strike frame must reproduce the source within the 50 -> 30 -> 50 Hz resampling error.
"""

from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "hope_training" / "whole_body_tracking" / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import gmr_video_clip_to_g1 as conv  # noqa: E402

# Same 29 joint names as GMR's g1_mocap_29dof.xml, in a different (qpos) order than the Isaac order.
VENDOR_XML = _REPO / "unitree" / "URDF" / "g1_description" / "mjcf" / "g1_29dof_rev_1_0.xml"
SRC = _REPO / "hope_training" / "motions" / "preprocessed" / "g1_hope_forehand"


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    src = np.load(SRC.with_suffix(".npz"))
    side = yaml.safe_load(SRC.with_suffix(".yaml").read_text())
    order = conv.base.g1_joint_order()
    names = conv.gmr_joint_names(str(VENDOR_XML))
    assert sorted(names) == sorted(order) and names != order

    t50 = np.arange(src["joint_pos"].shape[0]) / 50.0
    t30 = np.arange(0.0, t50[-1] + 1e-9, 1 / 30.0)
    q = src["joint_pos"].astype(np.float64)
    q30 = np.stack([np.interp(t30, t50, q[:, k]) for k in range(q.shape[1])], 1)
    root = src["body_pos_w"][:, 0].astype(np.float64)
    quat = src["body_quat_w"][:, 0].astype(np.float64)
    root30 = np.stack([np.interp(t30, t50, root[:, k]) for k in range(3)], 1)
    quat30 = np.stack([np.interp(t30, t50, quat[:, k]) for k in range(4)], 1)
    quat30 /= np.linalg.norm(quat30, axis=1, keepdims=True)

    tmp = tmp_path_factory.mktemp("rt")
    pkl = tmp / "fake_gmr.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"fps": 30, "root_pos": root30, "root_rot": quat30[:, [1, 2, 3, 0]],
                     "dof_pos": np.stack([q30[:, order.index(n)] for n in names], 1)}, fh)
    assert conv.main(["--gmr-pkl", str(pkl), "--gmr-xml", str(VENDOR_XML), "--name", "rt",
                      "--blend-end-s", "0", "--out-dir", str(tmp)]) == 0
    return src, side, np.load(tmp / "rt.npz"), yaml.safe_load((tmp / "rt.yaml").read_text())


def test_joints_and_bodies_reproduce_the_source(roundtrip):
    src, _, out, _ = roundtrip
    n = min(len(out["joint_pos"]), len(src["joint_pos"]))
    assert np.abs(out["joint_pos"][:n] - src["joint_pos"][:n]).max() < 0.03
    rel_out = out["body_pos_w"][:n] - out["body_pos_w"][:1, :1]
    rel_src = src["body_pos_w"][:n] - src["body_pos_w"][:1, :1]
    assert np.abs(rel_out - rel_src).max() < 0.01


def test_strike_frame_and_sidecar_schema(roundtrip):
    _, side_src, out, side = roundtrip
    assert abs(side["strike_frame"] - side_src["strike_frame"]) <= 1
    assert side["joint_order"] == conv.base.g1_joint_order()
    assert side["tracked_bodies"] == list(conv.base.G1_TRACKED_BODIES)
    assert side["strike_phase"] == pytest.approx(side["strike_frame"] / (side["frame_count"] - 1))
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        assert out[key].dtype == np.float32 and out[key].shape[0] == side["frame_count"]


def test_loop_blend_closes_the_wrap():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 29))
    pos = rng.normal(size=(50, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
    _, _, q2 = conv.blend_to_start(pos, quat, q.copy(), 15)
    assert np.allclose(q2[-1], q2[0]) and np.allclose(q2[:35], q[:35])
