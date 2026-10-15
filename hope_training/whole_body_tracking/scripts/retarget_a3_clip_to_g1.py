#!/usr/bin/env python3
# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Joint-space transfer of an Agibot A3 reference clip onto the Unitree G1 (host, MuJoCo FK).

The published HOPE clips are A3 joint trajectories; the human SMPL-X source they were retargeted
from is not published, so GMR cannot be re-run. The two robots do, however, share the joint
vocabulary (identical joint names, hinge axes and per-joint zero conventions; the A3 only adds two
head joints), so the swing can be carried over in joint space:

1. every G1 joint takes the A3 joint of the same name (head joints dropped), clamped to the G1
   URDF limits (the clamp magnitude is reported);
2. the root (pelvis) orientation is copied; the root displacement is scaled by the stand-height
   ratio of the two robots and the root height is offset once so the lowest foot sole over the
   clip rests on the floor;
3. joint velocities, tracked-body poses and velocities are recomputed on the G1 by MuJoCo forward
   kinematics (``unitree/URDF/g1_29dof_pingpang/mjcf``) and central finite differences at the
   clip's frame rate.

The result is a COMPACT clip (``hope_training/motions/README.md`` / ``docs/REPLACE_MOTIONS.md``):
``joint_pos``/``joint_vel`` in the canonical G1 Isaac order (``joint_order_unitree_g1.yaml``) and
the 14 ``G1_TRACKED_BODIES`` in configured order, plus a sidecar YAML.

LIMITS OF THIS TRANSFER (documented, not hidden): the G1 is ~30 % shorter with different limb
ratios, so the racket contact point, stride geometry and reach of the original swing are NOT
preserved -- only the joint-space style is. It is a usable imitation prior for training the G1
hitter (the racket-target command, not the clip, defines where the paddle must be), not a
kinematically faithful retarget. Replace it with a keypoint/IK retarget when one is available.

Usage (from the repo root, with ``mujoco`` + ``numpy`` + ``pyyaml``)::

    python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py          # both clips
    python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py --check  # committed == fresh
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "unitree").is_dir():
            return parent
    return here.parents[3]


REPO_ROOT = _repo_root()
MOTIONS_DIR = REPO_ROOT / "hope_training" / "motions" / "preprocessed"
G1_MJCF = REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang" / "mjcf" / "g1_29dof_pingpang.xml"
G1_URDF = REPO_ROOT / "unitree" / "URDF" / "g1_description" / "urdf" / "g1_29dof_rev_1_0.urdf"
G1_JOINT_ORDER_YAML = REPO_ROOT / "hope_training" / "config" / "joint_order_unitree_g1.yaml"

# Mirrors robots/g1.py (kept literal here so the script runs without Isaac).
G1_TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
G1_FEET_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
G1_MOUNT_OFFSET = (0.210, 0.015, 0.010)
# Stand pelvis heights used for the root-displacement scale (A3: robots/agibot_a3.py; G1: MJCF key).
A3_STAND_PELVIS_Z = 1.0684
G1_STAND_PELVIS_Z = 0.7585
DROPPED_A3_JOINTS = ("head_yaw_joint", "head_pitch_joint")

CLIPS = (("hope_forehand", "g1_hope_forehand"), ("hope_backhand", "g1_hope_backhand"))


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def _angular_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a (F, ..., 4) wxyz quaternion track (central differences)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # keep the track continuous (no sign flips) before differentiating
    for i in range(1, q.shape[0]):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0.0
        q[i][flip] *= -1.0
    dq = _central_diff(q, dt)
    omega_q = 2.0 * _quat_mul(dq, _quat_conj(q))  # world frame: 2 * dq * q^-1
    return omega_q[..., 1:]


def g1_joint_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(G1_URDF).getroot()
    return {
        j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
        for j in root.findall("joint")
        if j.get("type") == "revolute"
    }


def g1_joint_order() -> list[str]:
    with open(G1_JOINT_ORDER_YAML, "r", encoding="utf-8") as fh:
        return [str(n) for n in yaml.safe_load(fh)["joint_order"]]


class G1Kinematics:
    """MuJoCo forward kinematics on the racket-equipped G1 model."""

    def __init__(self, mjcf: pathlib.Path | None = None) -> None:
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf or G1_MJCF))
        self.data = mujoco.MjData(self.model)
        self.mjcf_joints = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.qpos_adr = {
            name: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in self.mjcf_joints
        }
        self.body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in G1_TRACKED_BODIES
        }
        missing = [n for n, i in self.body_ids.items() if i < 0]
        if missing:
            raise ValueError(f"tracked bodies missing from the G1 MJCF: {missing}")
        # Foot sole points: the sphere geoms on the ankle_roll bodies (their bottoms touch the floor).
        self.sole_geoms = [
            g for g in range(self.model.ngeom)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) in G1_FEET_BODIES
        ]
        if not self.sole_geoms:
            raise ValueError("no foot sole spheres found on the G1 ankle_roll bodies")

    def pose(self, root_pos, root_quat, joints: dict[str, float]):
        d = self.data
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        for name, value in joints.items():
            d.qpos[self.qpos_adr[name]] = value
        self.mj.mj_kinematics(self.model, d)
        pos = np.stack([d.xpos[self.body_ids[n]] for n in G1_TRACKED_BODIES])
        quat = np.stack([d.xquat[self.body_ids[n]] for n in G1_TRACKED_BODIES])  # wxyz
        sole_min = min(float(d.geom_xpos[g][2] - self.model.geom_size[g][0]) for g in self.sole_geoms)
        return pos.copy(), quat.copy(), sole_min


# --------------------------------------------------------------------------------------------------
# transfer
# --------------------------------------------------------------------------------------------------
def transfer_clip(src_npz: pathlib.Path, src_yaml: pathlib.Path, kin: G1Kinematics) -> tuple[dict, dict, dict]:
    """Return (npz arrays, sidecar dict, report dict) for one clip."""
    data = np.load(src_npz)
    with open(src_yaml, "r", encoding="utf-8") as fh:
        sidecar = yaml.safe_load(fh)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    a3_order = [str(n) for n in sidecar["joint_order"]]
    if len(a3_order) != data["joint_pos"].shape[1]:
        raise ValueError(f"{src_yaml.name}: joint_order lists {len(a3_order)} joints, npz has {data['joint_pos'].shape[1]}")
    a3_index = {name: i for i, name in enumerate(a3_order)}
    g1_order = g1_joint_order()
    limits = g1_joint_limits()
    missing = [n for n in g1_order if n not in a3_index]
    if missing:
        raise ValueError(f"A3 clip has no column for G1 joints {missing}")
    dropped = [n for n in a3_order if n not in set(g1_order)]
    if set(dropped) != set(DROPPED_A3_JOINTS):
        raise ValueError(f"unexpected A3-only joints {dropped}; expected {list(DROPPED_A3_JOINTS)}")

    frames = int(data["joint_pos"].shape[0])
    a3_q = np.asarray(data["joint_pos"], dtype=np.float64)
    q = np.stack([a3_q[:, a3_index[n]] for n in g1_order], axis=1)
    lo = np.array([limits[n][0] for n in g1_order])
    hi = np.array([limits[n][1] for n in g1_order])
    q_clamped = np.clip(q, lo, hi)
    clamp = np.abs(q_clamped - q)
    clamped_joints = {g1_order[j]: float(clamp[:, j].max()) for j in range(len(g1_order)) if clamp[:, j].max() > 1e-9}

    # Root: copy orientation, scale displacement, fit height so the lowest sole touches the floor.
    scale = G1_STAND_PELVIS_Z / A3_STAND_PELVIS_Z
    a3_root_pos = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float64)
    a3_root_quat = np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float64)
    root_pos = np.empty_like(a3_root_pos)
    root_pos[:, :2] = a3_root_pos[0, :2] + (a3_root_pos[:, :2] - a3_root_pos[0, :2]) * scale
    root_pos[:, 2] = a3_root_pos[:, 2] * scale
    root_quat = a3_root_quat / np.linalg.norm(a3_root_quat, axis=-1, keepdims=True)

    body_pos = np.empty((frames, len(G1_TRACKED_BODIES), 3))
    body_quat = np.empty((frames, len(G1_TRACKED_BODIES), 4))
    sole = np.empty(frames)
    for f in range(frames):
        joints = {n: float(q_clamped[f, j]) for j, n in enumerate(g1_order)}
        body_pos[f], body_quat[f], sole[f] = kin.pose(root_pos[f], root_quat[f], joints)
    z_shift = -float(sole.min())
    root_pos[:, 2] += z_shift
    body_pos[:, :, 2] += z_shift
    sole += z_shift

    joint_vel = _central_diff(q_clamped, dt)
    body_lin_vel = _central_diff(body_pos, dt)
    body_ang_vel = _angular_velocity(body_quat.copy(), dt)

    arrays = {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": q_clamped.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin_vel.astype(np.float32),
        "body_ang_vel_w": body_ang_vel.astype(np.float32),
    }
    report = {
        "frames": frames,
        "fps": fps,
        "root_scale": scale,
        "root_z_shift": z_shift,
        "root_z_range": (float(root_pos[:, 2].min()), float(root_pos[:, 2].max())),
        "max_sole_float": float(sole.max()),
        "clamped_joints": clamped_joints,
    }
    out_sidecar = dict(sidecar)
    out_sidecar.update(
        {
            "name": src_yaml.stem.replace("hope_", "g1_hope_"),
            "robot": "unitree_g1",
            "body_schema": "compact_tracked_bodies_v1",
            "provenance": (
                f"joint-space transfer of {src_npz.name} (Agibot A3) onto the Unitree G1 by "
                "hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py: joints copied by name "
                "(head dropped), clamped to the G1 URDF limits; root displacement scaled by "
                f"{scale:.4f}, root height offset {z_shift:+.4f} m for floor contact; body poses/velocities "
                "by MuJoCo FK + central differences. Joint-space style only: reach and contact geometry are "
                "the G1's, not the A3's."
            ),
            "source_clip": src_npz.name,
            "joint_order": list(g1_order),
            "tracked_bodies": list(G1_TRACKED_BODIES),
            "anchor_body": "torso_link",
            "root_body": "pelvis",
            "racket_link": "right_wrist_yaw_link",
            "racket_body": "pingpang_red_Link",
            "mount_offset_xyz": list(G1_MOUNT_OFFSET),
            "clamped_joints_rad": {k: round(v, 4) for k, v in clamped_joints.items()},
        }
    )
    return arrays, out_sidecar, report


def _write_sidecar(path: pathlib.Path, sidecar: dict) -> None:
    header = (
        f"# {sidecar['name']}: Unitree G1 reference clip derived from the published A3 clip "
        f"{sidecar['source_clip']} by joint-space transfer (see provenance). Regenerate with\n"
        "#   python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1.py\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(sidecar, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)


def generate(out_dir: pathlib.Path) -> list[dict]:
    kin = G1Kinematics()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for src_name, dst_name in CLIPS:
        arrays, sidecar, report = transfer_clip(MOTIONS_DIR / f"{src_name}.npz", MOTIONS_DIR / f"{src_name}.yaml", kin)
        np.savez(out_dir / f"{dst_name}.npz", **arrays)
        _write_sidecar(out_dir / f"{dst_name}.yaml", sidecar)
        report["clip"] = dst_name
        reports.append(report)
    return reports


def check(out_dir: pathlib.Path = MOTIONS_DIR) -> list[str]:
    """Regenerate into a temp dir; list committed files that differ (arrays compared with allclose)."""
    stale: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        generate(pathlib.Path(tmp))
        for _src, dst in CLIPS:
            for suffix in (".npz", ".yaml"):
                committed = out_dir / f"{dst}{suffix}"
                fresh = pathlib.Path(tmp) / f"{dst}{suffix}"
                if not committed.is_file():
                    stale.append(str(committed))
                    continue
                if suffix == ".yaml":
                    if committed.read_text(encoding="utf-8") != fresh.read_text(encoding="utf-8"):
                        stale.append(str(committed))
                    continue
                a, b = np.load(committed), np.load(fresh)
                if set(a.files) != set(b.files) or any(
                    a[k].shape != b[k].shape or not np.allclose(a[k], b[k], atol=1e-6) for k in a.files
                ):
                    stale.append(str(committed))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, default=MOTIONS_DIR, help="where the G1 clips are written")
    parser.add_argument("--check", action="store_true", help="fail if the committed G1 clips differ from a fresh transfer")
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.out_dir)
        for path in stale:
            print(f"[FAIL] stale generated clip: {path}", file=sys.stderr)
        if stale:
            return 1
        print("[retarget_a3_clip_to_g1] OK: committed G1 clips match a fresh transfer")
        return 0
    for report in generate(args.out_dir):
        print(
            f"[retarget_a3_clip_to_g1] {report['clip']}: {report['frames']} frames @ {report['fps']} Hz, "
            f"root scale {report['root_scale']:.4f}, z shift {report['root_z_shift']:+.4f} m, "
            f"root z {report['root_z_range'][0]:.3f}..{report['root_z_range'][1]:.3f} m, "
            f"max sole float {report['max_sole_float']:.4f} m, clamped joints: "
            f"{ {k: round(v, 3) for k, v in report['clamped_joints'].items()} or 'none'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
