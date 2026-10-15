#!/usr/bin/env python3
# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Keypoint / IK retarget of an Agibot A3 reference clip onto the Unitree G1 (host, MuJoCo).

The joint-space transfer (``retarget_a3_clip_to_g1.py``) copies joint angles, so on the ~30 %
shorter G1 the racket follows a different path than the A3's. This script keeps the A3 arm
*shape* instead: for every frame it takes the A3 elbow, wrist and racket-centre positions and the
blade-face normal, expresses them in the torso frame, scales each arm segment by the G1/A3 segment
length ratio about the G1's own shoulder, and solves the seven G1 arm joints by damped
least-squares IK (MuJoCo Jacobians) for both arms. Legs, waist and the root come from the
joint-space transfer (same floor fit), so the stride is unchanged.

Targets per arm (torso frame): elbow (weight 0.3), wrist_yaw link (0.3) and, on the racket arm,
the ``right_racket`` site position (1.0) and its blade plane (0.5; the face sign is free, the
backhand strikes with the black face). Every frame is solved from
two warm starts, the joint-space transfer pose of that frame and the previous frame's solution, and
the lower weighted residual wins (the previous-frame start alone drifts into a wrong local minimum
during the fast swing). Joints are clamped to the URDF limits and the arm tracks get a 5-tap moving
average before the final FK + finite differences.

Outputs ``g1ik_hope_{forehand,backhand}.{npz,yaml}`` in the compact 14-body schema (identical layout
to the joint-space clips); the sidecar records the strike-frame tracking error. Regenerate /
check from the repo root::

    python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py [--check]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from retarget_a3_clip_to_g1 import (  # noqa: E402
    CLIPS as JS_CLIPS,
    G1_MOUNT_OFFSET,
    G1_TRACKED_BODIES,
    G1Kinematics,
    MOTIONS_DIR,
    REPO_ROOT,
    _angular_velocity,
    _central_diff,
    g1_joint_limits,
    g1_joint_order,
    transfer_clip,
)

A3_MJCF = (
    REPO_ROOT / "a3_deploy" / "A3_MuJoCo_Sim" / "aimrt_mujoco_sim" / "src" / "models" / "bin" / "cfg" / "model"
    / "a3_pingpong" / "a3_pingpong.xml"
)
CLIPS = (("hope_forehand", "g1ik_hope_forehand"), ("hope_backhand", "g1ik_hope_backhand"))
#: --mount fist (D20): the same retarget on the fist-grip G1 (racket across the closed fist,
#: perpendicular to the forearm) -> g1ikfist_hope_*. The A3 racket centre / face normal stay the
#: targets, so the G1 wrist and forearm turn to put the perpendicular racket where the A3's was.
MOUNT_MODELS = {
    "forearm": (REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang" / "mjcf" / "g1_29dof_pingpang.xml", "g1ik_"),
    "fist": (REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang_fist" / "mjcf" / "g1_29dof_pingpang_fist.xml", "g1ikfist_"),
}
FIST_MOUNT = {"mount_offset_xyz": [0.083, 0.0226, 0.14], "mount_rpy": [0.0, -1.570796, 0.0]}
A3_BODY_OF_JOINT_OFFSET = 1  # complete_articulation_v1: body 0 = pelvis_link, body i+1 = child of joint i
ARM_JOINTS = {
    "right": ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"),
    "left": ("left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
             "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"),
}
# The racket centre and face normal are what the swing is about; the wrist/elbow keep the arm shape
# without over-constraining the (different) blade-in-wrist geometry of the two robots.
WEIGHTS = {"elbow": 0.3, "wrist": 0.3, "racket": 1.0, "normal": 0.5}
#: Racket arm with the fist mount: the A3 wrist position cannot be kept (its racket sat 21 cm along
#: the forearm; the fist racket is across it), so the racket arm follows the racket centre and face
#: normal and keeps only a weak elbow pull for the arm shape. The free arm keeps WEIGHTS.
FIST_RACKET_ARM_WEIGHTS = {"elbow": 0.05, "wrist": 0.0, "racket": 1.0, "normal": 0.5}
IK_ITERS = 40
IK_DAMPING = 1e-3
SMOOTH_TAPS = 5


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# --------------------------------------------------------------------------------------------------
# A3 side: keypoints from the complete-articulation clip
# --------------------------------------------------------------------------------------------------
class A3Keypoints:
    """Torso-frame elbow / wrist / racket-centre / face-normal tracks of both arms of an A3 clip."""

    def __init__(self, data, sidecar: dict):
        order = [str(n) for n in sidecar["joint_order"]]
        self.body_index = {"pelvis_link": 0}
        for i, joint in enumerate(order):
            self.body_index[joint.replace("_joint", "_Link")] = i + A3_BODY_OF_JOINT_OFFSET
        self.body_index["torso_Link"] = order.index("waist_pitch_joint") + A3_BODY_OF_JOINT_OFFSET
        self.pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        self.quat = np.asarray(data["body_quat_w"], dtype=np.float64)
        self.mount = np.asarray(sidecar["mount_offset_xyz"], dtype=np.float64)
        sign = float(sidecar.get("blade_normal_sign", 1.0))
        axis = {"x": 0, "y": 1, "z": 2}[str(sidecar.get("blade_normal_axis", "y"))]
        self.normal_local = np.zeros(3)
        self.normal_local[axis] = sign
        self.joint_order = order
        self.q = np.asarray(data["joint_pos"], dtype=np.float64)

    def validate_against_mjcf(self, frame: int = 0, tol: float = 0.02) -> float:
        """FK the clip's joint angles through the A3 MJCF and compare with the stored body positions."""
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(A3_MJCF))
        d = mujoco.MjData(m)
        d.qpos[:3] = self.pos[frame, 0]
        d.qpos[3:7] = self.quat[frame, 0]
        for j, name in enumerate(self.joint_order):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            d.qpos[m.jnt_qposadr[jid]] = self.q[frame, j]
        mujoco.mj_kinematics(m, d)
        worst = 0.0
        for body in ("torso_Link", "right_shoulder_pitch_Link", "right_elbow_Link", "right_wrist_yaw_Link",
                     "left_elbow_Link", "left_wrist_yaw_Link"):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                raise ValueError(f"A3 MJCF has no body {body}")
            err = float(np.linalg.norm(d.xpos[bid] - self.pos[frame, self.body_index[body]]))
            worst = max(worst, err)
        if worst > tol:
            raise ValueError(f"A3 clip body order does not match the assumed BFS layout (FK error {worst:.3f} m)")
        return worst

    def torso_frame(self, f: int):
        i = self.body_index["torso_Link"]
        return self.pos[f, i], _quat_to_mat(self.quat[f, i])

    def arm(self, f: int, side: str) -> dict[str, np.ndarray]:
        """Torso-frame keypoints of one arm at frame f."""
        t_pos, t_rot = self.torso_frame(f)
        to_torso = lambda p: t_rot.T @ (p - t_pos)  # noqa: E731
        sh = self.pos[f, self.body_index[f"{side}_shoulder_pitch_Link"]]
        el = self.pos[f, self.body_index[f"{side}_elbow_Link"]]
        wr_i = self.body_index[f"{side}_wrist_yaw_Link"]
        wr = self.pos[f, wr_i]
        out = {"shoulder": to_torso(sh), "elbow": to_torso(el), "wrist": to_torso(wr)}
        if side == "right":
            r_wr = _quat_to_mat(self.quat[f, wr_i])
            out["racket"] = to_torso(wr + r_wr @ self.mount)
            out["normal"] = t_rot.T @ (r_wr @ self.normal_local)
        return out


# --------------------------------------------------------------------------------------------------
# G1 side: arm IK on the racket-equipped model
# --------------------------------------------------------------------------------------------------
class G1ArmIK:
    def __init__(self, kin: G1Kinematics):
        self.kin = kin
        self.mj, self.m, self.d = kin.mj, kin.model, kin.data
        mj = self.mj
        self.body = {n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, n) for n in (
            "torso_link", "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link")}
        if min(self.body.values()) < 0:
            raise ValueError("G1 MJCF is missing an arm body")
        self.site = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_SITE, "right_racket")
        if self.site < 0:
            raise ValueError("G1 MJCF has no right_racket site")
        limits = g1_joint_limits()
        self.dofs, self.qadr, self.lo, self.hi = {}, {}, {}, {}
        for side, joints in ARM_JOINTS.items():
            ids = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_JOINT, j) for j in joints]
            self.dofs[side] = np.array([int(self.m.jnt_dofadr[i]) for i in ids])
            self.qadr[side] = np.array([int(self.m.jnt_qposadr[i]) for i in ids])
            self.lo[side] = np.array([limits[j][0] for j in joints])
            self.hi[side] = np.array([limits[j][1] for j in joints])
        self._jacp = np.zeros((3, self.m.nv))
        self._jacr = np.zeros((3, self.m.nv))

    def _fk(self):
        self.mj.mj_kinematics(self.m, self.d)

    def torso_frame(self):
        b = self.body["torso_link"]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def segment_lengths(self, side: str) -> tuple[float, float]:
        sh, el, wr = (self.d.xpos[self.body[f"{side}_{n}_link"]] for n in ("shoulder_pitch", "elbow", "wrist_yaw"))
        return float(np.linalg.norm(el - sh)), float(np.linalg.norm(wr - el))

    def solve(self, side: str, targets_w: dict[str, np.ndarray], q_init: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Damped least-squares IK of one arm toward world-frame targets; returns (q_arm, errors)."""
        mj, m, d = self.mj, self.m, self.d
        dofs, qadr = self.dofs[side], self.qadr[side]
        q = np.clip(q_init.copy(), self.lo[side], self.hi[side])
        elbow_b, wrist_b = self.body[f"{side}_elbow_link"], self.body[f"{side}_wrist_yaw_link"]
        W = self.racket_arm_weights if "racket" in targets_w else WEIGHTS
        for _ in range(IK_ITERS):
            d.qpos[qadr] = q
            self._fk()
            rows, errs = [], []
            for key, bid in (("elbow", elbow_b), ("wrist", wrist_b)):
                mj.mj_jacBody(m, d, self._jacp, self._jacr, bid)
                rows.append(W[key] * self._jacp[:, dofs])
                errs.append(W[key] * (targets_w[key] - d.xpos[bid]))
            if "racket" in targets_w:
                mj.mj_jacSite(m, d, self._jacp, self._jacr, self.site)
                rows.append(W["racket"] * self._jacp[:, dofs])
                errs.append(W["racket"] * (targets_w["racket"] - d.site_xpos[self.site]))
                n = d.site_xmat[self.site].reshape(3, 3)[:, 1]  # racket-local +Y = red face normal
                # Match the blade PLANE: the backhand hits with the black face, so the target normal
                # is taken with whichever sign is closer to the current face normal.
                n_t = targets_w["normal"] if np.dot(n, targets_w["normal"]) >= 0.0 else -targets_w["normal"]
                rows.append(W["normal"] * (-_skew(n) @ self._jacr[:, dofs]))
                errs.append(W["normal"] * (n_t - n))
            J = np.vstack(rows)
            e = np.concatenate(errs)
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(J.shape[0]), e)
            step = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + step, self.lo[side], self.hi[side])
            if np.abs(step).max() < 1e-5:
                break
        d.qpos[qadr] = q
        self._fk()
        errors = {
            "elbow": float(np.linalg.norm(targets_w["elbow"] - d.xpos[elbow_b])),
            "wrist": float(np.linalg.norm(targets_w["wrist"] - d.xpos[wrist_b])),
        }
        if "racket" in targets_w:
            n = d.site_xmat[self.site].reshape(3, 3)[:, 1]
            errors["racket"] = float(np.linalg.norm(targets_w["racket"] - d.site_xpos[self.site]))
            # plane angle (face sign ignored, see solve)
            errors["normal_deg"] = float(np.degrees(np.arccos(np.clip(abs(np.dot(n, targets_w["normal"])), 0.0, 1.0))))
        errors["residual"] = float(np.linalg.norm(e))
        return q, errors

    def solve_best(self, side: str, targets_w: dict[str, np.ndarray], starts) -> tuple[np.ndarray, dict[str, float]]:
        best = None
        for q0 in starts:
            q, err = self.solve(side, targets_w, np.asarray(q0, dtype=np.float64))
            if best is None or err["residual"] < best[1]["residual"]:
                best = (q, err)
        return best


def _moving_average(x: np.ndarray, taps: int) -> np.ndarray:
    if taps <= 1:
        return x
    pad = taps // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], pad, axis=0)], axis=0)
    kernel = np.ones(taps) / taps
    return np.stack([np.convolve(xp[:, j], kernel, mode="valid") for j in range(x.shape[1])], axis=1)


# --------------------------------------------------------------------------------------------------
# retarget
# --------------------------------------------------------------------------------------------------
def retarget_clip(src_npz: pathlib.Path, src_yaml: pathlib.Path, kin: G1Kinematics,
                  prefix: str = "g1ik_") -> tuple[dict, dict, dict]:
    data = np.load(src_npz)
    with open(src_yaml, "r", encoding="utf-8") as fh:
        sidecar = yaml.safe_load(fh)
    a3 = A3Keypoints(data, sidecar)
    a3_fk_err = a3.validate_against_mjcf()
    base_arrays, base_sidecar, base_report = transfer_clip(src_npz, src_yaml, kin)  # legs/waist/root + warm start
    g1_order = g1_joint_order()
    g1_index = {n: i for i, n in enumerate(g1_order)}
    ik = G1ArmIK(kin)
    ik.racket_arm_weights = FIST_RACKET_ARM_WEIGHTS if prefix == "g1ikfist_" else WEIGHTS
    frames = int(base_arrays["joint_pos"].shape[0])
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    q = np.asarray(base_arrays["joint_pos"], dtype=np.float64).copy()
    q_js = q.copy()  # joint-space transfer: warm start of every frame
    root_pos = np.asarray(base_arrays["body_pos_w"][:, 0, :], dtype=np.float64)
    root_quat = np.asarray(base_arrays["body_quat_w"][:, 0, :], dtype=np.float64)

    # Segment-length ratios from the stand poses (A3 frame 0 keypoints, G1 FK of the stand frame 0).
    kin.pose(root_pos[0], root_quat[0], {n: float(q[0, g1_index[n]]) for n in g1_order})
    ratios = {}
    for side in ("right", "left"):
        kp = a3.arm(0, side)
        a3_upper = float(np.linalg.norm(kp["elbow"] - kp["shoulder"]))
        a3_fore = float(np.linalg.norm(kp["wrist"] - kp["elbow"]))
        g1_upper, g1_fore = ik.segment_lengths(side)
        ratios[side] = (g1_upper / a3_upper, g1_fore / a3_fore)

    errors = {"right": [], "left": []}
    for f in range(frames):
        joints = {n: float(q[f, g1_index[n]]) for n in g1_order}
        kin.pose(root_pos[f], root_quat[f], joints)  # legs/waist/root of this frame -> torso frame
        t_pos, t_rot = ik.torso_frame()
        for side in ("right", "left"):
            kp = a3.arm(f, side)
            up, fo = ratios[side]
            g1_sh = t_rot.T @ (ik.d.xpos[ik.body[f"{side}_shoulder_pitch_link"]] - t_pos)
            elbow_t = g1_sh + (kp["elbow"] - kp["shoulder"]) * up
            wrist_t = elbow_t + (kp["wrist"] - kp["elbow"]) * fo
            targets = {"elbow": t_pos + t_rot @ elbow_t, "wrist": t_pos + t_rot @ wrist_t}
            if side == "right":
                targets["racket"] = t_pos + t_rot @ (wrist_t + (kp["racket"] - kp["wrist"]))
                targets["normal"] = t_rot @ kp["normal"]
            cols = [g1_index[j] for j in ARM_JOINTS[side]]
            starts = [q_js[f, cols]] + ([q[f - 1, cols]] if f > 0 else [])
            q_arm, err = ik.solve_best(side, targets, starts)
            q[f, cols] = q_arm
            errors[side].append(err)

    arm_cols = [g1_index[j] for side in ARM_JOINTS for j in ARM_JOINTS[side]]
    q[:, arm_cols] = _moving_average(q[:, arm_cols], SMOOTH_TAPS)
    limits = g1_joint_limits()
    q = np.clip(q, [limits[n][0] for n in g1_order], [limits[n][1] for n in g1_order])

    body_pos = np.empty((frames, len(G1_TRACKED_BODIES), 3))
    body_quat = np.empty((frames, len(G1_TRACKED_BODIES), 4))
    for f in range(frames):
        body_pos[f], body_quat[f], _ = kin.pose(root_pos[f], root_quat[f], {n: float(q[f, g1_index[n]]) for n in g1_order})
    arrays = {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": q.astype(np.float32),
        "joint_vel": _central_diff(q, dt).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _central_diff(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": _angular_velocity(body_quat.copy(), dt).astype(np.float32),
    }
    strike = int(sidecar["strike_frame"])
    summary = {
        side: {k: round(float(np.mean([e[k] for e in errs])), 4) for k in errs[0] if k != "residual"}
        for side, errs in errors.items()
    }
    strike_err = {k: round(float(v), 4) for k, v in errors["right"][strike].items() if k != "residual"}
    report = {
        "frames": frames, "fps": fps, "a3_fk_check_m": round(a3_fk_err, 4), "segment_ratios": ratios,
        "mean_errors": summary, "strike_frame_errors": strike_err, "root_z_shift": base_report["root_z_shift"],
    }
    out_sidecar = dict(base_sidecar)
    out_sidecar.update({
        "name": src_yaml.stem.replace("hope_", f"{prefix}hope_"),
        "provenance": (
            f"keypoint/IK retarget of {src_npz.name} (Agibot A3) onto the Unitree G1 by "
            "hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py: legs, waist and root from the "
            "joint-space transfer; both arms solved per frame by damped least-squares IK (MuJoCo) toward the A3 "
            "elbow / wrist / racket-centre positions and blade normal expressed in the torso frame and scaled by "
            f"the G1/A3 segment-length ratios (right upper {ratios['right'][0]:.3f}, forearm {ratios['right'][1]:.3f}). "
            "Preserves the A3 swing shape at G1 scale; the racket-target command, not the clip, defines contact."
        ),
        "ik_mean_errors_m": summary,
        "ik_strike_frame_errors": strike_err,
    })
    return arrays, out_sidecar, report


def _write_sidecar(path: pathlib.Path, sidecar: dict) -> None:
    header = (
        f"# {sidecar['name']}: Unitree G1 reference clip derived from the published A3 clip "
        f"{sidecar['source_clip']} by keypoint/IK retarget (see provenance). Regenerate with\n"
        "#   python3 hope_training/whole_body_tracking/scripts/retarget_a3_clip_to_g1_ik.py\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(sidecar, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)


def generate(out_dir: pathlib.Path, mount: str = "forearm") -> list[dict]:
    mjcf, prefix = MOUNT_MODELS[mount]
    kin = G1Kinematics(mjcf)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for src_name, dst_name in CLIPS:
        dst_name = dst_name.replace("g1ik_", prefix)
        arrays, sidecar, report = retarget_clip(MOTIONS_DIR / f"{src_name}.npz", MOTIONS_DIR / f"{src_name}.yaml", kin,
                                                prefix)
        if mount == "fist":
            sidecar.update(FIST_MOUNT)
        np.savez(out_dir / f"{dst_name}.npz", **arrays)
        _write_sidecar(out_dir / f"{dst_name}.yaml", sidecar)
        report["clip"] = dst_name
        reports.append(report)
    return reports


def check(out_dir: pathlib.Path = MOTIONS_DIR) -> list[str]:
    stale: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        generate(pathlib.Path(tmp))
        for _src, dst in CLIPS:
            for suffix in (".npz", ".yaml"):
                committed, fresh = out_dir / f"{dst}{suffix}", pathlib.Path(tmp) / f"{dst}{suffix}"
                if not committed.is_file():
                    stale.append(str(committed))
                    continue
                if suffix == ".yaml":
                    if committed.read_text(encoding="utf-8") != fresh.read_text(encoding="utf-8"):
                        stale.append(str(committed))
                    continue
                a, b = np.load(committed), np.load(fresh)
                if set(a.files) != set(b.files) or any(
                    a[k].shape != b[k].shape or not np.allclose(a[k], b[k], atol=1e-5) for k in a.files
                ):
                    stale.append(str(committed))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, default=MOTIONS_DIR)
    parser.add_argument("--check", action="store_true", help="fail if the committed IK clips differ from a fresh retarget")
    parser.add_argument("--mount", choices=sorted(MOUNT_MODELS), default="forearm",
                        help="racket mount of the G1 model (fist -> g1ikfist_hope_* clips, D20)")
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.out_dir)
        for path in stale:
            print(f"[FAIL] stale generated clip: {path}", file=sys.stderr)
        if stale:
            return 1
        print("[retarget_a3_clip_to_g1_ik] OK: committed IK clips match a fresh retarget")
        return 0
    for r in generate(args.out_dir, args.mount):
        print(
            f"[retarget_a3_clip_to_g1_ik] {r['clip']}: {r['frames']} frames @ {r['fps']} Hz, A3 FK check {r['a3_fk_check_m']} m, "
            f"segment ratios {{right: {tuple(round(v, 3) for v in r['segment_ratios']['right'])}, "
            f"left: {tuple(round(v, 3) for v in r['segment_ratios']['left'])}}}, mean errors {r['mean_errors']}, "
            f"strike-frame errors {r['strike_frame_errors']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
