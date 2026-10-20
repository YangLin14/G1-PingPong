#!/usr/bin/env python3
"""Re-solve a G1 reference clip's legs so both soles stay flat on the floor and the pelvis sits lower.

Triton Droids D23 (backhand, standing still). The IK clips (``retarget_a3_clip_to_g1_ik.py``) take
the legs from the A3 joint-space transfer, so their feet are not flat: in ``g1ikfist_hope_backhand``
the left foot rolls 10-12 deg onto its edge and the right sole dips up to 3 mm below the floor. A
standing-still policy that imitates that clip is taught to stand on one edge of a foot.

For every frame this script keeps the root orientation, the root's horizontal path, the waist and
both arms, lowers the root by a constant ``--pelvis-z`` - mean(root z), and solves the six joints
of each leg by damped least-squares IK (MuJoCo Jacobians) so that each ``*_ankle_roll_link`` sits
at ONE fixed pose for the whole clip:

* x, y: the foot's mean position in the source clip (same stance);
* yaw: the foot's mean yaw in the source clip (same toe-out);
* roll = pitch = 0 and the sole bottom exactly on the floor (ankle origin at ``SOLE_BOTTOM`` above it,
  the sole box of ``unitree/URDF/g1_29dof_pingpang_fist_sole``).

Then FK and finite differences give the body tracks and velocities, as in the IK retarget. The report
lists the remaining foot error, the lowest/highest sole corner, joint-limit margins and how far the
whole-body centre of mass leaves the support polygon (the two sole boxes).

    python3 hope_training/whole_body_tracking/scripts/plant_clip_feet.py \\
        --src g1ikfist_hope_backhand --dst g1ikfist_hope_backhand_flat063 --pelvis-z 0.63 --stance-width 0.32
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from retarget_a3_clip_to_g1 import (  # noqa: E402
    G1_TRACKED_BODIES,
    MOTIONS_DIR,
    REPO_ROOT,
    _angular_velocity,
    _central_diff,
    g1_joint_limits,
    g1_joint_order,
)

MJCF = REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang_fist_sole" / "mjcf" / "g1_29dof_pingpang_fist_sole.xml"
FEET = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}
LEG_JOINTS = ("hip_pitch_joint", "hip_roll_joint", "hip_yaw_joint", "knee_joint", "ankle_pitch_joint", "ankle_roll_joint")
#: Sole box in the ankle_roll_link frame (unitree/scripts/build_g1_pingpang_models.py SOLE_*).
SOLE_X = (-0.0658, 0.1424)
SOLE_HALF_Y = 0.0378
SOLE_BOTTOM = 0.0354
SOLE_CORNERS = np.array([[x, y, -SOLE_BOTTOM] for x in SOLE_X for y in (-SOLE_HALF_Y, SOLE_HALF_Y)])
IK_ITERS = 60
IK_DAMPING = 1e-4
ORI_WEIGHT = 0.3  # m per rad: 1 deg of foot tilt costs like 5 mm of position


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _yaw_of(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Legs:
    def __init__(self, mjcf: pathlib.Path):
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        self.adr = {}
        self.dof = {}
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.adr[name] = int(self.m.jnt_qposadr[j])
            self.dof[name] = int(self.m.jnt_dofadr[j])
        self.body = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n) for n in G1_TRACKED_BODIES}
        lim = g1_joint_limits()
        self.lim = lim

    def set(self, root_pos, root_quat, joints: dict[str, float]) -> None:
        d = self.d
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        for n, v in joints.items():
            d.qpos[self.adr[n]] = v
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_comPos(self.m, d)

    def foot_error(self, side: str, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        b = self.body[FEET[side]]
        ep = target_pos - self.d.xpos[b]
        eq = np.zeros(3)
        self.mj.mju_subQuat(eq, target_quat, self.d.xquat[b])  # body-frame rotation error
        er = self.d.xmat[b].reshape(3, 3) @ eq                  # -> world frame
        return np.concatenate([ep, ORI_WEIGHT * er])

    def solve(self, side: str, root_pos, root_quat, joints: dict[str, float], target_pos, target_quat):
        names = [f"{side}_{j}" for j in LEG_JOINTS]
        cols = [self.dof[n] for n in names]
        q = np.array([joints[n] for n in names])
        lo = np.array([self.lim[n][0] for n in names])
        hi = np.array([self.lim[n][1] for n in names])
        b = self.body[FEET[side]]
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(IK_ITERS):
            joints.update(zip(names, q))
            self.set(root_pos, root_quat, joints)
            e = self.foot_error(side, target_pos, target_quat)
            if np.linalg.norm(e) < 1e-6:
                break
            self.mj.mj_jacBody(self.m, self.d, jp, jr, b)
            J = np.vstack([jp[:, cols], ORI_WEIGHT * jr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), e)
            q = np.clip(q + dq, lo, hi)
        joints.update(zip(names, q))
        self.set(root_pos, root_quat, joints)
        return q, self.foot_error(side, target_pos, target_quat)

    def com_xy(self) -> np.ndarray:
        return self.d.subtree_com[0][:2].copy()


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _support_margin(points_xy: np.ndarray, com: np.ndarray) -> float:
    """Signed distance of ``com`` inside the convex hull of ``points_xy`` (positive = inside)."""
    pts = points_xy[np.lexsort((points_xy[:, 1], points_xy[:, 0]))]

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross2(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    margin = np.inf
    for a, b in zip(hull, hull[1:] + hull[:1]):
        edge = b - a
        n = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)  # outward for a CCW hull
        margin = min(margin, -float(n @ (com - a)))
    return margin


def plant(src: str, dst: str, pelvis_z: float, stance_width: float | None = None) -> dict:
    data = np.load(MOTIONS_DIR / f"{src}.npz")
    with open(MOTIONS_DIR / f"{src}.yaml", "r", encoding="utf-8") as fh:
        sidecar = yaml.safe_load(fh)
    order = g1_joint_order()
    if [str(n) for n in sidecar["joint_order"]] != order:
        raise ValueError(f"{src}: joint order differs from the G1 policy order")
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    q = np.asarray(data["joint_pos"], dtype=np.float64).copy()
    pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    frames = q.shape[0]
    idx = {n: i for i, n in enumerate(G1_TRACKED_BODIES)}
    root_pos = pos[:, idx["pelvis"]].copy()
    root_quat = quat[:, idx["pelvis"]].copy()
    dz = pelvis_z - float(root_pos[:, 2].mean())
    root_pos[:, 2] += dz

    legs = Legs(MJCF)
    targets = {}
    for side, body in FEET.items():
        p = pos[:, idx[body]]
        yaw = float(np.mean([_yaw_of(quat[f, idx[body]]) for f in range(frames)]))
        targets[side] = (np.array([p[:, 0].mean(), p[:, 1].mean(), SOLE_BOTTOM]), _yaw_quat(yaw))
    if stance_width is not None:
        # centre the feet under the mean pelvis (lateral) at the given width: with the soles flat, each
        # ankle roll equals its shank's sideways lean, and the G1 ankle roll range is only +-15 deg
        mid = float(root_pos[:, 1].mean())
        targets["left"][0][1] = mid + stance_width / 2
        targets["right"][0][1] = mid - stance_width / 2

    foot_err = {"left": [], "right": []}
    for f in range(frames):
        joints = {n: float(q[f, i]) for i, n in enumerate(order)}
        if f > 0:  # warm start the legs from the previous solved frame
            for side in FEET:
                for j in LEG_JOINTS:
                    joints[f"{side}_{j}"] = float(q[f - 1, order.index(f"{side}_{j}")])
        for side in FEET:
            q_leg, e = legs.solve(side, root_pos[f], root_quat[f], joints, *targets[side])
            for j, v in zip(LEG_JOINTS, q_leg):
                q[f, order.index(f"{side}_{j}")] = v
            foot_err[side].append(e)

    body_pos = np.empty((frames, len(G1_TRACKED_BODIES), 3))
    body_quat = np.empty((frames, len(G1_TRACKED_BODIES), 4))
    sole_z, com_margin, limit_margin = [], [], []
    for f in range(frames):
        legs.set(root_pos[f], root_quat[f], {n: float(q[f, i]) for i, n in enumerate(order)})
        d = legs.d
        body_pos[f] = np.stack([d.xpos[legs.body[n]] for n in G1_TRACKED_BODIES])
        body_quat[f] = np.stack([d.xquat[legs.body[n]] for n in G1_TRACKED_BODIES])
        corners = []
        for body in FEET.values():
            b = legs.body[body]
            corners.append(SOLE_CORNERS @ d.xmat[b].reshape(3, 3).T + d.xpos[b])
        corners = np.concatenate(corners)
        sole_z.append(corners[:, 2])
        com_margin.append(_support_margin(corners[:, :2], legs.com_xy()))
        limit_margin.append(min(min(q[f, i] - legs.lim[n][0], legs.lim[n][1] - q[f, i])
                                for i, n in enumerate(order) if n.split("_", 1)[1] in LEG_JOINTS))
    sole_z = np.array(sole_z)
    err = np.array(foot_err["left"] + foot_err["right"])
    report = {
        "frames": frames,
        "pelvis_z_shift_m": round(dz, 4),
        "pelvis_z_m": [round(float(root_pos[:, 2].min()), 4), round(float(root_pos[:, 2].max()), 4)],
        "foot_targets": {s: {"xy": [round(float(v), 4) for v in t[0][:2]], "yaw_deg": round(float(np.degrees(_yaw_of(t[1]))), 2)}
                         for s, t in targets.items()},
        "foot_pos_err_max_mm": round(float(np.abs(err[:, :3]).max()) * 1000, 3),
        "foot_tilt_err_max_deg": round(float(np.degrees(np.abs(err[:, 3:]).max() / ORI_WEIGHT)), 3),
        "sole_corner_z_mm": [round(float(sole_z.min()) * 1000, 3), round(float(sole_z.max()) * 1000, 3)],
        "com_support_margin_m": [round(float(min(com_margin)), 4), round(float(np.median(com_margin)), 4)],
        "leg_joint_limit_margin_rad": round(float(min(limit_margin)), 4),
        "ankle_roll_rad": {s: [round(float(q[:, order.index(f"{s}_ankle_roll_joint")].min()), 4),
                               round(float(q[:, order.index(f"{s}_ankle_roll_joint")].max()), 4)] for s in FEET},
    }
    arrays = {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": q.astype(np.float32),
        "joint_vel": _central_diff(q, dt).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": _central_diff(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": _angular_velocity(body_quat.copy(), dt).astype(np.float32),
    }
    out = dict(sidecar)
    out["name"] = dst
    out["source_clip"] = f"{src}.npz"
    out["provenance"] = (
        f"{src} with both legs re-solved per frame by plant_clip_feet.py (D23): each ankle_roll_link held at "
        f"one pose for the whole clip (mean source x/y and yaw, roll = pitch = 0, sole box bottom on the floor), "
        + (f"feet centred under the mean pelvis {stance_width:.3f} m apart, " if stance_width is not None else "")
        + f"root lowered by {-dz:.3f} m to a mean of {pelvis_z:.3f} m; root orientation and horizontal path, waist "
        f"and arms unchanged. " + str(sidecar.get("provenance", ""))
    )
    out["plant_report"] = report
    np.savez(MOTIONS_DIR / f"{dst}.npz", **arrays)
    with open(MOTIONS_DIR / f"{dst}.yaml", "w", encoding="utf-8") as fh:
        fh.write(f"# {dst}: {src} with flat, fixed feet and a lower pelvis. Regenerate with\n"
                 f"#   python3 hope_training/whole_body_tracking/scripts/plant_clip_feet.py --src {src} --dst {dst} "
                 f"--pelvis-z {pelvis_z}"
                 + (f" --stance-width {stance_width}" if stance_width is not None else "") + "\n")
        yaml.safe_dump(out, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="g1ikfist_hope_backhand")
    ap.add_argument("--dst", default="g1ikfist_hope_backhand_flat063")
    ap.add_argument("--stance-width", type=float, default=None,
                    help="centre the feet under the pelvis this far apart (m); default: the source stance")
    ap.add_argument("--pelvis-z", type=float, default=0.63, help="mean pelvis height of the output clip (m)")
    args = ap.parse_args(argv)
    print(yaml.safe_dump(plant(args.src, args.dst, args.pelvis_z, args.stance_width), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
