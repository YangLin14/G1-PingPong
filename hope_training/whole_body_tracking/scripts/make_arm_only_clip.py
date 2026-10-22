#!/usr/bin/env python3
"""Turn a standing G1 reference clip into an ARM-ONLY swing: pelvis and waist still, right arm re-solved.

Triton Droids D26 (backhand, standing still). In ``g1ikfist_hope_backhand_stand`` the torso turns ~21 deg
during the swing (pelvis yaw 0-8 deg + waist yaw up to 13 deg) and the racket at the strike frame is
~14 cm from the bank's median strike point. The club's ``coordinated-rl-overnight`` backhand returns
balls with the waist and left arm held at the ready pose and only the 7 right-arm joints moving.

For every frame this script

* holds the pelvis at the source frame-0 orientation and the mean source xy/z (no turn, no sway);
* holds the left arm (7) at its source frame-0 angles and the waist (3) at one fixed pose ``--waist``
  (default: 0.15 rad yaw toward the backhand side, 0.25 rad forward lean): with the waist at the source
  frame-0 pose the arm alone reaches only ~27 % of the bank's strike points (IK, 150 rows, 3 cm / 15 deg),
  with this fixed lean and turn ~95 %. The torso does not move during the swing;
* keeps the source right-arm joint angles (the arm's own part of the swing; the torso turn's part is
  dropped) and adds ONE joint-space correction ``dq`` faded in and out with a raised-cosine window
  (``--window`` frames each side of the strike, 1 at the strike frame, 0 outside). ``dq`` comes from
  damped least-squares IK (MuJoCo Jacobians, soft limits 0.9) that puts the ``right_racket`` site on the
  bank's median strike point with the bank's median face normal at the strike frame (bank rows are
  relative to the station; the pelvis stands on the station as in training). The window's slope is 0
  at the strike, so the arm's joint velocities there are the source's.

The legs are then re-solved by ``plant_clip_feet.plant`` (flat fixed soles under the still pelvis), which
also recomputes the body tracks and velocities and writes the final clip.

    python3 hope_training/whole_body_tracking/scripts/make_arm_only_clip.py \\
        --src g1ikfist_hope_backhand_stand --dst g1ikfist_hope_backhand_arm \\
        --bank hope_training/whole_body_tracking/assets/venue_tuple/g1_bh_center_bank.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plant_clip_feet import MJCF, plant  # noqa: E402
from retarget_a3_clip_to_g1 import G1_TRACKED_BODIES, MOTIONS_DIR, g1_joint_limits, g1_joint_order  # noqa: E402

ARM = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
HELD = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint")
SOFT_LIMIT_FACTOR = 0.9
NORMAL_WEIGHT = 0.3  # m per unit of normal error, as filter_bank_reach.py
IK_ITERS = 100
IK_DAMPING = 1e-3


def make(src: str, dst: str, bank_path: str, face_sign: float = -1.0, window: int = 20,
         waist: tuple[float, float, float] = (0.15, 0.0, 0.25)) -> dict:
    import mujoco

    data = np.load(MOTIONS_DIR / f"{src}.npz")
    with open(MOTIONS_DIR / f"{src}.yaml", "r", encoding="utf-8") as fh:
        sidecar = yaml.safe_load(fh)
    order = g1_joint_order()
    lim = g1_joint_limits()
    q_src = np.asarray(data["joint_pos"], dtype=np.float64)
    pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    quat = np.asarray(data["body_quat_w"], dtype=np.float64)
    frames = q_src.shape[0]
    strike = int(sidecar["strike_frame"])
    pelvis = G1_TRACKED_BODIES.index("pelvis")

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    adr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in order}
    dof = [int(m.jnt_dofadr[m.joint(n).id]) for n in ARM]
    qadr = [adr[n] for n in ARM]
    mid = np.array([(lim[n][0] + lim[n][1]) / 2 for n in ARM])
    half = np.array([(lim[n][1] - lim[n][0]) / 2 * SOFT_LIMIT_FACTOR for n in ARM])
    lo, hi = mid - half, mid + half
    site = m.site("right_racket").id

    def pose(root_pos, root_quat, q):
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat
        for i, n in enumerate(order):
            d.qpos[adr[n]] = q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    def racket():
        return d.site_xpos[site].copy(), face_sign * d.site_xmat[site].reshape(3, 3)[:, 1].copy()

    # still pelvis on the station frame (the pelvis xy mean stands on the station, as filter_bank_reach.py
    # and training place it) and the bank's median strike point / face normal
    root_pos = pos[:, pelvis].mean(axis=0)
    root_quat = quat[0, pelvis]
    bank = np.load(bank_path)
    target = np.median(bank["target_position_offset"].astype(np.float64), axis=0)
    target[:2] += root_pos[:2]
    target_normal = np.median(bank["target_normal"].astype(np.float64), axis=0)
    target_normal /= np.linalg.norm(target_normal)

    q = q_src.copy()
    held = [order.index(n) for n in HELD]
    q[:, held] = q_src[0, held]
    for n, v in zip(HELD[:3], waist):
        q[:, order.index(n)] = v
    arm = [order.index(n) for n in ARM]

    # strike-frame IK on the still torso, from the source arm
    pose(root_pos, root_quat, q[strike])
    x0, _ = racket()
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    for _ in range(IK_ITERS):
        x, n_cur = racket()
        e = np.concatenate([target - x, NORMAL_WEIGHT * (target_normal - n_cur)])
        if np.linalg.norm(e) < 1e-6:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        jn = np.cross(jr[:, dof].T, n_cur).T
        J = np.vstack([jp[:, dof], NORMAL_WEIGHT * jn])
        step = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + step, lo, hi)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
    dq = d.qpos[qadr] - q[strike, arm]
    frame = np.arange(frames)
    w = np.where(np.abs(frame - strike) < window, 0.5 * (1 + np.cos(np.pi * (frame - strike) / window)), 0.0)
    q[:, arm] = np.clip(q[:, arm] + w[:, None] * dq, lo, hi)

    path = np.empty((frames, 3))
    for f in range(frames):
        pose(root_pos, root_quat, q[f])
        path[f], _ = racket()

    # intermediate clip: still pelvis, held waist/left arm, new right arm; plant() re-solves the legs
    mid_name = f"{dst}_unplanted"
    body_pos = np.repeat(pos[:1], frames, axis=0)
    body_pos[:, pelvis] = root_pos
    body_quat = np.repeat(quat[:1], frames, axis=0)
    body_quat[:, pelvis] = root_quat
    np.savez(MOTIONS_DIR / f"{mid_name}.npz", fps=data["fps"], joint_pos=q.astype(np.float32),
             joint_vel=np.gradient(q, axis=0).astype(np.float32) * float(np.asarray(data["fps"]).reshape(-1)[0]),
             body_pos_w=body_pos.astype(np.float32), body_quat_w=body_quat.astype(np.float32),
             body_lin_vel_w=np.zeros_like(body_pos, dtype=np.float32), body_ang_vel_w=np.zeros_like(body_pos, dtype=np.float32))
    with open(MOTIONS_DIR / f"{mid_name}.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(sidecar, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)
    plant_report = plant(mid_name, dst, pelvis_z=float(root_pos[2]), stance_width=None)
    for ext in ("npz", "yaml"):
        (MOTIONS_DIR / f"{mid_name}.{ext}").unlink()

    # strike-frame check on the final clip
    out = np.load(MOTIONS_DIR / f"{dst}.npz")
    pose(out["body_pos_w"][strike, pelvis], out["body_quat_w"][strike, pelvis], out["joint_pos"][strike])
    x_strike, n_strike = racket()
    pose(out["body_pos_w"][strike + 1, pelvis], out["body_quat_w"][strike + 1, pelvis], out["joint_pos"][strike + 1])
    x_next, _ = racket()
    pose(out["body_pos_w"][strike - 1, pelvis], out["body_quat_w"][strike - 1, pelvis], out["joint_pos"][strike - 1])
    x_prev, _ = racket()
    fps = float(np.asarray(out["fps"]).reshape(-1)[0])
    report = {
        "window_frames": window,
        "strike_arm_correction_rad": {n: round(float(v), 3) for n, v in zip(ARM, dq)},
        "strike_racket_before_correction_m": [round(float(v), 4) for v in x0],
        "racket_lowest_z_m": round(float(path[:, 2].min()), 4),
        "strike_racket_m": [round(float(v), 4) for v in x_strike],
        "strike_target_m": [round(float(v), 4) for v in target],
        "strike_racket_speed_mps": round(float(np.linalg.norm(x_next - x_prev) * fps / 2), 3),
        "strike_face_vs_bank_median_normal_deg": round(float(np.degrees(np.arccos(np.clip(n_strike @ target_normal, -1, 1)))), 2),
        "waist_rad": {n: round(float(v), 4) for n, v in zip(HELD[:3], waist)},
        "right_arm_range_rad": {n: [round(float(q[:, order.index(n)].min()), 3), round(float(q[:, order.index(n)].max()), 3)] for n in ARM},
    }
    side = yaml.safe_load((MOTIONS_DIR / f"{dst}.yaml").read_text(encoding="utf-8"))
    side["source_clip"] = f"{src}.npz"
    side["provenance"] = (
        f"{src} made arm-only by make_arm_only_clip.py (D26): pelvis held at the source frame-0 orientation and mean "
        f"position, waist fixed at {list(waist)} rad (yaw, roll, pitch), left arm held at source frame 0, source right arm plus one IK correction faded in over "
        f"+-{window} frames so the strike-frame racket sits on the median bank strike point with the median bank face "
        f"normal; legs re-solved by plant_clip_feet.py. " + str(sidecar.get("provenance", ""))
    )
    side["arm_only_report"] = report
    side["plant_report"] = plant_report
    with open(MOTIONS_DIR / f"{dst}.yaml", "w", encoding="utf-8") as fh:
        fh.write(f"# {dst}: arm-only swing from {src}. Regenerate with\n"
                 f"#   python3 hope_training/whole_body_tracking/scripts/make_arm_only_clip.py --src {src} --dst {dst} "
                 f"--bank {bank_path} --window {window} --waist {' '.join(str(v) for v in waist)}\n")
        yaml.safe_dump(side, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="g1ikfist_hope_backhand_stand")
    ap.add_argument("--dst", default="g1ikfist_hope_backhand_arm")
    ap.add_argument("--bank", default="hope_training/whole_body_tracking/assets/venue_tuple/g1_bh_center_bank.npz")
    ap.add_argument("--face-sign", type=float, default=-1.0, help="striking face: -1 black (backhand)")
    ap.add_argument("--waist", type=float, nargs=3, default=(0.15, 0.0, 0.25), metavar=("YAW", "ROLL", "PITCH"),
                    help="fixed waist angles for the whole clip (rad)")
    ap.add_argument("--window", type=int, default=20, help="frames each side of the strike the correction fades over")
    args = ap.parse_args(argv)
    print(yaml.safe_dump(make(args.src, args.dst, args.bank, args.face_sign, args.window, tuple(args.waist)), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
