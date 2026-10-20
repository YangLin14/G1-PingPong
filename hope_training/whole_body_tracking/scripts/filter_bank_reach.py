#!/usr/bin/env python3
"""Keep only the bank rows the G1 can strike WITHOUT moving its feet (D23, standing-still drill).

For every row the robot stands in the reference clip's strike-frame pose at the row's station (legs
and pelvis exactly as in the clip, pelvis xy shifted onto the station), and damped least-squares IK
on the waist (3) + right arm (7) moves the ``right_racket`` site to the strike point with the
striking face along ``target_normal``. Joints are clamped to the SOFT limits the training uses
(``soft_joint_pos_limit_factor`` 0.9). A row is kept when the racket ends within ``--pos-tol`` of the
strike point and the face within ``--face-tol-deg`` of the target normal; the rest go to the
stepping policy later.

    python3 hope_training/whole_body_tracking/scripts/filter_bank_reach.py \\
        --src assets/venue_tuple/g1_bh_center_bank_raw.npz --out assets/venue_tuple/g1_bh_center_bank.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from retarget_a3_clip_to_g1 import MOTIONS_DIR, REPO_ROOT, g1_joint_limits, g1_joint_order  # noqa: E402

MJCF = REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang_fist_sole" / "mjcf" / "g1_29dof_pingpang_fist_sole.xml"
IK_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
             "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
SOFT_LIMIT_FACTOR = 0.9
NORMAL_WEIGHT = 0.3   # m per unit of normal error
IK_ITERS = 60
IK_DAMPING = 1e-3


def main(argv: list[str] | None = None) -> int:
    import mujoco

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="bank .npz (its .receipt.json beside it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", default="g1ikfist_hope_backhand_flat063")
    ap.add_argument("--face-sign", type=float, default=-1.0,
                    help="striking face: +1 red (forehand, racket +Y), -1 black (backhand)")
    ap.add_argument("--pos-tol", type=float, default=0.03)
    ap.add_argument("--face-tol-deg", type=float, default=15.0)
    ap.add_argument("--rows", type=int, default=None, help="keep at most this many reachable rows")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.src)
    bank = dict(np.load(src, allow_pickle=False))
    with open(MOTIONS_DIR / f"{args.clip}.yaml", "r", encoding="utf-8") as fh:
        side = yaml.safe_load(fh)
    clip = np.load(MOTIONS_DIR / f"{args.clip}.npz")
    f = int(side["strike_frame"])
    order = g1_joint_order()
    lim = g1_joint_limits()

    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    adr = {n: int(m.jnt_qposadr[m.joint(n).id]) for n in order}
    dof = [int(m.jnt_dofadr[m.joint(n).id]) for n in IK_JOINTS]
    qadr = [adr[n] for n in IK_JOINTS]
    lo, hi = [], []
    for n in IK_JOINTS:
        mid, half = (lim[n][0] + lim[n][1]) / 2, (lim[n][1] - lim[n][0]) / 2 * SOFT_LIMIT_FACTOR
        lo.append(mid - half)
        hi.append(mid + half)
    lo, hi = np.array(lo), np.array(hi)
    site = m.site("right_racket").id
    pelvis = clip["body_pos_w"][:, 0]
    pelvis_shift = pelvis[f, :2] - pelvis[:, :2].mean(axis=0)
    q_clip = clip["joint_pos"][f].astype(np.float64)

    def pose(station_xy):
        d.qpos[:] = 0
        d.qpos[:2] = np.asarray(station_xy) + pelvis_shift
        d.qpos[2] = pelvis[f, 2]
        d.qpos[3:7] = clip["body_quat_w"][f, 0]
        for i, n in enumerate(order):
            d.qpos[adr[n]] = q_clip[i]

    station = bank["station_xy"].astype(np.float64)
    strike = np.concatenate([station + bank["target_position_offset"][:, :2], bank["target_position_offset"][:, 2:]], 1)
    normal = bank["target_normal"].astype(np.float64)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    pos_err = np.zeros(len(station))
    face_err = np.zeros(len(station))
    for r in range(len(station)):
        pose(station[r])
        for _ in range(IK_ITERS):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            x = d.site_xpos[site]
            n_cur = args.face_sign * d.site_xmat[site].reshape(3, 3)[:, 1]
            e = np.concatenate([strike[r] - x, NORMAL_WEIGHT * (normal[r] - n_cur)])
            mujoco.mj_jacSite(m, d, jp, jr, site)
            jn = np.cross(jr[:, dof].T, n_cur).T  # d n / d q
            J = np.vstack([jp[:, dof], NORMAL_WEIGHT * jn])
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), e)
            d.qpos[qadr] = np.clip(d.qpos[qadr] + dq, lo, hi)
        mujoco.mj_kinematics(m, d)
        n_cur = args.face_sign * d.site_xmat[site].reshape(3, 3)[:, 1]
        pos_err[r] = np.linalg.norm(strike[r] - d.site_xpos[site])
        face_err[r] = np.degrees(np.arccos(np.clip(n_cur @ normal[r], -1, 1)))
        if r % 5000 == 0:
            print(f"[filter_bank_reach] {r}/{len(station)}", flush=True)

    ok = (pos_err <= args.pos_tol) & (face_err <= args.face_tol_deg)
    keep = np.where(ok)[0]
    if args.rows is not None:
        keep = keep[: args.rows]
    out = {k: (v[keep] if v.ndim and v.shape[0] == len(station) else v) for k, v in bank.items()}
    if "candidate_id" in out:
        out["candidate_id"] = np.arange(len(keep), dtype=out["candidate_id"].dtype)
    out_path = pathlib.Path(args.out)
    np.savez(out_path, **out)
    receipt = json.loads(src.with_name(src.stem + ".receipt.json").read_text(encoding="utf-8"))
    off = out["target_position_offset"]
    receipt["reach_filter"] = {
        "source_bank": src.name,
        "clip": args.clip,
        "strike_frame": f,
        "model": MJCF.name,
        "ik_joints": list(IK_JOINTS),
        "soft_limit_factor": SOFT_LIMIT_FACTOR,
        "pos_tol_m": args.pos_tol,
        "face_tol_deg": args.face_tol_deg,
        "reachable_fraction": float(ok.mean()),
        "kept_rows": int(len(keep)),
        "pos_err_m_median_all": float(np.median(pos_err)),
        "face_err_deg_median_all": float(np.median(face_err)),
        "kept_strike_offset_ranges": {a: [float(off[:, i].min()), float(off[:, i].max())] for i, a in enumerate("xyz")},
    }
    out_path.with_name(out_path.stem + ".receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(receipt["reach_filter"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
