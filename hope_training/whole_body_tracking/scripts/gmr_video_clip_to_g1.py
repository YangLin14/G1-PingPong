#!/usr/bin/env python3
"""Convert a GMR-retargeted human swing (video -> GVHMR -> GMR) into a HOPE G1 reference clip.

This is the HITTER reference pipeline (arXiv 2508.21043: GVHMR + GMR to the G1) feeding the
G1 motion schema used by ``retarget_a3_clip_to_g1.py``: ``joint_pos``/``joint_vel`` in the
canonical G1 Isaac order (``joint_order_unitree_g1.yaml``), the 14 ``G1_TRACKED_BODIES`` by MuJoCo
FK on the racket-equipped G1, and a sidecar YAML with the strike frame.

Steps:
  1. load the GMR pickle (``root_pos``, ``root_rot`` xyzw, ``dof_pos`` in the GMR G1 MJCF order),
     trim to ``--start``/``--end`` (source frames);
  2. resample to ``--fps`` (joints/root linear, root orientation slerp);
  3. heading/position normalization: frame 0 pelvis yaw -> 0 (facing +x, toward the table) and
     pelvis xy -> origin, like the published clips;
  4. reorder to the Isaac joint order, clamp to the G1 URDF limits;
  5. optional loop closure: blend the last ``--blend-end-s`` seconds back to frame 0 so the clip
     wraps without a jump (HOPE rallies replay the clip continuously);
  6. FK for the tracked bodies, lowest sole on the floor, velocities by central differences.

The strike frame is ``--strike-frame`` (output frame index) or, if omitted, the frame of peak
racket-centre speed; the top candidates are printed so the choice can be checked against the video.

  python3 hope_training/whole_body_tracking/scripts/gmr_video_clip_to_g1.py \\
      --gmr-pkl forehand_gmr.pkl --gmr-xml <GMR>/assets/unitree_g1/g1_mocap_29dof.xml \\
      --name g1vid_forehand --start 12 --end 80
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import retarget_a3_clip_to_g1 as base  # noqa: E402  (G1 FK, joint order/limits, finite differences)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation of two wxyz quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _rot_z(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    out = v.copy()
    out[..., 0] = c * v[..., 0] - s * v[..., 1]
    out[..., 1] = s * v[..., 0] + c * v[..., 1]
    return out


def gmr_joint_names(gmr_xml: str) -> list[str]:
    """Hinge/slide joints of the GMR robot MJCF in qpos order (= GMR ``dof_pos`` columns).

    MuJoCo lays out qpos in depth-first body order, i.e. XML document order, so the names are read
    from the XML directly (no mesh loading, no compile).
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(gmr_xml).getroot()
    names = []
    for el in root.find("worldbody").iter():
        if el.tag == "joint" and el.get("type", "hinge") in ("hinge", "slide"):
            names.append(el.get("name"))
    return names


def resample(root_pos, root_quat, q, src_fps: float, dst_fps: float):
    n = root_pos.shape[0]
    t_src = np.arange(n) / src_fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_fps)
    pos = np.stack([np.interp(t_dst, t_src, root_pos[:, k]) for k in range(3)], axis=1)
    joints = np.stack([np.interp(t_dst, t_src, q[:, k]) for k in range(q.shape[1])], axis=1)
    quat = np.empty((len(t_dst), 4))
    for i, t in enumerate(t_dst):
        f = min(t * src_fps, n - 1.0)
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, n - 1)
        quat[i] = _slerp(root_quat[i0], root_quat[i1], f - i0)
    return pos, _quat_normalize(quat), joints


def blend_to_start(root_pos, root_quat, q, frames: int):
    """Blend the last ``frames`` frames toward frame 0 (smoothstep weights) for a seamless wrap."""
    n = q.shape[0]
    frames = min(frames, n - 1)
    for k in range(frames):
        i = n - frames + k
        s = (k + 1) / frames
        w = s * s * (3 - 2 * s)
        q[i] = (1 - w) * q[i] + w * q[0]
        root_pos[i] = (1 - w) * root_pos[i] + w * root_pos[0]
        root_quat[i] = _slerp(root_quat[i], root_quat[0], w)
    return root_pos, root_quat, q


def racket_centre(kin: "base.G1Kinematics", root_pos, root_quat, joints: dict[str, float]) -> np.ndarray:
    import mujoco

    kin.pose(root_pos, root_quat, joints)
    bid = mujoco.mj_name2id(kin.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    R = kin.data.xmat[bid].reshape(3, 3)
    return kin.data.xpos[bid] + R @ np.asarray(base.G1_MOUNT_OFFSET)


def render(path: str, root_pos, root_quat, q, order: list[str], fps: int, strike: int) -> None:
    """Offscreen MP4 of the clip on the racket-equipped G1 (MUJOCO_GL=egl on headless nodes)."""
    import imageio
    import mujoco

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    kin = base.G1Kinematics()
    renderer = mujoco.Renderer(kin.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.8, 135.0, -15.0
    with imageio.get_writer(path, fps=fps) as w:
        for f in range(q.shape[0]):
            kin.pose(root_pos[f], root_quat[f], {n: float(q[f, j]) for j, n in enumerate(order)})
            cam.lookat[:] = [root_pos[f, 0], root_pos[f, 1], 0.8]
            renderer.update_scene(kin.data, cam)
            frame = renderer.render()
            if f == strike:  # flag the strike frame with a red border
                frame = frame.copy()
                frame[:8], frame[-8:], frame[:, :8], frame[:, -8:] = [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]
            w.append_data(frame)


def convert(args) -> tuple[dict, dict, dict]:
    with open(args.gmr_pkl, "rb") as fh:
        gmr = pickle.load(fh)
    src_fps = float(gmr["fps"])
    root_pos = np.asarray(gmr["root_pos"], dtype=np.float64)
    root_quat = np.asarray(gmr["root_rot"], dtype=np.float64)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    dof = np.asarray(gmr["dof_pos"], dtype=np.float64)
    end = dof.shape[0] if args.end is None else args.end
    root_pos, root_quat, dof = root_pos[args.start:end], root_quat[args.start:end], dof[args.start:end]
    if dof.shape[0] < 10:
        raise ValueError(f"only {dof.shape[0]} source frames after trimming")

    names = gmr_joint_names(args.gmr_xml)
    if len(names) != dof.shape[1]:
        raise ValueError(f"GMR xml lists {len(names)} joints, dof_pos has {dof.shape[1]} columns")
    order = base.g1_joint_order()
    missing = sorted(set(order) - set(names))
    if missing:
        raise ValueError(f"GMR robot has no column for G1 joints {missing}")
    col = {n: i for i, n in enumerate(names)}
    q = dof[:, [col[n] for n in order]]

    root_pos, root_quat, q = resample(root_pos, _quat_normalize(root_quat), q, src_fps, args.fps)

    yaw0 = _yaw(root_quat[0])
    undo = _yaw_quat(-yaw0)
    root_quat = _quat_normalize(np.stack([base._quat_mul(undo, r) for r in root_quat]))
    root_pos[:, :2] -= root_pos[0, :2]
    root_pos = _rot_z(root_pos, -yaw0)

    limits = base.g1_joint_limits()
    lo = np.array([limits[n][0] for n in order])
    hi = np.array([limits[n][1] for n in order])
    q_clamped = np.clip(q, lo, hi)
    clamp = np.abs(q_clamped - q).max(axis=0)
    clamped = {order[j]: round(float(clamp[j]), 4) for j in range(len(order)) if clamp[j] > 1e-6}

    wrap_gap_before = float(np.abs(q_clamped[-1] - q_clamped[0]).max())
    blend_frames = int(round(args.blend_end_s * args.fps))
    if blend_frames > 0:
        root_pos, root_quat, q_clamped = blend_to_start(root_pos, root_quat, q_clamped, blend_frames)
    wrap_gap_after = float(np.abs(q_clamped[-1] - q_clamped[0]).max())

    kin = base.G1Kinematics()
    frames = q_clamped.shape[0]
    dt = 1.0 / args.fps
    body_pos = np.empty((frames, len(base.G1_TRACKED_BODIES), 3))
    body_quat = np.empty((frames, len(base.G1_TRACKED_BODIES), 4))
    sole = np.empty(frames)
    racket = np.empty((frames, 3))
    for f in range(frames):
        joints = {n: float(q_clamped[f, j]) for j, n in enumerate(order)}
        body_pos[f], body_quat[f], sole[f] = kin.pose(root_pos[f], root_quat[f], joints)
        racket[f] = racket_centre(kin, root_pos[f], root_quat[f], joints)
    z_shift = -float(sole.min())
    root_pos[:, 2] += z_shift
    body_pos[:, :, 2] += z_shift
    racket[:, 2] += z_shift

    racket_speed = np.linalg.norm(base._central_diff(racket, dt), axis=-1)
    candidates = [int(i) for i in np.argsort(-racket_speed)[:5]]
    strike = int(args.strike_frame) if args.strike_frame is not None else candidates[0]
    if not 0 < strike < frames - 1:
        raise ValueError(f"strike frame {strike} outside (0, {frames - 1})")

    if args.render:
        render(args.render, root_pos, root_quat, q_clamped, order, args.fps, strike)

    arrays = {
        "fps": np.array([args.fps], dtype=np.int64),
        "joint_pos": q_clamped.astype(np.float32),
        "joint_vel": base._central_diff(q_clamped, dt).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": base._central_diff(body_pos, dt).astype(np.float32),
        "body_ang_vel_w": base._angular_velocity(body_quat.copy(), dt).astype(np.float32),
    }
    sidecar = {
        "name": args.name,
        "swing_side": 1 if args.side == "forehand" else -1,
        "fps": int(args.fps),
        "frame_count": frames,
        "frame_time_s": dt,
        "duration_s": round(frames * dt, 4),
        "body_schema": "compact_tracked_bodies_v1",
        "provenance": (
            f"human video -> GVHMR -> GMR (unitree_g1) -> {pathlib.Path(__file__).name}: source frames "
            f"[{args.start}, {end}) of {pathlib.Path(args.gmr_pkl).name} at {src_fps:g} Hz resampled to "
            f"{args.fps} Hz, frame-0 heading/xy normalized, clamped to the G1 URDF limits, last "
            f"{args.blend_end_s:g} s blended to frame 0, body poses by MuJoCo FK on the racket-equipped G1."
        ),
        "source_clip": pathlib.Path(args.gmr_pkl).name,
        "strike_frame": strike,
        "strike_phase": round(strike / (frames - 1), 8),
        "strike_time_s": round(strike * dt, 4),
        "recover_end_frame": frames - 1,
        "joint_order": list(order),
        "tracked_bodies": list(base.G1_TRACKED_BODIES),
        "anchor_body": "torso_link",
        "root_body": "pelvis",
        "racket_link": "right_wrist_yaw_link",
        "racket_body": "pingpang_red_Link",
        "mount_offset_xyz": list(base.G1_MOUNT_OFFSET),
        "blade_normal_axis": "y",
        "blade_normal_sign": 1.0 if args.side == "forehand" else -1.0,
        "robot": "unitree_g1",
        "clamped_joints_rad": clamped,
    }
    report = {
        "frames": frames,
        "strike_frame": strike,
        "strike_candidates": [(c, round(float(racket_speed[c]), 2)) for c in candidates],
        "racket_at_strike": [round(float(v), 3) for v in racket[strike]],
        "peak_racket_speed": round(float(racket_speed.max()), 2),
        "wrap_gap_rad": (round(wrap_gap_before, 3), round(wrap_gap_after, 3)),
        "root_z_range": (round(float(root_pos[:, 2].min()), 3), round(float(root_pos[:, 2].max()), 3)),
        "clamped_joints": clamped,
    }
    return arrays, sidecar, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmr-pkl", required=True, help="GMR scripts/gvhmr_to_robot.py --save_path output")
    p.add_argument("--gmr-xml", required=True, help="GMR's unitree_g1 MJCF (defines the dof_pos column order)")
    p.add_argument("--name", required=True, help="clip name, e.g. g1vid_forehand")
    p.add_argument("--side", choices=("forehand", "backhand"), default="forehand")
    p.add_argument("--start", type=int, default=0, help="first source frame (ready pose)")
    p.add_argument("--end", type=int, default=None, help="end source frame, exclusive (back at ready)")
    p.add_argument("--strike-frame", type=int, default=None, help="output frame of ball contact (default: peak racket speed)")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--blend-end-s", type=float, default=0.3, help="loop-closure blend window (0 disables)")
    p.add_argument("--out-dir", default=str(base.MOTIONS_DIR))
    p.add_argument("--render", default=None, help="also write an MP4 preview (strike frame has a red border)")
    args = p.parse_args(argv)

    arrays, sidecar, report = convert(args)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.name}.npz", **arrays)
    with open(out / f"{args.name}.yaml", "w", encoding="utf-8") as fh:
        fh.write(f"# {args.name}: Unitree G1 reference clip from human video (see provenance).\n")
        yaml.safe_dump(sidecar, fh, sort_keys=False, default_flow_style=None, allow_unicode=True)
    print(f"[gmr_video_clip_to_g1] wrote {out / args.name}.npz/.yaml")
    for k, v in report.items():
        print(f"[gmr_video_clip_to_g1]   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
