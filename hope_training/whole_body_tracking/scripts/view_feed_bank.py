#!/usr/bin/env python3
"""Watch the questions of a G1 forehand bank in MuJoCo: feed -> apex strike -> cross-court return.

For each sampled row of ``make_g1_forehand_bank.py``'s bank the ball is replayed with the same
venue spin model the training reward uses (:mod:`virtual_ball`):

  incoming flight (integrated backward from the feed bounce) -> table bounce in the feed zone ->
  rise to the apex = contact point -> outgoing flight -> first bounce in the target zone -> 0.3 s more.

Scene: ITTF table and net, feed zone (orange), training target zone (green), the other two target
presets (faint), the G1 in its stand pose at the row's station, and the racket target (a disc at the
contact point facing along the racket velocity, with a velocity arrow) shown while the ball is near.
The ball leaves a trail. This visualizes the QUESTIONS; it does not run a policy.

Interactive (macOS needs mjpython; from the repo root, quoted because the path has spaces):
  .venv/bin/python .venv/bin/mjpython hope_training/whole_body_tracking/scripts/view_feed_bank.py
Offscreen MP4 (cluster: MUJOCO_GL=egl):
  python hope_training/whole_body_tracking/scripts/view_feed_bank.py --record feed.mp4 --rows 6
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import make_g1_forehand_bank as mk  # noqa: E402
import retarget_a3_clip_to_g1 as base  # noqa: E402

fs, vb = mk.fs, mk.vb
DT = 0.005
TRAIL = 80
# Behind the robot's right shoulder looking down the table: (lookat, (distance, azimuth, elevation)).
CAMERA = ([1.2, 0.0, 0.85], (3.0, 20.0, -24.0))


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64)).reshape(1, 3)


def _fly(p, v, w, prm, steps: int, h: float = DT):
    out = [p.copy()]
    pt, vt, wt = _t(p), _t(v), _t(w)
    for _ in range(steps):
        pt, vt = vb.rk4_step(pt, vt, wt, h, prm)
        out.append(pt[0].numpy().copy())
    return np.array(out), vt[0].numpy()


def row_trajectory(bank: dict, i: int, prm, tprm, scfg) -> dict:
    """Ball positions at DT for one question, plus the index of the contact frame."""
    station_y = float(bank["station_xy"][i, 1])
    contact = bank["target_position_offset"][i].astype(np.float64).copy()
    contact[1] += station_y
    bounce = np.array([*bank["feed_bounce_xy"][i], scfg.table_surface_z + fs.BALL_RADIUS])
    v_pre, w_pre = bank["feed_velocity_pre"][i], bank["feed_spin_pre"][i]

    # incoming: integrate backward from the bounce until the ball is over the far half (or 0.8 s)
    back, _ = _fly(bounce, v_pre, w_pre, prm, steps=160, h=-DT)
    keep = np.where(back[:, 0] <= scfg.net_x + 0.6)[0]
    incoming = back[: keep[-1] + 1][::-1] if len(keep) else back[::-1]

    # bounce, then forward until the apex (vz <= 0) -- the contact point
    up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    v_post, w_post = vb.predict_paddle_contact(_t(v_pre), torch.zeros(1, 3, dtype=torch.float64), up, _t(w_pre), tprm)
    rise = [bounce.copy()]
    pt, vt, wt = _t(bounce), v_post, w_post
    for _ in range(200):
        pt, vt = vb.rk4_step(pt, vt, wt, DT, prm)
        rise.append(pt[0].numpy().copy())
        if float(vt[0, 2]) <= 0.0:
            break
    rise[-1] = contact  # snap the apex to the bank's contact point (same physics, coarser step)

    # outgoing: until the first table bounce, bounce it, then 0.3 s more
    out = [contact.copy()]
    pt, vt, wt = _t(contact), _t(bank["outgoing_velocity"][i]), _t(bank["outgoing_spin"][i])
    surface = scfg.table_surface_z + fs.BALL_RADIUS
    for _ in range(300):
        p_new, v_new = vb.rk4_step(pt, vt, wt, DT, prm)
        if float(p_new[0, 2]) <= surface and float(vt[0, 2]) < 0 and scfg.net_x < float(p_new[0, 0]) < scfg.far_x:
            p_new[0, 2] = surface
            v_new, wt = vb.predict_paddle_contact(v_new, torch.zeros_like(v_new), up, wt, tprm)
            out.append(p_new[0].numpy().copy())
            after, _ = _fly(p_new[0].numpy(), v_new[0].numpy(), wt[0].numpy(), prm, steps=60)
            out.extend(after[1:])
            break
        pt, vt = p_new, v_new
        out.append(pt[0].numpy().copy())
    path = np.concatenate([incoming, np.array(rise[1:]), np.array(out[1:])])
    return {"path": path, "contact_idx": len(incoming) + len(rise) - 2, "station_y": station_y, "contact": contact}


def build_model(scfg):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(base.G1_MJCF))
    wb = spec.worldbody
    z, L, W = scfg.table_surface_z, fs.TABLE_LENGTH, fs.TABLE_WIDTH
    x0 = scfg.table_near_x

    def box(name, pos, size, rgba):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=rgba, contype=0, conaffinity=0)

    box("table", [x0 + L / 2, 0, z - 0.025], [L / 2, W / 2, 0.025], [0.05, 0.22, 0.55, 1])
    box("centre_line", [x0 + L / 2, 0, z + 0.0005], [L / 2, 0.0015, 0.0005], [1, 1, 1, 1])
    box("net", [scfg.net_x, 0, z + fs.NET_HEIGHT / 2], [0.003, W / 2 + 0.15, fs.NET_HEIGHT / 2], [0.9, 0.9, 0.9, 0.5])
    for k, ((ax, ay), rgba) in enumerate([
        (scfg.feed_bounds(), [1.0, 0.55, 0.0, 0.55]),
        (scfg.zone_bounds(), [0.1, 0.9, 0.2, 0.6]),
        (scfg.with_target("center_line").zone_bounds(), [0.1, 0.9, 0.2, 0.18]),
        (scfg.with_target("opp_right_corner").zone_bounds(), [0.1, 0.9, 0.2, 0.18]),
    ]):
        box(f"zone_{k}", [(ax[0] + ax[1]) / 2, (ay[0] + ay[1]) / 2, z + 0.001 + 0.0005 * k],
            [(ax[1] - ax[0]) / 2, (ay[1] - ay[0]) / 2, 0.0005], rgba)
    ball = wb.add_body(name="ball_vis", mocap=True)
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[fs.BALL_RADIUS, 0, 0], rgba=[1, 0.95, 0.85, 1],
                  contype=0, conaffinity=0)
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720  # room for the MP4 renderer
    return model


def add_markers(scn, trail: list[np.ndarray], racket: tuple | None, reset: bool = True):
    """Draw the trail and racket target. ``reset`` clears ``scn`` first (the viewer's user_scn);
    pass False to append to a renderer scene that already holds the model geoms."""
    import mujoco

    if reset:
        scn.ngeom = 0
    first = scn.ngeom
    for j, p in enumerate(trail):
        if scn.ngeom >= scn.maxgeom:
            break
        a = 0.15 + 0.6 * j / max(len(trail), 1)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.009, 0, 0]),
                            p, np.eye(3).ravel(), np.array([1, 0.85, 0.3, a], dtype=np.float32))
        scn.ngeom += 1
    if racket is not None and scn.ngeom + 2 <= scn.maxgeom:
        c, v = racket
        n = v / (np.linalg.norm(v) + 1e-9)
        # disc frame: z axis = face normal
        x = np.cross([0, 0, 1.0], n)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
        R = np.stack([x, np.cross(n, x), n], axis=1)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.075, 0.004, 0]),
                            c, R.ravel(), np.array([0.85, 0.1, 0.1, 0.55], dtype=np.float32))
        scn.ngeom += 1
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), np.array([0.85, 0.1, 0.1, 0.9], dtype=np.float32))
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.008, c, c + 0.15 * v)
        scn.ngeom += 1
    for g in range(first, scn.ngeom):
        scn.geoms[g].label = ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default=str(mk.DEFAULT_OUT.with_suffix(".npz")))
    p.add_argument("--rows", type=int, default=20, help="number of random questions to play")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=0.5, help="playback speed (1 = real time)")
    p.add_argument("--record", default=None, help="write an MP4 instead of opening the viewer")
    args = p.parse_args(argv)

    import mujoco

    with np.load(args.bank) as f:
        bank = {k: f[k] for k in f.files}
    if "feed_velocity_pre" not in bank:
        raise SystemExit("bank lacks feed_velocity_pre/feed_spin_pre; regenerate it with make_g1_forehand_bank.py")
    venue = vb.default_venue_yaml_path()
    prm = vb.load_venue_params(venue)
    tprm = mk.table_params(prm, venue)
    scfg = fs.ForehandSuccessCfg()
    model = build_model(scfg)
    data = mujoco.MjData(model)
    key = model.key("stand").id
    ball_id = model.body("ball_vis").mocapid[0]
    rows = np.random.default_rng(args.seed).choice(len(bank["clip"]), size=args.rows, replace=False)
    trajs = {int(i): row_trajectory(bank, int(i), prm, tprm, scfg) for i in rows}

    def frames():
        for r, tr in trajs.items():
            mujoco.mj_resetDataKeyframe(model, data, key)
            data.qpos[1] = tr["station_y"]
            racket = (tr["contact"], bank["target_velocity"][r].astype(np.float64))
            for k, pos in enumerate(tr["path"]):
                data.mocap_pos[ball_id] = pos
                mujoco.mj_kinematics(model, data)
                show_racket = abs(k - tr["contact_idx"]) <= 30
                yield tr["path"][max(0, k - TRAIL): k + 1], racket if show_racket else None, int(r), k == tr["contact_idx"]
            for _ in range(int(0.4 / DT)):
                yield tr["path"][-TRAIL:], None, int(r), False

    if args.record:
        import imageio

        renderer = mujoco.Renderer(model, height=544, width=960)
        cam = mujoco.MjvCamera()
        cam.lookat[:], (cam.distance, cam.azimuth, cam.elevation) = CAMERA
        fps = int(round(args.speed / DT))
        with imageio.get_writer(args.record, fps=min(fps, 60)) as w:
            for n, (trail, racket, _, _) in enumerate(frames()):
                if fps > 60 and n % int(round(fps / 60)):
                    continue
                renderer.update_scene(data, cam)
                add_markers(renderer.scene, trail, racket, reset=False)
                w.append_data(renderer.render())
        print(f"[view_feed_bank] wrote {args.record} ({args.rows} questions)")
        return 0

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:], (viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation) = CAMERA
        for trail, racket, row, at_contact in frames():
            if not viewer.is_running():
                break
            t0 = time.time()
            with viewer.lock():
                add_markers(viewer.user_scn, trail, racket)
            viewer.sync()
            if at_contact:
                c = trajs[row]["contact"]
                print(f"[view_feed_bank] row {row}: contact {np.round(c, 3)} "
                      f"(apex {c[2] - scfg.table_surface_z:.2f} m above the table)", flush=True)
            time.sleep(max(0.0, DT / args.speed - (time.time() - t0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
