#!/usr/bin/env python3
"""Generate a G1 forehand question bank for the club's cross-court rally (fixed_balanced_bank_v1).

Every row is one physically consistent question in the env frame (robot at the origin facing +x,
near table edge at ``table_near_x``, surface at ``table_surface_z``), built with the venue-fitted
spin model the training reward uses (:mod:`virtual_ball`):

1. **feed**: the incoming ball bounces in ``FEED_ZONE`` -- middle depth third of the robot's half,
   middle band of the robot's right (-y) half -- with a sampled pre-bounce velocity/spin, and the
   table bounce is applied with the venue table contact constants. Integrated backward, the feed must
   have crossed the far end line over the opponent's right half at a hitting height and cleared the
   net: it is hit diagonally from the opponent's right, the diagonal the robot returns along;
2. **apex strike**: the post-bounce flight is integrated to its apex (vz = 0); only rows whose apex
   is inside the G1 forehand strike box are kept, so the ball is always struck at the top;
3. **return**: each row draws one of ``targets`` (default: all four ``TARGET_ZONES``, on the
   OPPONENT's right = robot's +y, recorded in ``target_zone_id``), a landing point uniformly inside
   it, and the racket velocity
   (face normal along it, the V14 velocity-normal contract) is solved by damped least squares on
   the predicted landing; rows that miss the target, clip the net or exceed the racket speed limit
   are rejected.

Writes ``<out>.npz`` with the schema the RacketTargetCommand loader validates, plus a receipt JSON.
All rows are forehand (clip 0); pair it with ``commands.racket_target.active_clip_ids: [0]``.

  python3 hope_training/whole_body_tracking/scripts/make_g1_forehand_bank.py --rows 50000
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

_WBT = pathlib.Path(__file__).resolve().parents[1]
_PKG = _WBT / "source" / "whole_body_tracking" / "whole_body_tracking"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fs = _load("_forehand_success", _PKG / "utils" / "forehand_success.py")
vb = fs.virtual_ball

SCHEMA_VERSION = 2
RECIPE_REVISION = 12  # the loader's receipt gate; this bank keeps the r12 array contract
DEFAULT_OUT = _WBT / "assets" / "venue_tuple" / "g1_fh_crosscourt_apex_bank"


@dataclasses.dataclass
class BankCfg:
    rows: int = 50000
    targets: tuple[str, ...] = ("opp_right_mid", "center_line_deep", "opp_right_deep", "opp_right_corner")
    seed: int = 0
    # pre-bounce incoming ball at the feed bounce (env frame, m/s); spin magnitude in rev/s
    vin_x: tuple[float, float] = (-5.5, -2.5)
    vin_y: tuple[float, float] = (-2.4, -0.3)  # diagonal: from the opponent's right (+y) to the robot's right
    vin_z: tuple[float, float] = (-3.5, -1.5)
    spin_in_rev_s: float = 5.0
    # G1 forehand strike box (station x = 0): reach plane around recipe v2's 0.46 m, recipe z box
    apex_x: tuple[float, float] = (0.42, 0.50)
    apex_z: tuple[float, float] = (0.80, 1.10)
    racket_offset_y: tuple[float, float] = (-0.38, -0.32)  # recipe v2 forehand y band
    station_y_max: float = 0.45
    racket_speed_limit: float = 3.5
    nominal_racket_vel: tuple[float, float, float] = (1.9, 0.3, 1.0)
    solve_iters: int = 12
    solve_damping: float = 1e-3  # weak pull: deep/edge targets need fast, flat swings
    max_landing_err: float = 0.05
    net_margin: float = 0.03
    net_clearance_target: float = 0.25  # flat rally ball, not a lob (solved jointly with the landing)
    net_clearance_weight: float = 1.0
    max_clearance_err: float = 0.10
    flight_h: float = 0.002
    flight_steps: int = 400


def _uniform(g: torch.Generator, n: int, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)


def feed_origin(bounce, v_pre, w_pre, prm, scfg, h: float, steps: int):
    """Integrate the pre-bounce flight BACKWARD from the bounce. Returns (y, z) where it crosses the far
    end line, the height where it crosses the net plane, whether both crossings happened, and the
    flight time from the end line to the bounce."""
    n = bounce.shape[0]
    p, v = bounce.clone(), v_pre.clone()
    end_yz = torch.zeros(n, 2, dtype=p.dtype)
    end_t = torch.zeros(n, dtype=p.dtype)
    net_z = torch.zeros(n, dtype=p.dtype)
    at_end = torch.zeros(n, dtype=torch.bool)
    at_net = torch.zeros(n, dtype=torch.bool)
    for k in range(steps):
        p1, v1 = vb.rk4_step(p, v, w_pre, -h, prm)
        for plane, hit, store in ((scfg.net_x, at_net, "net"), (scfg.far_x, at_end, "end")):
            cross = (~hit) & (p[:, 0] < plane) & (p1[:, 0] >= plane)
            f = ((plane - p[:, 0]) / (p1[:, 0] - p[:, 0]).clamp(min=1e-12)).unsqueeze(-1)
            q = p + f * (p1 - p)
            if store == "net":
                net_z = torch.where(cross, q[:, 2], net_z)
            else:
                end_yz = torch.where(cross.unsqueeze(-1), q[:, 1:], end_yz)
                end_t = torch.where(cross, (k + f.squeeze(-1)) * h, end_t)
            hit |= cross
        p, v = p1, v1
    return end_yz, net_z, at_end & at_net, end_t


def table_params(prm, venue_yaml: str):
    """The venue paddle-contact form with the table constants (v_r = 0, n = +z)."""
    with open(venue_yaml, "r", encoding="utf-8") as fh:
        t = yaml.safe_load(fh)["contact"]["table"]
    return dataclasses.replace(
        prm, paddle_a_t=float(t["a_t"]), paddle_b_t=float(t["b_t"]), paddle_mu=float(t["mu_safety"]),
        paddle_e_g1=float(t["e_eff"]), paddle_e_g2=0.0,
    )


def fly_to_apex(p, v, w, prm, h: float, steps: int):
    """Integrate until vz crosses 0 going down; linear interpolation of the crossing. Also returns the
    time to the apex."""
    found = torch.zeros(p.shape[0], dtype=torch.bool)
    p_apex, v_apex = p.clone(), v.clone()
    t_apex = torch.zeros(p.shape[0], dtype=p.dtype)
    for k in range(steps):
        p1, v1 = vb.rk4_step(p, v, w, h, prm)
        cross = (~found) & (v[:, 2] > 0) & (v1[:, 2] <= 0)
        f = (v[:, 2] / (v[:, 2] - v1[:, 2]).clamp(min=1e-12)).unsqueeze(-1)
        p_apex = torch.where(cross.unsqueeze(-1), p + f * (p1 - p), p_apex)
        v_apex = torch.where(cross.unsqueeze(-1), v + f * (v1 - v), v_apex)
        t_apex = torch.where(cross, (k + f.squeeze(-1)) * h, t_apex)
        found |= cross
        p, v = p1, v1
    v_apex[:, 2] = 0.0
    return p_apex, v_apex, found, t_apex


def landing(ball, v_in, w_in, v_r, prm, scfg):
    n = v_r / torch.linalg.norm(v_r, dim=-1, keepdim=True).clamp(min=1e-9)
    v_out, w_out = vb.predict_paddle_contact(v_in, v_r, n, w_in, prm)
    land = vb.coarse_landing(
        ball, v_out, w_out, prm, surface_z=scfg.table_surface_z + fs.BALL_RADIUS, net_x=scfg.net_x,
        h=scfg.rollout_h, n_steps=scfg.rollout_steps,
    )
    return land, v_out, w_out


def _residual_features(land, scfg, cfg: BankCfg) -> torch.Tensor:
    clearance = land["net_z"] - (scfg.table_surface_z + fs.NET_HEIGHT + fs.BALL_RADIUS)
    return torch.cat([land["land_xy"], cfg.net_clearance_weight * clearance.unsqueeze(-1)], dim=-1)


def solve_racket(ball, v_in, w_in, target_xy, prm, scfg, cfg: BankCfg):
    """Damped least squares on (landing xy, net clearance), regularized toward the nominal racket velocity."""
    v_nom = torch.tensor(cfg.nominal_racket_vel, dtype=torch.float64).expand_as(v_in)
    v_r = v_nom.clone()
    eps = 1e-3
    lam = cfg.solve_damping
    goal = torch.cat([target_xy, torch.full_like(target_xy[:, :1], cfg.net_clearance_weight * cfg.net_clearance_target)], -1)
    for _ in range(cfg.solve_iters):
        base = _residual_features(landing(ball, v_in, w_in, v_r, prm, scfg)[0], scfg, cfg)
        J = torch.empty(v_r.shape[0], 3, 3, dtype=torch.float64)
        for k in range(3):
            dv = torch.zeros_like(v_r)
            dv[:, k] = eps
            J[:, :, k] = (_residual_features(landing(ball, v_in, w_in, v_r + dv, prm, scfg)[0], scfg, cfg) - base) / eps
        r = goal - base
        # min |J dv - r|^2 + lam |v_r + dv - v_nom|^2
        A = J.transpose(1, 2) @ J + lam * torch.eye(3, dtype=torch.float64)
        b = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1) - lam * (v_r - v_nom)
        step = torch.linalg.solve(A, b)
        v_r = v_r + step.clamp(-1.0, 1.0)
    return v_r


def generate(cfg: BankCfg) -> tuple[dict, dict]:
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(cfg.seed)
    venue_yaml = vb.default_venue_yaml_path()
    prm = vb.load_venue_params(venue_yaml)
    tprm = table_params(prm, venue_yaml)
    scfg = fs.ForehandSuccessCfg()
    (fx, fy) = scfg.feed_bounds()
    (oy, oz) = scfg.feed_origin_bounds()
    zone_cfgs = [scfg.with_target(t) for t in cfg.targets]
    zones = torch.tensor([sum(map(list, c.zone_bounds()), []) for c in zone_cfgs])  # (Z, 4) x_lo x_hi y_lo y_hi

    accepted: dict[str, list[torch.Tensor]] = {}
    rejects = {"no_apex": 0, "apex_out_of_box": 0, "station_out_of_range": 0, "feed_not_from_opp_right": 0,
               "landing_miss": 0, "net": 0, "speed": 0, "not_in_zone": 0}
    total = 0
    batch = max(4 * cfg.rows, 20000)
    quota = -(-cfg.rows // len(cfg.targets))  # equal rows per target zone

    def per_zone_counts() -> list[int]:
        ids = torch.cat(accepted["target_zone_id"]) if accepted else torch.zeros(0, dtype=torch.int64)
        return [int((ids == z).sum()) for z in range(len(cfg.targets))]

    while min(per_zone_counts()) < quota:
        n = batch
        total += n
        bounce = torch.stack([
            _uniform(g, n, *fx), _uniform(g, n, *fy),
            torch.full((n,), scfg.table_surface_z + fs.BALL_RADIUS),
        ], dim=-1)
        v_pre = torch.stack([_uniform(g, n, *cfg.vin_x), _uniform(g, n, *cfg.vin_y), _uniform(g, n, *cfg.vin_z)], -1)
        axis = torch.randn(n, 3, generator=g)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
        w_pre = axis * _uniform(g, n, 0.0, cfg.spin_in_rev_s).unsqueeze(-1) * 2 * np.pi
        up = torch.tensor([0.0, 0.0, 1.0]).expand(n, 3)
        v_post, w_post = vb.predict_paddle_contact(v_pre, torch.zeros(n, 3), up, w_pre, tprm)

        p_apex, v_apex, found, t_apex = fly_to_apex(bounce, v_post, w_post, prm, cfg.flight_h, cfg.flight_steps)
        rejects["no_apex"] += int((~found).sum())
        in_box = found & (p_apex[:, 0] >= cfg.apex_x[0]) & (p_apex[:, 0] <= cfg.apex_x[1]) \
            & (p_apex[:, 2] >= cfg.apex_z[0]) & (p_apex[:, 2] <= cfg.apex_z[1])
        rejects["apex_out_of_box"] += int((found & ~in_box).sum())
        offset_y = _uniform(g, n, *cfg.racket_offset_y)
        station_y = p_apex[:, 1] - offset_y
        ok_station = station_y.abs() <= cfg.station_y_max
        rejects["station_out_of_range"] += int((in_box & ~ok_station).sum())
        keep = in_box & ok_station
        if not bool(keep.any()):
            continue
        idx = torch.where(keep)[0]
        end_yz, feed_net_z, crossed, end_t = feed_origin(bounce[idx], v_pre[idx], w_pre[idx], prm, scfg,
                                                         cfg.flight_h, cfg.flight_steps)
        from_opp_right = crossed & (end_yz[:, 0] >= oy[0]) & (end_yz[:, 0] <= oy[1]) \
            & (end_yz[:, 1] >= oz[0]) & (end_yz[:, 1] <= oz[1]) \
            & (feed_net_z > scfg.table_surface_z + fs.NET_HEIGHT + fs.BALL_RADIUS + cfg.net_margin)
        rejects["feed_not_from_opp_right"] += int((~from_opp_right).sum())
        idx, end_yz, end_t = idx[from_opp_right], end_yz[from_opp_right], end_t[from_opp_right]
        if len(idx) == 0:
            continue
        ball, vin, win = p_apex[idx], v_apex[idx], w_post[idx]
        zone_id = torch.randint(len(cfg.targets), (len(idx),), generator=g)
        zb = zones[zone_id]
        target = torch.stack([zb[:, 0] + (zb[:, 1] - zb[:, 0]) * torch.rand(len(idx), generator=g),
                              zb[:, 2] + (zb[:, 3] - zb[:, 2]) * torch.rand(len(idx), generator=g)], -1)
        v_r = solve_racket(ball, vin, win, target, prm, scfg, cfg)
        land, v_out, w_out = landing(ball, vin, win, v_r, prm, scfg)
        err = torch.linalg.norm(land["land_xy"] - target, dim=-1)
        clearance = land["net_z"] - (scfg.table_surface_z + fs.NET_HEIGHT + fs.BALL_RADIUS)
        hit = err <= cfg.max_landing_err
        net_ok = land["net_valid"] & (clearance >= cfg.net_margin) \
            & ((clearance - cfg.net_clearance_target).abs() <= cfg.max_clearance_err)
        speed_ok = torch.linalg.norm(v_r, dim=-1) <= cfg.racket_speed_limit
        n_r = v_r / torch.linalg.norm(v_r, dim=-1, keepdim=True)
        success = torch.zeros(len(idx), dtype=torch.bool)
        for z, zcfg in enumerate(zone_cfgs):
            m = zone_id == z
            if bool(m.any()):
                success[m] = fs.classify_strikes(ball[m], vin[m], win[m], ball[m], v_r[m], n_r[m], zcfg, prm)["success"]
        rejects["landing_miss"] += int((~hit).sum())
        rejects["net"] += int((hit & ~net_ok).sum())
        rejects["speed"] += int((hit & net_ok & ~speed_ok).sum())
        good = hit & net_ok & speed_ok
        rejects["not_in_zone"] += int((good & ~success).sum())
        good &= success
        if not bool(good.any()):
            continue
        sel = torch.where(good)[0]
        k = idx[sel]
        sy = station_y[k]
        rows = {
            "clip": torch.zeros(len(sel), dtype=torch.int64),
            "station_xy": torch.stack([torch.zeros_like(sy), sy], -1),
            "target_position_offset": torch.stack([ball[sel, 0], ball[sel, 1] - sy, ball[sel, 2]], -1),
            "target_velocity": v_r[sel],
            "target_normal": v_r[sel] / torch.linalg.norm(v_r[sel], dim=-1, keepdim=True),
            "incoming_velocity": vin[sel],
            "incoming_spin": win[sel],
            "intended_landing_xy": target[sel],
            "predicted_landing_xy": land["land_xy"][sel],
            "outgoing_velocity": v_out[sel],
            "outgoing_spin": w_out[sel],
            "net_clearance_m": clearance[sel],
            "planner_residual_m": err[sel],
            "feed_bounce_xy": bounce[k, :2],
            "feed_velocity_pre": v_pre[k],
            "feed_spin_pre": w_pre[k],
            "feed_origin_yz": end_yz[sel],
            # end line -> feed bounce -> apex: the time the robot has once the opponent's shot is seen
            "feed_flight_s": end_t[sel] + t_apex[k],
            "target_zone_id": zone_id[sel],
        }
        for name, value in rows.items():
            accepted.setdefault(name, []).append(value)
        print(f"[make_g1_forehand_bank] accepted per target {per_zone_counts()} / {quota} (drawn {total})", flush=True)

    merged = {name: torch.cat(parts) for name, parts in accepted.items()}
    take = torch.cat([torch.where(merged["target_zone_id"] == z)[0][:quota] for z in range(len(cfg.targets))])
    take = take[torch.randperm(len(take), generator=g)][: cfg.rows]
    arrays = {name: value[take].numpy() for name, value in merged.items()}
    for name, value in arrays.items():
        if value.dtype == np.float64:
            arrays[name] = value.astype(np.float32)
    arrays["candidate_id"] = np.arange(cfg.rows, dtype=np.int64)
    arrays["schema_version"] = np.array(SCHEMA_VERSION, dtype=np.int64)

    def stats(x):
        x = np.asarray(x, dtype=np.float64)
        return {"min": float(x.min()), "mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())}

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "contract": {
            "recipe_revision": RECIPE_REVISION,
            "task": "HOPEPingPongUnitreeG1_FH cross-court apex forehand",
            "robot": "unitree_g1",
            "frame": "env frame: robot at origin facing +x, +y = robot's left",
            "table_near_x": scfg.table_near_x,
            "table_surface_z": scfg.table_surface_z,
            "feed_zone_xy": [list(fx), list(fy)],
            "feed_origin": {"crosses_far_end_line_at_x": scfg.far_x, "y": list(oy), "z": list(oz)},
            "target_zones": {t: [list(b) for b in c.zone_bounds()] for t, c in zip(cfg.targets, zone_cfgs)},
            "target_zone_id": list(cfg.targets),
            "strike": "incoming ball at its post-bounce apex (vz = 0)",
            "racket_speed_limit_mps": cfg.racket_speed_limit,
            "net_clearance_margin_m": cfg.net_margin,
            "physics": str(pathlib.Path(venue_yaml).name),
        },
        "generation": dataclasses.asdict(cfg),
        "attempts": total,
        "reject_counts": rejects,
        "accepted_summary": {
            "racket_speed_mps": stats(np.linalg.norm(arrays["target_velocity"], axis=-1)),
            "apex_height_above_table_m": stats(arrays["target_position_offset"][:, 2] - scfg.table_surface_z),
            "incoming_speed_mps": stats(np.linalg.norm(arrays["incoming_velocity"], axis=-1)),
            "feed_speed_pre_bounce_mps": stats(np.linalg.norm(arrays["feed_velocity_pre"], axis=-1)),
            "feed_flight_s": stats(arrays["feed_flight_s"]),
            "net_clearance_m": stats(arrays["net_clearance_m"]),
            "landing_residual_m": stats(arrays["planner_residual_m"]),
            "station_y_m": stats(arrays["station_xy"][:, 1]),
            "rows_per_target": {t: int((arrays["target_zone_id"] == z).sum()) for z, t in enumerate(cfg.targets)},
        },
    }
    return arrays, receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=BankCfg.rows)
    p.add_argument("--targets", nargs="+", default=list(BankCfg.targets), choices=sorted(fs.TARGET_ZONES))
    p.add_argument("--seed", type=int, default=BankCfg.seed)
    p.add_argument("--vin-x", nargs=2, type=float, default=BankCfg.vin_x, metavar=("LO", "HI"),
                   help="pre-bounce feed velocity x at the feed bounce (m/s, negative = toward the robot)")
    p.add_argument("--vin-y", nargs=2, type=float, default=BankCfg.vin_y, metavar=("LO", "HI"))
    p.add_argument("--vin-z", nargs=2, type=float, default=BankCfg.vin_z, metavar=("LO", "HI"))
    p.add_argument("--spin-rev-s", type=float, default=BankCfg.spin_in_rev_s, help="max feed spin (rev/s)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output path without extension")
    args = p.parse_args(argv)
    arrays, receipt = generate(BankCfg(
        rows=args.rows, targets=tuple(args.targets), seed=args.seed, vin_x=tuple(args.vin_x),
        vin_y=tuple(args.vin_y), vin_z=tuple(args.vin_z), spin_in_rev_s=args.spin_rev_s,
    ))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"), **arrays)
    receipt["bank_path"] = out.with_suffix(".npz").name
    receipt["bank_sha256"] = hashlib.sha256(out.with_suffix(".npz").read_bytes()).hexdigest()
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reject_counts": receipt["reject_counts"], **receipt["accepted_summary"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
