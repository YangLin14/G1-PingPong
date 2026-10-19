"""Fixed-station copy of a forehand question bank: the robot stands in ONE place for every ball.

Every row keeps its strike point in the world (station + racket offset), its feed, spin, target
velocity/normal and landing; only the split changes: station_xy := (0, STATION_Y) for all rows and
target_position_offset[:, :2] := old station + old offset - new station. So nothing about the ball
or the return changes, only where the robot is asked to stand (no reason to step between balls).
IK check (fist model, pelvis 0.66 m, waist + right arm): from y = -0.30 the racket reaches 100 % of
300 sampled strike points (position < 3 cm, face < 15 deg); -0.25: 99 %, -0.20: 95 %, -0.10: 81 %.

    python3 hope_training/whole_body_tracking/scripts/make_fixed_station_bank.py \
        --src assets/venue_tuple/g1_fh_crosscourt_apex_bank_wide.npz --station-y -0.30
"""

import argparse
import json
import pathlib

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="bank .npz (its .receipt.json must sit beside it)")
    ap.add_argument("--station-y", type=float, default=-0.30)
    ap.add_argument("--out", default=None, help="default: <src>_fixed<cm>.npz")
    args = ap.parse_args()
    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out) if args.out else src.with_name(
        f"{src.stem}_fixed{int(round(-100 * args.station_y))}.npz")
    d = dict(np.load(src, allow_pickle=False))
    station = d["station_xy"].astype(np.float64)
    offset = d["target_position_offset"].astype(np.float64)
    strike_xy = station + offset[:, :2]
    new_station = np.zeros_like(station)
    new_station[:, 1] = args.station_y
    new_offset = offset.copy()
    new_offset[:, :2] = strike_xy - new_station
    d["station_xy"] = new_station.astype(d["station_xy"].dtype)
    d["target_position_offset"] = new_offset.astype(d["target_position_offset"].dtype)
    assert np.allclose(d["station_xy"] + d["target_position_offset"][:, :2], strike_xy, atol=1e-5)
    np.savez(out, **d)
    receipt = json.loads(src.with_name(src.stem + ".receipt.json").read_text(encoding="utf-8"))
    receipt["fixed_station"] = {
        "source_bank": src.name, "station_xy": [0.0, args.station_y],
        "note": "strike points, feeds and returns unchanged; only station/offset split changed",
        "racket_offset_y_range": [float(new_offset[:, 1].min()), float(new_offset[:, 1].max())],
    }
    out.with_name(out.stem + ".receipt.json").write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(f"[fixed_station_bank] {len(station)} rows -> {out} (station y {args.station_y:+.2f}; "
          f"racket offset y {new_offset[:, 1].min():+.3f}..{new_offset[:, 1].max():+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
