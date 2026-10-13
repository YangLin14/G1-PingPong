"""Score one policy under many camera models (``mujoco_eval_onnx.py --perception``) and tabulate.

Every condition gets the same bank rows and rally gaps (same --seed), so the rows of the table differ
only in what the planner sees. Runs the conditions in parallel processes (CPU only):

    .venv/bin/python hope_training/whole_body_tracking/scripts/perception_sweep.py \
        --onnx artifacts/evals/g1_fh_rally_model_5900/policy.onnx --serves 40 --out artifacts/perception/model_5900
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

CONDITIONS = [
    # (group, spec)
    ("reference", "oracle"),
    ("reference", "oracle_nospin"),
    ("systems", "mocap_360"),
    ("systems", "mocap_240"),
    ("systems", "stereo_125"),
    ("systems", "iphone_240"),
    ("systems", "iphone_120"),
    ("systems", "iphone_60"),
    ("systems", "webcam_30"),
    ("latency (mocap_360)", "mocap_360:latency_s=0.03"),
    ("latency (mocap_360)", "mocap_360:latency_s=0.06"),
    ("latency (mocap_360)", "mocap_360:latency_s=0.1"),
    ("noise (mocap_360)", "mocap_360:noise_m=0.005"),
    ("noise (mocap_360)", "mocap_360:noise_m=0.01"),
    ("noise (mocap_360)", "mocap_360:noise_m=0.02"),
    ("rate (5 ms, 1 mm)", "mocap_360:rate_hz=120"),
    ("rate (5 ms, 1 mm)", "mocap_360:rate_hz=60"),
    ("spin known", "mocap_360:spin=true"),
    ("spin known", "iphone_120:spin=true"),
    # spin read with a per-ball error (e.g. from the opponent's racket): how accurate must it be?
    ("spin read (mocap_360)", "mocap_360:spin=read,spin_rel_err=0.1,spin_ang_err_deg=10"),
    ("spin read (mocap_360)", "mocap_360:spin=read,spin_rel_err=0.25,spin_ang_err_deg=20"),
    ("spin read (mocap_360)", "mocap_360:spin=read,spin_rel_err=0.5,spin_ang_err_deg=40"),
    ("spin read (iphone_120)", "iphone_120:spin=read,spin_rel_err=0.1,spin_ang_err_deg=10"),
    ("spin read (iphone_120)", "iphone_120:spin=read,spin_rel_err=0.25,spin_ang_err_deg=20"),
    ("spin read (iphone_120)", "iphone_120:spin=read,spin_rel_err=0.5,spin_ang_err_deg=40"),
]


def run_one(spec: str, seed: int, args) -> dict:
    tag = spec.replace(":", "__").replace(",", "_").replace("=", "-")
    out = pathlib.Path(args.out) / f"{tag}__seed{seed}.json"
    if not out.is_file():
        cmd = [sys.executable, str(HERE / "mujoco_eval_onnx.py"), "--robot", "unitree_g1", "--onnx", args.onnx,
               "--serve-bank", args.bank, "--num-serves", str(args.serves), "--seed", str(seed),
               "--perception", spec, "--reset-after-fall", "--json-out", str(out)]
        if args.rally:
            cmd.append("--rally-pace")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--bank", default="default")
    ap.add_argument("--serves", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-rally", dest="rally", action="store_false", help="announce 0.9 s ahead instead of rally pace")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)

    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = {(spec, seed): pool.submit(run_one, spec, seed, args) for _, spec in CONDITIONS for seed in args.seeds}
        runs = {key: f.result() for key, f in futures.items()}

    def pooled(spec):
        """All balls of all seeds of one condition, re-summarized (rates over the pooled balls)."""
        serves = [r for seed in args.seeds for r in runs[(spec, seed)]["serves"]]
        n = len(serves)
        contact = [r for r in serves if r["contacted"]]
        net = [r for r in contact if r["net_clear"]]
        legal = [r for r in net if r["on_opponent_half"]]
        def med(key):
            vals = [abs(r[key]) for r in serves if r.get(key) is not None]
            return sorted(vals)[len(vals) // 2] if vals else None
        return n, {
            "contact_rate": len(contact) / n, "apex_rate": sum(r["apex_ok"] for r in contact) / n,
            "net_clear_rate": len(net) / n, "on_opponent_rate": len(legal) / n,
            "in_zone_rate": sum(r["in_zone"] for r in legal) / n, "fall_rate": sum(r["fell"] for r in serves) / n,
        }, {
            "first_command_time_to_strike_s_median": med("first_command_time_to_strike_s"),
            "apex_err_m_median": med("perceived_apex_err_m"), "time_err_s_median": med("perceived_time_err_s"),
        }

    head = ("| group | condition | first command (s before strike) | apex error (cm) | time error (ms) "
            "| contact | apex OK | over net | legal | in zone | falls |")
    lines = [head, "|" + "---|" * 11]
    for group, spec in CONDITIONS:
        n, o, p = pooled(spec)
        def pct(x):
            return f"{100 * x:.0f} %"
        def num(x, scale, fmt):
            return "–" if x is None else format(x * scale, fmt)
        lines.append(
            f"| {group} | `{spec}` | {num(p.get('first_command_time_to_strike_s_median'), 1, '.2f')} "
            f"| {num(p.get('apex_err_m_median'), 100, '.1f')} | {num(p.get('time_err_s_median'), 1000, '.0f')} "
            f"| {pct(o['contact_rate'])} | {pct(o['apex_rate'])} | {pct(o['net_clear_rate'])} "
            f"| {pct(o['on_opponent_rate'])} | {pct(o['in_zone_rate'])} | {pct(o['fall_rate'])} |")
    table = "\n".join(lines)
    (pathlib.Path(args.out) / "summary.md").write_text(
        f"policy `{args.onnx}`, {args.serves} bank balls x seeds {args.seeds} per condition "
        f"({args.serves * len(args.seeds)} balls), robot stood up again after a fall, "
        f"{'rally pace' if args.rally else 'announced 0.9 s ahead'}\n\n{table}\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
