#!/usr/bin/env python3
"""Pin the Unitree G1 articulation enumeration in Isaac Lab (run once per asset change).

Spawns the racket-equipped G1 articulation (``robots/g1.py`` ``G1_CYLINDER_CFG``) in a bare Isaac
scene — no motion clip, no task — and checks what every downstream contract assumes:

* 29 joints, enumerated EXACTLY as ``hope_training/config/joint_order_unitree_g1.yaml`` (the
  exporter hard-gates on this; the reference runner's ``UNITREE_G1`` profile and the ONNX columns
  follow it);
* the racket bodies ``right_hand_pingpang_Link`` / ``pingpang_red_Link`` / ``pingpang_black_Link``
  (or, when the importer merged the fixed links, at least ``right_wrist_yaw_link``) are present;
* the default joint pose from the articulation cfg resolves.

On a mismatch it prints both orders and exits non-zero. If Isaac enumerates differently from the
YAML, update the YAML AND ``a3_deploy/.../reference/a3_deploy_onnx_ref_pingpong/joint_order_unitree_g1.py``
together (``tests/test_policy_contract_g1.py`` pins them equal), regenerate the derived G1 clips with
``scripts/retarget_a3_clip_to_g1.py`` (their ``joint_pos`` columns follow the YAML), then re-run.

Usage (inside the Isaac environment, from hope_training/whole_body_tracking/):
    hope_isaac_py scripts/prepare_g1_isaac_asset.py --force
    hope_isaac_py scripts/check_g1_asset_isaac.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "unitree").is_dir():
            return parent
    return here.parents[3]


RACKET_BODIES = ("right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--joint-order-yaml",
        default=str(_repo_root() / "hope_training" / "config" / "joint_order_unitree_g1.yaml"),
        help="Canonical joint-order YAML to compare against.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.argv = sys.argv[:1]
    # Slurm/redirected stdout is block-buffered; flush per line so progress shows in the job log.
    sys.stdout.reconfigure(line_buffering=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app
    status = 0
    sim = None
    try:
        import yaml

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext

        from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, UNITREE_G1_JOINT_NAMES, UNITREE_G1_URDF_PATH

        if not pathlib.Path(UNITREE_G1_URDF_PATH).is_file():
            raise FileNotFoundError(
                f"prepared G1 asset missing: {UNITREE_G1_URDF_PATH}\n"
                "Run: hope_isaac_py scripts/prepare_g1_isaac_asset.py --force"
            )
        with open(args.joint_order_yaml, "r", encoding="utf-8") as fh:
            canonical = [str(n) for n in yaml.safe_load(fh)["joint_order"]]

        sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/defaultGroundPlane", ground)
        robot = Articulation(G1_CYLINDER_CFG.replace(prim_path="/World/Robot"))
        sim.reset()

        joint_names = list(robot.joint_names)
        body_names = list(robot.body_names)
        print(f"[check_g1_asset_isaac] {len(joint_names)} joints:")
        for i, name in enumerate(joint_names):
            print(f"  {i:2d}  {name}")
        print(f"[check_g1_asset_isaac] {len(body_names)} bodies: {body_names}")

        failures: list[str] = []
        if len(joint_names) != 29:
            failures.append(f"expected 29 joints, found {len(joint_names)}")
        if joint_names != canonical:
            failures.append(
                "articulation joint order != joint_order_unitree_g1.yaml\n"
                f"  articulation: {joint_names}\n  canonical:    {canonical}\n"
                "  -> update hope_training/config/joint_order_unitree_g1.yaml AND the reference package's "
                "joint_order_unitree_g1.py to the articulation order above, then regenerate the derived G1 "
                "clips (python3 scripts/retarget_a3_clip_to_g1.py; their joint columns follow the YAML)."
            )
        if joint_names != list(UNITREE_G1_JOINT_NAMES):
            failures.append("robots/g1.py UNITREE_G1_JOINT_NAMES differs from the articulation enumeration")
        missing_racket = [b for b in RACKET_BODIES if b not in body_names]
        if missing_racket:
            if "right_wrist_yaw_link" not in body_names:
                failures.append("right_wrist_yaw_link missing from the articulation")
            else:
                print(
                    f"[check_g1_asset_isaac] NOTE: racket links {missing_racket} were merged into "
                    "right_wrist_yaw_link by the importer; RacketTargetCommand falls back to the "
                    "wrist + mount_offset FK (G1_MOUNT_OFFSET)."
                )
        if failures:
            for failure in failures:
                print(f"[FAIL] {failure}")
            status = 1
        else:
            print("[check_g1_asset_isaac] OK: joint order pinned, racket bodies present.")
    except Exception:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        # A bare SimulationContext (no env) keeps Isaac Lab's timeline-STOP handler subscribed;
        # simulation_app.close() closes the stage, fires STOP, and that handler then spins in
        # `while not timeline.is_playing(): render()` forever. Envs clear this in their close();
        # do the same here before shutting the app down.
        if sim is not None:
            try:
                sim.clear_all_callbacks()
                sim.clear_instance()
            except Exception:
                pass
        simulation_app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
