#!/usr/bin/env python3
"""Isaac-vs-MuJoCo plant replay for the Unitree G1 (the sim-to-sim parity gate).

Drives the SAME scripted joint-target sequence through both plants and compares the resulting
joint trajectories and pelvis motion. A policy that stands in Isaac but collapses in MuJoCo is a
plant mismatch (gains, armature, limits, timing, contacts), not a policy failure, and this script
is how that is told apart.

Two halves, because the Isaac half needs the Kit app (cluster/container) and the MuJoCo half only
needs the host venv:

    # inside the Isaac environment (hope_isaac_py), from hope_training/whole_body_tracking/:
    hope_isaac_py scripts/sim2sim_replay_g1.py --isaac --out /tmp/replay_isaac.npz
    # on the host venv, from the repo root:
    MUJOCO_GL=egl python hope_training/whole_body_tracking/scripts/sim2sim_replay_g1.py \\
        --mujoco --compare /tmp/replay_isaac.npz --out /tmp/replay_mujoco.npz

The Isaac half spawns ``G1_CYLINDER_CFG`` (racket-equipped asset, implicit PD at 200 Hz, control
at 50 Hz, friction-1.0 ground, no domain randomization) in a bare scene and applies the targets with
``set_joint_position_target``. The MuJoCo half loads the generated ``g1_29dof_pingpang.xml`` through
the evaluator's ``PingPongRealPhysicsScene`` with the per-joint gains of
``hope_pingpong_runtime_unitree_g1.yaml`` (explicit PD torque at 1 kHz, 50 Hz control). Both record
joint positions in the canonical YAML order plus the pelvis pose at every control tick.

Sequence (deterministic): 2 s hold at the stand pose, then 4 s of slow sinusoids (arms 0.30 rad,
waist 0.10 rad, a 0.05/0.10 rad leg squat), ramped in over 1 s.
Gate (``--compare``): over the first ``--window`` seconds (default 1.0 s; a pure PD hold with the recipe's
ankle gains sags and topples in BOTH simulators, after which ground contact makes joint errors
meaningless) the RMS joint error must stay <= --max-q-rms (0.10 rad) and the pelvis position difference
<= --max-pelvis-diff (0.10 m), and if either plant topples (pelvis z < 0.4 m) both must, within
--max-fall-time-diff (0.25 s); otherwise exit 1 with a per-joint table.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

CONTROL_DT = 0.02
HOLD_S = 2.0
TOTAL_S = 6.0


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "unitree").is_dir():
            return parent
    return here.parents[3]


def canonical_joint_names(repo: pathlib.Path) -> list[str]:
    import yaml

    with open(repo / "hope_training" / "config" / "joint_order_unitree_g1.yaml", "r", encoding="utf-8") as fh:
        return [str(n) for n in yaml.safe_load(fh)["joint_order"]]


def target_sequence(names: list[str], q0: np.ndarray) -> np.ndarray:
    """[T, 29] joint targets at 50 Hz: hold, then slow sinusoids (arms / waist / squat)."""
    ticks = int(round(TOTAL_S / CONTROL_DT))
    t = np.arange(ticks) * CONTROL_DT
    q = np.repeat(np.asarray(q0, dtype=np.float64)[None, :], ticks, axis=0)
    active = t >= HOLD_S
    tt = t - HOLD_S
    for j, name in enumerate(names):
        amp, freq = 0.0, 0.5
        if any(k in name for k in ("shoulder", "elbow", "wrist")):
            amp = 0.30
        elif name.startswith("waist"):
            amp = 0.10
        elif "hip_pitch" in name:
            amp = -0.05
        elif "knee" in name:
            amp = 0.10
        elif "ankle_pitch" in name:
            amp = -0.05
        phase = 0.37 * j
        ramp = np.clip(tt / 1.0, 0.0, 1.0)  # smooth ramp-in so the first active tick is not a step
        q[active, j] = q0[j] + amp * ramp[active] * np.sin(2 * np.pi * freq * tt[active] + phase)
    return q


def _save(out: pathlib.Path, plant: str, names, q0, targets, logs) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, plant=plant, control_dt=CONTROL_DT, joint_names=np.array(names), q0=np.asarray(q0), targets=targets,
        q=np.asarray(logs["q"]), qd=np.asarray(logs["qd"]), pelvis_pos=np.asarray(logs["pos"]),
        pelvis_quat=np.asarray(logs["quat"]), tau=np.asarray(logs["tau"]),
    )
    z = np.asarray(logs["pos"])[:, 2]
    print(f"[sim2sim] {plant}: {targets.shape[0]} ticks, pelvis z start {z[0]:.3f} min {z.min():.3f} end {z[-1]:.3f}; wrote {out}")


def run_isaac(args, repo: pathlib.Path) -> int:
    sys.argv = sys.argv[:1]
    sys.stdout.reconfigure(line_buffering=True)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app
    status = 0
    sim = None
    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext

        from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, UNITREE_G1_URDF_PATH

        if not pathlib.Path(UNITREE_G1_URDF_PATH).is_file():
            raise FileNotFoundError(f"prepared G1 asset missing: {UNITREE_G1_URDF_PATH} (run prepare_g1_isaac_asset.py)")
        names = canonical_joint_names(repo)
        physics_dt = 0.005
        decimation = int(round(CONTROL_DT / physics_dt))
        sim = SimulationContext(SimulationCfg(dt=physics_dt, device=args.device))
        ground = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            )
        )
        ground.func("/World/defaultGroundPlane", ground)
        robot = Articulation(G1_CYLINDER_CFG.replace(prim_path="/World/Robot"))
        sim.reset()
        if list(robot.joint_names) != names:
            raise RuntimeError(f"articulation joint order != canonical YAML:\n{list(robot.joint_names)}\n{names}")
        q0 = robot.data.default_joint_pos[0].cpu().numpy().astype(np.float64)
        targets = target_sequence(names, q0)
        root = robot.data.default_root_state.clone()
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:])
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.reset()
        logs = {"q": [], "qd": [], "pos": [], "quat": [], "tau": []}
        for tick in range(targets.shape[0]):
            tgt = torch.as_tensor(targets[tick], dtype=torch.float32, device=args.device).unsqueeze(0)
            for _ in range(decimation):
                robot.set_joint_position_target(tgt)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(physics_dt)
            logs["q"].append(robot.data.joint_pos[0].cpu().numpy())
            logs["qd"].append(robot.data.joint_vel[0].cpu().numpy())
            logs["pos"].append(robot.data.root_pos_w[0].cpu().numpy())
            logs["quat"].append(robot.data.root_quat_w[0].cpu().numpy())
            logs["tau"].append(robot.data.applied_torque[0].cpu().numpy())
        _save(pathlib.Path(args.out), "isaac", names, q0, targets, logs)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        # Kit's close() ends the interpreter with status 0 (atexit included); hard-exit on failure.
        if sim is not None:
            try:
                sim.clear_all_callbacks()
                sim.clear_instance()
            except Exception:
                pass
        import os

        os._exit(1)
    finally:
        if sim is not None:
            try:
                sim.clear_all_callbacks()
                sim.clear_instance()
            except Exception:
                pass
        simulation_app.close()
    return status


def run_mujoco(args, repo: pathlib.Path) -> int:
    import importlib.util

    sys.path.insert(0, str(repo / "a3_deploy" / "a3_deploy_example" / "reference"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    spec = importlib.util.spec_from_file_location(
        "hope_success_metric",
        repo / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/success_metric.py",
    )
    metric = importlib.util.module_from_spec(spec)
    sys.modules["hope_success_metric"] = metric
    spec.loader.exec_module(metric)

    cfg = RuntimeConfig.load(repo / "a3_deploy/a3_deploy_example/config/hope_pingpong_runtime_unitree_g1.yaml", profile="unitree_g1")
    prof = cfg.profile
    names = canonical_joint_names(repo)
    if list(prof.joint_names) != names:
        raise RuntimeError("UNITREE_G1 profile joint order != canonical YAML")
    if abs(cfg.control_dt - CONTROL_DT) > 1e-9:
        raise RuntimeError(f"runtime control_dt {cfg.control_dt} != {CONTROL_DT}")
    scene = PingPongRealPhysicsScene(
        str(cfg.model_xml_path), metric.load_ball_physics_config(), prof.joint_names, control_dt=cfg.control_dt, near_edge_x=0.5,
        base_free_joint=prof.base_free_joint, gyro_sensor=prof.gyro_sensor, racket_site=prof.racket_site,
        racket_collision_geom=prof.racket_collision_geom, floor_geom=prof.floor_geom,
    )
    q0 = cfg.action_adapter.default_q.copy()
    targets = target_sequence(names, q0)
    kp, kd = cfg.sim_kp.copy(), cfg.sim_kd.copy()
    scene.reset_stand()
    logs = {"q": [], "qd": [], "pos": [], "quat": [], "tau": []}
    for tick in range(targets.shape[0]):
        scene.write_targets(targets[tick], kp, kd)
        scene.step()
        st = scene.read_robot_state()
        logs["q"].append(np.asarray(st.q, dtype=np.float64))
        logs["qd"].append(np.asarray(st.qd, dtype=np.float64))
        logs["pos"].append(np.asarray(st.base_pos_w, dtype=np.float64))
        logs["quat"].append(np.asarray(st.base_quat_w, dtype=np.float64))
        logs["tau"].append(np.asarray(scene.data.ctrl[scene._act_idx], dtype=np.float64))
    out = pathlib.Path(args.out)
    _save(out, "mujoco", names, q0, targets, logs)
    if args.compare:
        return compare(pathlib.Path(args.compare), out, args)
    return 0


def compare(ref_path: pathlib.Path, test_path: pathlib.Path, args) -> int:
    ref, test = np.load(ref_path), np.load(test_path)
    names = [str(n) for n in ref["joint_names"]]
    if names != [str(n) for n in test["joint_names"]]:
        print("[sim2sim] FAIL: joint name lists differ")
        return 1
    if not np.allclose(ref["targets"], test["targets"], atol=1e-6):
        print("[sim2sim] FAIL: target sequences differ (default pose mismatch?)")
        d = np.abs(ref["targets"][0] - test["targets"][0])
        for j in np.argsort(-d)[:5]:
            print(f"    {names[j]:28s} isaac q0={ref['targets'][0, j]:+.4f} mujoco q0={test['targets'][0, j]:+.4f}")
        return 1
    n = min(ref["q"].shape[0], test["q"].shape[0])
    dq = test["q"][:n] - ref["q"][:n]
    rms_t = np.sqrt((dq**2).mean(axis=1))
    dpos = np.linalg.norm(test["pelvis_pos"][:n] - ref["pelvis_pos"][:n], axis=1)
    hold_ticks = int(round(HOLD_S / CONTROL_DT))
    print(f"[sim2sim] {n} ticks compared; hold phase {HOLD_S}s, active {TOTAL_S - HOLD_S}s")
    print("   t(s)  q_rms(rad)  pelvis_diff(m)  isaac_z  mujoco_z")
    for tick in list(range(0, n, int(0.5 / CONTROL_DT))) + [n - 1]:
        print(f"  {tick * CONTROL_DT:5.2f}   {rms_t[tick]:8.4f}     {dpos[tick]:8.4f}      {ref['pelvis_pos'][tick, 2]:.3f}    {test['pelvis_pos'][tick, 2]:.3f}")
    per_joint = np.sqrt((dq**2).mean(axis=0))
    print("[sim2sim] worst joints (RMS over the whole run):")
    for j in np.argsort(-per_joint)[:6]:
        print(f"    {names[j]:28s} {per_joint[j]:.4f} rad")
    worst_q, worst_pos = float(rms_t.max()), float(dpos.max())
    hold_q, hold_pos = float(rms_t[:hold_ticks].max()), float(dpos[:hold_ticks].max())
    print(f"[sim2sim] max q_rms={worst_q:.4f} rad (hold phase {hold_q:.4f}), max pelvis diff={worst_pos:.4f} m (hold phase {hold_pos:.4f})")
    # A static PD hold with the recipe's ankle gains sags and eventually topples in BOTH simulators
    # (the policy balances actively; the plants agree on that too). Once a plant is on the ground the
    # contact chaos makes joint errors meaningless, so the gate is evaluated over the first --window
    # seconds and additionally requires both plants to reach the ground (pelvis z < 0.4 m) within
    # --max-fall-time-diff of each other if either does.
    win = min(n, int(round(args.window / CONTROL_DT)))
    win_q, win_pos = float(rms_t[:win].max()), float(dpos[:win].max())
    fall_ref = _fall_tick(ref["pelvis_pos"][:n, 2])
    fall_test = _fall_tick(test["pelvis_pos"][:n, 2])
    fall_txt = f"fall onset isaac={_tick_s(fall_ref)} mujoco={_tick_s(fall_test)}"
    ok = win_q <= args.max_q_rms and win_pos <= args.max_pelvis_diff
    if (fall_ref is None) != (fall_test is None):
        ok = False
    elif fall_ref is not None and abs(fall_ref - fall_test) * CONTROL_DT > args.max_fall_time_diff:
        ok = False
    print(f"[sim2sim] window {args.window}s: max q_rms={win_q:.4f} rad, max pelvis diff={win_pos:.4f} m; {fall_txt}")
    print(f"[sim2sim] {'PASS' if ok else 'FAIL'} (gate over the window: q_rms <= {args.max_q_rms}, pelvis diff <= {args.max_pelvis_diff}, "
          f"fall onset within {args.max_fall_time_diff}s)")
    return 0 if ok else 1


def _fall_tick(z: np.ndarray, thresh: float = 0.4):
    idx = np.where(z < thresh)[0]
    return int(idx[0]) if idx.size else None


def _tick_s(tick) -> str:
    return "never" if tick is None else f"{tick * CONTROL_DT:.2f}s"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--isaac", action="store_true", help="Run the Isaac Lab half (needs the Kit app).")
    mode.add_argument("--mujoco", action="store_true", help="Run the MuJoCo half (host venv).")
    mode.add_argument("--compare-only", nargs=2, metavar=("ISAAC_NPZ", "MUJOCO_NPZ"), help="Compare two recordings.")
    parser.add_argument("--out", default="/tmp/sim2sim_replay_g1.npz", help="Recording to write.")
    parser.add_argument("--compare", default=None, help="MuJoCo half: Isaac recording to compare against.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-q-rms", type=float, default=0.10, help="Gate on RMS joint error inside --window (rad).")
    parser.add_argument("--max-pelvis-diff", type=float, default=0.10, help="Gate on pelvis position difference inside --window (m).")
    parser.add_argument("--window", type=float, default=1.0, help="Seconds from the start over which the gate is evaluated.")
    parser.add_argument("--max-fall-time-diff", type=float, default=0.25, help="Allowed difference in fall onset time (s).")
    args = parser.parse_args(argv)
    repo = _repo_root()
    if args.compare_only:
        return compare(pathlib.Path(args.compare_only[0]), pathlib.Path(args.compare_only[1]), args)
    if args.isaac:
        return run_isaac(args, repo)
    return run_mujoco(args, repo)


if __name__ == "__main__":
    raise SystemExit(main())
