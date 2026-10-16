# Copyright (c) 2025, Intelligent Racing Inc. (dba Hitch Interactive).
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo sim-to-sim evaluation of the exported HOPE ONNX policy.

This drives the exported ``policy.onnx`` (single layout: observation[1, 110]
``hitter_pure`` -> raw_action[1, 31] for the Agibot A3; observation[1, 104]
``hitter_pure_unitree_g1`` -> raw_action[1, 29] with ``--robot unitree_g1``, no
observation normalization) through the SAME observation builder, ActionAdapter,
swing lifecycle and RacketCommand that the clean-room reference deploy runner
uses, so this is a faithful test of the deploy contract rather than a second,
divergent implementation. The robot is selected explicitly (``--robot``), never
inferred from a joint count.

``success_rate`` is measured from an ACTUAL simulated ball. A real ball (free joint,
sphere) is served from the opponent side toward the robot; it flies under gravity +
no-spin aerodynamic drag and really bounces off the racket, table and net inside
MuJoCo. A return succeeds when ALL of the following are observed on the real
trajectory:

  * **contact**   -- an actual MuJoCo contact between the ball geom and the racket
                     collision geom (or the ball passes within the contact radius of
                     the racket site while the racket is moving toward it);
  * **net clear** -- after contact the ball's x crosses the net plane with its centre
                     above the net top; and
  * **opponent-half first bounce** -- the ball's first downward crossing of the table
                     surface (z reaches ball_radius descending) lands on the opponent
                     half, inside the table bounds.

    success_rate = successful_return_tasks / incoming_balls_that_entered_a_strike_task

There is NO analytic predicted-landing substitute: every event above is read off the
real simulated ball. Forehand, backhand and all serves merge into the one number.
Emits ``{"success_rate": <float>, ...}`` to stdout (and optionally --json-out) -- the shipped
metric plus per-serve diagnostics (contact / net-clear / on-opponent / fall rates, the minimum
racket-ball distance and a ``serves`` list) so a 0 % run can be diagnosed; a
one-line human summary goes to stderr.

Two evaluation modes (``--eval-mode``):

  * ``continuous`` (default) — deploy-faithful continuous rally: the robot, policy
    ``last_action``, lifecycle and fixed station are initialized ONCE; between serves
    the policy keeps running through its follow-through/recovery with no state reset
    and no teleport, exactly like the deploy runner. The serve side follows the
    pattern FH, FH, BH, BH, ... so all four adjacent transitions (FH->FH, FH->BH,
    BH->BH, BH->FH) are exercised; the transitions actually seen are reported on
    stderr. A fall keeps affecting later serves — that is the honest continuous
    measurement.
  * ``independent`` — independent-strike evaluation: every serve resets the robot to
    its stand with a fresh lifecycle/last_action/station. This measures isolated
    swings only; it does NOT validate between-swing transitions.

Club forehand mode (``--serve-bank``, Triton Droids): every serve is a question from the G1 forehand
bank (``assets/venue_tuple/g1_fh_crosscourt_apex_bank.npz``): the real feed (diagonal from the
opponent's right, with spin and Magnus lift) is launched from the far end line, a deploy-style planner
(``mujoco_forehand_bank.ApexPlanner``) predicts the real ball's post-bounce apex every tick and
re-solves the racket velocity for the row's target zone once the ball has bounced, and each serve is
also scored with the club definition (``forehand_success``): apex timing read off the REAL ball at
contact, contact, net, opponent half, inside the row's zone. ``--zones`` restricts the questions to
some zones; ``--feed-lead-s`` announces the question that long before the apex (training condition;
0 = only once the ball is launched, the deploy condition). The target zones are drawn on the table.
``--rally-pace`` plays the bank at real rally cadence: the opponent hits the next feed
``--rally-gap-s`` after our contact (our ball's flight to them), the question is only known at that
moment (lead 0 unless ``--feed-lead-s`` is given), and the strike-to-strike interval is reported.

Requires ``mujoco``, ``onnxruntime``, ``pyyaml`` and ``numpy`` plus the shipped
reference deploy package (``a3_deploy/a3_deploy_example``) and the
``a3_pingpong`` MJCF. Runs OUTSIDE Isaac.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "a3_deploy").is_dir() and (parent / "hope_training").is_dir():
            return parent
    return here.parents[3]


def _reference_pkg_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "reference"


_RUNTIME_CONFIGS = {
    "agibot_a3": "hope_pingpong_runtime.yaml",
    "unitree_g1": "hope_pingpong_runtime_unitree_g1.yaml",
}
DEFAULT_ROBOT = "agibot_a3"

# Example serve distribution per robot: strike-point y band per side (MuJoCo/robot frame, forehand
# = robot's right = -y) and the strike height band above the table surface. The A3 values are the
# original neutral examples; the G1 entry starts EQUAL (TODO: retune to the G1's shorter reach once a
# G1 policy exists). ``--serve-y-fh/--serve-y-bh/--serve-z`` override whichever robot is selected.
# Robot-to-table placement and the fixed intercept plane, table-frame x (0 = the near edge), as the
# TRAINING recipes define them: the base stands 0.50 m before the near edge (vb_table_near_x) and the
# strike plane is 0.58 m from the base on the A3 (HOPEPingPong.yaml) / 0.46 m on the G1
# (config/g1/g1_recipe.py). With the old defaults (edge 0.30, plane 0.0) the ball arrived 0.28 m
# closer than any policy was trained for and model_21800 never got within 0.4 m of it.
TABLE_GEOMETRY = {
    "agibot_a3": {"near_edge_x": 0.50, "strike_plane_x": 0.08},
    "unitree_g1": {"near_edge_x": 0.50, "strike_plane_x": -0.04},
}

SERVE_RANGES = {
    "agibot_a3": {"y_fh": (-0.45, -0.12), "y_bh": (0.05, 0.38), "z": (0.10, 0.22)},
    # G1: lateral bands x0.79 (shoulder-to-blade reach 0.62 m vs 0.786 m), strike height band lowered
    # 2 cm with the shoulder (1.05 m vs 1.36 m); see config/g1/g1_recipe.py.
    "unitree_g1": {"y_fh": (-0.36, -0.10), "y_bh": (0.04, 0.30), "z": (0.08, 0.20)},
}


def _default_runtime_config(repo_root: pathlib.Path, robot: str = DEFAULT_ROBOT) -> pathlib.Path:
    return repo_root / "a3_deploy" / "a3_deploy_example" / "config" / _RUNTIME_CONFIGS[robot]


def _load_success_metric(repo_root: pathlib.Path):
    """Import the shared ``success_metric`` module by file path (pure NumPy).

    Importing it as ``whole_body_tracking.utils.success_metric`` would execute the
    training package ``__init__`` (which registers Isaac Lab gym tasks and needs
    gymnasium); this eval only needs the metric's SuccessRate accumulator, the
    TableGeometry.on_opponent_half definition, and the physics-config loader, so we
    load the single file standalone.
    """
    import importlib.util

    path = (
        repo_root / "hope_training" / "whole_body_tracking" / "source" / "whole_body_tracking"
        / "whole_body_tracking" / "utils" / "success_metric.py"
    )
    spec = importlib.util.spec_from_file_location("hope_success_metric", path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (sys.modules[__module__]) works.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------------
# Example serve distribution (planner-less). These are NEUTRAL example incoming balls that arrive in
# front of the robot with a return toward the opponent half; they are NOT the training sampling boxes.
# Replace them (or feed your own planner's RacketCommand stream) for a deployment-faithful evaluation.
# All positions/velocities below are in the MuJoCo world (robot) frame, metres / m·s^-1.
# ---------------------------------------------------------------------------------------------------
def _sample_serve(rng, side, scene, physics, args):
    """Sample (strike_point, serve_origin, serve_velocity) in the MuJoCo world frame.

    A strike point in the robot's reachable zone is chosen (forehand to the robot's
    right, backhand to its left), a serve origin is placed on the opponent side, and
    the serve velocity is the ballistic (no-drag) solution that would carry the ball
    from the origin to the strike point in ``T`` seconds. Real drag then makes the
    ball arrive a little short/low, but the closed-loop intercept predictor re-reads
    the true ball each tick, so the exact serve is not relied upon.
    """
    strike_x = scene.near_edge_x + args.strike_plane_x  # MuJoCo x of the fixed strike plane
    ranges = args.serve_ranges
    if side >= 0:  # FOREHAND -> robot's right = -y in the MuJoCo/robot frame
        y = rng.uniform(*ranges["y_fh"])
    else:          # BACKHAND -> robot's left = +y
        y = rng.uniform(*ranges["y_bh"])
    z = scene.table_height + rng.uniform(*ranges["z"])
    strike_pt = np.array([strike_x, y, z], dtype=np.float64)

    origin = np.array(
        [
            scene.near_edge_x + rng.uniform(1.9, 2.4),
            y + rng.uniform(-0.15, 0.15),
            scene.table_height + rng.uniform(0.15, 0.35),
        ],
        dtype=np.float64,
    )
    flight_t = rng.uniform(0.75, 1.0)
    accel = np.array([0.0, 0.0, -physics.gravity])
    serve_vel = (strike_pt - origin) / flight_t - 0.5 * accel * flight_t
    return strike_pt, origin, serve_vel


def _predict_command(ball_pos, ball_vel, side, scene, physics, args, task_id, revision, RacketCommand):
    """Lightweight inline no-spin intercept predictor -> a RacketCommand.

    Forward-integrates the true ball with the shared no-spin model (gravity +
    quadratic drag) until it crosses the fixed strike plane, producing the racket
    target position (where the ball will be), the time-to-strike, and a target racket
    velocity aimed at the fixed opponent-half landing point. ``swing_sign`` is the
    side locked for this task (harness-internal target semantics only — the policy
    never observes it). This mirrors the planner's job but reads the ball directly
    (the evaluator is the ground truth, so no mocap emulation is needed).
    """
    strike_x = scene.near_edge_x + args.strike_plane_x
    p = np.asarray(ball_pos, dtype=np.float64).copy()
    v = np.asarray(ball_vel, dtype=np.float64).copy()
    dt = 0.002
    t = 0.0
    y_cross, z_cross, tts = float(p[1]), float(p[2]), 0.0
    for _ in range(int(2.0 / dt)):
        a = physics.acceleration(v)
        p_new = p + v * dt + 0.5 * a * dt * dt
        v_new = v + a * dt
        if p[0] >= strike_x > p_new[0]:  # ball moving -x through the strike plane
            dx = p_new[0] - p[0]
            frac = (strike_x - p[0]) / dx if abs(dx) > 1e-12 else 0.5
            y_cross = float(p[1] + frac * (p_new[1] - p[1]))
            z_cross = float(p[2] + frac * (p_new[2] - p[2]))
            tts = t + frac * dt
            break
        p, v = p_new, v_new
        t += dt

    target_pos = np.array([strike_x, y_cross, z_cross], dtype=np.float64)

    # Target racket velocity: the ballistic velocity that would send the ball from the
    # strike point to the fixed opponent-half landing target in a fixed time.
    landing_table = np.array(
        [(scene.net_x_table + scene.length) / 2.0, -scene.width / 2.0, 0.0], dtype=np.float64
    )
    landing_mujoco = landing_table + scene.offset
    t_out = 0.6
    accel = np.array([0.0, 0.0, -physics.gravity])
    target_vel = (landing_mujoco - target_pos) / t_out - 0.5 * accel * t_out

    return RacketCommand(
        task_id=task_id,
        task_revision=revision,
        swing_sign=side,
        position=target_pos,
        velocity=target_vel,
        time_to_strike=float(tts),
    )


def run_eval(args) -> dict:
    repo_root = _repo_root()

    # Make the reference deploy package importable (shared 110-D obs / ActionAdapter /
    # lifecycle / RacketCommand / ONNX wrapper).
    ref_dir = pathlib.Path(args.reference_dir) if args.reference_dir else _reference_pkg_dir(repo_root)
    sys.path.insert(0, str(ref_dir))
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig
    from a3_deploy_onnx_ref_pingpong.lifecycle import SwingLifecycle
    from a3_deploy_onnx_ref_pingpong.observation import build_observation
    from a3_deploy_onnx_ref_pingpong.onnx_policy import OnnxPolicy
    from a3_deploy_onnx_ref_pingpong.racket_command import (
        BACKHAND,
        FOREHAND,
        QueueRacketCommandSource,
        RacketCommand,
    )

    # Shared success metric + physics config (pure NumPy; the DEFINITION only -- fed
    # real simulated events here, never an analytic predicted-landing rollout). Loaded
    # directly by file path so importing it does NOT drag in the Isaac Lab training
    # package (its __init__ registers gym tasks that need gymnasium/Isaac).
    metric = _load_success_metric(repo_root)
    BallPhysics = metric.BallPhysics
    SuccessRate = metric.SuccessRate
    TableGeometry = metric.TableGeometry
    load_ball_physics_config = metric.load_ball_physics_config

    # The MuJoCo scene builder lives next to this script.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    robot = getattr(args, "robot", DEFAULT_ROBOT) or DEFAULT_ROBOT
    runtime_cfg = RuntimeConfig.load(
        args.runtime_config or _default_runtime_config(repo_root, robot), profile=robot
    )
    if getattr(args, "clock", None):
        runtime_cfg.lifecycle.clock = args.clock
    if getattr(args, "tts_offset_s", None) is not None:
        runtime_cfg.lifecycle.tts_offset_s = float(args.tts_offset_s)
    profile = runtime_cfg.profile  # joint order / dims / passive columns / MJCF names
    onnx_path = args.onnx or str(runtime_cfg.onnx_path)
    robot_xml = args.model_xml or str(runtime_cfg.model_xml_path)
    args.serve_ranges = _resolve_serve_ranges(args, profile.name)
    geometry = TABLE_GEOMETRY[profile.name]
    if args.near_edge_x is None:
        args.near_edge_x = geometry["near_edge_x"]
    if args.strike_plane_x is None:
        args.strike_plane_x = geometry["strike_plane_x"]

    ball_cfg = load_ball_physics_config()
    physics = BallPhysics.from_config(ball_cfg)
    table = TableGeometry.from_config(ball_cfg)
    accumulator = SuccessRate()

    policy = OnnxPolicy(onnx_path, profile=profile)
    club = None
    if getattr(args, "serve_bank", None):
        import mujoco_forehand_bank as mfb

        bank_path = mfb.default_bank_path() if args.serve_bank == "default" else pathlib.Path(args.serve_bank)
        venue_prm = mfb.vb.load_venue_params()
        club_cfg = mfb.fs.ForehandSuccessCfg(table_near_x=args.near_edge_x)
        club = {
            "mfb": mfb,
            "bank": mfb.ForehandBank(str(bank_path), zones=args.zones),
            "prm": venue_prm,
            "cfg": club_cfg,
            "planner": mfb.ApexPlanner(venue_prm, club_cfg),
        }
        club["zone_names"] = club["bank"].zone_names
        club["zone_bounds"] = [club_cfg.with_target(z).zone_bounds() for z in club["zone_names"]]
        print(f"[mujoco_eval] club forehand bank: {bank_path} ({len(club['bank'].index)} rows"
              f"{', zones ' + ','.join(args.zones) if args.zones else ''})", file=sys.stderr)
    scene = PingPongRealPhysicsScene(
        robot_xml,
        ball_cfg,
        profile.joint_names,
        control_dt=runtime_cfg.control_dt,
        near_edge_x=args.near_edge_x,
        launch_viewer=args.view,
        base_free_joint=profile.base_free_joint,
        gyro_sensor=profile.gyro_sensor,
        racket_site=profile.racket_site,
        racket_collision_geom=profile.racket_collision_geom,
        floor_geom=profile.floor_geom,
        magnus_k=club["prm"].k_m if club else 0.0,
        zone_rects=[(*bx, *by) for bx, by in club["zone_bounds"]] if club else (),
        table_bounce_fn=club["planner"].table_bounce if club else None,
        racket_contact_fn=club["planner"].racket_hit if club and args.racket_contact == "venue" else None,
    )

    # Aim compensation (landing space): solve the racket for the target moved by the measured mean miss.
    aim_offset = np.asarray(getattr(args, "aim_offset", None) or (0.0, 0.0), dtype=np.float64)

    cam = tracker = None
    if getattr(args, "perception", None):
        if club is None:
            raise ValueError("--perception needs --serve-bank")
        from ball_perception import BallTracker, camera_from_spec

        cam = camera_from_spec(args.perception)
        tracker = BallTracker(cam, np.random.default_rng(args.seed + 7919), club["planner"].surface, club_cfg.net_x)
        scene.ball_track = []
        print(f"[mujoco_eval] perception: {cam}", file=sys.stderr)

    # Optional offscreen mp4 of the session (every --video-every-th tick; needs MUJOCO_GL=egl/osmesa).
    capture = None
    if getattr(args, "video_out", None):
        from mujoco_video import RolloutVideo

        az, el, dist = (float(v) for v in args.video_camera)
        lookat = np.asarray(scene.base_pos_w(), dtype=np.float64) + np.asarray(args.video_lookat_offset, dtype=np.float64)
        capture = RolloutVideo(
            scene.model,
            scene.data,
            args.video_out,
            control_dt=runtime_cfg.control_dt,
            every=int(args.video_every),
            lookat=lookat,
            azimuth=az,
            elevation=el,
            distance=dist,
        )

    default_q = runtime_cfg.action_adapter.default_q.copy()
    kp, kd = runtime_cfg.sim_kp.copy(), runtime_cfg.sim_kd.copy()
    head_idx = list(profile.passive_indices)   # (3, 4) on the A3; empty on the G1
    num_joints = profile.num_joints
    net_clear_z = table.net_height + physics.ball_radius
    contact_radius = args.contact_radius
    dt = runtime_cfg.control_dt
    max_ticks = max(1, int(round(args.max_trial_seconds / dt)))
    rng = np.random.default_rng(args.seed)
    push_rng = np.random.default_rng(args.seed + 7919)  # own stream: the ball sequence is the same with or without --push
    continuous = args.eval_mode == "continuous"

    clock = [0]  # control ticks since the session started (rally cadence bookkeeping)
    trace = [] if getattr(args, "trace_out", None) else None

    def _policy_tick(lifecycle, source, last_action, base_target_xy):
        """One 50 Hz policy step (identical to the deploy runner's tick).

        Returns (events, applied_action) — the caller keeps ``applied_action`` as
        the next tick's ``last_action``.
        """
        state = scene.read_robot_state()
        target = lifecycle.update(source.poll(), state, now=float(scene.data.time))
        obs = build_observation(state, target, last_action, default_q, base_target_xy, num_joints=num_joints)
        raw_action = policy.infer_target(
            obs,
            target.time_to_strike,
            lifecycle.swing_sign,
            runtime_cfg.control_dt,
        )
        # Applied action = raw with the passive head columns zeroed, matching both
        # the deploy runner and training's zeroed last_action feedback.
        applied_action = np.asarray(raw_action, dtype=np.float64).copy()
        if runtime_cfg.passive_neck and head_idx:
            applied_action[head_idx] = 0.0
        q_des = runtime_cfg.action_adapter.decode(applied_action)
        if runtime_cfg.passive_neck and head_idx:
            q_des[head_idx] = default_q[head_idx]
        scene.write_targets(q_des, kp, kd)
        result = scene.step()
        clock[0] += 1
        if capture is not None:
            capture.tick()
        return result, applied_action

    def _park_ball():
        """Drop the ball out of play (past the far edge) between rally serves."""
        scene.set_ball(
            [scene.near_edge_x + scene.length + 1.0, 0.0, scene.table_height + 0.5],
            [0.0, 0.0, 0.0],
        )

    # Continuous mode: ONE initialization for the whole session — robot state,
    # last_action, lifecycle and the base target (the fixed station of this
    # harness, captured at spawn) all persist across serves (matching the deploy
    # runner). Independent mode re-creates them per serve.
    scene.reset_stand()
    lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
    source = QueueRacketCommandSource()
    last_action = np.zeros(num_joints, dtype=np.float64)
    base_target_xy = scene.base_pos_w()[:2].copy()
    spawn_xy = base_target_xy.copy()
    if club is not None and np.linalg.norm(spawn_xy) > 0.05:
        raise RuntimeError(f"--serve-bank assumes the robot spawns at the env origin; stand keyframe base xy = {spawn_xy}")
    max_rest_ticks = max(1, int(round(args.max_rest_seconds / dt)))
    rally = bool(getattr(args, "rally_pace", False))
    if rally and (club is None or not continuous):
        raise ValueError("--rally-pace needs --serve-bank and --eval-mode continuous")
    if getattr(args, "feed_lead_s", None) is None:
        args.feed_lead_s = 0.0 if rally else 0.9
    rally_ref_tick = 0
    last_contact_clock = None
    transitions_seen: set = set()
    prev_side = None
    task_counter = 0
    serve_records: list[dict] = []

    for trial in range(args.num_serves):
        # Side pattern FH, FH, BH, BH, ... exercises all four adjacent side
        # transitions (FH->FH, FH->BH, BH->BH, BH->FH) across the session.
        side = FOREHAND if club is not None or (trial % 4) < 2 else BACKHAND
        if prev_side is not None:
            transitions_seen.add((prev_side, side))
        prev_side = side
        serve_spin = None
        if club is not None:
            row = club["bank"].sample(rng)
            serve_pos, serve_vel, serve_spin = club["mfb"].feed_start(row, club["prm"], club["cfg"])
            _strike_pt = row["target_position_offset"] + np.array([*row["station_xy"], 0.0])
            zone_idx = int(row["target_zone_id"])
            row_v_r = np.asarray(row["target_velocity"], dtype=np.float64)
            scene.highlight_zone(zone_idx)
        else:
            _strike_pt, serve_pos, serve_vel = _sample_serve(rng, side, scene, physics, args)

        if not continuous:
            # Independent-strike evaluation: fresh robot + policy state per serve.
            scene.reset_stand()
            lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
            source = QueueRacketCommandSource()
            last_action = np.zeros(num_joints, dtype=np.float64)
            base_target_xy = scene.base_pos_w()[:2].copy()
            if club is not None:
                base_target_xy = spawn_xy + np.asarray(row["station_xy"], dtype=np.float64)
        elif trial > 0:
            if getattr(args, "reset_after_fall", False) and (serve_records[-1]["fell"] or scene.base_fallen()):
                # a person stands the robot back up; otherwise one fall decides every later ball
                scene.reset_stand()
                lifecycle = SwingLifecycle(runtime_cfg.lifecycle)
                source = QueueRacketCommandSource()
                last_action = np.zeros(num_joints, dtype=np.float64)
                rally_ref_tick = clock[0]
            if club is not None:  # the question's station: the robot steps there during recovery
                base_target_xy = spawn_xy + np.asarray(row["station_xy"], dtype=np.float64)
            # Continuous rally: keep the policy running (no reset, no teleport)
            # through follow-through/recovery until it is ready for the next ball.
            _park_ball()
            if rally:
                # Real cadence: the opponent hits the next ball rally_gap after our contact (or after
                # the missed ball went past), whether or not the robot is ready.
                gap_ticks = int(round(rng.uniform(*args.rally_gap_s) / dt))
                for _ in range(max(0, gap_ticks - (clock[0] - rally_ref_tick))):
                    _events, last_action = _policy_tick(lifecycle, source, last_action, base_target_xy)
            else:
                for _ in range(max_rest_ticks):
                    _events, last_action = _policy_tick(
                        lifecycle, source, last_action, base_target_xy
                    )
                    if lifecycle.phase.value == "ready":
                        break

        if capture is not None and trial > 0:
            capture.mark()
        task_counter += 1
        task_id = task_counter
        revision = 0
        if club is not None:
            if trial == 0:
                base_target_xy = spawn_xy + np.asarray(row["station_xy"], dtype=np.float64)
            # Announce the question --feed-lead-s before the apex (the training condition: the swing
            # clip needs ~0.83 s); the ball is launched from the far end line when its own flight time
            # is left. 0 = announce only at launch (the deploy condition).
            pred0 = club["planner"].predict(serve_pos, serve_vel, serve_spin, after_bounce=False)
            if pred0 is None:
                raise RuntimeError(f"bank row {row['row']}: planner found no apex for the feed")
            wait_ticks = max(0, int(round((float(args.feed_lead_s) - pred0[3]) / dt)))
            _park_ball()
            for k in range(wait_ticks):
                source.submit(RacketCommand(
                    task_id=task_id, task_revision=revision, swing_sign=side, position=pred0[0],
                    velocity=row_v_r, time_to_strike=float(pred0[3] + (wait_ticks - k) * dt),
                    strike_time=float(scene.data.time) + float(pred0[3] + (wait_ticks - k) * dt),
                ))
                revision += 1
                _events, last_action = _policy_tick(lifecycle, source, last_action, base_target_xy)
            club_state = {"bounced": False, "resolved": False, "v_r": row_v_r, "apex_vz": None,
                          "pre_contact_vel": None, "lead_s": wait_ticks * dt + pred0[3],
                          "launch_tick": clock[0], "contact_clock": None,
                          "first_cmd_tts": None, "apex_err_m": None, "tts_err_s": None}
        scene.set_ball(serve_pos, serve_vel, serve_spin)
        if tracker is not None:
            scene.ball_track.clear()
            tracker.reset(float(scene.data.time))
        contacted = False
        net_clear = False
        first_bounce = None
        # per-serve diagnostics (so a 0 % run can be read: never reached the ball / fell over /
        # hit into the net / landed out)
        min_dist = float("inf")
        fell = False
        contact_tick = None
        ticks_run = 0
        # --push: one random shove per ball, like Isaac's push_by_setting_velocity (adds a random
        # base velocity), at a random moment of this ball's window
        push = getattr(args, "push", None)
        push_tick, push_dv = None, None
        if push is not None:
            lo, hi = args.push_window_s
            push_tick = int(round(push_rng.uniform(lo, hi) / dt))
            push_dv = np.concatenate([push_rng.uniform(-push[0], push[0], 2), [0.0],
                                      push_rng.uniform(-push[1], push[1], 3)])

        posture_log = []
        for _tick in range(max_ticks):
            ticks_run = _tick + 1
            ball_pos, ball_vel = scene.ball_state()
            if club is not None:
                cmd = None
                if not contacted:
                    club_state["pre_contact_vel"] = ball_vel.copy()
                    spin_now = scene.ball_spin_w()
                    if tracker is None:
                        pred = club["planner"].predict(ball_pos, ball_vel, spin_now, after_bounce=club_state["bounced"])
                        seen_bounce = club_state["bounced"]
                    else:
                        pred, seen_bounce = _perceived_apex(
                            tracker, cam, scene, club["planner"], ball_pos, ball_vel, spin_now, club_state)
                    if pred is not None:
                        if seen_bounce and not club_state["resolved"]:
                            # re-solve the racket velocity on the real ball's apex state, once
                            club_state["v_r"] = club["planner"].solve_racket(
                                pred[0], pred[1], pred[2],
                                np.asarray(row["intended_landing_xy"], dtype=np.float64) + aim_offset)
                            club_state["resolved"] = True
                        cmd = RacketCommand(
                            task_id=task_id, task_revision=revision, swing_sign=side, position=pred[0],
                            velocity=club_state["v_r"], time_to_strike=float(pred[3]),
                            strike_time=float(scene.data.time) + float(pred[3]),
                        )
                        if club_state["first_cmd_tts"] is None:
                            club_state["first_cmd_tts"] = float(pred[3])
            else:
                cmd = _predict_command(
                    ball_pos, ball_vel, side, scene, physics, args, task_id, revision, RacketCommand
                )
            if cmd is not None:
                revision += 1
                source.submit(cmd)

            if push_tick is not None and _tick == push_tick:
                scene.data.qvel[scene._base_vadr:scene._base_vadr + 6] += push_dv  # free joint: lin (world), ang (body)
            events, last_action = _policy_tick(lifecycle, source, last_action, base_target_xy)
            if trace is not None and club is not None:
                rp, rv = scene.racket_site_state()
                bp_, bv_ = scene.ball_state()
                trace.append({
                    "serve": trial, "tick": _tick, "t": round(float(scene.data.time), 4),
                    "cmd_tts": None if cmd is None else round(float(cmd.time_to_strike), 4),
                    "cmd_pos": None if cmd is None else [round(float(v), 4) for v in cmd.position],
                    "cmd_vel": None if cmd is None else [round(float(v), 4) for v in cmd.velocity],
                    "phase": lifecycle.phase.value,
                    "ball_pos": [round(float(v), 4) for v in bp_], "ball_vel": [round(float(v), 4) for v in bv_],
                    "racket_pos": [round(float(v), 4) for v in rp], "racket_vel": [round(float(v), 4) for v in rv],
                    "racket_normal": [round(float(v), 4) for v in scene.racket_normal_w()],
                    "ball_spin": [round(float(v), 3) for v in scene.ball_spin_w()],
                    "geom_contact": bool(events.ball_racket_contact), "contacted_before": bool(contacted),
                })
            if club is not None and not club_state["bounced"] and not contacted:
                club_state["bounced"] = any(
                    z_sign < 0.0 and x_t < scene.net_x_table for x_t, _y, z_sign in events.surface_crossings
                )

            # --- contact: real MuJoCo ball<->racket contact, or the allowed proximity
            #     fallback (ball within contact_radius of the racket site with the
            #     racket moving toward it). ---
            if scene.base_fallen():
                fell = True
            pose = scene.posture()
            if pose is not None and not fell:
                posture_log.append((pose["pelvis_z"], pose["com_z"], pose["com_offset"]))
            if not contacted:
                r_pos, r_vel = scene.racket_site_state()
                b_pos, _b_vel = scene.ball_state()
                delta = b_pos - r_pos
                min_dist = min(min_dist, float(np.linalg.norm(delta)))
                if events.ball_racket_contact:
                    contacted = True
                elif np.linalg.norm(delta) <= contact_radius and float(np.dot(r_vel, delta)) > 0.0:
                    contacted = True
                if contacted:
                    contact_tick = _tick
                    if club is not None and club_state["pre_contact_vel"] is not None:
                        club_state["apex_vz"] = float(club_state["pre_contact_vel"][2])
                    if club is not None:
                        club_state["contact_clock"] = clock[0]

            # --- after contact, watch the REAL outgoing ball for net clearance and its
            #     first bounce (both read off the simulated trajectory). ---
            if contacted and first_bounce is None:
                if not net_clear:
                    for z_cross, x_sign in events.net_crossings:
                        if x_sign > 0.0 and z_cross > net_clear_z:
                            net_clear = True
                            break
                for x_t, y_t, z_sign in events.surface_crossings:
                    if z_sign < 0.0:
                        first_bounce = (x_t, y_t)
                        break

            if first_bounce is not None:
                break
            # Incoming ball that flew past the robot without ever being contacted is a
            # definite miss (it can no longer produce an opponent-half return): stop the
            # trial early instead of waiting out the timeout.
            if not contacted and scene.ball_state()[0][0] < scene.near_edge_x - 0.8:
                break

        on_opponent = bool(first_bounce is not None and table.on_opponent_half(first_bounce[0], first_bounce[1]))
        success = bool(contacted and net_clear and on_opponent)
        accumulator.add_bool(success)
        club_record = {}
        if club is not None:
            apex_ok = club_state["apex_vz"] is not None and abs(club_state["apex_vz"]) <= club["cfg"].apex_vz_tol
            in_zone = False
            land_w = None
            if first_bounce is not None:
                land_w = [float(first_bounce[0] + scene.offset[0]), float(first_bounce[1] + scene.offset[1])]
                (zx0, zx1), (zy0, zy1) = club["zone_bounds"][zone_idx]
                in_zone = zx0 <= land_w[0] <= zx1 and zy0 <= land_w[1] <= zy1
            club_record = {
                "bank_row": int(row["row"]),
                "zone": row["zone"],
                "apex_vz_at_contact": None if club_state["apex_vz"] is None else round(club_state["apex_vz"], 3),
                "apex_ok": bool(apex_ok),
                "landing_xy": None if land_w is None else [round(v, 3) for v in land_w],
                "intended_landing_xy": [round(float(v), 3) for v in row["intended_landing_xy"]],
                "in_zone": bool(on_opponent and in_zone),
                "club_success": bool(success and apex_ok and in_zone),
                "announced_s_before_apex": round(club_state["lead_s"], 3),
                "first_command_time_to_strike_s": None if club_state["first_cmd_tts"] is None
                else round(club_state["first_cmd_tts"], 3),
                "perceived_apex_err_m": None if club_state["apex_err_m"] is None else round(club_state["apex_err_m"], 4),
                "perceived_time_err_s": None if club_state["tts_err_s"] is None else round(club_state["tts_err_s"], 4),
                "strike_interval_s": None if club_state["contact_clock"] is None or last_contact_clock is None
                else round((club_state["contact_clock"] - last_contact_clock) * dt, 3),
            }
            if club_state["contact_clock"] is not None:
                last_contact_clock = club_state["contact_clock"]
            rally_ref_tick = club_state["contact_clock"] if club_state["contact_clock"] is not None else clock[0]
        serve_records.append(
            {
                "serve": trial,
                "side": "FH" if side == FOREHAND else "BH",
                "success": success,
                "contacted": bool(contacted),
                "contact_tick": contact_tick,
                "min_racket_ball_distance_m": None if not np.isfinite(min_dist) else round(min_dist, 4),
                "net_clear": bool(net_clear),
                "first_bounce_xy": None if first_bounce is None else [round(float(first_bounce[0]), 3), round(float(first_bounce[1]), 3)],
                "on_opponent_half": on_opponent,
                "fell": bool(fell),
                "push": None if push_dv is None else {"t_s": round(push_tick * dt, 3),
                                                      "dv": [round(float(v), 3) for v in push_dv]},
                "ticks": ticks_run,
                "posture": None if not posture_log else {
                    "pelvis_z_mean": round(float(np.mean([p[0] for p in posture_log])), 4),
                    "com_z_mean": round(float(np.mean([p[1] for p in posture_log])), 4),
                    "com_offset_mean": round(float(np.mean([p[2] for p in posture_log])), 4),
                    "com_offset_max": round(float(np.max([p[2] for p in posture_log])), 4)},
                "strike_point_w": [round(float(v), 3) for v in _strike_pt],
                **club_record,
            }
        )

    if capture is not None:
        n_frames = capture.num_frames
        written = capture.close()
        print(
            f"[mujoco_eval] video: {written} ({n_frames} frames)" if written else "[mujoco_eval] video: no frames captured",
            file=sys.stderr,
        )
    scene.close()
    if continuous:
        _names = {FOREHAND: "FH", BACKHAND: "BH"}
        seen = sorted(f"{_names[a]}->{_names[b]}" for a, b in transitions_seen)
        all_four = len(transitions_seen) == 4
        print(
            f"[mujoco_eval] adjacent side transitions exercised: {', '.join(seen) or 'none'}"
            + ("" if all_four else "  (increase --num-serves >= 5 to cover all four)"),
            file=sys.stderr,
        )
    n_serves = max(1, len(serve_records))
    diagnostics = {
        "contact_rate": sum(r["contacted"] for r in serve_records) / n_serves,
        "net_clear_rate": sum(r["net_clear"] for r in serve_records) / n_serves,
        "on_opponent_rate": sum(r["on_opponent_half"] for r in serve_records) / n_serves,
        "fall_rate": sum(r["fell"] for r in serve_records) / n_serves,
        "mean_min_racket_ball_distance_m": float(np.mean([
            r["min_racket_ball_distance_m"] for r in serve_records if r["min_racket_ball_distance_m"] is not None
        ])) if any(r["min_racket_ball_distance_m"] is not None for r in serve_records) else None,
    }
    print(
        f"[mujoco_eval] robot={profile.name} mode={args.eval_mode} serves={accumulator.attempts} "
        f"returns={accumulator.successes} success_rate={accumulator.value:.4f} "
        f"contact_rate={diagnostics['contact_rate']:.2f} net_clear_rate={diagnostics['net_clear_rate']:.2f} "
        f"fall_rate={diagnostics['fall_rate']:.2f}",
        file=sys.stderr,
    )
    out = accumulator.as_dict()
    out.update(diagnostics)
    if club is not None:
        out["club"] = _club_summary(serve_records, club["zone_names"])
        intervals = [r["strike_interval_s"] for r in serve_records if r.get("strike_interval_s") is not None]
        out["club"]["rally_pace"] = rally
        out["club"]["feed_lead_s"] = float(args.feed_lead_s)
        out["club"]["announced_s_before_apex_median"] = float(np.median(
            [r["announced_s_before_apex"] for r in serve_records]))
        out["club"]["strike_interval_s_median"] = float(np.median(intervals)) if intervals else None
        if cam is not None:
            def _med(key):
                vals = [abs(r[key]) for r in serve_records if r.get(key) is not None]
                return float(np.median(vals)) if vals else None
            out["club"]["perception"] = {
                **{k: getattr(cam, k) for k in ("name", "rate_hz", "latency_s", "noise_m", "dropout", "window_s",
                                                "spin", "spin_rel_err", "spin_ang_err_deg")},
                "first_command_time_to_strike_s_median": _med("first_command_time_to_strike_s"),
                "apex_err_m_median": _med("perceived_apex_err_m"),
                "time_err_s_median": _med("perceived_time_err_s"),
            }
        o = out["club"]["overall"]
        print(
            f"[mujoco_eval] club forehand: success={o['success_rate']:.3f} contact={o['contact_rate']:.2f} "
            f"apex={o['apex_rate']:.2f} net={o['net_clear_rate']:.2f} legal={o['on_opponent_rate']:.2f} "
            f"in_zone={o['in_zone_rate']:.2f} (n={o['attempts']})",
            file=sys.stderr,
        )
    out["serves"] = serve_records
    if trace is not None:
        pathlib.Path(args.trace_out).write_text(json.dumps(trace))
    return out


def _perceived_apex(tracker, cam, scene, planner, ball_pos, ball_vel, spin_now, club_state):
    """Planner prediction from what the tracking system reports instead of the true ball.

    Returns (prediction with time-to-strike counted from NOW, bounce seen by the tracker). Also records
    the error of the commanded apex against the planner run on the true state (after the true bounce),
    so the last value before contact is what the policy was finally told."""
    t_now = float(scene.data.time)
    tracker.ingest(scene.ball_track)
    scene.ball_track.clear()
    if cam.oracle:
        p_e, v_e, seen, age = ball_pos, ball_vel, club_state["bounced"], 0.0
    else:
        est = tracker.estimate(t_now)
        if est is None:
            return None, False
        p_e, v_e, seen, age = est.pos, est.vel, est.bounced, t_now - est.t
    w_e = tracker.spin_estimate(spin_now)
    pred = planner.predict(p_e, v_e, w_e, after_bounce=seen)
    if pred is None or pred[3] - age <= 0.0:
        return None, seen
    pred = (pred[0], pred[1], pred[2], pred[3] - age)
    if club_state["bounced"]:
        truth = planner.predict(ball_pos, ball_vel, spin_now, after_bounce=True)
        if truth is not None:
            club_state["apex_err_m"] = float(np.linalg.norm(pred[0] - truth[0]))
            club_state["tts_err_s"] = float(pred[3] - truth[3])
    return pred, seen


def _club_summary(records: list[dict], zone_names) -> dict:
    """Per-zone funnel under the club definition: contact -> + net -> + opponent half -> + zone."""

    def summarize(rs):
        n = max(1, len(rs))
        contact = [r for r in rs if r["contacted"]]
        net = [r for r in contact if r["net_clear"]]
        legal = [r for r in net if r["on_opponent_half"]]
        zone = [r for r in legal if r["in_zone"]]
        return {
            "attempts": len(rs),
            "success_rate": sum(r["club_success"] for r in rs) / n,
            "contact_rate": len(contact) / n,
            "apex_rate": sum(r["apex_ok"] for r in rs) / n,
            "net_clear_rate": len(net) / n,
            "on_opponent_rate": len(legal) / n,
            "in_zone_rate": len(zone) / n,
            "fall_rate": sum(r["fell"] for r in rs) / n,
        }

    return {"overall": summarize(records),
            "per_zone": {z: summarize([r for r in records if r["zone"] == z]) for z in zone_names}}


def _resolve_serve_ranges(args, robot: str) -> dict:
    """The robot's example serve bands, with any ``--serve-*`` CLI overrides applied."""
    ranges = {key: tuple(value) for key, value in SERVE_RANGES[robot].items()}
    for key, attr in (("y_fh", "serve_y_fh"), ("y_bh", "serve_y_bh"), ("z", "serve_z")):
        override = getattr(args, attr, None)
        if override is not None:
            lo, hi = (float(override[0]), float(override[1]))
            if lo > hi:
                raise ValueError(f"--{attr.replace('_', '-')} expects LO HI with LO <= HI, got {override}")
            ranges[key] = (lo, hi)
    return ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot", choices=sorted(_RUNTIME_CONFIGS), default=DEFAULT_ROBOT,
        help="Robot profile: agibot_a3 (31 joints, 110-D; default) or unitree_g1 (29 joints, 104-D). "
             "Selects the default runtime config, joint order, MJCF names and serve bands.",
    )
    parser.add_argument("--onnx", default=None, help="Exported policy.onnx (default: from runtime config).")
    parser.add_argument("--model-xml", default=None, help="Robot MJCF (default: from runtime config).")
    parser.add_argument("--runtime-config", default=None, help="Runtime YAML (default: the shipped one for --robot).")
    parser.add_argument("--serve-y-fh", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                        help="Forehand strike-point y band (robot frame, m); default per robot.")
    parser.add_argument("--serve-y-bh", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                        help="Backhand strike-point y band (robot frame, m); default per robot.")
    parser.add_argument("--serve-z", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                        help="Strike height band above the table surface (m); default per robot.")
    parser.add_argument("--reference-dir", default=None, help="Dir containing the reference deploy package.")
    parser.add_argument("--num-serves", type=int, default=50, help="Number of served balls (denominator).")
    parser.add_argument(
        "--eval-mode", choices=["continuous", "independent"], default="continuous",
        help="continuous (default): one uninterrupted rally session — robot/policy state persist "
             "across serves and all four adjacent side transitions are exercised. "
             "independent: reset the robot per serve (isolated-swing evaluation only).",
    )
    parser.add_argument(
        "--max-rest-seconds", type=float, default=2.0,
        help="continuous mode: max seconds the policy gets between serves to finish "
             "follow-through/recovery before the next ball is served.",
    )
    parser.add_argument(
        "--max-trial-seconds", type=float, default=3.0, help="Max simulated seconds per serve before scoring."
    )
    parser.add_argument("--contact-radius", type=float, default=0.10, help="Racket-site proximity contact fallback (m).")
    parser.add_argument(
        "--near-edge-x", type=float, default=None,
        help="MuJoCo x of the table's near edge (sets the robot-to-table placement). Default: the robot's "
             "training placement (TABLE_GEOMETRY, 0.50 m for both robots).",
    )
    parser.add_argument(
        "--strike-plane-x", type=float, default=None,
        help="Strike plane, table-frame x (0 = the near table edge); the fixed intercept plane. Default: the "
             "robot's training strike plane (TABLE_GEOMETRY: +0.08 A3, -0.04 G1).",
    )
    parser.add_argument("--serve-bank", default=None, metavar="NPZ",
                        help="Club forehand mode: serve questions from this G1 forehand bank ('default' = "
                             "assets/venue_tuple/g1_fh_crosscourt_apex_bank.npz) instead of the example serves.")
    parser.add_argument("--zones", nargs="+", default=None, metavar="ZONE",
                        help="With --serve-bank: only questions for these target zones (default: all).")
    parser.add_argument("--feed-lead-s", type=float, default=None,
                        help="With --serve-bank: announce the question this many seconds before the apex "
                             "(the swing clip needs ~0.83 s; the feed itself flies 0.5-0.8 s). 0 = at launch. "
                             "Default 0.9, or 0 with --rally-pace.")
    parser.add_argument("--rally-pace", action="store_true",
                        help="With --serve-bank (continuous): real rally cadence, the next feed is hit "
                             "--rally-gap-s after our contact and announced at launch.")
    parser.add_argument("--rally-gap-s", nargs=2, type=float, default=(0.7, 1.0), metavar=("LO", "HI"),
                        help="--rally-pace: seconds from our contact to the opponent's contact (default 0.7 1.0).")
    parser.add_argument("--perception", default=None, metavar="PRESET[:k=v,...]",
                        help="With --serve-bank: the planner sees the ball through a camera model (ball_perception.py "
                             "PRESETS: oracle, oracle_nospin, mocap_360, mocap_240, stereo_125, iphone_240, iphone_120, "
                             "iphone_60, webcam_30; overrides e.g. mocap_360:latency_s=0.05). Default: true state.")
    parser.add_argument("--aim-offset", nargs=2, type=float, default=None, metavar=("DX", "DY"),
                        help="With --serve-bank: aim the re-solved racket velocity at target + (DX, DY) m, to "
                             "cancel a systematic landing miss (e.g. 0.35 -0.08 if balls land 35 cm short, 8 cm wide).")
    parser.add_argument("--racket-contact", choices=("mujoco", "venue"), default="mujoco",
                        help="With --serve-bank: MuJoCo's soft racket contact, or the venue-fitted paddle model "
                             "the policy is trained against (analytic, like the table bounce).")
    parser.add_argument("--clock", choices=("countdown", "strike_time"), default=None,
                        help="Lifecycle reference clock (default: the runtime config's, i.e. countdown).")
    parser.add_argument("--tts-offset-s", type=float, default=None,
                        help="Override the runtime config's lifecycle.tts_offset_s (timing calibration, s).")
    parser.add_argument("--trace-out", default=None,
                        help="With --serve-bank: write a per-tick JSON trace (command, ball, racket, contact).")
    parser.add_argument("--push", nargs=2, type=float, default=None, metavar=("V_XY", "W_RPY"),
                        help="Balance test: one shove per ball, a random base velocity change of up to "
                             "V_XY m/s (x, y) and W_RPY rad/s (roll, pitch, yaw), as Isaac's push event.")
    parser.add_argument("--push-window-s", nargs=2, type=float, default=(0.0, 1.0), metavar=("LO", "HI"),
                        help="When the shove comes, in s after the ball is fed (uniform).")
    parser.add_argument("--reset-after-fall", action="store_true",
                        help="Continuous mode: stand the robot back up before the next ball if it fell.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the example serve distribution.")
    parser.add_argument("--view", action="store_true", help="Launch the MuJoCo passive viewer (debug only).")
    parser.add_argument("--json-out", default=None, help="Also write {'success_rate': ...} to this file.")
    parser.add_argument("--video-out", default=None,
                        help="Write an offscreen mp4 of the whole session here (needs MUJOCO_GL=egl or osmesa, "
                             "imageio + imageio-ffmpeg).")
    parser.add_argument("--video-every", type=int, default=2,
                        help="Render every N-th control tick (default 2 -> 25 fps at 50 Hz control).")
    parser.add_argument("--video-camera", nargs=3, type=float, default=(330.0, -22.0, 3.4), metavar=("AZ", "EL", "DIST"),
                        help="Free-camera azimuth/elevation (deg) and distance (m) around the look-at point.")
    parser.add_argument("--video-lookat-offset", nargs=3, type=float, default=(0.5, 0.0, 0.1), metavar=("DX", "DY", "DZ"),
                        help="Look-at point = robot base at spawn + this offset (default leans toward the table).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_eval(args)
    print(json.dumps(result))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
