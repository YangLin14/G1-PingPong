"""Per-zone success rate of a G1 forehand policy under the club's success definition.

Runs the policy in Isaac on the training task (default ``HOPEPingPongUnitreeG1_FH_XC``: every
question comes from the cross-court apex bank) and scores every question with
:mod:`whole_body_tracking.utils.forehand_success`:

    success = struck at the bounce apex AND racket contact AND over the net AND first bounce on the
              opponent half AND inside the question's own target zone

* one attempt per bank question issued after warm-up; a question whose env falls (resets) before the
  strike frame is a failed attempt of its zone, so misses never leave the denominator;
* the strike state is read at the command term's exact-strike frame, the same instant the training
  virtual ball scores (ball = the planned contact point, incoming velocity/spin from the bank row,
  racket = achieved position / velocity / face normal);
* the question's zone is recovered from its ``intended_landing_xy`` (``forehand_success.zone_index``);
* the time-out is disabled for the evaluation, so only falls end an episode.

Timing caveat: in this simulator the ball is virtual and reaches the contact point exactly at the
strike frame, and the bank places that frame at the apex, so the timing condition checks the
question rather than the robot. Robot-side timing is tested with a real simulated ball (MuJoCo) or on
the robot.

Usage:
    hope_isaac_py scripts/evaluate_forehand.py \
        checkpoint=logs/rsl_rl/unitree_g1_hitter_pingpong/<run>/model_<i>.pt json_out=/tmp/fh.json
"""

import importlib.util
import json
import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf


def _train_module():
    """scripts/train.py, for its task-override and motion-path helpers (scripts/ is not a package)."""
    path = pathlib.Path(__file__).resolve().with_name("train.py")
    spec = importlib.util.spec_from_file_location("_hope_train_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(cfg) -> dict:
    import gymnasium as gym
    import torch

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    importlib.import_module("whole_body_tracking.tasks")  # registers the gym tasks
    from whole_body_tracking.utils import forehand_success as fs
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    train = _train_module()
    checkpoint = os.path.abspath(str(cfg.checkpoint))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=int(cfg.num_envs))
    applied: list = []
    train._apply_task_overrides(env_cfg, cfg, applied)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.episode_length_s = 1.0e5  # no time-out: only a fall ends an episode (and fails its question)
    motion_files = train._resolve_motion_sources(cfg)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    print(f"[evaluate_forehand] task={cfg.task.name} envs={cfg.num_envs} overrides={len(applied)}", flush=True)

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)
    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=base.device)

    cmd = base.command_manager.get_term("racket_target")
    if getattr(cmd, "_venue_tuple_mix_mode", None) != "fixed_balanced_bank_v1" or not bool(
        cmd.cfg.venue_tuple_unconditional_outcomes
    ):
        raise RuntimeError("the task must use fixed_balanced_bank_v1 with venue_tuple_unconditional_outcomes")
    if float(cmd.cfg.venue_tuple_final_mix_prob) < 1.0:
        print("[evaluate_forehand] WARNING: venue_tuple_final_mix_prob < 1: only bank questions are scored", flush=True)

    scfg = fs.ForehandSuccessCfg(
        table_near_x=float(cmd.cfg.vb_table_near_x),
        table_surface_z=float(cmd.cfg.vb_table_surface_z),
        capture_radius=float(cmd.cfg.vb_capture_radius),
        min_approach_speed=float(cmd.cfg.vb_min_approach_speed),
    )
    report = fs.ZoneSuccessReport(scfg)
    origins = base.scene.env_origins
    n_env = base.num_envs
    # fresh[e]: env e holds a scored question (issued after warm-up) that is not resolved yet.
    with torch.inference_mode():  # the env writes inference tensors; keep ours in the same mode
        fresh = torch.zeros(n_env, dtype=torch.bool, device=base.device)
    scoring = [False]
    strikes: list[dict] = []
    missed: list = []
    counts = {"questions": 0, "falls": 0}

    def _zone(mask):
        return fs.zone_index(cmd._venue_intended_landing_xy[mask], scfg)

    # Hooks on the command term's instance (the term calls these through ``self``).
    orig_resolve = cmd._resolve_pending_venue_outcomes_as_failures
    orig_resample = cmd._resample_command
    orig_vb = cmd._vb_evaluate

    def resolve_hook(ids):
        lost = torch.zeros_like(fresh)
        lost[ids] = True
        lost &= fresh & cmd._venue_tuple_outcome_pending
        if bool(lost.any()):  # reset before the strike frame: a failed attempt of its zone
            missed.append(_zone(lost))
            fresh[lost] = False
        return orig_resolve(ids)

    def resample_hook(env_ids):
        out = orig_resample(env_ids)
        if scoring[0] and len(env_ids) > 0:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=base.device)
            new = cmd._venue_tuple_selected[ids] & cmd._venue_tuple_outcome_pending[ids]
            fresh[ids] = new
            counts["questions"] += int(new.sum())
        return out

    def vb_hook(exact_strike, pos_err):
        m = exact_strike & fresh & cmd._venue_tuple_selected & cmd._venue_tuple_outcome_pending
        if bool(m.any()):
            o = origins[m]
            strikes.append({
                "zone": _zone(m),
                "ball_pos": cmd.racket_target_pos_w[m] - o,
                "ball_vel_in": cmd.vb_vel_in_w[m].clone(),
                "ball_spin_in": cmd.vb_spin_in_w[m].clone(),
                "racket_pos": cmd.racket_pos_w[m] - o,
                "racket_vel": cmd.racket_lin_vel_w[m].clone(),
                "racket_normal": cmd.racket_normal_w[m].clone(),
            })
            fresh[m] = False
        return orig_vb(exact_strike, pos_err)

    cmd._resolve_pending_venue_outcomes_as_failures = resolve_hook
    cmd._resample_command = resample_hook
    cmd._vb_evaluate = vb_hook

    obs = env.get_observations().to(str(cfg.device))
    warmup, total = int(cfg.warmup_steps), int(cfg.warmup_steps) + int(cfg.num_steps)
    with torch.inference_mode():
        for step in range(total):
            scoring[0] = step >= warmup
            obs, _, dones, _ = env.step(policy(obs))
            if scoring[0]:
                counts["falls"] += int(dones.sum())
            if step % 500 == 0:
                done = sum(len(s["zone"]) for s in strikes)
                print(f"[evaluate_forehand] step {step}/{total} strikes={done} misses={sum(len(z) for z in missed)}", flush=True)

    if strikes:
        cat = {k: torch.cat([s[k] for s in strikes]) for k in strikes[0]}
        report.add_strikes(cat.pop("zone"), **cat)
    for z in missed:
        report.add_missed(z)
    result = report.as_dict()
    result["meta"] = {
        "checkpoint": checkpoint,
        "task": str(cfg.task.name),
        "num_envs": n_env,
        "scored_steps": int(cfg.num_steps),
        "questions_issued": counts["questions"],
        "unresolved_at_end": int(fresh.sum()),
        "falls": counts["falls"],
        "zones": {n: [list(b) for b in c.zone_bounds()] for n, c in report.zone_cfgs.items()},
    }
    env.close()
    return result


def _print_table(result: dict) -> None:
    cols = ("attempts", "success_rate", "contact_rate", "net_clear_rate", "on_opponent_rate", "in_zone_rate")
    print(f"{'zone':<18}" + "".join(f"{c:>18}" for c in cols), flush=True)
    rows = list(result["per_zone"].items()) + [("OVERALL", result["overall"])]
    for name, r in rows:
        cells = "".join(f"{r[c]:>18d}" if c == "attempts" else f"{r[c]:>18.3f}" for c in cols)
        print(f"{name:<18}{cells}", flush=True)


@hydra.main(version_base=None, config_path="../cfg", config_name="evaluate_forehand")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device)).app
    failed = False
    try:
        result = _run(cfg)
        _print_table(result)
        print(json.dumps(result), flush=True)
        if cfg.json_out:
            with open(str(cfg.json_out), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
    except Exception:
        import traceback

        print("\n[evaluate_forehand] ERROR:", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
