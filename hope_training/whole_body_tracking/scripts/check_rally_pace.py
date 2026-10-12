"""Isaac check of the rally-pace reference clock and the virtual-ball timing jitter.

Builds the rally task with rally pace forced to 100 % and no ramps, runs a checkpoint, and reports
what the clock actually did: time-to-strike when each swing starts, strike-to-strike intervals,
share of rally-pace swings, the jitter in effect, and falls. Uses the evaluate_forehand config:

    hope_isaac_py scripts/check_rally_pace.py task=HOPEPingPongUnitreeG1_FH_XC_RALLY \
        checkpoint=logs/rsl_rl/unitree_g1_hitter_pingpong/<run>/model_<i>.pt num_envs=256 num_steps=600
"""

import json
import os
import sys

import hydra
from omegaconf import OmegaConf


def _run(cfg) -> dict:
    import gymnasium as gym
    import importlib
    import numpy as np
    import torch

    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    importlib.import_module("whole_body_tracking.tasks")
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate_forehand import _train_module

    train = _train_module()
    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=int(cfg.num_envs))
    applied: list = []
    train._apply_task_overrides(env_cfg, cfg, applied)
    env_cfg.seed = int(cfg.seed)
    env_cfg.commands.motion.rally_pace_prob = 1.0
    env_cfg.commands.motion.rally_pace_ramp_steps = 0
    env_cfg.commands.racket_target.vb_timing_jitter_ramp_steps = 0
    motion_files = train._resolve_motion_sources(cfg)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    env = gym.make(task_id, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env)
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.device = str(cfg.device)
    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(str(cfg.checkpoint)))
    policy = runner.get_inference_policy(device=base.device)

    motion = base.command_manager.get_term("motion")
    cmd = base.command_manager.get_term("racket_target")
    dt = float(base.step_dt)
    n = base.num_envs
    start_tts, intervals, rally_flags, jitters = [], [], [], []
    last_strike = np.full(n, np.nan)
    prev_tts = cmd.time_to_strike.clone()
    falls = 0
    obs = env.get_observations().to(str(cfg.device))
    with torch.inference_mode():
        for step in range(int(cfg.num_steps)):
            obs, _, dones, _ = env.step(policy(obs))
            tts = cmd.time_to_strike
            resampled = motion.just_resampled & ~dones.bool()
            if bool(resampled.any()):
                start_tts += tts[resampled].tolist()
                rally_flags += motion.metrics["rally_pace_swing"][resampled].tolist()
            struck = (prev_tts > 0) & (tts <= 0) & ~dones.bool()
            for e in torch.where(struck)[0].tolist():
                if not np.isnan(last_strike[e]):
                    intervals.append((step - last_strike[e]) * dt)
                last_strike[e] = step
            reset = dones.bool().cpu().numpy()
            last_strike[reset] = np.nan
            falls += int(dones.sum())
            if "virtual_timing_jitter_s" in cmd.metrics:
                jitters.append(float(cmd.metrics["virtual_timing_jitter_s"].max()))
            prev_tts = tts.clone()

    def q(a):
        a = np.asarray(a, dtype=float)
        return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())} if a.size else None

    result = {
        "swing_start_time_to_strike_s": q(start_tts),
        "rally_pace_share_of_wrap_swings": float(np.mean(rally_flags)) if rally_flags else None,
        "strike_to_strike_s": q(intervals),
        "timing_jitter_s": max(jitters) if jitters else None,
        "falls": falls,
        "env_steps": int(cfg.num_steps) * n,
    }
    env.close()
    return result


@hydra.main(version_base=None, config_path="../cfg", config_name="evaluate_forehand")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=str(cfg.device)).app
    failed = False
    try:
        print("[check_rally_pace] " + json.dumps(_run(cfg)), flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        failed = True
    finally:
        app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
