"""Trace where a G1 policy moves its base between balls in Isaac (the training task, no learning).

Logs every policy step for every env: base xy and heading, the command's base target (the ball's
station), the motion clip frame, the strike frame (the command's exact-strike instant) and resets.
Used to check whether the sideways step after the hit seen in MuJoCo also happens in training, and
what the base target is at that time. ``no_push=true`` turns the shoves off.

Usage:
    hope_isaac_py scripts/trace_base_motion.py task=HOPEPingPongUnitreeG1_FH_FIST_STABLE_SWING \
        checkpoint=logs/rsl_rl/unitree_g1_hitter_pingpong/<run>/model_<i>.pt num_envs=256 num_steps=1500 \
        json_out=/tmp/trace.npz [+no_push=true]
"""

import importlib.util
import os
import pathlib
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf


def _train_module():
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

    importlib.import_module("whole_body_tracking.tasks")
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    train = _train_module()
    checkpoint = os.path.abspath(str(cfg.checkpoint))
    task_id = str(cfg.task.gym_task)
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=int(cfg.num_envs))
    applied: list = []
    train._apply_task_overrides(env_cfg, cfg, applied)
    if bool(cfg.get("no_push", False)):
        env_cfg.events.push_robot = None
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    env_cfg.episode_length_s = 1.0e5  # only falls reset
    motion_files = train._resolve_motion_sources(cfg)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]
    print(f"[trace_base] task={cfg.task.name} envs={cfg.num_envs} overrides={len(applied)} "
          f"no_push={bool(cfg.get('no_push', False))}", flush=True)

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
    motion = base.command_manager.get_term("motion")
    robot = base.scene["robot"]
    origins = base.scene.env_origins
    n_env = base.num_envs
    with torch.inference_mode():
        strike_now = torch.zeros(n_env, dtype=torch.bool, device=base.device)
    orig_vb = cmd._vb_evaluate

    def vb_hook(exact_strike, pos_err):
        strike_now[:] |= exact_strike
        return orig_vb(exact_strike, pos_err)

    cmd._vb_evaluate = vb_hook

    log = {k: [] for k in ("base_xy", "yaw", "target_xy", "racket_target_xy", "clip_frame", "strike", "done")}
    obs = env.get_observations().to(str(cfg.device))
    with torch.inference_mode():
        for step in range(int(cfg.num_steps)):
            strike_now[:] = False
            obs, _, dones, _ = env.step(policy(obs))
            q = robot.data.root_quat_w  # w, x, y, z
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
            log["base_xy"].append((robot.data.root_pos_w[:, :2] - origins[:, :2]).cpu().numpy())
            log["yaw"].append(yaw.cpu().numpy())
            log["target_xy"].append((cmd.base_target_pos_w - origins[:, :2]).cpu().numpy())
            log["racket_target_xy"].append((cmd.racket_target_pos_w[:, :2] - origins[:, :2]).cpu().numpy())
            log["clip_frame"].append(motion.time_steps.cpu().numpy().copy())
            log["strike"].append(strike_now.cpu().numpy().copy())
            log["done"].append(dones.cpu().numpy().astype(bool))
            if step % 250 == 0:
                print(f"[trace_base] step {step}/{cfg.num_steps}", flush=True)
    out = {k: np.stack(v) for k, v in log.items()}
    out["step_dt"] = np.array(float(base.step_dt))
    env.close()
    return out


@hydra.main(version_base=None, config_path="../cfg", config_name="evaluate_forehand")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device)).app
    failed = False
    try:
        out = _run(cfg)
        np.savez_compressed(str(cfg.json_out), **out)
        print(f"[trace_base] wrote {cfg.json_out}: strikes={int(out['strike'].sum())} resets={int(out['done'].sum())}",
              flush=True)
    except Exception:
        import traceback

        print("\n[trace_base] ERROR:", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
