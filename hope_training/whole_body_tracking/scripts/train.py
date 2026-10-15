"""Hydra training entry for the HOPE Agibot A3 policy.

Single task, single algo. Build the ``HOPE-HitterPingPong-AgibotA3-v0`` environment (110-D
``hitter_pure`` actor observation, privileged critic, 50 Hz control, ``wrap_teleport: false``), a
rsl_rl PPO runner, and train. Checkpoints are written locally (periodic every ``save_interval`` and
a final one). There is no Weights & Biases, no external logging service, no gate / lineage
machinery.

Usage:
    python scripts/train.py task=HOPEPingPong algo=ppo headless=true

Override any field on the CLI, e.g.:
    python scripts/train.py task=HOPEPingPong num_envs=2048 max_iterations=20000 seed=1 \
        motion_file=/abs/hope_forehand.npz motion_file_2=/abs/hope_backhand.npz

Tune training by editing cfg/task/HOPEPingPong.yaml and cfg/algo/ppo.yaml.
"""

import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf


def _repo_root() -> pathlib.Path:
    """Repo root = the directory that contains ``hope_training/`` (walk up from this file)."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir():
            return parent
    return here.parents[2]


def _resolve_motion_path(value: str) -> str:
    """Resolve a clip path: absolute / cwd-relative first, then repo-root-relative."""
    p = pathlib.Path(str(value))
    if p.is_file():
        return str(p.resolve())
    repo_root = _repo_root()
    candidates = (
        repo_root / value,
        repo_root / "hope_training" / "whole_body_tracking" / value,
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    # Return the training-root candidate because task YAML paths are relative
    # to hope_training/whole_body_tracking, independent of the launch cwd.
    return str(candidates[-1].resolve())


def _resolve_motion_sources(cfg) -> list[str]:
    """Return the list of local clip paths [forehand, backhand] (CLI overrides the task cfg)."""
    primary = cfg.motion_file if cfg.motion_file is not None else cfg.task.get("motion_file")
    secondary = cfg.motion_file_2 if cfg.motion_file_2 is not None else cfg.task.get("motion_file_2")
    clips = [primary]
    if secondary is not None:
        clips.append(secondary)
    resolved = [_resolve_motion_path(c) for c in clips if c is not None]
    if not resolved:
        raise RuntimeError(
            "No motion clip configured. Set motion_file (and motion_file_2) on the CLI or in "
            "cfg/task/HOPEPingPong.yaml."
        )
    for clip in resolved:
        if not pathlib.Path(clip).is_file():
            raise FileNotFoundError(
                f"motion clip not found: {clip}\nUse the published clips under "
                "hope_training/motions/preprocessed/ or provide your own pair "
                "(see docs/REPLACE_MOTIONS.md)."
            )
    return resolved


def _set_dotted(obj, dotted: str, value, applied: list, where: str) -> None:
    """Set ``obj.<a>.<b>... = value`` if the attribute chain exists; else warn and skip."""
    parts = dotted.split(".")
    node = obj
    for attr in parts[:-1]:
        if not hasattr(node, attr):
            print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{attr}'; skipped.", flush=True)
            return
        node = getattr(node, attr)
    leaf = parts[-1]
    if not hasattr(node, leaf):
        print(f"[train.py] WARNING: {where}: '{dotted}' — no attribute '{leaf}'; skipped.", flush=True)
        return
    setattr(node, leaf, value)
    applied.append(f"{dotted} = {value}")


def _apply_domain_rand(env_cfg, dr, applied: list) -> None:
    """Apply the shared link-mass / PD-gain randomization knobs.

    The event terms are named ``events.randomize_link_mass`` and
    ``events.randomize_pd_gains`` in :class:`HOPEEventCfg` — the override MUST target
    those exact fields (see ``tests/test_domain_rand_overrides.py``). Semantics per
    range knob:
      * absent          -> keep the env-cfg default;
      * ``null``        -> disable the event entirely (set the term to None);
      * ``[lo, hi]``    -> override the distribution parameters.

    PD gains have two modes:
      * ``pd_mode`` absent (legacy generic scale DR): ``pd_gain_range`` drives the
        stiffness/damping distribution params of ``events.randomize_pd_gains``;
      * ``pd_mode: a3_message_passive_nominal_cohort_v1`` (the shipped
        HitterPingPong recipe, ``mdp.randomize_a3_message_pd_gains``):
        ``pd_gain_range`` MUST be null (it only retires the generic scale DR — the
        a3-message term stays), and ``pd_alpha_range`` / ``pd_beta_range`` /
        ``pd_nominal_fraction`` override the term's alpha/beta/nominal params.
    """
    if dr is None:
        return
    events = getattr(env_cfg, "events", None)
    if events is None:
        return

    def _apply(range_key: str, event_name: str, param_keys: tuple[str, ...]) -> None:
        if range_key not in dr:
            return
        if not hasattr(events, event_name):
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} does not "
                "exist on this env cfg; skipped.",
                flush=True,
            )
            return
        rng = dr.get(range_key)
        if rng is None:
            if getattr(events, event_name) is not None:
                setattr(events, event_name, None)
                applied.append(f"events.{event_name} = None (disabled)")
            return
        term = getattr(events, event_name)
        if term is None:
            print(
                f"[train.py] WARNING: domain_rand.{range_key}: events.{event_name} is already "
                "disabled in the env cfg; range ignored.",
                flush=True,
            )
            return
        lo, hi = float(rng[0]), float(rng[1])
        for key in param_keys:
            term.params[key] = (lo, hi)
        applied.append(f"events.{event_name} = {(lo, hi)}")

    _apply("link_mass_range", "randomize_link_mass", ("mass_distribution_params",))
    _apply_push(events, dr, applied)

    pd_mode = dr.get("pd_mode")
    if pd_mode is None:
        _apply(
            "pd_gain_range",
            "randomize_pd_gains",
            ("stiffness_distribution_params", "damping_distribution_params"),
        )
        return

    # a3-message PD cohort (the env cfg already installs randomize_a3_message_pd_gains
    # for this recipe): pd_gain_range null only documents that the generic scale DR is
    # off; the alpha/beta/nominal knobs refine the installed term.
    if dr.get("pd_gain_range") is not None:
        print(
            f"[train.py] WARNING: domain_rand.pd_mode={pd_mode!r} requires "
            "pd_gain_range: null (the a3-message term replaces the generic scale DR); "
            "pd_gain_range ignored.",
            flush=True,
        )
    term = getattr(events, "randomize_pd_gains", None)
    if term is None:
        print(
            f"[train.py] WARNING: domain_rand.pd_mode={pd_mode!r}: "
            "events.randomize_pd_gains is disabled in the env cfg; PD knobs ignored.",
            flush=True,
        )
        return
    for range_key, param_key in (
        ("pd_alpha_range", "alpha_range"),
        ("pd_beta_range", "beta_range"),
    ):
        rng = dr.get(range_key)
        if rng is not None:
            term.params[param_key] = (float(rng[0]), float(rng[1]))
            applied.append(f"events.randomize_pd_gains.{param_key} = {term.params[param_key]}")
    nominal = dr.get("pd_nominal_fraction")
    if nominal is not None:
        term.params["nominal_fraction"] = float(nominal)
        applied.append(f"events.randomize_pd_gains.nominal_fraction = {float(nominal)}")


def _apply_push(events, dr, applied: list) -> None:
    """Install random shoves (``events.push_robot``) from ``domain_rand.push_*`` (balance training).

    The HOPE recipe has no pushes (HITTER alignment, ``HOPEEventCfg.push_robot = None``). With
    ``push_velocity_range`` set, a ``mdp.push_by_setting_velocity_ramped`` interval event is installed:
      * ``push_velocity_range``: {x, y, z, roll, pitch, yaw: [lo, hi]} base velocity change (m/s, rad/s);
      * ``push_interval_s``: [lo, hi] seconds between shoves per env (default [2, 4]);
      * ``push_ramp_steps`` / ``push_ramp_start_steps``: env steps (``env.common_step_counter``) over
        which the range grows from 0 to full, and before which there are none (default 0 / 0).
    Absent or null ``push_velocity_range`` keeps the env cfg as it is.
    """
    rng = dr.get("push_velocity_range")
    if rng is None:
        return
    from isaaclab.managers import EventTermCfg

    from whole_body_tracking.tasks.tracking import mdp

    velocity_range = {str(k): (float(v[0]), float(v[1])) for k, v in dict(rng).items()}
    unknown = set(velocity_range) - {"x", "y", "z", "roll", "pitch", "yaw"}
    if unknown:
        raise ValueError(f"domain_rand.push_velocity_range: unknown axes {sorted(unknown)}")
    interval = dr.get("push_interval_s") or (2.0, 4.0)
    events.push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity_ramped,
        mode="interval",
        interval_range_s=(float(interval[0]), float(interval[1])),
        params={
            "velocity_range": velocity_range,
            "ramp_steps": int(dr.get("push_ramp_steps") or 0),
            "start_steps": int(dr.get("push_ramp_start_steps") or 0),
        },
    )
    applied.append(
        f"events.push_robot = ramped push {velocity_range} every {tuple(events.push_robot.interval_range_s)} s, "
        f"ramp {events.push_robot.params['ramp_steps']} steps from step {events.push_robot.params['start_steps']}"
    )


def _apply_g1_racket_mount(env_cfg, mount, applied: list) -> None:
    """``g1_racket_mount: fist`` (task YAML) swaps the G1 racket mount: Isaac URDF, racket-target mount
    offset and quat (``robots/g1.py G1_RACKET_MOUNTS``). Absent / ``forearm`` keeps the env cfg as built.
    The fist asset lives in its own prepared directory and is prepared here if missing."""
    if mount is None or str(mount) == "forearm":
        return
    from whole_body_tracking.robots.g1 import G1_RACKET_MOUNTS

    mount = str(mount)
    if mount not in G1_RACKET_MOUNTS:
        raise ValueError(f"g1_racket_mount={mount!r}; choose from {sorted(G1_RACKET_MOUNTS)}")
    offset, quat, urdf = G1_RACKET_MOUNTS[mount]
    urdf_path = pathlib.Path(urdf)
    if not urdf_path.is_file():
        import subprocess

        repo = pathlib.Path(__file__).resolve().parents[3]
        subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve().parent / "prepare_g1_isaac_asset.py"),
             "--source-root", str(repo / "unitree" / "URDF" / f"g1_29dof_pingpang_{mount}"),
             "--output-root", str(urdf_path.parents[1]), "--force"],
            check=True,
        )
    env_cfg.scene.robot.spawn.asset_path = str(urdf_path)
    env_cfg.commands.racket_target.mount_offset = tuple(offset)
    env_cfg.commands.racket_target.mount_quat = tuple(quat)
    applied.append(f"g1_racket_mount={mount}: asset {urdf_path}, mount_offset {tuple(offset)}, mount_quat {tuple(quat)}")


def _apply_task_overrides(env_cfg, cfg, applied: list) -> None:
    """Apply the launcher-level knobs + generic dotted-path overrides from the task cfg."""
    from whole_body_tracking.utils.task_reward_overrides import apply_reward_overrides

    task = cfg.task
    # episode length (top-level on ManagerBasedRLEnvCfg).
    env_block = task.get("env")
    if env_block is not None and env_block.get("episode_length_s") is not None:
        _set_dotted(env_cfg, "episode_length_s", float(env_block.get("episode_length_s")), applied, "env")
    # continuous multi-rally lifecycle: no teleport on clip wrap.
    motion_block = task.get("motion")
    if motion_block is not None and motion_block.get("wrap_teleport") is not None:
        _set_dotted(
            env_cfg, "commands.motion.wrap_teleport", bool(motion_block.get("wrap_teleport")), applied, "motion"
        )
    # G1 racket mount (forearm = every run before 2026-10-14; fist = D20).
    _apply_g1_racket_mount(env_cfg, task.get("g1_racket_mount"), applied)
    # domain randomization.
    _apply_domain_rand(env_cfg, task.get("domain_rand"), applied)
    # Reward terms are registered by the EnvCfg, but Hydra's rewards mapping is
    # not applied by Isaac Lab automatically. Keep the YAML as the recipe source
    # of truth and fail loudly on any unmapped or misspelled field.
    apply_reward_overrides(env_cfg.rewards, task.get("rewards"), applied)
    # generic overrides map (dotted attribute paths -> value).
    overrides = task.get("overrides")
    if overrides:
        for dotted, value in OmegaConf.to_container(overrides, resolve=True).items():
            _set_dotted(env_cfg, str(dotted), value, applied, "overrides")


def _run(cfg):
    import gymnasium as gym
    import torch
    from datetime import datetime

    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    import importlib

    importlib.import_module("whole_body_tracking.tasks")  # registers the gym task
    from whole_body_tracking.utils.my_on_policy_runner import HOPEOnPolicyRunner
    from whole_body_tracking.utils.ppo_cfg import runner_kwargs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    task_id = str(cfg.task.gym_task)
    num_envs = int(cfg.num_envs) if cfg.num_envs is not None else int(cfg.task.env.num_envs)

    # 1) environment cfg from the registered gym task + task-cfg overrides.
    env_cfg = parse_env_cfg(task_id, device=str(cfg.device), num_envs=num_envs)
    applied: list = []
    _apply_task_overrides(env_cfg, cfg, applied)
    env_cfg.seed = int(cfg.seed)
    env_cfg.sim.device = str(cfg.device)
    print(f"[train.py] task={task_id} num_envs={num_envs} — applied {len(applied)} task override(s):", flush=True)
    for line in applied:
        print(f"[train.py]     {line}", flush=True)

    # 2) reference motion clips (local .npz; clip 0 = forehand, clip 1 = backhand).
    motion_files = _resolve_motion_sources(cfg)
    for i, mf in enumerate(motion_files):
        print(f"[train.py] motion clip {i}: {mf}", flush=True)
    env_cfg.commands.motion.motion_file = motion_files if len(motion_files) > 1 else motion_files[0]

    # 3) PPO runner cfg from cfg/algo/ppo.yaml.
    algo = OmegaConf.to_container(cfg.algo, resolve=True)
    agent_cfg = RslRlOnPolicyRunnerCfg(**runner_kwargs(algo, str(cfg.task.experiment_name)))
    agent_cfg.seed = int(cfg.seed)
    agent_cfg.device = str(cfg.device)
    if cfg.max_iterations is not None:
        agent_cfg.max_iterations = int(cfg.max_iterations)
    if cfg.run_name is not None:
        agent_cfg.run_name = str(cfg.run_name)
    agent_cfg.logger = str(cfg.get("logger", "tensorboard")).lower()
    agent_cfg.wandb_project = str(cfg.get("wandb_project", "hope"))

    # 4) local logging directory.
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[train.py] experiment={agent_cfg.experiment_name} | log_dir={log_dir}", flush=True)
    if agent_cfg.logger == "wandb":
        # Keep the W&B run files next to the local run instead of a stray ./wandb in the launch dir.
        os.makedirs(log_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", log_dir)
        print(f"[train.py] logger=wandb project={agent_cfg.wandb_project} (WANDB_DIR={os.environ['WANDB_DIR']})", flush=True)

    # 5) build env, (optionally) record video, wrap for rsl_rl.
    render_mode = "rgb_array" if cfg.video else None
    env = gym.make(task_id, cfg=env_cfg, render_mode=render_mode)

    # Validate the actor observation contract when the task declares one (guarded import).
    # The name (e.g. hitter_pure, 110-D) is resolved generically through the contract registry.
    expected_contract = cfg.task.get("actor_obs_contract")
    if expected_contract is not None:
        try:
            from whole_body_tracking.tasks.tracking.actor_observation_contract import (
                validate_actor_observation_contract,
            )

            contract = validate_actor_observation_contract(env.unwrapped, str(expected_contract))
            print(
                f"[train.py] actor observation contract validated: {contract.name} "
                f"({contract.total_dim}D)",
                flush=True,
            )
        except ImportError:
            print("[train.py] NOTE: actor_observation_contract validator not available; skipping.", flush=True)

    video_env = None
    if cfg.video:
        from whole_body_tracking.utils.wandb_video import WandbRecordVideo

        video_interval = int(cfg.video_interval)
        # Also capture one early clip (iteration 2) so a fresh run shows a rollout right away.
        early_step = 2 * int(agent_cfg.num_steps_per_env)
        video_env = env = WandbRecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % video_interval == 0 or step == early_step,
            video_length=int(cfg.video_length),
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = HOPEOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if video_env is not None:
        video_env.step_provider = lambda: int(runner.current_learning_iteration)
    runner.add_git_repo_to_log(__file__)

    # 6) optional resume from a local checkpoint (strict: weights + optimizer + iteration counter).
    ckpt = getattr(cfg, "checkpoint_path", None)
    if ckpt is not None:
        ckpt = os.path.abspath(str(ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"[train.py] checkpoint_path does not exist: {ckpt}")
        runner.load(ckpt)
        print(f"[train.py] resumed from checkpoint: {ckpt}", flush=True)
        if bool(getattr(cfg, "restore_step_counter", False)):
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            buffers = (saved.get("hope_exact_resume_state") or {}).get("environment_resume_state", {})
            counter = (buffers or {}).get("env_buffers", {}).get("common_step_counter")
            if counter is not None:
                env.unwrapped.common_step_counter = int(counter)
                print(f"[train.py] restored common_step_counter={int(counter)}", flush=True)
            else:
                print("[train.py] WARNING: checkpoint has no common_step_counter; ramps restart at 0", flush=True)

    # 7) dump the resolved configuration + train.
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # Launch Isaac Sim BEFORE importing isaaclab modules. Clear argv so Kit does not try to parse
    # Hydra's task=.../algo=... overrides.
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=bool(cfg.headless), device=str(cfg.device), enable_cameras=bool(cfg.video))
    simulation_app = app_launcher.app

    failed = False
    try:
        _run(cfg)
    except Exception:
        import traceback

        print("\n[train.py] ERROR during run:", flush=True)
        traceback.print_exc()
        failed = True
    finally:
        simulation_app.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
