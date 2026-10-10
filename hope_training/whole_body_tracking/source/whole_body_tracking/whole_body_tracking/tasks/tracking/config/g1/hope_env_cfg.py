"""Unitree G1 — HOPE ping-pong environments (racket-target tracking on the 29-DoF G1).

Two environments, both derived from the Agibot A3 HOPE configs so every reward, observation,
termination and numeric delta of the validated A3 recipe is reused verbatim and only the robot
changes:

* :class:`HOPEPingPongUnitreeG1EnvCfg` — the "full" HOPE stack (BeyondMimic imitation + HITTER
  racket objective; research env, no deploy contract);
* :class:`HitterPingPongUnitreeG1EnvCfg` — the deploy-grade ``hitter_pure`` rally task (mirror of
  ``HitterPingPongAgibotA3EnvCfg``). Its 29-joint actor observation registers as the
  ``hitter_pure_unitree_g1`` contract (104-D; see ``actor_observation_contract.py``).

What :func:`apply_unitree_g1` changes after the A3 ``__post_init__`` chain has run:

1. the articulation (``G1_CYLINDER_CFG``, racket-equipped URDF) and its action scale;
2. motion anchor / tracked bodies, racket wrist body + mount offset (``robots/g1.py``);
3. no passive joints (the G1 has no neck) — the passive-head action columns and the
   passive-head penalty disappear;
4. the A3 message/passive PD randomizer (which refuses non-A3 assets) is replaced by the generic
   actuator-gain randomization;
5. every remaining A3 body name in the config tree is retargeted mechanically
   (``utils/g1_retarget.py``: ``*_Link -> *_link``, ``pelvis_link -> pelvis``, racket links kept),
   including A3 default arguments of MDP functions, and the result is validated to contain no
   A3 name.

NOT retuned here (copied from the A3 recipe, marked for retune once G1 clips exist): the
racket-target sampling boxes (A3 reach), station ranges, ready-reach values and reward stds.

Reference motions: the published ``hope_forehand/backhand.npz`` pair is A3-retargeted (31
joints, 32 bodies) and CANNOT drive these environments — the motion loader rejects it with both
joint counts. ``cfg/task/HOPEPingPongUnitreeG1.yaml`` defaults to ``g1_hope_forehand/backhand.npz``,
joint-space transfers of that pair produced by ``scripts/retarget_a3_clip_to_g1.py`` (swing style
only; G1 reach). See ``docs/UNITREE_G1.md``.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.robots.g1 import (
    G1_ACTION_SCALE,
    G1_ANCHOR_BODY,
    G1_CYLINDER_CFG,
    G1_MOUNT_OFFSET,
    G1_MOUNT_QUAT,
    G1_RACKET_BODY,
    G1_TRACKED_BODIES,
    G1_WRIST_BODY,
)
from whole_body_tracking.tasks.tracking.config.agibot_a3.hope_env_cfg import (
    HitterPingPongAgibotA3EnvCfg,
    HOPEPingPongAgibotA3EnvCfg,
)
from whole_body_tracking.utils.g1_retarget import find_residual_a3_names, retarget_cfg_tree
from whole_body_tracking.tasks.tracking.config.g1 import g1_recipe

# Generic actuator-gain DR for the G1 (the A3 recipe's ``randomize_a3_message_pd_gains`` encodes
# the A3 message-kd / passive-damping split and raises on any other asset). Same ±15 % band as the
# A3 alpha/beta cohort; ``cfg/task/HOPEPingPongUnitreeG1.yaml`` ``domain_rand.pd_gain_range``
# overrides the distribution params through ``train.py``.
G1_PD_GAIN_RANGE = (0.85, 1.15)


def _generic_pd_gain_event() -> EventTerm:
    return EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": G1_PD_GAIN_RANGE,
            "damping_distribution_params": G1_PD_GAIN_RANGE,
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


def apply_unitree_g1(cfg) -> list[str]:
    """Swap a fully built A3 HOPE env cfg onto the Unitree G1 (in place). Returns the retarget log."""
    # 1. robot + action scale
    cfg.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = G1_ACTION_SCALE

    # 2. motion / racket wiring
    cfg.commands.motion.anchor_body_name = G1_ANCHOR_BODY
    cfg.commands.motion.body_names = list(G1_TRACKED_BODIES)
    racket = cfg.commands.racket_target
    racket.wrist_body_name = G1_WRIST_BODY
    racket.racket_body_name = G1_RACKET_BODY
    racket.mount_offset = tuple(G1_MOUNT_OFFSET)
    racket.mount_quat = tuple(G1_MOUNT_QUAT)

    # 3. no passive (neck) joints on the G1
    if hasattr(cfg.actions.joint_pos, "passive_joint_names"):
        cfg.actions.joint_pos.passive_joint_names = ()
    if getattr(cfg.rewards, "passive_head_raw_action", None) is not None:
        cfg.rewards.passive_head_raw_action = None

    # 4. A3-only PD randomizer -> generic actuator-gain DR
    if getattr(cfg.events, "randomize_pd_gains", None) is not None:
        cfg.events.randomize_pd_gains = _generic_pd_gain_event()

    # 5. mechanical body-name retarget of everything else (A3 default arguments included)
    log = retarget_cfg_tree(cfg, skip_attrs=("scene",))
    residual = find_residual_a3_names(cfg, skip_attrs=("scene",))
    if residual:
        details = "\n".join(f"  {path}: {text!r}" for path, text in residual[:12])
        raise ValueError(
            "Unitree G1 env cfg still references Agibot A3 body/joint names after retargeting:\n"
            f"{details}\nExtend whole_body_tracking/utils/g1_retarget.py or clear the term."
        )
    return log


@configclass
class HOPEPingPongUnitreeG1EnvCfg(HOPEPingPongAgibotA3EnvCfg):
    """Full HOPE stack (imitation + racket objective) on the Unitree G1."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.wrap_teleport = False
        apply_unitree_g1(self)


@configclass
class HitterPingPongUnitreeG1EnvCfg(HitterPingPongAgibotA3EnvCfg):
    """Deploy-grade ``hitter_pure`` rally task on the Unitree G1 (104-D actor, 29-D action).

    Identical recipe to ``HitterPingPongAgibotA3EnvCfg`` (same observation/action/reward/termination
    classes and ``__post_init__`` deltas), retargeted to the G1 articulation, then rescaled to the
    G1's geometry by ``g1_recipe.apply_recipe`` (strike plane / boxes x0.81, shoulder-relative
    heights, pelvis-height terms, stance width; see that module for the rule and the evidence).
    """

    def __post_init__(self):
        super().__post_init__()
        apply_unitree_g1(self)
        g1_recipe.apply_recipe(self)
