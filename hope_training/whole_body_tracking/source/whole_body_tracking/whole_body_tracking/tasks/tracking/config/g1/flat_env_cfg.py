from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import (
    G1_ACTION_SCALE,
    G1_ANCHOR_BODY,
    G1_CONTACT_EXCLUDE_REGEX,
    G1_CYLINDER_CFG,
    G1_FEET_BODIES,
    G1_HAND_BODIES,
    G1_TRACKED_BODIES,
)
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # The racket-equipped G1 asset (unitree/URDF/g1_29dof_pingpang via scripts/prepare_g1_isaac_asset.py).
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = G1_ANCHOR_BODY
        self.commands.motion.body_names = list(G1_TRACKED_BODIES)

        # Mirror AgibotA3FlatEnvCfg: the paddle links on the right wrist must not count as undesired
        # contacts, and the end-effector fall check watches feet + wrists.
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[G1_CONTACT_EXCLUDE_REGEX]
        )
        if self.terminations.ee_body_pos is not None:
            self.terminations.ee_body_pos.params["body_names"] = G1_FEET_BODIES + G1_HAND_BODIES


@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
