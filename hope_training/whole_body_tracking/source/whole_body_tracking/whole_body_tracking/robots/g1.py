"""Unitree G1 (29-DoF, ``g1_29dof_rev_1_0``) — BeyondMimic recipe plus the HOPE racket-equipped asset.

The articulation spawns the racket-equipped URDF generated from the vendored Unitree description
(``unitree/URDF/g1_29dof_pingpang/``) and prepared into the git-ignored training asset directory by
``scripts/prepare_g1_isaac_asset.py`` (mirror of the A3 flow). Nothing here touches the filesystem at
import time, so the G1 tasks register without the asset present.

PD gains / armature keep the BeyondMimic G1 recipe (10 Hz natural frequency, damping ratio 2, motor
armatures). NOTE: upstream ``g1_29dof_rev_1_0`` lists 35 N·m for the ankle and waist roll/pitch
motors while the recipe uses 50 N·m; changing an effort limit changes the action scale
(``0.25 * effort / stiffness``), so that is a deliberate, separate decision — keep them in sync with
``G1_ACTION_SCALE`` and ``a3_deploy/a3_deploy_example/config/action_adapter_unitree_g1.yaml`` if you do.

The name constants below mirror ``robots/agibot_a3.py`` so the HOPE config code can be retargeted
mechanically (see ``utils/g1_retarget.py``). G1 body names are lowercase ``*_link`` and the root is
``pelvis`` (the A3 uses ``*_Link`` and ``pelvis_link``); the G1 has no head joints, so its policy
vector is 29-D with no passive columns.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# Asset path — the racket-equipped G1 URDF prepared from unitree/URDF/g1_29dof_pingpang/ by
# scripts/prepare_g1_isaac_asset.py (same layout as the A3's assets/agibot_a3/).
##
UNITREE_G1_ASSET_ROOT = f"{ASSET_DIR}/unitree_g1"
UNITREE_G1_URDF_PATH = f"{UNITREE_G1_ASSET_ROOT}/urdf/model.urdf"
# Fist-grip variant (unitree/URDF/g1_29dof_pingpang_fist/, D20): prepared into its OWN directory so a
# fist run never overwrites the asset that forearm-mount runs (g1_fh_diag, ...) load on requeue.
UNITREE_G1_FIST_ASSET_ROOT = f"{ASSET_DIR}/unitree_g1_fist"
UNITREE_G1_FIST_URDF_PATH = f"{UNITREE_G1_FIST_ASSET_ROOT}/urdf/model.urdf"

##
# Body / joint name constants (real names from the Unitree G1 URDF).
##
G1_ROOT_BODY = "pelvis"
G1_ANCHOR_BODY = "torso_link"

# Bodies tracked by the BeyondMimic motion command (the 14-body G1 set).
G1_TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Feet + hands; used for contact/termination exclusions.
G1_FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
G1_HAND_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

# UPPER-body tracked bodies (torso + both arms); mirror of A3_UPPER_TRACKED.
G1_UPPER_TRACKED = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

# Canonical 29-joint policy order — MUST equal hope_training/config/joint_order_unitree_g1.yaml
# (tests pin the two together) and the live Isaac articulation enumeration (pinned once with
# scripts/check_g1_asset_isaac.py). Isaac/PhysX enumerate breadth-first over the kinematic tree.
UNITREE_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# Racket mount (right arm) — see unitree/scripts/build_g1_pingpang_models.py, the single source of
# truth for the generated URDF/MJCF. A test parses the committed URDF's pingpang_red_joint origin and
# asserts it equals G1_MOUNT_OFFSET, so the two cannot drift apart.
G1_WRIST_BODY = "right_wrist_yaw_link"          # last actuated link of the paddle arm
G1_RACKET_BODY = "pingpang_red_Link"            # racket-centre body (A3 name kept on purpose)
G1_MOUNT_OFFSET = (0.210, 0.015, 0.010)         # wrist_yaw -> blade centre, wrist frame (m)
G1_MOUNT_QUAT = (1.0, 0.0, 0.0, 0.0)            # identity: blade plane XZ, +Y = red face normal
# Fist grip (D20; build script mount FIST): the handle runs across the closed fist, so the racket
# stands perpendicular to the forearm (racket-local +X = wrist +Z, +Y = red face = palm normal).
G1_FIST_MOUNT_OFFSET = (0.083, 0.0226, 0.14)    # wrist_yaw -> blade mesh origin, wrist frame (m)
G1_FIST_MOUNT_QUAT = (0.70710678, 0.0, -0.70710678, 0.0)  # rpy (0, -pi/2, 0)
#: Task-level choice (task YAML ``g1_racket_mount``): mount offset, mount quat, Isaac URDF.
G1_RACKET_MOUNTS = {
    "forearm": (G1_MOUNT_OFFSET, G1_MOUNT_QUAT, UNITREE_G1_URDF_PATH),
    "fist": (G1_FIST_MOUNT_OFFSET, G1_FIST_MOUNT_QUAT, UNITREE_G1_FIST_URDF_PATH),
}

# Contact bodies allowed to touch (feet) or carrying the paddle (wrists + racket links); the
# undesired-contact penalty excludes them via this negative-lookahead regex.
G1_CONTACT_EXCLUDE_REGEX = (
    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
    r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
    r"(?!right_hand_pingpang_Link$)(?!pingpang_red_Link$)(?!pingpang_black_Link$).+$"
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=UNITREE_G1_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

G1_ACTION_SCALE = {}
for a in G1_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
