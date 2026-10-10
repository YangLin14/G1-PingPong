"""Unitree G1 HitterPingPong recipe: the A3 numbers rescaled to the G1's geometry (variant A).

Pure Python (no Isaac imports) so the host tests can load it by path and pin
``cfg/task/HOPEPingPongUnitreeG1.yaml`` to it. Applied by
``HitterPingPongUnitreeG1EnvCfg.__post_init__`` (``apply_recipe``); ``train.py`` applies the
``rewards:`` block of the YAML on top, which must carry the same values (tested).

Why (from the first G1 run, job 4047, 24.5k iterations, W&B triton-pupper/hope-g1):
  * the strike matrix showed a systematic -9 cm x / -10 cm z (high targets) racket position error:
    the G1 cannot reach the A3 strike boxes. Shoulder-to-blade reach (sum of the right-arm joint
    origins from shoulder_roll to wrist_yaw + 0.21 m racket) is 0.62 m on the G1 vs 0.786 m on
    the A3 -> ratio 0.79, and the G1 shoulder stands 0.31 m lower (1.05 m vs 1.36 m);
  * ``rally_ready_root_height_debt`` asked for a 0.98 m pelvis from a robot whose pelvis stands at
    0.76 m: a constant -1.2/episode debt (the largest negative term of the run).

Scaling rule (documented in docs/UNITREE_G1.md):
  horizontal reach x0.79 (REACH_RATIO); heights shoulder-relative (the A3 z box [0.85, 1.30] is
  [shoulder-0.51, shoulder-0.06] -> G1 [0.54, 0.99], clamped to >= table top + 4 cm and extended
  to shoulder + 5 cm so balls above the table stay hittable); pelvis-height terms by the pelvis
  ratio 0.76/1.068; stance by the hip spacing ratio 0.233/0.269 (STANCE_RATIO).
Everything that is world-scale or robot-agnostic (racket velocity boxes, strike window / phases,
station steps, domain randomization, PPO) stays the A3 value.
"""

from __future__ import annotations

# --- geometry (metres), from the two URDFs standing in their init poses ---------------------------
A3_PELVIS_Z = 1.0684          # robots/agibot_a3.py init_state
G1_PELVIS_Z = 0.76            # robots/g1.py init_state
A3_SHOULDER_Z = 1.36          # pelvis + waist_pitch (0.005) + shoulder_pitch origin (0.284)
G1_SHOULDER_Z = 1.05          # pelvis + waist_roll (0.044) + shoulder_pitch origin (0.248)
A3_ARM_REACH = 0.786          # shoulder_roll..wrist_yaw joint-origin chain 0.576 + 0.21 racket
G1_ARM_REACH = 0.62           # chain 0.41 + 0.21 racket
REACH_RATIO = 0.79            # G1_ARM_REACH / A3_ARM_REACH
STANCE_RATIO = 0.87           # hip spacing 0.233 m / 0.269 m
TABLE_TOP_Z = 0.76

# --- A3 reference values (HOPEPingPong.yaml / HitterPingPongAgibotA3EnvCfg) ------------------------
A3_STRIKE_PLANE_X = 0.58
A3_POS_RANGE_PER_CLIP = (((0.58, 0.58), (-0.48, -0.40), (0.85, 1.30)), ((0.58, 0.58), (-0.13, -0.05), (0.85, 1.30)))
A3_VENUE_CONTACT_Z_RANGE = (0.98, 1.26)
A3_ROOT_HEIGHT_MIN = 0.98
A3_BASE_TOO_LOW = 0.5
A3_STANCE_WIDTH = (0.25, 0.35)
A3_HOLD_READY_REACH = 0.3

# --- G1 values -------------------------------------------------------------------------------------
STRIKE_PLANE_X = 0.46                     # 0.58 x 0.79 (4 cm before the table's near edge at 0.50)
Z_BOX = (0.80, 1.10)                      # see the scaling rule above
POS_RANGE_PER_CLIP = (
    ((0.46, 0.46), (-0.38, -0.32), Z_BOX),  # forehand: y (-0.48, -0.40) x 0.79
    ((0.46, 0.46), (-0.10, -0.04), Z_BOX),  # backhand: y (-0.13, -0.05) x 0.79
)
# Generic boxes used by the non-per-clip sampling paths (A3: x (0.25, 0.55), y_abs (0.05, 0.45), z (0.70, 1.15)).
POS_X_RANGE = (0.20, 0.45)
POS_Y_ABS_RANGE = (0.04, 0.36)
POS_Z_RANGE = (0.70, 1.10)
VENUE_CONTACT_Z_RANGE = (0.84, 1.10)      # A3 (0.98, 1.26) shifted by the shoulder delta, clamped to the z box
ROOT_HEIGHT_MIN = 0.70                    # 0.98 x (0.76 / 1.068)
BASE_TOO_LOW = 0.40                       # 0.5 x (0.76 / 1.068), rounded
STANCE_WIDTH = (0.22, 0.30)               # (0.25, 0.35) x 0.87
HOLD_READY_REACH = 0.24                   # 0.3 x 0.79

#: ``rewards:`` keys of cfg/task/HOPEPingPongUnitreeG1.yaml that must carry these values.
YAML_REWARD_VALUES = {
    "rally_ready_root_height_debt_min_height": ROOT_HEIGHT_MIN,
    "ready_stance_width_lo": STANCE_WIDTH[0],
    "ready_stance_width_hi": STANCE_WIDTH[1],
    "stance_width_lo": STANCE_WIDTH[0],
    "stance_width_hi": STANCE_WIDTH[1],
    "hold_ready_reach": HOLD_READY_REACH,
}


def apply_recipe(cfg) -> list[str]:
    """Write the G1 values into a built HitterPingPong env cfg (in place). Returns a change log."""
    log: list[str] = []
    racket = cfg.commands.racket_target
    racket.racket_pos_range_per_clip = tuple(tuple(tuple(r) for r in clip) for clip in POS_RANGE_PER_CLIP)
    racket.racket_pos_x_range = POS_X_RANGE
    racket.racket_pos_y_abs_range = POS_Y_ABS_RANGE
    racket.racket_pos_z_range = POS_Z_RANGE
    racket.venue_tuple_contact_z_range = VENUE_CONTACT_Z_RANGE
    log += [
        f"commands.racket_target.racket_pos_range_per_clip = {POS_RANGE_PER_CLIP}",
        f"commands.racket_target.racket_pos_x_range = {POS_X_RANGE}",
        f"commands.racket_target.racket_pos_y_abs_range = {POS_Y_ABS_RANGE}",
        f"commands.racket_target.racket_pos_z_range = {POS_Z_RANGE}",
        f"commands.racket_target.venue_tuple_contact_z_range = {VENUE_CONTACT_Z_RANGE}",
    ]
    rewards = cfg.rewards
    term = getattr(rewards, "rally_ready_root_height_debt", None)
    if term is not None:
        term.params["min_height"] = ROOT_HEIGHT_MIN
        log.append(f"rewards.rally_ready_root_height_debt.min_height = {ROOT_HEIGHT_MIN}")
    for name in ("ready_stance_width", "stance_width"):
        term = getattr(rewards, name, None)
        if term is not None:
            term.params["lo"], term.params["hi"] = STANCE_WIDTH
            log.append(f"rewards.{name}.lo/hi = {STANCE_WIDTH}")
    term = getattr(rewards, "hold_ready", None)
    if term is not None:
        term.params["reach"] = HOLD_READY_REACH
        log.append(f"rewards.hold_ready.reach = {HOLD_READY_REACH}")
    term = getattr(cfg.terminations, "base_too_low", None)
    if term is not None:
        term.params["minimum_height"] = BASE_TOO_LOW
        log.append(f"terminations.base_too_low.minimum_height = {BASE_TOO_LOW}")
    return log
