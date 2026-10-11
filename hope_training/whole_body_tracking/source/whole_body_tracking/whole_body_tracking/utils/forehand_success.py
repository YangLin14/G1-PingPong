"""Forehand-rally success definition (Triton Droids), Isaac-free.

A return counts as a success only when ALL of:

1. **timing** -- the robot strikes the ball at the top of its bounce on the robot's side: the
   incoming ball's vertical velocity at contact is within ``apex_vz_tol`` of zero
   (0.3 m/s ~= +-30 ms around the apex at g = 9.81);
2. **contact** -- the racket actually meets the ball: racket centre within ``capture_radius`` of the
   ball and the face moving into it faster than ``min_approach_speed`` (the training capture gate);
3. **legal** -- the outgoing ball clears the net (centre above net top + ball radius) and its first
   bounce is on the opponent half;
4. **zone** -- that first bounce lands in the selected target zone on the opponent half.

Court layout (club spec). All frames below are the ROBOT's: env frame, robot faces +x, +y = robot's
left. The robot plays a cross-court forehand from its right side (-y) to the OPPONENT's right side,
which is the robot's left (+y) half of the far table.

* feed: the incoming ball's own-side bounce lands in the middle depth third of the robot's half,
  middle band of the robot's right half (``FEED_ZONE``);
* landing targets (``TARGET_ZONES``), opponent-perspective names:
  ``center_line`` (middle depth, on the centre line), ``opp_right_mid`` (middle depth third, middle
  band of the opponent's right half -- the first training target), ``opp_right_corner`` (deep third,
  outer band of the opponent's right half, near the corner).

Outgoing ball state comes from the same venue-fitted spin contact model and Magnus flight that the
training reward uses (:mod:`virtual_ball`), evaluated on the ACHIEVED racket state, so the metric
and the reward never disagree about physics.

Frame: the env frame of the HOPE tracking task -- robot root at the origin facing +x, near table
edge at ``table_near_x``, table surface at ``table_surface_z``, table centred on y = 0.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, replace

import torch

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_HEIGHT = 0.1525
BALL_RADIUS = 0.02

# (depth_frac, lateral_frac) presets; see the module docstring for the frame conventions.
TARGET_ZONES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "center_line": ((1.0 / 3.0, 2.0 / 3.0), (-0.1, 0.1)),
    "opp_right_mid": ((1.0 / 3.0, 2.0 / 3.0), (0.25, 0.75)),
    "opp_right_corner": ((2.0 / 3.0, 0.95), (0.6, 0.95)),
}
FEED_ZONE: tuple[tuple[float, float], tuple[float, float]] = ((1.0 / 3.0, 2.0 / 3.0), (-0.75, -0.25))


def _load_virtual_ball():
    """``virtual_ball`` is torch/yaml only, but importing it through its package pulls in Isaac."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tasks", "tracking", "mdp", "virtual_ball.py",
    )
    spec = importlib.util.spec_from_file_location("_hope_virtual_ball", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


virtual_ball = _load_virtual_ball()


@dataclass(frozen=True)
class ForehandSuccessCfg:
    table_near_x: float = 0.5
    table_surface_z: float = 0.76
    apex_vz_tol: float = 0.3
    capture_radius: float = 0.095
    min_approach_speed: float = 0.3
    # Target zone on the opponent half (default: TARGET_ZONES["opp_right_mid"]). Depth is the fraction
    # of the half measured from the net toward the far end line; lateral is the fraction of the half
    # width along +y (robot's left = opponent's right), negative toward the robot's right.
    zone_depth_frac: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
    zone_lateral_frac: tuple[float, float] = (0.25, 0.75)
    rollout_h: float = 0.005
    rollout_steps: int = 300

    @property
    def net_x(self) -> float:
        return self.table_near_x + TABLE_LENGTH / 2.0

    @property
    def far_x(self) -> float:
        return self.table_near_x + TABLE_LENGTH

    def zone_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x_lo, x_hi), (y_lo, y_hi)) of the target zone in the env frame."""
        half_len = TABLE_LENGTH / 2.0
        half_w = TABLE_WIDTH / 2.0
        x = (self.net_x + self.zone_depth_frac[0] * half_len, self.net_x + self.zone_depth_frac[1] * half_len)
        y = (self.zone_lateral_frac[0] * half_w, self.zone_lateral_frac[1] * half_w)
        return x, y

    def feed_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x_lo, x_hi), (y_lo, y_hi)) of FEED_ZONE (incoming bounce on the robot's half)."""
        half_len = TABLE_LENGTH / 2.0
        half_w = TABLE_WIDTH / 2.0
        (d0, d1), (l0, l1) = FEED_ZONE
        # depth on the robot's half is measured from the net back toward the robot's end line
        x = (self.net_x - d1 * half_len, self.net_x - d0 * half_len)
        return x, (l0 * half_w, l1 * half_w)

    def with_target(self, name: str) -> "ForehandSuccessCfg":
        depth, lateral = TARGET_ZONES[name]
        return replace(self, zone_depth_frac=depth, zone_lateral_frac=lateral)


def classify_strikes(
    ball_pos: torch.Tensor,
    ball_vel_in: torch.Tensor,
    ball_spin_in: torch.Tensor,
    racket_pos: torch.Tensor,
    racket_vel: torch.Tensor,
    racket_normal: torch.Tensor,
    cfg: ForehandSuccessCfg = ForehandSuccessCfg(),
    prm=None,
) -> dict[str, torch.Tensor]:
    """Per-strike success flags. All inputs ``(N, 3)`` in the env frame at the strike instant.

    Returns boolean ``(N,)`` tensors ``apex``, ``contact``, ``net_clear``, ``on_opponent``,
    ``in_zone``, ``success`` plus ``landing_xy`` ``(N, 2)``, ``net_z`` and ``landed`` for reporting.
    Outcome flags are only meaningful where ``contact`` is true; ``success`` already requires it.
    """
    prm = prm or virtual_ball.load_venue_params()
    apex = ball_vel_in[:, 2].abs() <= cfg.apex_vz_tol

    n_or = virtual_ball.orient_normal(racket_normal, ball_vel_in, racket_vel)
    approach = torch.sum(racket_vel * n_or, dim=-1)
    near = torch.linalg.norm(racket_pos - ball_pos, dim=-1) <= cfg.capture_radius
    contact = near & (approach > cfg.min_approach_speed)

    v_out, w_out = virtual_ball.predict_paddle_contact(ball_vel_in, racket_vel, racket_normal, ball_spin_in, prm)
    land = virtual_ball.coarse_landing(
        ball_pos, v_out, w_out, prm,
        surface_z=cfg.table_surface_z + BALL_RADIUS,
        net_x=cfg.net_x,
        h=cfg.rollout_h,
        n_steps=cfg.rollout_steps,
    )
    lx, ly = land["land_xy"][:, 0], land["land_xy"][:, 1]
    net_clear = land["net_valid"] & (land["net_z"] > cfg.table_surface_z + NET_HEIGHT + BALL_RADIUS)
    on_opponent = land["land_valid"] & (lx > cfg.net_x) & (lx <= cfg.far_x) & (ly.abs() <= TABLE_WIDTH / 2.0)
    (x_lo, x_hi), (y_lo, y_hi) = cfg.zone_bounds()
    in_zone = on_opponent & (lx >= x_lo) & (lx <= x_hi) & (ly >= y_lo) & (ly <= y_hi)
    success = apex & contact & net_clear & on_opponent & in_zone
    return {
        "apex": apex,
        "contact": contact,
        "net_clear": net_clear,
        "on_opponent": on_opponent,
        "in_zone": in_zone,
        "success": success,
        "landing_xy": land["land_xy"],
        "landed": land["land_valid"],
        "net_z": land["net_z"],
    }


class ForehandSuccessTally:
    """Unconditional counts: every incoming ball is an attempt, including balls that reset or
    fell before the strike (``add_missed``), so misses can never leave the denominator."""

    KEYS = ("apex", "contact", "net_clear", "on_opponent", "in_zone", "success")

    def __init__(self) -> None:
        self.attempts = 0
        self.counts = {k: 0 for k in self.KEYS}
        self.zone_errors: list[float] = []

    def add_strikes(self, flags: dict[str, torch.Tensor], cfg: ForehandSuccessCfg = ForehandSuccessCfg()) -> None:
        self.attempts += int(flags["success"].numel())
        for k in self.KEYS:
            self.counts[k] += int(flags[k].sum())
        (x_lo, x_hi), (y_lo, y_hi) = cfg.zone_bounds()
        centre = torch.tensor([(x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0], device=flags["landing_xy"].device)
        landed = flags["contact"] & flags["on_opponent"]
        self.zone_errors += torch.linalg.norm(flags["landing_xy"][landed] - centre, dim=-1).tolist()

    def add_missed(self, n: int) -> None:
        self.attempts += int(n)

    def as_dict(self) -> dict:
        a = max(self.attempts, 1)
        out = {"attempts": self.attempts}
        out.update({f"{k}_rate": self.counts[k] / a for k in self.KEYS})
        out["success_rate"] = out.pop("success_rate")
        if self.zone_errors:
            errs = torch.tensor(self.zone_errors)
            out["landing_err_to_zone_centre_m_median"] = float(errs.median())
        return out
