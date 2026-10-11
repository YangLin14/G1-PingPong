"""Host tests for the forehand-rally success definition (utils/forehand_success.py)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "forehand_success.py"
)
_spec = importlib.util.spec_from_file_location("forehand_success", _PATH)
fs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fs
_spec.loader.exec_module(fs)

CFG = fs.ForehandSuccessCfg()
PRM = fs.virtual_ball.load_venue_params()
BALL = torch.tensor([0.46, -0.30, 0.95])  # struck on the robot's right
V_APEX = torch.tensor([-3.0, 0.0, 0.0])


def _rows(n, v):
    return torch.as_tensor(v, dtype=torch.float32).expand(n, 3).clone()


def _classify(v_in, v_r, racket_offset=(0.0, 0.0, 0.0)):
    n = v_r.shape[0]
    ball = _rows(n, BALL)
    return fs.classify_strikes(
        ball, _rows(n, v_in), torch.zeros(n, 3), ball + torch.tensor(racket_offset),
        v_r, v_r / torch.linalg.norm(v_r, dim=-1, keepdim=True), CFG, PRM,
    )


@pytest.fixture(scope="module")
def zone_hit():
    """A racket velocity (face normal along it, the hitter_pure convention) that returns an
    apex ball into the target zone, found by grid search so the test pins physics, not a guess."""
    g = torch.meshgrid(
        torch.linspace(0.5, 5.0, 46), torch.linspace(-2.0, 1.0, 31), torch.linspace(0.0, 3.0, 31),
        indexing="ij",
    )
    v_r = torch.stack([x.reshape(-1) for x in g], dim=-1)
    flags = _classify(V_APEX, v_r)
    ok = torch.where(flags["success"])[0]
    assert len(ok) > 0, "no racket velocity returns an apex ball into the zone"
    return v_r[ok[len(ok) // 2]].unsqueeze(0)


def test_default_zone_is_opponent_right_middle_band():
    (x_lo, x_hi), (y_lo, y_hi) = CFG.zone_bounds()
    assert CFG.net_x < x_lo < x_hi < CFG.far_x
    assert x_lo == pytest.approx(CFG.net_x + 1.37 / 3) and x_hi == pytest.approx(CFG.net_x + 2 * 1.37 / 3)
    # opponent's right = robot's left (+y), inside the side line
    assert 0.0 < y_lo < y_hi < fs.TABLE_WIDTH / 2
    assert CFG.with_target("opp_right_mid").zone_bounds() == CFG.zone_bounds()


def test_target_presets_and_feed_zone():
    (cx, cy) = CFG.with_target("center_line").zone_bounds()
    assert cy[0] < 0.0 < cy[1]
    (kx, ky) = CFG.with_target("opp_right_corner").zone_bounds()
    assert kx[0] > CFG.zone_bounds()[0][1] - 1e-9 and kx[1] < CFG.far_x and ky[1] < fs.TABLE_WIDTH / 2
    (fx, fy) = CFG.feed_bounds()
    # feed bounce: robot's half, middle depth third, middle band of the robot's right (-y)
    assert CFG.table_near_x < fx[0] < fx[1] < CFG.net_x
    assert fx[0] == pytest.approx(CFG.net_x - 2 * 1.37 / 3) and fx[1] == pytest.approx(CFG.net_x - 1.37 / 3)
    assert -fs.TABLE_WIDTH / 2 < fy[0] < fy[1] < 0.0


def test_zone_return_succeeds(zone_hit):
    f = _classify(V_APEX, zone_hit)
    assert all(bool(f[k][0]) for k in ("apex", "contact", "net_clear", "on_opponent", "in_zone", "success"))


def test_falling_ball_fails_timing(zone_hit):
    f = _classify(torch.tensor([-3.0, 0.2, -1.2]), zone_hit)
    assert not bool(f["apex"][0]) and not bool(f["success"][0])


def test_racket_away_from_ball_is_no_contact(zone_hit):
    f = _classify(V_APEX, zone_hit, racket_offset=(0.0, 0.0, 0.2))
    assert not bool(f["contact"][0]) and not bool(f["success"][0])


def test_soft_tap_does_not_clear_net():
    f = _classify(V_APEX, torch.tensor([[0.6, 0.0, 0.0]]))
    assert bool(f["contact"][0]) and not bool(f["net_clear"][0]) and not bool(f["success"][0])


def test_down_the_line_landing_is_out_of_zone(zone_hit):
    mirrored = zone_hit.clone()
    mirrored[:, 1] = -1.5  # down the line to the robot's right = opponent's LEFT
    f = _classify(V_APEX, mirrored)
    if bool(f["on_opponent"][0]):
        assert f["landing_xy"][0, 1] < 0 and not bool(f["in_zone"][0])
    assert not bool(f["success"][0])


def test_tally_counts_misses_in_denominator(zone_hit):
    t = fs.ForehandSuccessTally()
    t.add_strikes(_classify(V_APEX, zone_hit), CFG)
    t.add_missed(3)
    d = t.as_dict()
    assert d["attempts"] == 4 and d["success_rate"] == pytest.approx(0.25)
