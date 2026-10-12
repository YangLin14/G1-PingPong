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


def test_four_target_zones():
    half_w, half_len = fs.TABLE_WIDTH / 2, 1.37
    b = {name: CFG.with_target(name).zone_bounds() for name in fs.TARGET_ZONES}
    assert set(b) == {"opp_right_mid", "center_line_deep", "opp_right_deep", "opp_right_corner"}
    end_band = CFG.far_x - 0.25 * half_len  # within the outer quarter of the half: next to the end line
    side_band = 0.75 * half_w                # outer quarter of the half width: next to the side line
    # the three edge zones are lined up along the end line: same depth band, centre -> middle -> corner
    deep = ["center_line_deep", "opp_right_deep", "opp_right_corner"]
    assert all(b[n][0] == b[deep[0]][0] for n in deep)
    assert b[deep[0]][0][0] > end_band and b[deep[0]][0][1] < CFG.far_x
    (_, y) = b["center_line_deep"]
    assert y[0] > fs.BALL_RADIUS and y[1] - y[0] == pytest.approx(0.2 * half_w)  # off the centre line
    assert b["opp_right_deep"][1] == b["opp_right_mid"][1]  # same lateral band as the open-court zone
    (_, y) = b["opp_right_corner"]
    assert y[0] > side_band and y[1] < half_w
    assert b["center_line_deep"][1][1] < b["opp_right_deep"][1][0] < b["opp_right_deep"][1][1] < b["opp_right_corner"][1][0]
    names = list(b)  # no two zones overlap
    for i, m in enumerate(names):
        for n in names[i + 1:]:
            (mx, my), (nx, ny) = b[m], b[n]
            assert mx[1] <= nx[0] or nx[1] <= mx[0] or my[1] <= ny[0] or ny[1] <= my[0], (m, n)


def test_feed_zone_and_origin():
    (oy, oz) = CFG.feed_origin_bounds()  # hit from the opponent's right = robot's +y, above the table
    assert 0.0 < oy[0] < oy[1] < fs.TABLE_WIDTH / 2 and CFG.table_surface_z < oz[0] < oz[1]
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


def test_zone_index_recovers_each_zone_centre():
    names = tuple(fs.TARGET_ZONES)
    centres = []
    for name in names:
        (x0, x1), (y0, y1) = CFG.with_target(name).zone_bounds()
        centres.append([(x0 + x1) / 2, (y0 + y1) / 2])
    pts = torch.tensor(centres + [[0.0, 0.0], [CFG.far_x - 0.01, -0.5]])
    assert fs.zone_index(pts, CFG).tolist() == list(range(len(names))) + [-1, -1]


def test_zone_report_counts_misses_and_scores_against_own_zone():
    report = fs.ZoneSuccessReport(CFG)
    n = 3
    v_r = _rows(n, [2.5, 0.6, 1.2])
    ball = _rows(n, BALL)
    report.add_strikes(
        torch.tensor([0, 1, 2]), PRM, ball_pos=ball, ball_vel_in=_rows(n, V_APEX), ball_spin_in=torch.zeros(n, 3),
        racket_pos=ball, racket_vel=v_r, racket_normal=v_r / torch.linalg.norm(v_r, dim=-1, keepdim=True),
    )
    report.add_missed(torch.tensor([0, 0, 3, -1]))
    out = report.as_dict()
    assert out["per_zone"]["opp_right_mid"]["attempts"] == 3
    assert out["per_zone"]["opp_right_corner"]["attempts"] == 1
    assert out["overall"]["attempts"] == 6
    assert out["unassigned"] == 1
    # identical strikes, different zones: at most one of them can land in its own zone
    assert sum(out["per_zone"][z]["success_rate"] > 0 for z in fs.TARGET_ZONES) <= 1
