"""Host tests for the MuJoCo club-forehand serve tools (scripts/mujoco_forehand_bank.py)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mujoco_forehand_bank.py"
_spec = importlib.util.spec_from_file_location("mujoco_forehand_bank", _PATH)
mb = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mb
_spec.loader.exec_module(mb)

BANK = mb.default_bank_path()
pytestmark = pytest.mark.skipif(not BANK.is_file(), reason="forehand bank not generated")


@pytest.fixture(scope="module")
def tools():
    prm = mb.vb.load_venue_params()
    return mb.ForehandBank(str(BANK)), prm, mb.ApexPlanner(prm)


def test_feed_from_end_line_reproduces_bank_apex_and_racket(tools):
    bank, prm, planner = tools
    rng = np.random.default_rng(0)
    for _ in range(5):
        row = bank.sample(rng)
        p, v, w = mb.feed_start(row, prm)
        assert p[0] == pytest.approx(planner.scfg.far_x) and v[0] < 0 and p[1] > 0  # from the opponent's right
        apex, v_apex, w_apex, tts = planner.predict(p, v, w, after_bounce=False)
        bank_apex = row["target_position_offset"] + np.array([*row["station_xy"], 0.0])
        assert np.linalg.norm(apex - bank_apex) < 0.005
        assert 0.3 < tts < 1.0
        v_r = planner.solve_racket(apex, v_apex, w_apex, row["intended_landing_xy"])
        assert np.linalg.norm(v_r - row["target_velocity"]) < 0.05


def test_zone_filter(tools):
    bank, _, _ = tools
    only = mb.ForehandBank(str(BANK), zones=["opp_right_corner"])
    rng = np.random.default_rng(1)
    assert {only.sample(rng)["zone"] for _ in range(20)} == {"opp_right_corner"}
    assert len(only.index) < len(bank.index)
    with pytest.raises(ValueError):
        mb.ForehandBank(str(BANK), zones=["nowhere"])


def test_predict_after_apex_is_none(tools):
    _, _, planner = tools
    assert planner.predict([0.4, -0.4, 1.0], [-3.0, 0.0, -0.5], [0.0, 0.0, 0.0], after_bounce=True) is None
