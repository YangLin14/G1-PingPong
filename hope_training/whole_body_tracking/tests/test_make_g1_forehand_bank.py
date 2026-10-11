"""Host test of scripts/make_g1_forehand_bank.py: every generated row obeys the club spec."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import torch

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import make_g1_forehand_bank as mk  # noqa: E402

LOADER_FIELDS = {
    "clip": (None,), "station_xy": (None, 2), "target_position_offset": (None, 3),
    "target_velocity": (None, 3), "target_normal": (None, 3), "incoming_velocity": (None, 3),
    "incoming_spin": (None, 3), "intended_landing_xy": (None, 2), "predicted_landing_xy": (None, 2),
    "outgoing_velocity": (None, 3),
}


@pytest.fixture(scope="module")
def bank():
    cfg = mk.BankCfg(rows=64, seed=3)
    arrays, receipt = mk.generate(cfg)
    torch.set_default_dtype(torch.float32)
    return cfg, arrays, receipt


def test_loader_schema(bank):
    cfg, a, receipt = bank
    assert int(a["schema_version"]) == mk.SCHEMA_VERSION and a["schema_version"].shape == ()
    for name, shape in LOADER_FIELDS.items():
        assert a[name].shape == tuple(cfg.rows if s is None else s for s in shape), name
        if name != "clip":
            assert np.isfinite(a[name]).all()
    assert (a["clip"] == 0).all()
    assert np.allclose(np.linalg.norm(a["target_normal"], axis=-1), 1.0, atol=2e-4)
    assert receipt["contract"]["recipe_revision"] == 12 and receipt["schema_version"] == mk.SCHEMA_VERSION


def test_feed_apex_strike_and_speed(bank):
    cfg, a, _ = bank
    scfg = mk.fs.ForehandSuccessCfg()
    (fx, fy) = scfg.feed_bounds()
    b = a["feed_bounce_xy"]
    assert ((b[:, 0] >= fx[0]) & (b[:, 0] <= fx[1]) & (b[:, 1] >= fy[0]) & (b[:, 1] <= fy[1])).all()
    assert np.abs(a["incoming_velocity"][:, 2]).max() < 1e-6  # struck at the apex
    contact = a["target_position_offset"]
    assert (contact[:, 0] >= cfg.apex_x[0]).all() and (contact[:, 0] <= cfg.apex_x[1]).all()
    assert (contact[:, 2] >= cfg.apex_z[0]).all() and (contact[:, 2] <= cfg.apex_z[1]).all()
    assert (np.linalg.norm(a["target_velocity"], axis=-1) <= cfg.racket_speed_limit + 1e-5).all()


def test_every_row_is_a_success_into_the_target_zone(bank):
    cfg, a, _ = bank
    scfg = mk.fs.ForehandSuccessCfg().with_target(cfg.target)
    t = lambda k: torch.as_tensor(a[k], dtype=torch.float64)  # noqa: E731
    ball = t("target_position_offset").clone()
    ball[:, 1] += t("station_xy")[:, 1]  # contact xy is stored relative to the station
    v_r = t("target_velocity")
    flags = mk.fs.classify_strikes(ball, t("incoming_velocity"), t("incoming_spin"), ball, v_r,
                                   v_r / torch.linalg.norm(v_r, dim=-1, keepdim=True), scfg)
    assert bool(flags["success"].all())
    (zx, zy) = scfg.zone_bounds()
    assert (a["intended_landing_xy"][:, 1] > 0).all()  # opponent's right = robot's +y
    assert np.linalg.norm(a["predicted_landing_xy"] - a["intended_landing_xy"], axis=-1).max() <= cfg.max_landing_err
    assert np.abs(a["net_clearance_m"] - cfg.net_clearance_target).max() <= cfg.max_clearance_err
