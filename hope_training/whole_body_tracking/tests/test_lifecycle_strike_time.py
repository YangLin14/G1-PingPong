"""Deploy reference clock: original countdown vs the absolute ``strike_time`` clock."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference"))

from a3_deploy_onnx_ref_pingpong.lifecycle import LifecycleConfig, Phase, SwingLifecycle  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import RobotState  # noqa: E402
from a3_deploy_onnx_ref_pingpong.racket_command import FOREHAND, RacketCommand  # noqa: E402
from a3_deploy_onnx_ref_pingpong.ros_command_source import parse_flat_racket_command  # noqa: E402

DT = 0.02
STATE = RobotState(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(29), np.zeros(29))


def _cmd(tts, strike_time=None, rev=0):
    return RacketCommand(task_id=1, task_revision=rev, swing_sign=FOREHAND, position=[0.45, -0.35, 0.95],
                         velocity=[1.9, 0.3, 1.0], time_to_strike=tts, strike_time=strike_time)


def test_countdown_is_unchanged_and_ages_a_fresh_command_by_one_tick():
    life = SwingLifecycle(LifecycleConfig(dt=DT))
    assert life.update(_cmd(0.5), STATE).time_to_strike == pytest.approx(0.5 - DT)
    assert life.update(None, STATE).time_to_strike == pytest.approx(0.5 - 2 * DT)
    # now is ignored by the original clock
    life2 = SwingLifecycle(LifecycleConfig(dt=DT))
    assert life2.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0).time_to_strike == pytest.approx(0.5 - DT)


def test_strike_time_clock_reports_time_left_from_now_whatever_the_message_age():
    life = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time"))
    # fresh message: exactly the planner's value, no extra tick
    assert life.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0).time_to_strike == pytest.approx(0.5)
    # next ticks without messages follow the wall clock
    assert life.update(None, STATE, now=100.02).time_to_strike == pytest.approx(0.48)
    # a revision sent at 100.000 ("0.52 s left") that arrives 35 ms late still gives the right time left
    out = life.update(_cmd(0.52, strike_time=100.52, rev=1), STATE, now=100.035)
    assert out.time_to_strike == pytest.approx(0.485)
    assert life.clock_rejections == 0
    # contact at the deadline, then negative through the follow-through
    life.update(None, STATE, now=100.52)
    assert life.phase == Phase.FOLLOW_THROUGH
    assert life.update(None, STATE, now=100.6).time_to_strike == pytest.approx(-0.08)


def test_strike_time_clock_stamps_commands_without_a_deadline_and_falls_back_without_now():
    life = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time"))
    assert life.update(_cmd(0.4), STATE, now=50.0).time_to_strike == pytest.approx(0.4)
    assert life.update(None, STATE, now=50.1).time_to_strike == pytest.approx(0.3)
    old = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time"))
    assert old.update(_cmd(0.4, strike_time=50.4), STATE).time_to_strike == pytest.approx(0.4 - DT)


def test_ros_parser_reads_strike_time_and_treats_zero_as_missing():
    head = [1.0, 1.0, 1.0, 0.45, -0.35, 0.95, 1.9, 0.3, 1.0, 0.5]
    assert parse_flat_racket_command(head + [1234.5]).strike_time == pytest.approx(1234.5)
    assert parse_flat_racket_command(head + [0.0]).strike_time is None


def test_unknown_clock_is_rejected():
    with pytest.raises(ValueError):
        SwingLifecycle(LifecycleConfig(clock="bogus")).update(None, STATE)


def test_clock_guard_rejects_a_deadline_from_an_unsynced_clock():
    life = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time", max_message_age_s=0.1))
    # planner clock 2 s ahead: implied message age is -2 s -> ignore the deadline, stamp on receipt
    assert life.update(_cmd(0.5, strike_time=102.5), STATE, now=100.0).time_to_strike == pytest.approx(0.5)
    assert life.clock_rejections == 1
    assert life.update(None, STATE, now=100.1).time_to_strike == pytest.approx(0.4)
    # planner clock 0.3 s behind: implied age 0.3 s > 0.1 s -> also rejected
    life2 = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time", max_message_age_s=0.1))
    assert life2.update(_cmd(0.5, strike_time=100.2), STATE, now=100.0).time_to_strike == pytest.approx(0.5)
    assert life2.clock_rejections == 1


def test_g1_runtime_config_uses_strike_time_and_a3_keeps_countdown():
    from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig

    cfg_dir = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "config")
    g1 = RuntimeConfig.load(os.path.join(cfg_dir, "hope_pingpong_runtime_unitree_g1.yaml"), profile="unitree_g1")
    assert g1.lifecycle.clock == "strike_time" and g1.lifecycle.max_message_age_s == pytest.approx(0.1)
    assert g1.lifecycle.tts_offset_s == pytest.approx(0.04)
    a3 = RuntimeConfig.load(os.path.join(cfg_dir, "hope_pingpong_runtime.yaml"))
    assert a3.lifecycle.clock == "countdown" and a3.lifecycle.tts_offset_s == 0.0


def test_timing_offset_shifts_both_clocks_and_the_contact_phase():
    life = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time", tts_offset_s=0.04))
    assert life.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0).time_to_strike == pytest.approx(0.54)
    life.update(None, STATE, now=100.5)
    assert life.phase == Phase.SWING          # true deadline passed, calibrated one not yet
    life.update(None, STATE, now=100.54)
    assert life.phase == Phase.FOLLOW_THROUGH
    old = SwingLifecycle(LifecycleConfig(dt=DT, tts_offset_s=0.04))
    assert old.update(_cmd(0.5), STATE).time_to_strike == pytest.approx(0.52)


def test_hold_last_target_keeps_the_ball_target_and_runs_the_clock_down_to_the_floor():
    cfg = LifecycleConfig(dt=DT, clock="strike_time", follow_through_s=0.6, recovery_s=0.8,
                          hold_last_target=True, hold_tts_floor=-1.26)
    life = SwingLifecycle(cfg)
    # before any ball: the ready reach (base + reach), clock pinned
    pre = life.update(None, STATE, now=99.0)
    assert pre.time_to_strike == pytest.approx(cfg.ready_time_to_strike)
    life.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0)
    for t in np.arange(100.02, 102.5, DT):
        out = life.update(None, STATE, now=float(t))
    # recovery is over (phase READY) but the observed goal is still the ball's target
    assert life.phase == Phase.READY
    np.testing.assert_allclose(out.pos_w, [0.45, -0.35, 0.95])
    np.testing.assert_allclose(out.vel_w, [1.9, 0.3, 1.0])
    assert out.time_to_strike == pytest.approx(-1.26)
    mid = SwingLifecycle(cfg)
    mid.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0)
    for t in np.arange(100.02, 101.4 + 1e-9, DT):
        out = mid.update(None, STATE, now=float(t))
    assert mid.phase == Phase.RECOVERY and out.time_to_strike == pytest.approx(-0.9)
    # the next ball engages as usual
    nxt = RacketCommand(task_id=2, task_revision=0, swing_sign=FOREHAND, position=[0.47, -0.30, 1.0],
                        velocity=[2.0, 0.2, 1.1], time_to_strike=0.6, strike_time=103.1)
    out = life.update(nxt, STATE, now=102.5)
    assert life.phase == Phase.SWING and out.time_to_strike == pytest.approx(0.6)


def test_without_hold_the_original_ready_reach_is_shown_after_follow_through():
    life = SwingLifecycle(LifecycleConfig(dt=DT, clock="strike_time"))
    life.update(_cmd(0.5, strike_time=100.5), STATE, now=100.0)
    for t in np.arange(100.02, 101.2 + 1e-9, DT):  # 0.7 s after contact: recovery
        out = life.update(None, STATE, now=float(t))
    assert life.phase == Phase.RECOVERY and out.time_to_strike == pytest.approx(1.0)
