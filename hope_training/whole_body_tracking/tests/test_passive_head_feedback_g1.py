"""Unitree G1 reference-runner tick: 29-D, no passive columns (mirror of test_passive_head_feedback.py).

The REAL ``PingPongReferenceRunner`` loop runs against a fake 29-joint bridge with the G1 runtime
config and a fake policy emitting ones: every column of ``last_action`` must survive (nothing is
zeroed), the observation's actions slice ``[61:90]`` must equal the applied action, and every
joint target must be the adapter's decode of that action (no column held at default).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_RUNTIME_YAML = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "config", "hope_pingpong_runtime_unitree_g1.yaml")

sys.path.insert(0, _REFERENCE_DIR)

from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.joint_order_unitree_g1 import NUM_JOINTS  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import RobotState  # noqa: E402
from a3_deploy_onnx_ref_pingpong.racket_command import QueueRacketCommandSource  # noqa: E402
from a3_deploy_onnx_ref_pingpong.robot_profile import UNITREE_G1  # noqa: E402
from a3_deploy_onnx_ref_pingpong.runner import PingPongReferenceRunner  # noqa: E402
from a3_deploy_onnx_ref_pingpong.sim_bridge import SimBridge  # noqa: E402

_ACTIONS_SLICE = UNITREE_G1.actions_slice  # [61:90] of the 104-D hitter_pure_unitree_g1 contract


class _FakeBridge(SimBridge):
    def __init__(self, default_q: np.ndarray) -> None:
        self._q = default_q.copy()
        self.written_q_des: list[np.ndarray] = []

    def reset(self) -> None:
        pass

    def read_state(self) -> RobotState:
        return RobotState(
            base_pos_w=np.array([0.0, 0.0, 0.76]),
            base_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
            base_ang_vel_b=np.zeros(3),
            q=self._q.copy(),
            qd=np.zeros(NUM_JOINTS),
        )

    def write_targets(self, q_des, kp, kd) -> None:
        self.written_q_des.append(np.asarray(q_des, dtype=np.float64).copy())

    def step(self) -> None:
        pass


class _OnesPolicy:
    def __init__(self) -> None:
        self.seen_obs: list[np.ndarray] = []

    def infer(self, obs: np.ndarray) -> np.ndarray:
        self.seen_obs.append(np.asarray(obs, dtype=np.float64).copy())
        return np.ones(NUM_JOINTS, dtype=np.float32)


def _run_ticks(n: int = 3):
    cfg = RuntimeConfig.load(_RUNTIME_YAML)
    assert cfg.profile is UNITREE_G1 and not cfg.passive_neck
    bridge = _FakeBridge(cfg.action_adapter.default_q)
    policy = _OnesPolicy()
    runner = PingPongReferenceRunner(cfg, bridge, QueueRacketCommandSource(), policy=policy)
    runner.run(max_ticks=n, status_every=0)
    return cfg, bridge, policy, runner


def test_last_action_is_29_wide_with_no_zeroed_columns():
    _cfg, _bridge, _policy, runner = _run_ticks(3)
    assert runner.last_action.shape == (NUM_JOINTS,)
    assert np.all(runner.last_action == 1.0)


def test_observation_is_104_d_and_feeds_back_the_full_applied_action():
    _cfg, _bridge, policy, _runner = _run_ticks(3)
    assert policy.seen_obs[0].shape == (104,)
    assert np.all(policy.seen_obs[0][_ACTIONS_SLICE] == 0.0)  # zero-initialized last_action
    for obs in policy.seen_obs[1:]:
        assert np.all(obs[_ACTIONS_SLICE] == 1.0)


def test_every_joint_target_is_the_decoded_action():
    cfg, bridge, _policy, _runner = _run_ticks(2)
    expected = cfg.action_adapter.decode(np.ones(NUM_JOINTS))
    for q_des in bridge.written_q_des:
        assert q_des.shape == (NUM_JOINTS,)
        np.testing.assert_allclose(q_des, expected)
