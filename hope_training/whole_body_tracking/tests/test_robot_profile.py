"""Robot profiles in the reference deploy package: A3 defaults unchanged, Unitree G1 explicit.

Host-only (NumPy + PyYAML; no MuJoCo / onnxruntime). The G1 is 29 joints, no passive columns,
104-D ``hitter_pure_unitree_g1``; the A3 profile must equal the historical module constants so
every existing caller keeps its behaviour.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_EXAMPLE = os.path.join(_REPO, "a3_deploy", "a3_deploy_example")
_REFERENCE = os.path.join(_EXAMPLE, "reference")
_A3_RUNTIME = os.path.join(_EXAMPLE, "config", "hope_pingpong_runtime.yaml")
_G1_RUNTIME = os.path.join(_EXAMPLE, "config", "hope_pingpong_runtime_unitree_g1.yaml")
_G1_MJCF = os.path.join(_REPO, "unitree", "URDF", "g1_29dof_pingpang", "mjcf", "g1_29dof_pingpang.xml")

sys.path.insert(0, _REFERENCE)

from a3_deploy_onnx_ref_pingpong import joint_order as a3_order  # noqa: E402
from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig, _expand_pd_gains  # noqa: E402
from a3_deploy_onnx_ref_pingpong.observation import (  # noqa: E402
    CONTRACT_NAME,
    OBS_DIM,
    ObsTarget,
    RobotState,
    build_observation,
    obs_dim_for,
)
from a3_deploy_onnx_ref_pingpong.onnx_policy import (  # noqa: E402
    OnnxPolicy,
    validate_embedded_contract,
    validate_embedded_joint_order,
)
from a3_deploy_onnx_ref_pingpong.robot_profile import (  # noqa: E402
    AGIBOT_A3,
    PROFILES,
    UNITREE_G1,
    resolve_profile,
)


def test_a3_profile_equals_legacy_constants():
    assert AGIBOT_A3.joint_names == tuple(a3_order.JOINT_NAMES)
    assert AGIBOT_A3.num_joints == a3_order.NUM_JOINTS == 31
    assert AGIBOT_A3.passive_indices == tuple(a3_order.HEAD_INDICES) == (3, 4)
    assert AGIBOT_A3.contract_name == CONTRACT_NAME == "hitter_pure"
    assert AGIBOT_A3.obs_dim == OBS_DIM == 110
    assert (AGIBOT_A3.joint_pos_slice, AGIBOT_A3.joint_vel_slice, AGIBOT_A3.actions_slice) == (
        slice(3, 34), slice(34, 65), slice(65, 96)
    )
    assert AGIBOT_A3.group_ranges == {
        "waist": tuple(range(0, 3)), "neck": tuple(range(3, 5)), "arm": tuple(range(5, 19)), "leg": tuple(range(19, 31))
    }
    assert AGIBOT_A3.base_free_joint == "pelvis_free_joint" and AGIBOT_A3.racket_site == "right_racket"


def test_g1_profile_dims_and_slices():
    assert UNITREE_G1.num_joints == 29 and UNITREE_G1.passive_indices == ()
    assert UNITREE_G1.contract_name == "hitter_pure_unitree_g1"
    assert UNITREE_G1.obs_dim == 104 == obs_dim_for(29)
    assert (UNITREE_G1.joint_pos_slice, UNITREE_G1.joint_vel_slice, UNITREE_G1.actions_slice) == (
        slice(3, 32), slice(32, 61), slice(61, 90)
    )
    groups = UNITREE_G1.group_ranges
    assert set(groups) == {"waist", "arm", "leg"}
    assert sorted(i for idx in groups.values() for i in idx) == list(range(29))
    assert all(UNITREE_G1.joint_names[i].startswith("waist_") for i in groups["waist"])
    assert len(groups["arm"]) == 14 and len(groups["leg"]) == 12
    # The two robots share MJCF element names by construction of the G1 racket variant.
    assert UNITREE_G1.base_free_joint == "pelvis_free_joint"
    assert UNITREE_G1.gyro_sensor == "pelvis_imu_gyro"
    assert UNITREE_G1.racket_collision_geom == "right_racket_collision"


def test_resolve_profile():
    assert resolve_profile(None) is AGIBOT_A3
    assert resolve_profile("agibot_a3") is AGIBOT_A3
    assert resolve_profile("UNITREE_G1") is UNITREE_G1
    assert resolve_profile(UNITREE_G1) is UNITREE_G1
    assert set(PROFILES) == {"agibot_a3", "unitree_g1"}
    with pytest.raises(ValueError, match="bogus"):
        resolve_profile("bogus")


def _state(n: int) -> RobotState:
    return RobotState(
        base_pos_w=np.array([0.1, 0.2, 0.9]),
        base_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel_b=np.array([0.01, 0.02, 0.03]),
        q=np.arange(n, dtype=np.float64) * 0.01,
        qd=np.arange(n, dtype=np.float64) * 0.1,
    )


def test_build_observation_sizes_the_joint_blocks():
    target = ObsTarget(pos_w=np.array([0.5, -0.3, 1.0]), vel_w=np.array([1.0, 0.0, 0.5]), time_to_strike=0.4)
    for n, expected_dim in ((31, 110), (29, 104)):
        last = np.full(n, 0.5)
        obs = build_observation(_state(n), target, last, np.zeros(n), np.array([0.1, 0.2]), num_joints=n)
        assert obs.shape == (expected_dim,) and obs.dtype == np.float32
        prof = AGIBOT_A3 if n == 31 else UNITREE_G1
        np.testing.assert_allclose(obs[prof.joint_pos_slice], np.arange(n) * 0.01, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(obs[prof.joint_vel_slice], np.arange(n) * 0.1, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(obs[prof.actions_slice], 0.5)
        assert obs[-1] == pytest.approx(0.4)
    # The default remains the 31-joint A3 call signature.
    obs = build_observation(_state(31), target, np.zeros(31), np.zeros(31), np.zeros(2))
    assert obs.shape == (110,)
    with pytest.raises(ValueError):
        build_observation(_state(29), target, np.zeros(29), np.zeros(29), np.zeros(2))  # 29 joints, A3 default


def test_metadata_validators_accept_profile_arguments():
    validate_embedded_joint_order(",".join(UNITREE_G1.joint_names), UNITREE_G1.joint_names)
    validate_embedded_contract("hitter_pure_unitree_g1", UNITREE_G1.contract_name, UNITREE_G1.obs_dim)
    with pytest.raises(ValueError, match="joint_order"):
        validate_embedded_joint_order(",".join(UNITREE_G1.joint_names))  # A3 default rejects the G1 order
    with pytest.raises(ValueError, match="hitter_pure"):
        validate_embedded_contract("hitter_pure", UNITREE_G1.contract_name, UNITREE_G1.obs_dim)
    # A3 defaults unchanged.
    validate_embedded_joint_order(",".join(AGIBOT_A3.joint_names))
    validate_embedded_contract("hitter_pure")


class _FakeSession:
    def __init__(self, n: int) -> None:
        self.n = n
        self.feeds = None

    def run(self, output_names, feeds):
        self.feeds = feeds
        return [np.arange(self.n, dtype=np.float32).reshape(1, self.n)]


def test_g1_joint_columns_round_trip_between_sdk_and_policy_order():
    """G1 copy of the model_21800 permutation check: columns permute through the 104-D slices."""
    policy = OnnxPolicy.__new__(OnnxPolicy)
    policy._profile = UNITREE_G1
    policy._sess = _FakeSession(29)
    policy._input_name = "obs"
    policy._time_step_name = None
    policy._output_name = "actions"
    policy._policy_to_sdk = np.arange(28, -1, -1, dtype=np.int64)

    obs = np.arange(104, dtype=np.float32)
    action = policy.infer(obs)
    fed = policy._sess.feeds["obs"]
    assert fed.shape == (1, 104)
    np.testing.assert_array_equal(fed[0, 3:32], obs[3:32][::-1])
    np.testing.assert_array_equal(fed[0, 32:61], obs[32:61][::-1])
    np.testing.assert_array_equal(fed[0, 61:90], obs[61:90][::-1])
    np.testing.assert_array_equal(fed[0, 90:], obs[90:])
    np.testing.assert_array_equal(action, np.arange(29, dtype=np.float32)[::-1])
    # Objects built without __init__ default to the A3 (class-level profile).
    assert OnnxPolicy.__new__(OnnxPolicy)._profile is AGIBOT_A3


def test_g1_runtime_config_resolves_the_profile_and_per_joint_gains():
    cfg = RuntimeConfig.load(_G1_RUNTIME)
    assert cfg.profile is UNITREE_G1
    assert cfg.contract == "hitter_pure_unitree_g1"
    assert cfg.passive_neck is False
    assert cfg.action_adapter.num_joints == 29 and cfg.action_adapter.joint_names == UNITREE_G1.joint_names
    assert cfg.sim_kp.shape == (29,) and cfg.sim_kd.shape == (29,)
    assert np.all(cfg.sim_kp > 0) and np.all(cfg.sim_kd > 0)
    assert os.path.samefile(cfg.model_xml_path, _G1_MJCF)
    with open(_G1_RUNTIME) as fh:
        doc = yaml.safe_load(fh)
    kp_by_name = doc["simulation"]["pd_gains"]["per_joint"]["stiffness"]
    np.testing.assert_allclose(cfg.sim_kp, [kp_by_name[n] for n in UNITREE_G1.joint_names])
    # Explicit selection agrees with the YAML; a conflicting one is rejected.
    assert RuntimeConfig.load(_G1_RUNTIME, profile="unitree_g1").profile is UNITREE_G1
    with pytest.raises(ValueError, match="mismatch"):
        RuntimeConfig.load(_G1_RUNTIME, profile="agibot_a3")


def test_a3_runtime_config_default_profile_unchanged():
    cfg = RuntimeConfig.load(_A3_RUNTIME)
    assert cfg.profile is AGIBOT_A3 and cfg.passive_neck is True and cfg.contract == "hitter_pure"
    assert cfg.action_adapter.num_joints == 31 and cfg.sim_kp.shape == (31,)
    assert RuntimeConfig.load(_A3_RUNTIME, profile="agibot_a3").profile is AGIBOT_A3
    with pytest.raises(ValueError, match="mismatch"):
        RuntimeConfig.load(_A3_RUNTIME, profile="unitree_g1")


def test_group_gains_expand_by_profile(tmp_path):
    spec = {"groups": {"waist": {"kp": 1, "kd": 0.1}, "arm": {"kp": 2, "kd": 0.2}, "leg": {"kp": 3, "kd": 0.3}}}
    kp, kd = _expand_pd_gains(spec, tmp_path, UNITREE_G1)
    assert kp.shape == (29,)
    for i, name in enumerate(UNITREE_G1.joint_names):
        expected = 1 if name.startswith("waist_") else (3 if any(k in name for k in ("hip_", "knee_", "ankle_")) else 2)
        assert kp[i] == expected and kd[i] == pytest.approx(expected / 10)
    with pytest.raises(ValueError, match="neck"):
        _expand_pd_gains(spec, tmp_path, AGIBOT_A3)  # the A3 still needs its neck group
