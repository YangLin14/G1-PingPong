"""Unitree G1 MuJoCo plant parity (host-only; needs numpy, pyyaml, mujoco).

The sim-to-sim gate for the G1: the generated MJCF, the runtime YAML's per-joint PD gains and the
action adapter must all carry the SAME numbers as the Isaac articulation (``robots/g1.py``), mirrored
in ``unitree/scripts/g1_actuation.py``. The Isaac-vs-MuJoCo trajectory replay itself
(``scripts/sim2sim_replay_g1.py``) needs the Kit app and runs in the Slurm smoke job.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[3]
REF = REPO / "a3_deploy" / "a3_deploy_example" / "reference"
CFG_DIR = REPO / "a3_deploy" / "a3_deploy_example" / "config"
MJCF = REPO / "unitree" / "URDF" / "g1_29dof_pingpang" / "mjcf" / "g1_29dof_pingpang.xml"
SCRIPTS = REPO / "hope_training" / "whole_body_tracking" / "scripts"
for path in (str(REF), str(REPO / "unitree" / "scripts"), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import g1_actuation  # noqa: E402
from a3_deploy_onnx_ref_pingpong.config import RuntimeConfig  # noqa: E402
from a3_deploy_onnx_ref_pingpong.robot_profile import UNITREE_G1  # noqa: E402


@pytest.fixture(scope="module")
def runtime() -> RuntimeConfig:
    return RuntimeConfig.load(CFG_DIR / "hope_pingpong_runtime_unitree_g1.yaml", profile="unitree_g1")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MJCF))


def test_generated_yamls_match_generator():
    proc = subprocess.run(
        [sys.executable, str(REPO / "unitree" / "scripts" / "generate_g1_runtime_yamls.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_runtime_gains_equal_isaac_recipe(runtime):
    names = list(UNITREE_G1.joint_names)
    kp = np.asarray([g1_actuation.stiffness(n) for n in names])
    kd = np.asarray([g1_actuation.damping(n) for n in names])
    np.testing.assert_allclose(runtime.sim_kp, kp, rtol=1e-5)
    np.testing.assert_allclose(runtime.sim_kd, kd, rtol=1e-5)
    # the leg group alone spans 28.5 .. 99.1 N.m/rad: per-joint, never group-averaged
    assert runtime.sim_kp.max() / runtime.sim_kp.min() > 3.0


def test_action_adapter_equals_isaac_recipe(runtime):
    names = list(UNITREE_G1.joint_names)
    np.testing.assert_allclose(runtime.action_adapter.default_q, [g1_actuation.init_joint_pos(n) for n in names], atol=1e-9)
    np.testing.assert_allclose(runtime.action_adapter.action_scale, [g1_actuation.action_scale(n) for n in names], rtol=1e-5)


def test_mjcf_armature_limits_and_keyframe(model):
    names = list(UNITREE_G1.joint_names)
    assert model.nu == 29 and model.nq == 7 + 29 and model.nkey >= 1
    trn_joint = model.actuator_trnid[:, 0]
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert jid >= 0, n
        dof = model.jnt_dofadr[jid]
        assert np.isclose(model.dof_armature[dof], g1_actuation.armature(n), rtol=1e-4), (n, model.dof_armature[dof])
        assert model.dof_damping[dof] == 0.0 and model.dof_frictionloss[dof] == 0.0, n
        act = np.where(trn_joint == jid)[0]
        assert act.size == 1, n
        assert model.actuator_ctrllimited[act[0]]
        np.testing.assert_allclose(model.actuator_ctrlrange[act[0]], (-g1_actuation.effort_limit(n), g1_actuation.effort_limit(n)))
        # stand keyframe = Isaac init_state
        assert np.isclose(model.key_qpos[0, model.jnt_qposadr[jid]], g1_actuation.init_joint_pos(n), atol=1e-9), n
    assert abs(model.key_qpos[0, 2] - g1_actuation.INIT_PELVIS_Z) < 0.01  # grounded stand vs Isaac drop-in height
    assert model.opt.timestep == pytest.approx(0.001)


def test_scene_timing_and_joint_mapping(runtime):
    from mujoco_pingpong_scene import PingPongRealPhysicsScene

    spec = importlib.util.spec_from_file_location(
        "hope_success_metric_plant_test",
        REPO / "hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/utils/success_metric.py",
    )
    metric = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metric
    spec.loader.exec_module(metric)
    prof = runtime.profile
    scene = PingPongRealPhysicsScene(
        str(runtime.model_xml_path), metric.load_ball_physics_config(), prof.joint_names, control_dt=runtime.control_dt, near_edge_x=0.5,
        base_free_joint=prof.base_free_joint, gyro_sensor=prof.gyro_sensor, racket_site=prof.racket_site,
        racket_collision_geom=prof.racket_collision_geom, floor_geom=prof.floor_geom,
    )
    assert runtime.control_dt == pytest.approx(0.02) and scene._substeps == 20
    scene.reset_stand()
    state = scene.read_robot_state()
    np.testing.assert_allclose(state.q, runtime.action_adapter.default_q, atol=1e-6)
    # policy column i drives MJCF joint named prof.joint_names[i]
    for i, n in enumerate(prof.joint_names):
        jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert scene.model.actuator_trnid[scene._act_idx[i], 0] == jid
    # 0.2 s of pure PD hold: the plant must not blow up (gain-order bugs show up here). A static PD
    # hold with the recipe's ankle gains sags in BOTH simulators (the policy balances actively), so
    # only gross behaviour is asserted; the quantitative gate is scripts/sim2sim_replay_g1.py.
    scene.write_targets(runtime.action_adapter.default_q, runtime.sim_kp, runtime.sim_kd)
    for _ in range(10):
        scene.step()
    assert not scene.base_fallen()
    assert abs(scene.base_pos_w()[2] - state.base_pos_w[2]) < 0.05
