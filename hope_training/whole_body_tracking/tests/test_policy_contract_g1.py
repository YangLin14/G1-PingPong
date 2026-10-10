"""Unitree G1 policy contract: ``hitter_pure_unitree_g1`` (104-D / 29) across registry, YAML, exporter, runner.

Host-only mirror of ``test_policy_contract.py``. The A3 entries are asserted unchanged alongside.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(os.path.dirname(_ROOT))
_TRACKING = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "tasks", "tracking")
_UTILS = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "utils")
_JOINT_ORDER_YAML = os.path.join(_REPO, "hope_training", "config", "joint_order_unitree_g1.yaml")
_A3_JOINT_ORDER_YAML = os.path.join(_REPO, "hope_training", "config", "joint_order_agibot_a3.yaml")
_TASK_YAML = os.path.join(_ROOT, "cfg", "task", "HOPEPingPongUnitreeG1.yaml")
_A3_TASK_YAML = os.path.join(_ROOT, "cfg", "task", "HOPEPingPong.yaml")
_REFERENCE_DIR = os.path.join(_REPO, "a3_deploy", "a3_deploy_example", "reference")
_G1_ROBOT_PY = os.path.join(_ROOT, "source", "whole_body_tracking", "whole_body_tracking", "robots", "g1.py")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


registry = _load("hope_actor_obs_contract_g1", os.path.join(_TRACKING, "actor_observation_contract.py"))
export_script = _load("hope_export_onnx_script_g1", os.path.join(_ROOT, "scripts", "export_onnx.py"))
adapter_cfg = _load("hope_action_adapter_config_g1", os.path.join(_UTILS, "action_adapter_config.py"))

sys.path.insert(0, _REFERENCE_DIR)
from a3_deploy_onnx_ref_pingpong import joint_order_unitree_g1 as ref_g1  # noqa: E402
from a3_deploy_onnx_ref_pingpong.robot_profile import AGIBOT_A3, UNITREE_G1  # noqa: E402


def _yaml_order(path: str) -> tuple[str, ...]:
    with open(path) as f:
        return tuple(str(n) for n in yaml.safe_load(f)["joint_order"])


def test_factory_reproduces_the_a3_contract_and_sizes_the_g1():
    a3 = registry.hitter_pure_contract(31)
    assert a3.layout == registry.HITTER_PURE.layout and a3.total_dim == 110
    assert registry.resolve_actor_observation_contract("hitter_pure") is registry.HITTER_PURE  # identity kept
    g1 = registry.HITTER_PURE_UNITREE_G1
    assert registry.resolve_actor_observation_contract("hitter_pure_unitree_g1") is g1
    assert g1.total_dim == 104 == 17 + 3 * 29
    assert [name for name, _ in g1.layout] == [name for name, _ in registry.HITTER_PURE.layout]
    assert dict(g1.layout)["joint_pos"] == dict(g1.layout)["joint_vel"] == dict(g1.layout)["actions"] == 29
    assert sum(dim for _, dim in g1.layout) == 104
    assert g1.layout[-1] == ("time_to_strike", 1)


def test_task_yaml_declares_the_g1_contract():
    with open(_TASK_YAML) as f:
        task = yaml.safe_load(f)
    assert task["gym_task"] == "HOPE-HitterPingPong-UnitreeG1-v0"
    assert task["actor_obs_contract"] == "hitter_pure_unitree_g1"
    assert task["robot"] == "unitree_g1"
    assert task["experiment_name"] == "unitree_g1_hitter_pingpong"
    assert int(task["build_contract"]["actor_observation_dim"]) == 104
    assert int(task["build_contract"]["action_dim"]) == 29
    # The derived G1 clips (joint-space transfers of the A3 pair) are the defaults.
    assert task["motion_file"].endswith("g1_hope_forehand.npz") and task["motion_file_2"].endswith("g1_hope_backhand.npz")
    # The passive-head penalty does not exist on the G1 env; the reward translator would fail on it.
    assert not any(key.startswith("passive_head_raw_action") for key in task["rewards"])
    # The A3-only PD randomizer mode is not requested.
    assert "pd_mode" not in task["domain_rand"] and task["domain_rand"]["pd_gain_range"] == [0.85, 1.15]
    assert all("_Link" not in body for body in task["friction_contract"]["robot_contact_bodies"])
    # Every other key of the A3 recipe is present (the recipe is reused verbatim).
    with open(_A3_TASK_YAML) as f:
        a3 = yaml.safe_load(f)
    assert set(a3["rewards"]) - set(task["rewards"]) == {
        "passive_head_raw_action_weight", "passive_head_raw_action_std", "passive_head_raw_action_huber_delta"
    }
    # racket: block: identical to the A3 recipe except the documented G1 v2 geometry retune
    # (strike boxes and the venue contact band; see config/g1/g1_recipe.py + test_g1_recipe_scaling.py).
    g1_racket = dict(task["racket"])
    assert set(g1_racket) - set(a3["racket"]) == {"venue_tuple_contact_z_range"}
    g1_racket.pop("venue_tuple_contact_z_range")
    assert g1_racket.pop("pos_range_per_clip") != a3["racket"]["pos_range_per_clip"]
    a3_racket = dict(a3["racket"]); a3_racket.pop("pos_range_per_clip")
    assert a3_racket == g1_racket


def test_manifest_schema_for_29_joints():
    joint_names = [f"j{i}" for i in range(29)]
    manifest = export_script.build_manifest(
        registry.HITTER_PURE_UNITREE_G1,
        joint_names,
        action_adapter_relpath=export_script.ACTION_ADAPTER_RELPATHS["unitree_g1"],
    )
    assert manifest["contract_name"] == "hitter_pure_unitree_g1"
    assert manifest["obs_dim"] == 104 and manifest["action_dim"] == 29
    sig = manifest["onnx_signature"]
    assert sig["input"]["shape"] == [1, 104] and sig["output"]["shape"] == [1, 29]
    assert manifest["action_adapter_config"].endswith("action_adapter_unitree_g1.yaml")
    # The A3 default manifest is unchanged.
    a3 = export_script.build_manifest(registry.HITTER_PURE, [f"j{i}" for i in range(31)])
    assert a3["obs_dim"] == 110 and a3["action_dim"] == 31
    assert a3["action_adapter_config"] == "a3_deploy/a3_deploy_example/config/action_adapter.yaml"


def test_joint_order_yaml_has_29_unique_joints_without_head():
    order = _yaml_order(_JOINT_ORDER_YAML)
    assert len(order) == 29 and len(set(order)) == 29
    assert not any(name.startswith("head_") for name in order)
    assert all(name.endswith("_joint") for name in order)
    # Same joint SET as the A3 minus the neck (the two robots share the joint vocabulary).
    a3 = set(_yaml_order(_A3_JOINT_ORDER_YAML))
    assert set(order) == a3 - {"head_yaw_joint", "head_pitch_joint"}


def test_training_yaml_matches_reference_runner_and_robot_module():
    order = _yaml_order(_JOINT_ORDER_YAML)
    assert order == tuple(ref_g1.JOINT_NAMES) == UNITREE_G1.joint_names
    assert ref_g1.NUM_JOINTS == 29 and ref_g1.HEAD_INDICES == ()
    # robots/g1.py (imports isaaclab) is read as source.
    import ast

    with open(_G1_ROBOT_PY, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "UNITREE_G1_JOINT_NAMES" for t in node.targets)
    )
    assert tuple(names) == order


def test_training_loader_is_robot_keyed():
    assert adapter_cfg.load_joint_order(robot="unitree_g1") == _yaml_order(_JOINT_ORDER_YAML)
    assert adapter_cfg.load_joint_order() == _yaml_order(_A3_JOINT_ORDER_YAML)  # A3 default unchanged
    assert adapter_cfg.find_shared_action_adapter_config("unitree_g1").endswith("action_adapter_unitree_g1.yaml")
    assert adapter_cfg.find_shared_action_adapter_config().endswith("action_adapter.yaml")
    import pytest

    with pytest.raises(ValueError):
        adapter_cfg.load_joint_order(robot="bogus")


def test_export_gate_uses_the_robot_source_name():
    import pytest

    order = list(_yaml_order(_JOINT_ORDER_YAML))
    export_script.assert_canonical_joint_order(order, order, source=export_script.JOINT_ORDER_RELPATHS["unitree_g1"])
    permuted = list(order)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(RuntimeError, match="joint_order_unitree_g1"):
        export_script.assert_canonical_joint_order(
            permuted, order, source=export_script.JOINT_ORDER_RELPATHS["unitree_g1"]
        )


def test_a3_profile_is_untouched():
    assert AGIBOT_A3.joint_names == _yaml_order(_A3_JOINT_ORDER_YAML)
    assert AGIBOT_A3.passive_indices == (3, 4) and AGIBOT_A3.obs_dim == 110
