"""The Unitree G1 recipe (variant A) follows its documented scaling rule and the YAML mirrors it (host-only)."""

from __future__ import annotations

import importlib.util
import pathlib
import xml.etree.ElementTree as ET

import pytest

yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[3]
WBT = REPO / "hope_training" / "whole_body_tracking"
RECIPE = WBT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/g1_recipe.py"
G1_YAML = WBT / "cfg" / "task" / "HOPEPingPongUnitreeG1.yaml"
A3_YAML = WBT / "cfg" / "task" / "HOPEPingPong.yaml"
A3_URDF = REPO / "agibot" / "URDF" / "A3T2.5-URDF-std-pingpang" / "urdf" / "URDF-JOINT-LINK.urdf"
G1_URDF = REPO / "unitree" / "URDF" / "g1_description" / "urdf" / "g1_29dof_rev_1_0.urdf"


def _load_recipe():
    spec = importlib.util.spec_from_file_location("g1_recipe_under_test", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joint_origins(urdf: pathlib.Path) -> dict[str, tuple[float, float, float]]:
    root = ET.parse(urdf).getroot()
    out = {}
    for j in root.iter("joint"):
        o = j.find("origin")
        out[j.get("name")] = tuple(float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split())
    return out


def _norm(v):
    return sum(x * x for x in v) ** 0.5


def _arm_reach(origins: dict, racket: float = 0.21) -> float:
    """shoulder_pitch -> wrist_yaw link chain length + racket (metres)."""
    chain = ("right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
             "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
    return sum(_norm(origins[j]) for j in chain) + racket


def test_reach_ratio_matches_the_urdfs():
    recipe = _load_recipe()
    a3, g1 = _joint_origins(A3_URDF), _joint_origins(G1_URDF)
    a3_reach, g1_reach = _arm_reach(a3), _arm_reach(g1)
    assert a3_reach == pytest.approx(recipe.A3_ARM_REACH, abs=0.02)
    assert g1_reach == pytest.approx(recipe.G1_ARM_REACH, abs=0.02)
    assert g1_reach / a3_reach == pytest.approx(recipe.REACH_RATIO, abs=0.02)
    # shoulder heights: pelvis + torso chain + shoulder_pitch origin z
    a3_shoulder = recipe.A3_PELVIS_Z + a3["waist_pitch_joint"][2] + a3["right_shoulder_pitch_joint"][2]
    g1_shoulder = recipe.G1_PELVIS_Z + g1["waist_roll_joint"][2] + g1["right_shoulder_pitch_joint"][2]
    assert a3_shoulder == pytest.approx(recipe.A3_SHOULDER_Z, abs=0.02)
    assert g1_shoulder == pytest.approx(recipe.G1_SHOULDER_Z, abs=0.02)


def test_g1_values_follow_the_scaling_rule():
    r = _load_recipe()
    assert r.STRIKE_PLANE_X == pytest.approx(r.A3_STRIKE_PLANE_X * r.REACH_RATIO, abs=0.01)
    for a3_clip, g1_clip in zip(r.A3_POS_RANGE_PER_CLIP, r.POS_RANGE_PER_CLIP):
        assert g1_clip[0] == (r.STRIKE_PLANE_X, r.STRIKE_PLANE_X)
        for a3_y, g1_y in zip(a3_clip[1], g1_clip[1]):
            assert g1_y == pytest.approx(a3_y * r.REACH_RATIO, abs=0.01)
        assert g1_clip[2] == r.Z_BOX
    # z box: shoulder-relative lower bound clamped to table + 4 cm; upper bound = shoulder + 5 cm
    assert r.Z_BOX[0] == pytest.approx(max(r.G1_SHOULDER_Z - (r.A3_SHOULDER_Z - 0.85), r.TABLE_TOP_Z + 0.04), abs=0.01)
    assert r.Z_BOX[1] == pytest.approx(r.G1_SHOULDER_Z + 0.05, abs=0.01)
    assert r.VENUE_CONTACT_Z_RANGE[1] == r.Z_BOX[1] and r.Z_BOX[0] <= r.VENUE_CONTACT_Z_RANGE[0] < r.VENUE_CONTACT_Z_RANGE[1]
    assert r.ROOT_HEIGHT_MIN == pytest.approx(r.A3_ROOT_HEIGHT_MIN * r.G1_PELVIS_Z / r.A3_PELVIS_Z, abs=0.01)
    assert r.ROOT_HEIGHT_MIN < r.G1_PELVIS_Z  # must be satisfiable standing
    assert r.BASE_TOO_LOW == pytest.approx(r.A3_BASE_TOO_LOW * r.G1_PELVIS_Z / r.A3_PELVIS_Z, abs=0.05)
    assert r.STANCE_WIDTH == pytest.approx(tuple(v * r.STANCE_RATIO for v in r.A3_STANCE_WIDTH), abs=0.01)
    assert r.HOLD_READY_REACH == pytest.approx(r.A3_HOLD_READY_REACH * r.REACH_RATIO, abs=0.01)


def test_yaml_mirrors_the_recipe():
    r = _load_recipe()
    g1 = yaml.safe_load(G1_YAML.read_text(encoding="utf-8"))
    a3 = yaml.safe_load(A3_YAML.read_text(encoding="utf-8"))
    for key, value in r.YAML_REWARD_VALUES.items():
        assert g1["rewards"][key] == pytest.approx(value), key
    box = g1["racket"]["pos_range_per_clip"]
    for clip, name in zip(r.POS_RANGE_PER_CLIP, ("forehand", "backhand")):
        for axis, rng in zip(("x", "y", "z"), clip):
            assert tuple(box[name][axis]) == pytest.approx(rng), (name, axis)
    assert tuple(g1["racket"]["venue_tuple_contact_z_range"]) == pytest.approx(r.VENUE_CONTACT_Z_RANGE)
    # the A3 recipe is untouched and still carries the A3 numbers the module records
    assert a3["rewards"]["rally_ready_root_height_debt_min_height"] == pytest.approx(r.A3_ROOT_HEIGHT_MIN)
    assert tuple(a3["racket"]["pos_range_per_clip"]["forehand"]["z"]) == pytest.approx(r.A3_POS_RANGE_PER_CLIP[0][2])
    # every other reward key of the A3 recipe is present with the same value (only the retuned ones differ)
    retuned = set(r.YAML_REWARD_VALUES) | {"passive_head_raw_action_weight", "passive_head_raw_action_std", "passive_head_raw_action_huber_delta"}
    for key, value in a3["rewards"].items():
        if key in retuned:
            continue
        assert g1["rewards"].get(key) == value, key


def test_apply_recipe_writes_every_field():
    r = _load_recipe()

    class _Term:
        def __init__(self, **params):
            self.params = dict(params)

    class _NS:
        pass

    cfg = _NS()
    cfg.commands = _NS(); cfg.commands.racket_target = _NS()
    cfg.rewards = _NS()
    cfg.rewards.rally_ready_root_height_debt = _Term(min_height=0.98)
    cfg.rewards.ready_stance_width = _Term(lo=0.25, hi=0.35)
    cfg.rewards.stance_width = _Term(lo=0.25, hi=0.35)
    cfg.rewards.hold_ready = _Term(reach=0.3)
    cfg.terminations = _NS(); cfg.terminations.base_too_low = _Term(minimum_height=0.5)
    log = r.apply_recipe(cfg)
    rt = cfg.commands.racket_target
    assert rt.racket_pos_range_per_clip == r.POS_RANGE_PER_CLIP
    assert rt.venue_tuple_contact_z_range == r.VENUE_CONTACT_Z_RANGE
    assert cfg.rewards.rally_ready_root_height_debt.params["min_height"] == r.ROOT_HEIGHT_MIN
    assert (cfg.rewards.ready_stance_width.params["lo"], cfg.rewards.ready_stance_width.params["hi"]) == r.STANCE_WIDTH
    assert cfg.rewards.hold_ready.params["reach"] == r.HOLD_READY_REACH
    assert cfg.terminations.base_too_low.params["minimum_height"] == r.BASE_TOO_LOW
    assert len(log) == 10
