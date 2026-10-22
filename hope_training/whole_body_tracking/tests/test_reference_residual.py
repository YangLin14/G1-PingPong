"""D26 reference-residual action: the deploy/MuJoCo side matches the training side.

Training (``mdp/hope_actions.py`` ``ClampedJointPositionAction._apply_reference_residual``) drives the
listed joints as ``clip q(time step) + bound * tanh(scale * a / bound)``; the runner
(``scripts/reference_residual.py``) gets the clip frame back from the observed time to strike. Host-side:
the training method runs on a stand-in object (torch only, no Isaac), the runner on the real D26 clip
and task yaml.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys
import types

import numpy as np
import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
_TASK = _ROOT / "cfg" / "task" / "HOPEPingPongUnitreeG1_BH_FIST_ARM.yaml"
_HOPE_ACTIONS = _ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp" / "hope_actions.py"

from reference_residual import ReferenceResidual  # noqa: E402


def _joint_order() -> list[str]:
    with open(_ROOT.parent / "config" / "joint_order_unitree_g1.yaml", "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return [str(n) for n in (doc["joint_order"] if isinstance(doc, dict) else doc)]


def _runner(scale: float = 0.5) -> ReferenceResidual:
    n = len(_joint_order())
    return ReferenceResidual.from_task(_TASK, _joint_order(), np.full(n, scale), np.full(n, -10.0), np.full(n, 10.0))


def _training_method():
    """Compile _apply_reference_residual out of hope_actions.py without importing Isaac Lab."""
    torch = pytest.importorskip("torch")
    tree = ast.parse(_HOPE_ACTIONS.read_text(encoding="utf-8"))
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "ClampedJointPositionAction":
            for fn in cls.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_apply_reference_residual":
                    ns = {"torch": torch}
                    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_HOPE_ACTIONS), "exec"), ns)
                    return torch, ns["_apply_reference_residual"]
    raise AssertionError("_apply_reference_residual not found")


def test_task_declares_17_upper_body_joints():
    r = _runner()
    order = _joint_order()
    names = {order[c] for c in r.cols}
    assert len(names) == 17
    assert all(n.startswith(("right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
                             "left_wrist", "waist")) for n in names)
    assert r.clip_name == "g1ikfist_hope_backhand_arm.npz"


def test_frame_from_time_to_strike():
    r = _runner()
    assert np.allclose(r.reference(0.0), r.q_clip[r.strike_frame, r.cols])
    assert np.allclose(r.reference(r.strike_frame * r.dt), r.q_clip[0, r.cols])
    assert np.allclose(r.reference(5.0), r.q_clip[0, r.cols])          # ready / hold -> ready pose
    assert np.allclose(r.reference(-50.0), r.q_clip[-1, r.cols])       # long after the strike -> last frame
    mid = 0.5 * (r.q_clip[r.strike_frame - 3, r.cols] + r.q_clip[r.strike_frame - 2, r.cols])
    assert np.allclose(r.reference(2.5 * r.dt), mid)


def test_residual_is_bounded_and_zero_action_follows_clip():
    r = _runner(scale=0.5)
    q0 = np.zeros(len(_joint_order()))
    out = r.apply(q0, np.zeros_like(q0), 0.0)
    assert np.allclose(out[r.cols], r.q_clip[r.strike_frame, r.cols])
    big = r.apply(q0, np.full_like(q0, 1e3), 0.0)
    assert np.all(big[r.cols] - r.q_clip[r.strike_frame, r.cols] <= r.bounds + 1e-9)
    others = np.setdiff1d(np.arange(len(q0)), r.cols)
    assert np.all(big[others] == 0.0)  # non-residual joints keep the decoded default + scale * a


def test_training_and_runner_agree():
    torch, method = _training_method()
    r = _runner(scale=0.5)
    order = _joint_order()
    n = len(order)
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(4, n))
    steps = np.array([0, 10, r.strike_frame, r.q_clip.shape[0] - 1])
    clip = torch.as_tensor(r.q_clip, dtype=torch.float32)
    motion = types.SimpleNamespace(motion=types.SimpleNamespace(joint_pos=clip), time_steps=torch.as_tensor(steps))
    env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: motion))
    cols = torch.as_tensor(r.cols)
    fake = types.SimpleNamespace(
        _env=env,
        cfg=types.SimpleNamespace(reference_command_name="motion"),
        _ref_action_cols=cols,
        _ref_joint_ids=cols,  # action columns == articulation joint ids for the G1 (canonical order)
        _ref_bounds=torch.as_tensor(r.bounds, dtype=torch.float32),
        _scale=torch.full((4, n), 0.5),
        _raw_actions=torch.as_tensor(raw, dtype=torch.float32),
        _processed_actions=torch.zeros((4, n)),
    )
    method(fake)
    for i, s in enumerate(steps):
        tts = (r.strike_frame - s) * r.dt
        expect = r.apply(np.zeros(n), raw[i], tts)
        assert np.allclose(fake._processed_actions[i, r.cols].numpy(), expect[r.cols], atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
