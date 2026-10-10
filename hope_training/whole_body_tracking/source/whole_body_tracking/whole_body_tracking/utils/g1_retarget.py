# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Mechanical Agibot A3 -> Unitree G1 body-name retarget for HOPE environment configs.

The HOPE ping-pong environment configs (``tasks/tracking/config/agibot_a3/hope_env_cfg.py``) are a
long chain of ``configclass`` layers that name A3 bodies (``torso_Link``, ``left_wrist_yaw_Link``,
``pelvis_link`` ...) in term parameters, ``SceneEntityCfg.body_names`` lists, regex strings and plain
string fields, and a few MDP functions carry A3 names as *default arguments*. The G1 uses the same
kinematic vocabulary with different spelling (lowercase ``*_link``, root ``pelvis``), no head joints,
and keeps the A3 racket link names (``pingpang_red_Link`` ...) by construction of its racket-equipped
URDF. This module rewrites a fully built A3 config tree into G1 names in place:

* :func:`retarget_name` maps one string (exact names *and* names embedded in regexes);
* :func:`retarget_cfg_tree` walks a dataclass/dict/list tree, rewrites every string, and injects
  explicit ``params`` for MDP functions whose default arguments name an A3 body;
* :func:`find_residual_a3_names` lists anything that still looks like an A3 name afterwards so the
  env config can fail loudly instead of resolving a body that does not exist.

Pure Python (``dataclasses`` + ``inspect`` + ``re``): importable and testable without Isaac / torch.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from collections.abc import Callable, Iterable
from typing import Any

#: A3 body -> G1 body for every A3 body whose name does not follow the plain ``*_Link -> *_link``
#: rule. ``None`` marks an A3 body with no G1 counterpart; encountering one is an error.
A3_TO_G1_BODY_NAMES: dict[str, str | None] = {
    "pelvis_link": "pelvis",
    "imu_in_pelvis_Link": "imu_in_pelvis",
    "imu_in_torso_Link": "imu_in_torso",
    "left_hand_Link": "left_rubber_hand",
    "right_hand_Link": "right_rubber_hand",
    # Racket links keep their A3 names in unitree/URDF/g1_29dof_pingpang (by design).
    "right_hand_pingpang_Link": "right_hand_pingpang_Link",
    "pingpang_red_Link": "pingpang_red_Link",
    "pingpang_black_Link": "pingpang_black_Link",
    # The vendor calibration sphere is removed from both prepared assets; a negative-lookahead on
    # it in a contact regex is harmless, so the literal is preserved rather than rejected.
    "pingbang_ball_Link": "pingbang_ball_Link",
    # No G1 counterpart.
    "head_yaw_Link": None,
    "head_pitch_Link": None,
    "torso_shell_Link": None,
    "left_knee_shell_Link": None,
    "right_knee_shell_Link": None,
}

#: A3 joints with no G1 counterpart (the G1 has no neck); any surviving reference is an error.
A3_ONLY_JOINT_NAMES: tuple[str, ...] = ("head_yaw_joint", "head_pitch_joint")

#: Attributes never descended into (callables / class references / the scene with its asset cfg).
_SKIP_ATTRS: frozenset[str] = frozenset({"func", "class_type"})

_LINK_RE = re.compile(r"\b([A-Za-z0-9_]+?)_Link\b")
_PELVIS_RE = re.compile(r"\bpelvis_link\b")
_A3_RESIDUAL_RE = re.compile(r"\b(?:[A-Za-z0-9_]+_Link|pelvis_link|head_yaw_joint|head_pitch_joint)\b")
_RACKET_LINKS = frozenset(
    {"right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link", "pingbang_ball_Link"}
)


class UnmappableA3NameError(ValueError):
    """An A3 body/joint without a G1 counterpart was referenced by the config."""


def retarget_name(text: str) -> str:
    """Rewrite every A3 body name inside ``text`` (a plain name or a regex) into its G1 name."""

    def _link(match: re.Match) -> str:
        full = match.group(0)
        if full in A3_TO_G1_BODY_NAMES:
            mapped = A3_TO_G1_BODY_NAMES[full]
            if mapped is None:
                raise UnmappableA3NameError(f"A3 body {full!r} has no Unitree G1 counterpart (in {text!r})")
            return mapped
        return f"{match.group(1)}_link"

    out = _LINK_RE.sub(_link, text)
    out = _PELVIS_RE.sub(A3_TO_G1_BODY_NAMES["pelvis_link"], out)
    for joint in A3_ONLY_JOINT_NAMES:
        if re.search(rf"\b{joint}\b", out):
            raise UnmappableA3NameError(f"A3 joint {joint!r} has no Unitree G1 counterpart (in {text!r})")
    return out


def _is_cfg_object(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _term_callable(func: Any) -> Callable | None:
    if func is None:
        return None
    if inspect.isclass(func):
        call = getattr(func, "__call__", None)
        return call if callable(call) else None
    return func if callable(func) else None


def _inject_default_name_params(term: Any, log: list[str], path: str) -> None:
    """Make A3-named default arguments explicit (retargeted) in ``term.params``."""
    func = _term_callable(getattr(term, "func", None))
    params = getattr(term, "params", None)
    if func is None or not isinstance(params, dict):
        return
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return
    for name, parameter in signature.parameters.items():
        default = parameter.default
        if not isinstance(default, str) or name in params:
            continue
        mapped = retarget_name(default)
        if mapped != default:
            params[name] = mapped
            log.append(f"{path}.params[{name!r}] = {mapped!r} (was default {default!r})")


def retarget_cfg_tree(root: Any, *, skip_attrs: Iterable[str] = ("scene",), _path: str = "cfg") -> list[str]:
    """Rewrite A3 names to G1 names everywhere under ``root`` (in place). Returns a change log.

    ``skip_attrs`` names top-level attributes that are not descended into; the default skips the
    scene (its ``robot`` ArticulationCfg is replaced wholesale by the caller).
    """
    log: list[str] = []
    seen: set[int] = set()
    skip = frozenset(skip_attrs)

    def _walk(value: Any, path: str, top: bool) -> Any:
        if isinstance(value, str):
            mapped = retarget_name(value)
            if mapped != value:
                log.append(f"{path}: {value!r} -> {mapped!r}")
            return mapped
        if isinstance(value, (bool, int, float, type(None))) or callable(value) and not _is_cfg_object(value):
            return value
        if isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = _walk(value[key], f"{path}[{key!r}]", False)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = _walk(item, f"{path}[{index}]", False)
            return value
        if isinstance(value, tuple):
            return type(value)(_walk(item, f"{path}[{index}]", False) for index, item in enumerate(value))
        if _is_cfg_object(value):
            if id(value) in seen:
                return value
            seen.add(id(value))
            _inject_default_name_params(value, log, path)
            for attr in list(vars(value).keys()):
                if attr in _SKIP_ATTRS or (top and attr in skip):
                    continue
                current = getattr(value, attr)
                updated = _walk(current, f"{path}.{attr}", False)
                if updated is not current:
                    setattr(value, attr, updated)
            return value
        return value

    _walk(root, _path, True)
    return log


def find_residual_a3_names(root: Any, *, skip_attrs: Iterable[str] = ("scene",)) -> list[tuple[str, str]]:
    """Return ``(path, string)`` pairs that still contain an A3 body/joint name (racket links excepted)."""
    found: list[tuple[str, str]] = []
    seen: set[int] = set()
    skip = frozenset(skip_attrs)

    def _check(text: str, path: str) -> None:
        for match in _A3_RESIDUAL_RE.finditer(text):
            if match.group(0) not in _RACKET_LINKS:
                found.append((path, text))
                return

    def _walk(value: Any, path: str, top: bool) -> None:
        if isinstance(value, str):
            _check(value, path)
        elif isinstance(value, dict):
            for key, item in value.items():
                _walk(item, f"{path}[{key!r}]", False)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                _walk(item, f"{path}[{index}]", False)
        elif _is_cfg_object(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            for attr, item in vars(value).items():
                if attr in _SKIP_ATTRS or (top and attr in skip):
                    continue
                _walk(item, f"{path}.{attr}", False)

    _walk(root, "cfg", True)
    return found
