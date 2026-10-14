# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration loader for the reference runner.

Reads ``config/hope_pingpong_runtime.yaml`` (the clean 110-D ``hitter_pure``
runtime config) and resolves the ActionAdapter, the example simulation PD gains,
and the lifecycle timing into ready-to-use arrays. All relative paths in the YAML
are resolved against the YAML file's own directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .action_adapter import ActionAdapter
from .joint_order import JOINT_NAMES, NUM_JOINTS  # noqa: F401  (A3 defaults, kept for importers)
from .lifecycle import LifecycleConfig
from .observation import CONTRACT_NAME  # noqa: F401
from .robot_profile import AGIBOT_A3, A3_GROUP_RANGES, RobotProfile, resolve_profile

# Index groups of the four A3 joint groups (used to expand example PD gains); profile-driven now.
_GROUP_RANGES = A3_GROUP_RANGES


@dataclass
class RuntimeConfig:
    control_hz: float
    onnx_path: Path
    model_xml_path: Path
    action_adapter: ActionAdapter
    sim_kp: np.ndarray
    sim_kd: np.ndarray
    lifecycle: LifecycleConfig
    passive_neck: bool = True
    contract: str = CONTRACT_NAME
    config_dir: Path = field(default_factory=Path)
    # Which robot this runtime drives (joint order, dims, passive columns, MJCF names). The A3 is
    # the default; ``policy.joint_order`` in the YAML and/or ``load(profile=...)`` select another.
    profile: RobotProfile = AGIBOT_A3

    @property
    def control_dt(self) -> float:
        return 1.0 / float(self.control_hz)

    @classmethod
    def load(cls, path: str | Path, profile: str | RobotProfile | None = None) -> "RuntimeConfig":
        path = Path(path).resolve()
        cfg_dir = path.parent
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)

        norm = str(doc.get("observation_normalization", "none")).lower()
        if norm != "none":
            raise ValueError(
                f"observation_normalization must be 'none' (raw obs), got '{norm}'"
            )

        # Robot selection: explicit argument and/or the YAML's policy.joint_order (now load-bearing;
        # the shipped A3 config says ``agibot_a3``). Absent -> the A3. They must agree if both given.
        resolved = _select_profile(profile, doc.get("policy", {}).get("joint_order"), path)

        contract = str(doc.get("contract", resolved.contract_name))
        if contract != resolved.contract_name:
            raise ValueError(
                f"runtime config contract must be '{resolved.contract_name}' (the "
                f"{resolved.obs_dim}-D observation layout this runner implements for the "
                f"{resolved.name}), got '{contract}'"
            )

        control_hz = float(doc.get("control_hz", 50.0))
        dt = 1.0 / control_hz

        onnx_path = _resolve(cfg_dir, doc["policy"]["onnx_path"])
        model_xml_path = _resolve(cfg_dir, doc["simulation"]["model_xml_path"])
        adapter_path = _resolve(cfg_dir, doc["action_adapter"]["config_path"])
        adapter = ActionAdapter.from_yaml(adapter_path, joint_names=resolved.joint_names)

        sim_kp, sim_kd = _expand_pd_gains(doc["simulation"]["pd_gains"], cfg_dir, resolved)

        life_doc = doc.get("lifecycle", {})
        lifecycle = LifecycleConfig(
            dt=dt,
            follow_through_s=float(life_doc.get("follow_through_s", 0.6)),
            recovery_s=float(life_doc.get("recovery_s", 0.8)),
            ready_time_to_strike=float(life_doc.get("ready_time_to_strike", 1.0)),
            clock=str(life_doc.get("clock", "countdown")),
            ready_reach_x=float(life_doc.get("ready_reach_x", 0.40)),
            ready_reach_y=float(life_doc.get("ready_reach_y", 0.20)),
            ready_reach_z=float(life_doc.get("ready_reach_z", -0.05)),
        )

        return cls(
            control_hz=control_hz,
            onnx_path=onnx_path,
            model_xml_path=model_xml_path,
            action_adapter=adapter,
            sim_kp=sim_kp,
            sim_kd=sim_kd,
            lifecycle=lifecycle,
            # A robot without passive columns (the G1) never holds a neck, whatever the YAML says.
            passive_neck=bool(doc.get("passive_neck", True)) and bool(resolved.passive_indices),
            contract=contract,
            config_dir=cfg_dir,
            profile=resolved,
        )


def _select_profile(explicit, yaml_name, path: Path) -> RobotProfile:
    requested = resolve_profile(explicit) if explicit is not None else None
    from_yaml = None
    if yaml_name is not None and str(yaml_name).strip():
        from_yaml = resolve_profile(str(yaml_name))  # unknown name -> ValueError
    if requested is not None and from_yaml is not None and requested is not from_yaml:
        raise ValueError(
            f"robot profile mismatch: requested '{requested.name}' but {path} declares "
            f"policy.joint_order: {from_yaml.name}"
        )
    return requested or from_yaml or AGIBOT_A3


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (base / p).resolve()


def _expand_pd_gains(
    spec: dict, cfg_dir: Path, profile: RobotProfile = AGIBOT_A3
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve PD gains into per-joint arrays in the profile's canonical joint order.

    Three forms: a model bundle's ``deploy_config_path`` (joint_sdk_names/stiffness/damping),
    example ``groups`` (one kp/kd per joint group of the profile), or explicit ``per_joint``
    ``{stiffness: {name: kp}, damping: {name: kd}}`` maps.
    """
    names_canonical = profile.joint_names
    n = profile.num_joints
    deploy_relpath = spec.get("deploy_config_path")
    if deploy_relpath:
        deploy_path = _resolve(cfg_dir, deploy_relpath)
        with open(deploy_path, "r", encoding="utf-8") as fh:
            deploy = yaml.safe_load(fh)
        names = tuple(str(name) for name in deploy["joint_sdk_names"])
        if len(names) != n or set(names) != set(names_canonical):
            raise ValueError(f"deploy.yaml joint_sdk_names must name all {n} {profile.name} joints once")

        def reorder(values, field_name: str) -> np.ndarray:
            if len(values) != n:
                raise ValueError(f"deploy.yaml {field_name} must be length {n}")
            by_name = {name: value for name, value in zip(names, values)}
            return np.asarray([by_name[name] for name in names_canonical], dtype=np.float64)

        return reorder(deploy["stiffness"], "stiffness"), reorder(deploy["damping"], "damping")

    per_joint = spec.get("per_joint")
    if per_joint:
        def by_name(field_name: str) -> np.ndarray:
            table = per_joint.get(field_name) or {}
            missing = [name for name in names_canonical if name not in table]
            if missing:
                raise ValueError(f"simulation.pd_gains.per_joint.{field_name} is missing joints: {missing[:3]}...")
            return np.asarray([float(table[name]) for name in names_canonical], dtype=np.float64)

        return by_name("stiffness"), by_name("damping")

    kp = np.zeros(n, dtype=np.float64)
    kd = np.zeros(n, dtype=np.float64)
    groups = spec.get("groups", {})
    for name, indices in profile.group_ranges.items():
        if name not in groups:
            raise ValueError(f"simulation.pd_gains.groups is missing '{name}'")
        g = groups[name]
        for i in indices:
            kp[i] = float(g["kp"])
            kd[i] = float(g["kd"])
    return kp, kd
