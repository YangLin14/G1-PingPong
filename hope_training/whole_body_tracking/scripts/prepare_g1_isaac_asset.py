#!/usr/bin/env python3
"""Prepare (or verify) the racket-equipped Unitree G1 URDF as an Isaac Lab asset.

Sibling of ``prepare_a3_isaac_asset.py``. The source is the generated racket-equipped variant
``unitree/URDF/g1_29dof_pingpang/`` (see ``unitree/README.md``), whose URDF references the
verbatim upstream Unitree meshes in ``unitree/URDF/g1_description/meshes/`` and the two blade
meshes next to it. This script copies every referenced mesh into one flat ``meshes/`` directory,
rewrites the references to ``../meshes/<file>`` and writes ``urdf/model.urdf`` under the training
package's git-ignored ``assets/unitree_g1/`` directory so Isaac Lab can load the model from a
fresh clone without ROS package resolution.

Usage (from the repo root, inside the Isaac environment or any Python 3):
    python3 hope_training/whole_body_tracking/scripts/prepare_g1_isaac_asset.py --force
    python3 hope_training/whole_body_tracking/scripts/prepare_g1_isaac_asset.py --check
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import xml.etree.ElementTree as ET


def _repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "hope_training").is_dir() and (parent / "unitree").is_dir():
            return parent
    return here.parents[3]


REPO_ROOT = _repo_root()
DEFAULT_SOURCE_ROOT = REPO_ROOT / "unitree" / "URDF" / "g1_29dof_pingpang"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "unitree_g1"
)
RACKET_LINKS = ("right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link")
EXPECTED_MOVABLE_JOINTS = 29


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=pathlib.Path, default=DEFAULT_SOURCE_ROOT,
                        help="Racket-equipped G1 package root (contains urdf/ and meshes/).")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Where to write the prepared Isaac asset.")
    parser.add_argument("--force", action="store_true", help="Recreate the output root before copying.")
    parser.add_argument("--check", action="store_true", help="Only verify the already-prepared output asset.")
    return parser.parse_args()


def _find_source_urdf(source_root: pathlib.Path) -> pathlib.Path:
    urdf_dir = source_root / "urdf"
    candidates = sorted(urdf_dir.glob("*.urdf")) if urdf_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"no .urdf found under {urdf_dir}. Regenerate it with "
            "python3 unitree/scripts/build_g1_pingpang_models.py"
        )
    return candidates[0]


def prepare(source_root: pathlib.Path, output_root: pathlib.Path, force: bool) -> None:
    source_urdf = _find_source_urdf(source_root)
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    urdf_dir = output_root / "urdf"
    mesh_dir = output_root / "meshes"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    copied: dict[str, pathlib.Path] = {}
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            raise ValueError(f"unexpected package:// mesh reference in the G1 source URDF: {filename}")
        source_mesh = (source_urdf.parent / filename).resolve()
        if not source_mesh.is_file():
            raise FileNotFoundError(f"mesh referenced by {source_urdf.name} not found: {source_mesh}")
        basename = source_mesh.name
        previous = copied.get(basename)
        if previous is not None and previous != source_mesh:
            raise ValueError(f"two different meshes share the basename {basename}: {previous} vs {source_mesh}")
        if previous is None:
            shutil.copy2(source_mesh, mesh_dir / basename)
            copied[basename] = source_mesh
        mesh.set("filename", f"../meshes/{basename}")

    # Isaac Sim 5.x may mis-parse an <axis> on a fixed joint; the generated G1 URDF has none, but the
    # cleanup stays so a hand-edited source cannot reintroduce the problem (same as the A3 script).
    for joint in root.findall(".//joint"):
        axis = joint.find("axis")
        if joint.get("type") == "fixed" and axis is not None:
            joint.remove(axis)
    tree.write(urdf_dir / "model.urdf", encoding="utf-8", xml_declaration=True)
    print(f"[prepare_g1_isaac_asset] prepared: {output_root} ({len(copied)} meshes)")


def check(output_root: pathlib.Path) -> bool:
    model_urdf = output_root / "urdf" / "model.urdf"
    if not model_urdf.exists():
        print(f"[FAIL] missing prepared URDF: {model_urdf}")
        return False

    root = ET.parse(model_urdf).getroot()
    refs = [m.get("filename") for m in root.findall(".//mesh") if m.get("filename")]
    failures: list[str] = []

    stale = [r for r in refs if r.startswith("package://")]
    if stale:
        failures.append("stale package:// mesh refs remain: " + ", ".join(sorted(set(stale))[:8]))
    not_flat = [r for r in refs if not r.startswith("../meshes/")]
    if not_flat:
        failures.append("mesh refs not rewritten to ../meshes/: " + ", ".join(sorted(set(not_flat))[:8]))
    for ref in refs:
        if ref.startswith("../meshes/") and not (model_urdf.parent / ref).resolve().exists():
            failures.append(f"missing mesh referenced by URDF: {ref}")

    links = {link.get("name") for link in root.findall("link")}
    for name in RACKET_LINKS:
        if name not in links:
            failures.append(f"racket link missing from the prepared articulation: {name}")
    if "pingbang_ball_Link" in links:
        failures.append("racket calibration ball must not be part of the Isaac articulation")

    fixed_axes = [
        joint.get("name", "<unnamed>")
        for joint in root.findall(".//joint")
        if joint.get("type") == "fixed" and joint.find("axis") is not None
    ]
    if fixed_axes:
        failures.append("fixed joints still carry unnecessary axis tags: " + ", ".join(fixed_axes[:8]))

    movable = 0
    for joint in root.findall(".//joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        movable += 1
        axis = joint.find("axis")
        raw = None if axis is None else axis.get("xyz")
        try:
            values = [] if raw is None else [float(value) for value in raw.split()]
        except ValueError:
            values = []
        if len(values) != 3 or sum(value * value for value in values) <= 0.0:
            failures.append(f"movable joint has an invalid axis: {joint.get('name', '<unnamed>')} xyz={raw!r}")
    if movable != EXPECTED_MOVABLE_JOINTS:
        failures.append(f"expected {EXPECTED_MOVABLE_JOINTS} movable joints, found {movable}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return False
    print(
        f"[prepare_g1_isaac_asset] OK: {len(refs)} mesh refs, all repo-relative and resolving; "
        f"{movable} movable joints; racket links present."
    )
    return True


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not args.check:
        prepare(source_root, output_root, args.force)
    return 0 if check(output_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
