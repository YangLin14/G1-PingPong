"""Fresh-clone Unitree G1 Isaac asset preparation (mirror of test_prepare_a3_isaac_asset.py)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parents[1]
_SCRIPT = _ROOT / "scripts" / "prepare_g1_isaac_asset.py"
_SOURCE = _REPO / "unitree" / "URDF" / "g1_29dof_pingpang"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("hope_prepare_g1_asset", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_asset = _load_prepare_module()


def test_default_source_is_the_generated_racket_variant() -> None:
    assert prepare_asset.DEFAULT_SOURCE_ROOT == _SOURCE
    assert (_SOURCE / "urdf" / "g1_29dof_pingpang.urdf").is_file()
    assert prepare_asset.DEFAULT_OUTPUT_ROOT.name == "unitree_g1"
    assert prepare_asset.DEFAULT_OUTPUT_ROOT.parent.name == "assets"


def test_prepare_flattens_meshes_and_passes_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "unitree_g1"
        prepare_asset.prepare(_SOURCE, output, force=True)

        model = output / "urdf" / "model.urdf"
        root = ET.parse(model).getroot()
        refs = [m.get("filename") for m in root.findall(".//mesh")]
        assert refs and all(r.startswith("../meshes/") for r in refs)
        assert all((model.parent / r).resolve().is_file() for r in refs)
        # 35 upstream body meshes + 2 blade meshes, flattened into one directory.
        assert len({r for r in refs}) == 37
        assert len(list((output / "meshes").glob("*.STL"))) == 37
        links = {link.get("name") for link in root.findall("link")}
        assert {"right_hand_pingpang_Link", "pingpang_red_Link", "pingpang_black_Link"} <= links
        assert "pingbang_ball_Link" not in links
        assert all(j.find("axis") is None for j in root.findall(".//joint") if j.get("type") == "fixed")
        assert prepare_asset.check(output)


def test_check_rejects_missing_racket_link_and_bad_axis() -> None:
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        source = tmp / "source"
        (source / "urdf").mkdir(parents=True)
        (source / "meshes").mkdir(parents=True)
        (source / "meshes" / "body.STL").write_bytes(b"solid body\nendsolid body\n")
        (source / "urdf" / "model.urdf").write_text(
            """<?xml version="1.0"?>
<robot name="fixture">
  <link name="base"><visual><geometry><mesh filename="../meshes/body.STL"/></geometry></visual></link>
  <link name="child"/>
  <joint name="moving_joint" type="revolute">
    <parent link="base"/><child link="child"/><axis xyz="0.0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""",
            encoding="utf-8",
        )
        output = tmp / "output"
        prepare_asset.prepare(source, output, force=True)
        assert not prepare_asset.check(output)
