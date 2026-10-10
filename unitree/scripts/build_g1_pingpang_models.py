#!/usr/bin/env python3
# Copyright (c) 2026 Intelligent Racing Inc. (dba Hitch Interactive)
# SPDX-License-Identifier: Apache-2.0
"""Generate the racket-equipped Unitree G1 URDF + MJCF from the vendored Unitree description.

The racket mount is defined ONCE here (``G1_RACKET_MOUNT_XYZ`` / ``G1_RACKET_MOUNT_RPY`` and the
handle/collision primitives) and written into both

    unitree/URDF/g1_29dof_pingpang/urdf/g1_29dof_pingpang.urdf   (Isaac Lab training asset source)
    unitree/URDF/g1_29dof_pingpang/mjcf/g1_29dof_pingpang.xml    (MuJoCo sim-to-sim evaluation model)

so the two simulators agree byte-for-byte on where the paddle is.  The generated files are
COMMITTED; ``--check`` regenerates them into a temporary directory and fails if the committed copies
differ (``unitree/tests/test_g1_pingpang_models.py`` runs the same comparison).

The upstream ``g1_29dof_rev_1_0`` URDF/MJCF (``unitree/URDF/g1_description/``, BSD-3-Clause, pinned
commit in ``unitree/README.md``) is read verbatim and never modified.

Racket convention (identical to the Agibot A3 ping-pong asset, so every HOPE default carries over):

* link names ``right_hand_pingpang_Link`` -> ``pingpang_red_Link`` / ``pingpang_black_Link`` on the
  right wrist (``right_wrist_yaw_link``);
* the blade plane is the racket-local XZ plane; racket-local **+Y is the red (forehand) face
  normal** (``RacketTargetCommandCfg.mount_normal_axis = 1``, identity ``mount_quat``);
* the blade centre sits ``G1_RACKET_MOUNT_XYZ`` from the wrist-yaw origin, expressed in the
  ``right_wrist_yaw_link`` frame (the A3 uses 0.210 m along the forearm axis as well).

MODELLING ASSUMPTION: no physical G1 racket adapter has been measured.  The mount below places the
blade 21 cm along the forearm axis (A3-like reach) with the handle running through the centre of the
G1 rubber fist.  Re-measure ``G1_RACKET_MOUNT_XYZ`` / ``G1_RACKET_MOUNT_RPY`` against the real
adapter before any hardware work and regenerate.

Usage (from the repo root)::

    python3 unitree/scripts/build_g1_pingpang_models.py            # (re)write the committed files
    python3 unitree/scripts/build_g1_pingpang_models.py --check    # committed == regenerated ?
"""

from __future__ import annotations

import argparse
import copy
import math
import pathlib
import sys
import tempfile
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------
UNITREE_ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_DIR = UNITREE_ROOT / "URDF" / "g1_description"
UPSTREAM_URDF = UPSTREAM_DIR / "urdf" / "g1_29dof_rev_1_0.urdf"
UPSTREAM_MJCF = UPSTREAM_DIR / "mjcf" / "g1_29dof_rev_1_0.xml"
OUTPUT_DIR = UNITREE_ROOT / "URDF" / "g1_29dof_pingpang"
OUTPUT_URDF_REL = pathlib.Path("urdf") / "g1_29dof_pingpang.urdf"
OUTPUT_MJCF_REL = pathlib.Path("mjcf") / "g1_29dof_pingpang.xml"

MODEL_NAME = "g1_29dof_pingpang"

# Mesh references written into the generated files (relative to the generated file's directory).
URDF_BODY_MESH_PREFIX = "../../g1_description/meshes/"
URDF_RACKET_MESH_PREFIX = "../meshes/"
MJCF_MESHDIR = "../../g1_description/meshes"
MJCF_RACKET_MESH_PREFIX = "../../g1_29dof_pingpang/meshes/"

# --------------------------------------------------------------------------------------------------
# Racket mount -- the single source of truth (metres / radians, right_wrist_yaw_link frame)
# --------------------------------------------------------------------------------------------------
WRIST_LINK = "right_wrist_yaw_link"
HAND_LINK = "right_hand_pingpang_Link"
RED_LINK = "pingpang_red_Link"
BLACK_LINK = "pingpang_black_Link"
RACKET_MESHES = {RED_LINK: "pingpang_red_Link.STL", BLACK_LINK: "pingpang_black_Link.STL"}
RED_RGBA = "1 0 0 1"
BLACK_RGBA = "0.1 0.1 0.1 1"

#: Blade-centre (mesh origin) position in the right_wrist_yaw_link frame.
G1_RACKET_MOUNT_XYZ = (0.210, 0.015, 0.010)
#: Blade orientation; (0, 0, 0) keeps the A3 convention: blade plane = XZ, +Y = red face normal.
G1_RACKET_MOUNT_RPY = (0.0, 0.0, 0.0)

# The blade STLs are centred on their own origin but not perfectly symmetric
# (x, z in [-0.0842, +0.0762]; red face y in [-0.003, 0], black face y in [-0.013, -0.010]).
# The collision cylinder / clearance proxy are centred on the true blade centre.
BLADE_CENTRE_OFFSET = (-0.004, -0.0066, -0.004)
BLADE_RADIUS = 0.081          # m, covers the 160 mm blade
BLADE_HALF_THICKNESS = 0.007  # m, red face through black face

#: Handle capsule (visual + collision) running through the rubber fist to the blade edge.
HANDLE_FROM = (0.060, 0.015, 0.010)
HANDLE_TO = (0.145, 0.015, 0.010)
HANDLE_RADIUS = 0.014

#: Racket (handle + blade + rubbers) lumped mass at the blade centre, thin-disc inertia.
RACKET_MASS = 0.18
RACKET_DISC_RADIUS = 0.08

#: Grounded stand keyframe for the MJCF.  Joint values mirror ``G1_CYLINDER_CFG.init_state`` in
#: ``hope_training/.../robots/g1.py`` (BeyondMimic G1 recipe); the pelvis height puts the lowest
#: foot contact sphere on the floor for that leg pose.
STAND_PELVIS_Z = 0.7585
STAND_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

# Names the MuJoCo evaluator / reference runner look up in the MJCF.
FREE_JOINT_NAME = "pelvis_free_joint"
RACKET_SITE = "right_racket"
RACKET_COLLISION_GEOM = "right_racket_collision"
RACKET_HANDLE_COLLISION_GEOM = "right_racket_handle_collision"
RACKET_CLEARANCE_PROXY_GEOM = "right_racket_clearance_proxy"
PELVIS_SITE = "pelvis_site"
PELVIS_GYRO_SENSOR = "pelvis_imu_gyro"
MJCF_TIMESTEP = 0.001


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _parse_xml(path: pathlib.Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _write_xml(tree: ET.ElementTree, path: pathlib.Path, declaration: bool) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ET.tostring(tree.getroot(), encoding="unicode")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        if declaration:
            fh.write('<?xml version="1.0" encoding="utf-8"?>\n')
        fh.write(data)
        fh.write("\n")


def _rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate(quat_wxyz, vec):
    w, x, y, z = quat_wxyz
    vx, vy, vz = vec
    # q * v * q^-1
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def blade_centre_xyz() -> tuple[float, float, float]:
    """True blade centre in the wrist frame (mesh origin + mesh asymmetry offset)."""
    q = _rpy_to_quat_wxyz(G1_RACKET_MOUNT_RPY)
    d = _rotate(q, BLADE_CENTRE_OFFSET)
    return tuple(m + o for m, o in zip(G1_RACKET_MOUNT_XYZ, d))


def racket_disc_inertia() -> dict[str, float]:
    """Thin disc of mass RACKET_MASS, radius RACKET_DISC_RADIUS, normal along racket-local Y."""
    i_axis = 0.5 * RACKET_MASS * RACKET_DISC_RADIUS**2
    i_plane = 0.25 * RACKET_MASS * RACKET_DISC_RADIUS**2
    return {"ixx": i_plane, "iyy": i_axis, "izz": i_plane, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}


def joint_effort_limits(urdf_root: ET.Element) -> dict[str, float]:
    limits: dict[str, float] = {}
    for joint in urdf_root.findall("joint"):
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("effort") is None:
            raise ValueError(f"upstream URDF joint {joint.get('name')} has no effort limit")
        limits[joint.get("name")] = float(limit.get("effort"))
    return limits


# --------------------------------------------------------------------------------------------------
# URDF
# --------------------------------------------------------------------------------------------------
def _urdf_link(name: str, *, mass: float, inertia: dict[str, float], inertial_xyz, visual=None,
               collision=None) -> ET.Element:
    link = ET.Element("link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(inertial_xyz), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
    ET.SubElement(
        inertial, "inertia",
        ixx=f"{inertia['ixx']:.6g}", ixy=f"{inertia['ixy']:.6g}", ixz=f"{inertia['ixz']:.6g}",
        iyy=f"{inertia['iyy']:.6g}", iyz=f"{inertia['iyz']:.6g}", izz=f"{inertia['izz']:.6g}",
    )
    for tag, spec in (("visual", visual), ("collision", collision)):
        if spec is None:
            continue
        el = ET.SubElement(link, tag)
        ET.SubElement(el, "origin", xyz=_fmt(spec["xyz"]), rpy=_fmt(spec["rpy"]))
        geometry = ET.SubElement(el, "geometry")
        ET.SubElement(geometry, spec["geom_tag"], **spec["geom_attrs"])
        if tag == "visual" and spec.get("rgba"):
            material = ET.SubElement(el, "material", name="")
            ET.SubElement(material, "color", rgba=spec["rgba"])
    return link


def _urdf_fixed_joint(name: str, parent: str, child: str, xyz, rpy) -> ET.Element:
    # Fixed joints carry no <axis>: Isaac Sim 5.x may mis-parse an axis on a fixed joint
    # (see hope_training/.../scripts/prepare_a3_isaac_asset.py).
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    return joint


def build_urdf(upstream: ET.ElementTree) -> ET.ElementTree:
    tree = copy.deepcopy(upstream)
    root = tree.getroot()
    root.set("name", MODEL_NAME)

    # Body meshes live in the verbatim upstream directory.
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("meshes/"):
            mesh.set("filename", URDF_BODY_MESH_PREFIX + filename[len("meshes/"):])

    # Handle capsule: a URDF cylinder along the handle axis (Isaac converts cylinders to capsules
    # when replace_cylinders_with_capsules=True; this is a primitive, not a mesh).
    hx = tuple((a + b) / 2 for a, b in zip(HANDLE_FROM, HANDLE_TO))
    handle_len = math.dist(HANDLE_FROM, HANDLE_TO)
    handle_spec = {
        "xyz": hx,
        "rpy": (0.0, math.pi / 2, 0.0),  # URDF cylinder axis is Z; rotate onto +X (forearm axis)
        "geom_tag": "cylinder",
        "geom_attrs": {"radius": f"{HANDLE_RADIUS:.6g}", "length": f"{handle_len:.6g}"},
        "rgba": "0.35 0.25 0.15 1",
    }
    hand_link = _urdf_link(
        HAND_LINK,
        mass=RACKET_MASS,
        inertia=racket_disc_inertia(),
        inertial_xyz=G1_RACKET_MOUNT_XYZ,
        visual=handle_spec,
        collision=dict(handle_spec, rgba=None),
    )
    hand_joint = _urdf_fixed_joint("right_hand_pingpang_joint", WRIST_LINK, HAND_LINK, (0, 0, 0), (0, 0, 0))

    zero_inertia = {k: 0.0 for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    blade_elements = []
    for link_name, rgba in ((RED_LINK, RED_RGBA), (BLACK_LINK, BLACK_RGBA)):
        spec = {
            "xyz": (0, 0, 0),
            "rpy": (0, 0, 0),
            "geom_tag": "mesh",
            "geom_attrs": {"filename": URDF_RACKET_MESH_PREFIX + RACKET_MESHES[link_name]},
            "rgba": rgba,
        }
        blade_elements.append(
            _urdf_link(link_name, mass=0.0, inertia=zero_inertia, inertial_xyz=(0, 0, 0),
                       visual=spec, collision=dict(spec, rgba=None))
        )
        blade_elements.append(
            _urdf_fixed_joint(link_name.replace("_Link", "_joint"), HAND_LINK, link_name,
                              G1_RACKET_MOUNT_XYZ, G1_RACKET_MOUNT_RPY)
        )

    # Append after the upstream right_rubber_hand link (the last element of the upstream file).
    children = list(root)
    anchor = next(i for i, el in enumerate(children) if el.tag == "link" and el.get("name") == "right_rubber_hand")
    insert_at = anchor + 1
    for el in [hand_link, hand_joint, *blade_elements]:
        root.insert(insert_at, el)
        insert_at += 1
    return tree


# --------------------------------------------------------------------------------------------------
# MJCF
# --------------------------------------------------------------------------------------------------
def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise ValueError(f"body '{name}' not found in the upstream MJCF")


def _mjcf_joint_names(root: ET.Element) -> list[str]:
    return [j.get("name") for j in root.iter("joint") if j.get("type") != "free"]


def build_mjcf(upstream: ET.ElementTree, effort_limits: dict[str, float]) -> ET.ElementTree:
    tree = copy.deepcopy(upstream)
    root = tree.getroot()
    root.set("model", MODEL_NAME)
    children = list(root)

    compiler = root.find("compiler")
    compiler.set("meshdir", MJCF_MESHDIR)
    compiler.set("autolimits", "true")

    # <option> and <default> right after <compiler>.
    option = ET.Element("option", timestep=f"{MJCF_TIMESTEP:g}", gravity="0 0 -9.81")
    default = ET.Element("default")
    collision_default = ET.SubElement(default, "default", attrib={"class": "collision"})
    ET.SubElement(collision_default, "geom", group="3", contype="1", conaffinity="7", condim="3",
                  friction="1.5 0.005 0.0001", solref="0.005 1")
    idx = children.index(compiler) + 1
    root.insert(idx, option)
    root.insert(idx + 1, default)

    # Blade mesh assets (first <asset> block is the robot's).
    asset = root.find("asset")
    for link_name in (RED_LINK, BLACK_LINK):
        ET.SubElement(asset, "mesh", name=link_name, file=MJCF_RACKET_MESH_PREFIX + RACKET_MESHES[link_name])

    # Free joint rename + pelvis site.
    pelvis = _find_body(root, "pelvis")
    for joint in pelvis.findall("joint"):
        if joint.get("type") == "free":
            joint.set("name", FREE_JOINT_NAME)
    pelvis_children = list(pelvis)
    imu_site = next(el for el in pelvis_children if el.tag == "site" and el.get("name") == "imu_in_pelvis")
    pelvis.insert(pelvis_children.index(imu_site) + 1,
                  ET.Element("site", name=PELVIS_SITE, size="0.01", pos="0 0 0", rgba="0 0 0 0"))

    # Racket on the right wrist.
    wrist = _find_body(root, WRIST_LINK)
    mount_quat = _rpy_to_quat_wxyz(G1_RACKET_MOUNT_RPY)
    centre = blade_centre_xyz()
    # MuJoCo cylinder/box axes: the blade normal is racket-local +Y, i.e. rotate local Z onto Y.
    y_axis_quat = _quat_mul(mount_quat, (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    wrist.append(ET.Comment(
        f" HOPE racket mount: blade centre {_fmt(G1_RACKET_MOUNT_XYZ)} rpy {_fmt(G1_RACKET_MOUNT_RPY)} in the "
        f"{WRIST_LINK} frame; +Y of the racket frame is the red (forehand) face normal. "
        "Generated by unitree/scripts/build_g1_pingpang_models.py; do not edit by hand. "
    ))
    for link_name, rgba in ((RED_LINK, RED_RGBA), (BLACK_LINK, BLACK_RGBA)):
        wrist.append(ET.Element(
            "geom", name=f"{link_name}_visual", pos=_fmt(G1_RACKET_MOUNT_XYZ), quat=_fmt(mount_quat),
            type="mesh", contype="0", conaffinity="0", group="1", density="0", rgba=rgba, mesh=link_name,
        ))
    wrist.append(ET.Element(
        "geom", name="right_racket_handle_visual", type="capsule",
        fromto=_fmt([*HANDLE_FROM, *HANDLE_TO]), size=f"{HANDLE_RADIUS:.6g}",
        contype="0", conaffinity="0", group="1", density="0", rgba="0.35 0.25 0.15 1",
    ))
    wrist.append(ET.Element(
        "geom", name=RACKET_COLLISION_GEOM, attrib={"class": "collision"}, type="cylinder",
        pos=_fmt(centre), quat=_fmt(y_axis_quat), size=f"{BLADE_RADIUS:.6g} {BLADE_HALF_THICKNESS:.6g}",
        mass=f"{RACKET_MASS:.6g}",
    ))
    wrist.append(ET.Element(
        "geom", name=RACKET_HANDLE_COLLISION_GEOM, attrib={"class": "collision"}, type="capsule",
        fromto=_fmt([*HANDLE_FROM, *HANDLE_TO]), size=f"{HANDLE_RADIUS:.6g}", density="0",
    ))
    wrist.append(ET.Comment(" Conservative box around the blade collision cylinder. Diagnostic only. "))
    wrist.append(ET.Element(
        "geom", name=RACKET_CLEARANCE_PROXY_GEOM, type="box", pos=_fmt(centre), quat=_fmt(mount_quat),
        size=f"{BLADE_RADIUS + 0.001:.6g} {BLADE_HALF_THICKNESS + 0.001:.6g} {BLADE_RADIUS + 0.001:.6g}",
        contype="0", conaffinity="0", group="4", density="0", rgba="0 0 0 0",
    ))
    wrist.append(ET.Element(
        "site", name=RACKET_SITE, size="0.001", pos=_fmt(G1_RACKET_MOUNT_XYZ), quat=_fmt(mount_quat),
        rgba="0 0 0 0",
    ))

    # Torque motors with ctrlrange = +/- URDF effort (the PD law lives in the harness and is clipped).
    actuator = root.find("actuator")
    for motor in actuator.findall("motor"):
        joint = motor.get("joint")
        effort = effort_limits[joint]
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", _fmt((-effort, effort)))

    # Sensors the evaluator / reference runner read.
    sensor = root.find("sensor")
    ET.SubElement(sensor, "gyro", name=PELVIS_GYRO_SENSOR, site="imu_in_pelvis")
    ET.SubElement(sensor, "framepos", name="pelvis_framepos", objtype="site", objname=PELVIS_SITE)
    ET.SubElement(sensor, "framepos", name="right_racket_framepos", objtype="site", objname=RACKET_SITE)
    ET.SubElement(sensor, "framequat", name="right_racket_framequat", objtype="site", objname=RACKET_SITE)

    # Grounded stand keyframe (qpos = free joint 7 + 29 hinges in tree order).
    joint_names = _mjcf_joint_names(root)
    if len(joint_names) != 29:
        raise ValueError(f"expected 29 hinge joints in the upstream MJCF, found {len(joint_names)}")
    unknown = set(STAND_JOINT_POS) - set(joint_names)
    if unknown:
        raise ValueError(f"stand pose names unknown joints: {sorted(unknown)}")
    qpos = [0.0, 0.0, STAND_PELVIS_Z, 1.0, 0.0, 0.0, 0.0] + [STAND_JOINT_POS.get(n, 0.0) for n in joint_names]
    keyframe = ET.Element("keyframe")
    ET.SubElement(keyframe, "key", name="stand", qpos=_fmt(qpos))
    root.insert(list(root).index(sensor) + 1, keyframe)
    return tree


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------
def generate(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    upstream_urdf = _parse_xml(UPSTREAM_URDF)
    upstream_mjcf = _parse_xml(UPSTREAM_MJCF)
    efforts = joint_effort_limits(upstream_urdf.getroot())
    urdf_path = output_dir / OUTPUT_URDF_REL
    mjcf_path = output_dir / OUTPUT_MJCF_REL
    _write_xml(build_urdf(upstream_urdf), urdf_path, declaration=True)
    _write_xml(build_mjcf(upstream_mjcf, efforts), mjcf_path, declaration=False)
    return urdf_path, mjcf_path


def check(output_dir: pathlib.Path = OUTPUT_DIR) -> list[str]:
    """Regenerate into a temp dir and return a list of committed files that differ (empty = OK)."""
    stale: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_urdf, tmp_mjcf = generate(pathlib.Path(tmp))
        for rel, fresh in ((OUTPUT_URDF_REL, tmp_urdf), (OUTPUT_MJCF_REL, tmp_mjcf)):
            committed = output_dir / rel
            if not committed.is_file() or committed.read_bytes() != fresh.read_bytes():
                stale.append(str(committed))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUTPUT_DIR,
                        help="where urdf/ and mjcf/ are written (default: the committed location)")
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail if the committed files differ from a fresh generation")
    args = parser.parse_args(argv)
    if args.check:
        stale = check(args.output_dir)
        if stale:
            for path in stale:
                print(f"[FAIL] stale generated file: {path}", file=sys.stderr)
            print("Regenerate with: python3 unitree/scripts/build_g1_pingpang_models.py", file=sys.stderr)
            return 1
        print("[build_g1_pingpang_models] OK: committed URDF/MJCF match a fresh generation")
        return 0
    urdf_path, mjcf_path = generate(args.output_dir)
    print(f"[build_g1_pingpang_models] wrote {urdf_path}")
    print(f"[build_g1_pingpang_models] wrote {mjcf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
