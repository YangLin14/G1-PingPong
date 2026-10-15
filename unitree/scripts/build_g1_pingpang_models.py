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
from dataclasses import dataclass

# --------------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------------
UNITREE_ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_DIR = UNITREE_ROOT / "URDF" / "g1_description"
UPSTREAM_URDF = UPSTREAM_DIR / "urdf" / "g1_29dof_rev_1_0.urdf"
UPSTREAM_MJCF = UPSTREAM_DIR / "mjcf" / "g1_29dof_rev_1_0.xml"

MODEL_NAME = "g1_29dof_pingpang"

# Mesh references written into the generated files (relative to the generated file's directory).
URDF_BODY_MESH_PREFIX = "../../g1_description/meshes/"
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

# The blade STLs are centred on their own origin but not perfectly symmetric
# (x, z in [-0.0842, +0.0762]; red face y in [-0.003, 0], black face y in [-0.013, -0.010]).
# The collision cylinder / clearance proxy are centred on the true blade centre.
BLADE_CENTRE_OFFSET = (-0.004, -0.0066, -0.004)
BLADE_RADIUS = 0.081          # m, covers the 160 mm blade
BLADE_HALF_THICKNESS = 0.007  # m, red face through black face
HANDLE_RADIUS = 0.014

RUBBER_HAND_LINK = "right_rubber_hand"
RUBBER_HAND_XYZ = (0.0415, -0.003, 0.0)   # right_hand_palm_joint origin in the wrist frame (upstream)
FIST_MESH = "right_fist.STL"


@dataclass(frozen=True)
class Mount:
    """One racket mount: where the blade sits on the wrist and what the right hand looks like.

    ``xyz`` / ``rpy``: blade mesh origin and orientation in the right_wrist_yaw_link frame (blade plane
    = racket-local XZ, racket-local +Y = red / forehand face normal, racket-local +X = from the handle
    towards the blade tip). ``fist``: draw the right hand closed round the handle (generated mesh)
    instead of the upstream open rubber hand.
    """

    name: str
    xyz: tuple
    rpy: tuple
    handle_from: tuple
    handle_to: tuple
    fist: bool = False
    finger_s: tuple = ()        # finger positions along the handle (from the grip centre), pinky -> index
    thumb: tuple = ()           # thumb capsule chain, wrist frame
    blade_mesh_prefix_urdf: str = "../meshes/"

    @property
    def out_dir(self) -> pathlib.Path:
        return UNITREE_ROOT / "URDF" / self.name

    @property
    def handle_axis(self) -> tuple:
        d = [b - a for a, b in zip(self.handle_from, self.handle_to)]
        n = math.sqrt(sum(v * v for v in d))
        return tuple(v / n for v in d)


#: A3-style mount (the original G1 port): blade 21 cm out ALONG the forearm, handle lying on the open
#: rubber hand. Every run up to 2026-10-14 (g1_fh, g1_fh_diag, g1_fh_rally) is trained on this one.
#: A real hand cannot hold a racket like this: a closed fist puts the handle across the palm (D20).
FOREARM = Mount(
    name="g1_29dof_pingpang",
    xyz=(0.210, 0.015, 0.010),
    rpy=(0.0, 0.0, 0.0),
    handle_from=(0.060, 0.015, 0.010),
    handle_to=(0.145, 0.015, 0.010),
)

#: Fist grip (Triton Droids D20, 2026-10-14): the right hand is closed into a fist and the handle runs
#: through it ACROSS the palm, from the little-finger side to the thumb side, so the racket stands
#: perpendicular to the forearm (like a shakehand grip seen in photos). The forehand (red) face looks
#: the same way as the palm (+Y), as in a shakehand grip. FIST_HANDLE_ANGLE_DEG is the angle between
#: the forearm (+X) and the handle, towards the thumb (+Z); 90 = perpendicular.
FIST_HANDLE_ANGLE_DEG = 90.0
FIST_GRIP_CENTRE = (0.083, 0.016, 0.0)    # handle axis point in the middle of the four fingers
FIST_BLADE_ALONG = 0.140                  # grip centre -> blade mesh origin along the handle (m)


def _fist_mount() -> Mount:
    a = math.radians(FIST_HANDLE_ANGLE_DEG)
    axis = (math.cos(a), 0.0, math.sin(a))
    c = FIST_GRIP_CENTRE
    at = lambda s: tuple(round(c[k] + s * axis[k], 6) for k in range(3))  # noqa: E731
    # blade origin: on the handle axis, shifted along +Y by the mesh's half thickness so the blade is
    # centred on the handle
    origin = at(FIST_BLADE_ALONG)
    origin = (origin[0], round(origin[1] - BLADE_CENTRE_OFFSET[1], 6), origin[2])
    return Mount(
        name="g1_29dof_pingpang_fist",
        xyz=origin,
        rpy=(0.0, -a, 0.0),                # racket-local +X (handle -> tip) onto the handle axis
        handle_from=at(-0.050),            # butt sticks out past the little finger
        handle_to=at(0.075),               # into the blade throat
        fist=True,
        finger_s=(-0.027, -0.009, 0.009, 0.027),
        thumb=((0.052, -0.004, 0.034), (0.070, 0.028, 0.044), (0.088, 0.034, 0.064)),
        blade_mesh_prefix_urdf="../../g1_29dof_pingpang/meshes/",
    )


FIST = _fist_mount()
MOUNTS = (FOREARM, FIST)

# Back-compat names (the FOREARM mount, which every existing checkpoint uses).
G1_RACKET_MOUNT_XYZ = FOREARM.xyz
G1_RACKET_MOUNT_RPY = FOREARM.rpy
HANDLE_FROM = FOREARM.handle_from
HANDLE_TO = FOREARM.handle_to

#: Fist mesh (palm, fingers curled round the handle, thumb), FIST mount only.
FIST_PALM_X = (0.050, 0.100)
FIST_PALM_YZ = ((-0.010, -0.024), (-0.010, -0.008), (-0.010, 0.008), (-0.010, 0.024))
FIST_PALM_RADIUS = 0.011
FIST_FINGER_RADIUS = (0.0075, 0.0085, 0.0088, 0.0085)
#: Finger path round the handle axis in the plane across it: (u, w) with u towards the knuckles and
#: w = +Y (palm normal). Starts at the knuckle, wraps over the handle, tip back against the palm.
FIST_FINGER_ARC_DEG = (-20, 20, 60, 100, 140, 180, 208)
FIST_FINGER_BASE_UW = (0.040, -0.011)
FIST_THUMB_RADIUS = 0.0095

#: Racket (handle + blade + rubbers) lumped mass at the blade centre, thin-disc inertia.
RACKET_MASS = 0.18
RACKET_DISC_RADIUS = 0.08

# Shared actuation tables (armature / stiffness / damping / effort / init pose) — pure Python
# mirror of robots/g1.py ``G1_CYLINDER_CFG`` so the MJCF carries the SAME per-joint armature as the
# Isaac implicit drive (sim-to-sim parity: a joint without its rotor inertia responds differently to
# the same PD torque).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import g1_actuation  # noqa: E402

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


def blade_centre_xyz(mount: Mount = FOREARM) -> tuple[float, float, float]:
    """True blade centre in the wrist frame (mesh origin + mesh asymmetry offset)."""
    q = _rpy_to_quat_wxyz(mount.rpy)
    d = _rotate(q, BLADE_CENTRE_OFFSET)
    return tuple(m + o for m, o in zip(mount.xyz, d))


def racket_disc_inertia() -> dict[str, float]:
    """Thin disc of mass RACKET_MASS, radius RACKET_DISC_RADIUS, normal along racket-local Y."""
    i_axis = 0.5 * RACKET_MASS * RACKET_DISC_RADIUS**2
    i_plane = 0.25 * RACKET_MASS * RACKET_DISC_RADIUS**2
    return {"ixx": i_plane, "iyy": i_axis, "izz": i_plane, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}


def _sphere_tris(c, r, n_lat=8, n_lon=14):
    pts = [[(c[0] + r * math.sin(math.pi * i / n_lat) * math.cos(2 * math.pi * j / n_lon),
             c[1] + r * math.sin(math.pi * i / n_lat) * math.sin(2 * math.pi * j / n_lon),
             c[2] + r * math.cos(math.pi * i / n_lat)) for j in range(n_lon)] for i in range(n_lat + 1)]
    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            a, b = pts[i][j], pts[i][(j + 1) % n_lon]
            c2, d = pts[i + 1][j], pts[i + 1][(j + 1) % n_lon]
            if i > 0:
                tris.append((a, c2, b))
            if i < n_lat - 1:
                tris.append((b, c2, d))
    return tris


def _cylinder_tris(p, q, r, n=14):
    axis = [q[k] - p[k] for k in range(3)]
    length = math.sqrt(sum(a * a for a in axis))
    ax = [a / length for a in axis]
    ref = (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = [ax[1] * ref[2] - ax[2] * ref[1], ax[2] * ref[0] - ax[0] * ref[2], ax[0] * ref[1] - ax[1] * ref[0]]
    n1 = math.sqrt(sum(a * a for a in e1))
    e1 = [a / n1 for a in e1]
    e2 = [ax[1] * e1[2] - ax[2] * e1[1], ax[2] * e1[0] - ax[0] * e1[2], ax[0] * e1[1] - ax[1] * e1[0]]
    ring = [[r * (math.cos(2 * math.pi * j / n) * e1[k] + math.sin(2 * math.pi * j / n) * e2[k]) for k in range(3)]
            for j in range(n)]
    tris = []
    for j in range(n):
        a = tuple(p[k] + ring[j][k] for k in range(3))
        b = tuple(p[k] + ring[(j + 1) % n][k] for k in range(3))
        c = tuple(q[k] + ring[j][k] for k in range(3))
        d = tuple(q[k] + ring[(j + 1) % n][k] for k in range(3))
        tris += [(a, b, c), (b, d, c)]
    return tris


def _tube_tris(points, r):
    """Capsule chain through ``points``: spheres at the nodes, cylinders between them."""
    tris = []
    for i, p in enumerate(points):
        tris += _sphere_tris(p, r)
        if i:
            tris += _cylinder_tris(points[i - 1], p, r)
    return tris


def fist_triangles(mount: Mount = FIST) -> list[tuple]:
    """The right fist in the right_rubber_hand frame: palm, fingers curled round the handle, thumb."""
    axis = mount.handle_axis
    # grip centre = point of the handle axis nearest the palm centre
    hf = mount.handle_from
    rel = [FIST_GRIP_CENTRE[k] - hf[k] for k in range(3)]
    t = sum(rel[k] * axis[k] for k in range(3))
    c = tuple(hf[k] + t * axis[k] for k in range(3))
    e_w = (0.0, 1.0, 0.0)                                   # palm normal
    e_u = (axis[2], 0.0, -axis[0])                          # across the handle, towards the knuckles
    if e_u[0] < 0:
        e_u = tuple(-v for v in e_u)

    def pt(s_along, u, w):
        return tuple(c[k] + s_along * axis[k] + u * e_u[k] + w * e_w[k] for k in range(3))

    tris = []
    for y, z in FIST_PALM_YZ:
        tris += _tube_tris([(FIST_PALM_X[0], y, z), (FIST_PALM_X[1], y, z)], FIST_PALM_RADIUS)
    for s_along, r in zip(mount.finger_s, FIST_FINGER_RADIUS):
        arc_r = HANDLE_RADIUS + r + 0.001
        pts = [pt(s_along, *FIST_FINGER_BASE_UW)]
        pts += [pt(s_along, arc_r * math.cos(math.radians(a)), arc_r * math.sin(math.radians(a)))
                for a in FIST_FINGER_ARC_DEG]
        tris += _tube_tris(pts, r)
    tris += _tube_tris(list(mount.thumb), FIST_THUMB_RADIUS)
    off = RUBBER_HAND_XYZ
    return [tuple(tuple(round(p[k] - off[k], 6) for k in range(3)) for p in t) for t in tris]


def fist_stl_bytes(mount: Mount = FIST) -> bytes:
    import struct

    tris = fist_triangles(mount)
    out = [f"{mount.name} right fist (generated)".encode().ljust(80, b" "), struct.pack("<I", len(tris))]
    for a, b, c in tris:
        u = [b[k] - a[k] for k in range(3)]
        v = [c[k] - a[k] for k in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = math.sqrt(sum(x * x for x in n)) or 1.0
        out.append(struct.pack("<12fH", *(x / length for x in n), *a, *b, *c, 0))
    return b"".join(out)


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


def build_urdf(upstream: ET.ElementTree, mount: Mount = FOREARM) -> ET.ElementTree:
    tree = copy.deepcopy(upstream)
    root = tree.getroot()
    root.set("name", mount.name)

    # Body meshes live in the verbatim upstream directory.
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("meshes/"):
            mesh.set("filename", URDF_BODY_MESH_PREFIX + filename[len("meshes/"):])

    # Right hand: with the fist grip the generated fist replaces the open rubber hand's visual mesh.
    if mount.fist:
        rubber = next(el for el in root.iter("link") if el.get("name") == RUBBER_HAND_LINK)
        rubber.find("visual/geometry/mesh").set("filename", "../meshes/" + FIST_MESH)

    # Handle capsule: a URDF cylinder along the handle axis (Isaac converts cylinders to capsules
    # when replace_cylinders_with_capsules=True; this is a primitive, not a mesh).
    hx = tuple((a + b) / 2 for a, b in zip(mount.handle_from, mount.handle_to))
    handle_len = math.dist(mount.handle_from, mount.handle_to)
    ax = mount.handle_axis  # in the XZ plane
    handle_spec = {
        "xyz": hx,
        # URDF cylinder axis is Z; pitch it onto the handle axis (FOREARM: +X, pi/2)
        "rpy": (0.0, math.atan2(ax[0], ax[2]), 0.0),
        "geom_tag": "cylinder",
        "geom_attrs": {"radius": f"{HANDLE_RADIUS:.6g}", "length": f"{handle_len:.6g}"},
        "rgba": "0.35 0.25 0.15 1",
    }
    hand_link = _urdf_link(
        HAND_LINK,
        mass=RACKET_MASS,
        inertia=racket_disc_inertia(),
        inertial_xyz=mount.xyz,
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
            "geom_attrs": {"filename": mount.blade_mesh_prefix_urdf + RACKET_MESHES[link_name]},
            "rgba": rgba,
        }
        blade_elements.append(
            _urdf_link(link_name, mass=0.0, inertia=zero_inertia, inertial_xyz=(0, 0, 0),
                       visual=spec, collision=dict(spec, rgba=None))
        )
        blade_elements.append(
            _urdf_fixed_joint(link_name.replace("_Link", "_joint"), HAND_LINK, link_name,
                              mount.xyz, mount.rpy)
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


def build_mjcf(upstream: ET.ElementTree, effort_limits: dict[str, float], mount: Mount = FOREARM) -> ET.ElementTree:
    tree = copy.deepcopy(upstream)
    root = tree.getroot()
    root.set("model", mount.name)
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
    if mount.fist:
        ET.SubElement(asset, "mesh", name="right_fist", file=f"../../{mount.name}/meshes/{FIST_MESH}")
        hand_geoms = [g for g in root.iter("geom") if g.get("mesh") == RUBBER_HAND_LINK]
        if len(hand_geoms) != 1:
            raise ValueError(f"expected one {RUBBER_HAND_LINK} geom in the upstream MJCF, found {len(hand_geoms)}")
        hand_geoms[0].set("mesh", "right_fist")

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
    mount_quat = _rpy_to_quat_wxyz(mount.rpy)
    centre = blade_centre_xyz(mount)
    # MuJoCo cylinder/box axes: the blade normal is racket-local +Y, i.e. rotate local Z onto Y.
    y_axis_quat = _quat_mul(mount_quat, (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))
    wrist.append(ET.Comment(
        f" HOPE racket mount ({mount.name}): blade origin {_fmt(mount.xyz)} rpy {_fmt(mount.rpy)} in the "
        f"{WRIST_LINK} frame; +Y of the racket frame is the red (forehand) face normal. "
        "Generated by unitree/scripts/build_g1_pingpang_models.py; do not edit by hand. "
    ))
    for link_name, rgba in ((RED_LINK, RED_RGBA), (BLACK_LINK, BLACK_RGBA)):
        wrist.append(ET.Element(
            "geom", name=f"{link_name}_visual", pos=_fmt(mount.xyz), quat=_fmt(mount_quat),
            type="mesh", contype="0", conaffinity="0", group="1", density="0", rgba=rgba, mesh=link_name,
        ))
    wrist.append(ET.Element(
        "geom", name="right_racket_handle_visual", type="capsule",
        fromto=_fmt([*mount.handle_from, *mount.handle_to]), size=f"{HANDLE_RADIUS:.6g}",
        contype="0", conaffinity="0", group="1", density="0", rgba="0.35 0.25 0.15 1",
    ))
    wrist.append(ET.Element(
        "geom", name=RACKET_COLLISION_GEOM, attrib={"class": "collision"}, type="cylinder",
        pos=_fmt(centre), quat=_fmt(y_axis_quat), size=f"{BLADE_RADIUS:.6g} {BLADE_HALF_THICKNESS:.6g}",
        mass=f"{RACKET_MASS:.6g}",
    ))
    wrist.append(ET.Element(
        "geom", name=RACKET_HANDLE_COLLISION_GEOM, attrib={"class": "collision"}, type="capsule",
        fromto=_fmt([*mount.handle_from, *mount.handle_to]), size=f"{HANDLE_RADIUS:.6g}", density="0",
    ))
    wrist.append(ET.Comment(" Conservative box around the blade collision cylinder. Diagnostic only. "))
    wrist.append(ET.Element(
        "geom", name=RACKET_CLEARANCE_PROXY_GEOM, type="box", pos=_fmt(centre), quat=_fmt(mount_quat),
        size=f"{BLADE_RADIUS + 0.001:.6g} {BLADE_HALF_THICKNESS + 0.001:.6g} {BLADE_RADIUS + 0.001:.6g}",
        contype="0", conaffinity="0", group="4", density="0", rgba="0 0 0 0",
    ))
    wrist.append(ET.Element(
        "site", name=RACKET_SITE, size="0.001", pos=_fmt(mount.xyz), quat=_fmt(mount_quat),
        rgba="0 0 0 0",
    ))

    # Torque motors with ctrlrange = +/- the TRAINING recipe's effort limit (robots/g1.py
    # effort_limit_sim, mirrored in g1_actuation): the PD law lives in the harness and is clipped the
    # way the Isaac implicit drive clips it. NOTE the recipe uses 50 N.m for the ankle and waist
    # roll/pitch motors where the upstream URDF says 35 N.m (see robots/g1.py); the MuJoCo plant must
    # match the plant the policy was trained on, not the vendor sheet.
    actuator = root.find("actuator")
    for motor in actuator.findall("motor"):
        joint = motor.get("joint")
        effort = g1_actuation.effort_limit(joint)
        motor.set("ctrllimited", "true")
        motor.set("ctrlrange", _fmt((-effort, effort)))

    # Per-joint rotor armature exactly as the Isaac articulation (robots/g1.py); no implicit joint
    # damping or dry friction, since the harness computes the full PD torque itself and Isaac's
    # URDF import adds none either.
    for joint in root.iter("joint"):
        if joint.get("type") == "free" or joint.get("name") is None:
            continue
        joint.set("armature", f"{g1_actuation.armature(joint.get('name')):.6g}")
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        effort = g1_actuation.effort_limit(joint.get("name"))
        joint.set("actuatorfrcrange", _fmt((-effort, effort)))

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
def generate(output_dir: pathlib.Path, mount: Mount = FOREARM) -> tuple[pathlib.Path, pathlib.Path]:
    """Write one mount's URDF + MJCF (and its fist mesh) under ``output_dir``."""
    upstream_urdf = _parse_xml(UPSTREAM_URDF)
    upstream_mjcf = _parse_xml(UPSTREAM_MJCF)
    efforts = joint_effort_limits(upstream_urdf.getroot())
    urdf_path = output_dir / "urdf" / f"{mount.name}.urdf"
    mjcf_path = output_dir / "mjcf" / f"{mount.name}.xml"
    _write_xml(build_urdf(upstream_urdf, mount), urdf_path, declaration=True)
    _write_xml(build_mjcf(upstream_mjcf, efforts, mount), mjcf_path, declaration=False)
    if mount.fist:
        fist_path = output_dir / "meshes" / FIST_MESH
        fist_path.parent.mkdir(parents=True, exist_ok=True)
        fist_path.write_bytes(fist_stl_bytes(mount))
    return urdf_path, mjcf_path


def _generated_files(mount: Mount) -> list[pathlib.Path]:
    rels = [pathlib.Path("urdf") / f"{mount.name}.urdf", pathlib.Path("mjcf") / f"{mount.name}.xml"]
    return rels + ([pathlib.Path("meshes") / FIST_MESH] if mount.fist else [])


def check(output_dir: pathlib.Path | None = None) -> list[str]:
    """Regenerate every mount into a temp dir; return the committed files that differ (empty = OK).

    ``output_dir`` checks only the FOREARM mount there (legacy single-model call)."""
    stale: list[str] = []
    targets = [(FOREARM, output_dir)] if output_dir is not None else [(m, m.out_dir) for m in MOUNTS]
    for mount, out in targets:
        with tempfile.TemporaryDirectory() as tmp:
            generate(pathlib.Path(tmp), mount)
            for rel in _generated_files(mount):
                committed, fresh = out / rel, pathlib.Path(tmp) / rel
                if not committed.is_file() or committed.read_bytes() != fresh.read_bytes():
                    stale.append(str(committed))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None,
                        help="write only the FOREARM mount here (default: every mount to its committed location)")
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
    targets = [(FOREARM, args.output_dir)] if args.output_dir is not None else [(m, m.out_dir) for m in MOUNTS]
    for mount, out in targets:
        urdf_path, mjcf_path = generate(out, mount)
        print(f"[build_g1_pingpang_models] wrote {urdf_path}")
        print(f"[build_g1_pingpang_models] wrote {mjcf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
