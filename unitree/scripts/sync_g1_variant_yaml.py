#!/usr/bin/env python3
"""Regenerate cfg/task/HOPEPingPongUnitreeG1_IK.yaml (variant B) from HOPEPingPongUnitreeG1.yaml.

The two recipes differ only in ``name`` and the two motion files; keeping variant B generated means a
retune of variant A can never silently diverge from it (tests/test_g1_ik_retarget.py checks).
Usage (repo root): python3 unitree/scripts/sync_g1_variant_yaml.py [--check]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TASKS = REPO / "hope_training" / "whole_body_tracking" / "cfg" / "task"
SRC, DST = TASKS / "HOPEPingPongUnitreeG1.yaml", TASKS / "HOPEPingPongUnitreeG1_IK.yaml"
HEADER = """# Unitree G1 HitterPingPong, VARIANT B: identical to HOPEPingPongUnitreeG1.yaml (recipe v2) except that
# the reference motions are the keypoint/IK retargeted pair (scripts/retarget_a3_clip_to_g1_ik.py:
# the A3 arm shape scaled to the G1 and solved by IK, instead of copying joint angles).
# tests/test_g1_ik_retarget.py asserts the two files differ only in name/motion_file/motion_file_2.
# GENERATED: edit HOPEPingPongUnitreeG1.yaml and run
#   python3 unitree/scripts/sync_g1_variant_yaml.py   (sbatch knob: HOPE_G1_TASK=HOPEPingPongUnitreeG1_IK)
"""


def render() -> str:
    src = SRC.read_text(encoding="utf-8")
    out = src.replace("name: HitterPingPongUnitreeG1\n", "name: HitterPingPongUnitreeG1_IK\n", 1)
    old = "motion_file: ../motions/preprocessed/g1_hope_forehand.npz\nmotion_file_2: ../motions/preprocessed/g1_hope_backhand.npz\n"
    new = "motion_file: ../motions/preprocessed/g1ik_hope_forehand.npz\nmotion_file_2: ../motions/preprocessed/g1ik_hope_backhand.npz\n"
    if out.count(old) != 1 or "name: HitterPingPongUnitreeG1_IK" not in out:
        raise RuntimeError("HOPEPingPongUnitreeG1.yaml no longer has the expected name/motion lines")
    return HEADER + out.replace(old, new, 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not DST.is_file() or DST.read_text(encoding="utf-8") != text:
            print(f"[sync_g1_variant_yaml] STALE: {DST.relative_to(REPO)} (run python3 unitree/scripts/sync_g1_variant_yaml.py)", file=sys.stderr)
            return 1
        print("[sync_g1_variant_yaml] OK")
        return 0
    DST.write_text(text, encoding="utf-8")
    print(f"[sync_g1_variant_yaml] wrote {DST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
