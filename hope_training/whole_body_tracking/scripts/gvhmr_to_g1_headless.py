#!/usr/bin/env python3
"""Headless GVHMR -> GMR retarget to the Unitree G1 (runs in the GMR venv, no display needed).

Same retarget as GMR's ``scripts/gvhmr_to_robot.py`` (30 Hz SMPL-X frames -> ``GMR(...).retarget``)
without its MuJoCo viewer, which needs a display that cluster nodes lack. Writes the same pickle
(``fps``, ``root_pos``, ``root_rot`` xyzw, ``dof_pos``) consumed by ``gmr_video_clip_to_g1.py``.

  python gvhmr_to_g1_headless.py --gmr-root <GMR checkout> \\
      --gvhmr-pred <GVHMR>/outputs/demo/<video>/hmr4d_results.pt --save-path forehand_gmr.pkl
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gmr-root", required=True, help="GMR checkout (assets/body_models/smplx must hold SMPL-X)")
    p.add_argument("--gvhmr-pred", required=True, help="GVHMR hmr4d_results.pt")
    p.add_argument("--save-path", required=True)
    p.add_argument("--robot", default="unitree_g1")
    args = p.parse_args(argv)

    root = pathlib.Path(args.gmr_root).resolve()
    sys.path.insert(0, str(root))
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast, load_gvhmr_pred_file

    smplx_data, body_model, smplx_output, height = load_gvhmr_pred_file(
        args.gvhmr_pred, root / "assets" / "body_models"
    )
    frames, fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    retarget = GMR(actual_human_height=height, src_human="smplx", tgt_robot=args.robot)
    qpos = np.array([retarget.retarget(f) for f in frames])

    out = pathlib.Path(args.save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(
            {
                "fps": fps,
                "root_pos": qpos[:, :3],
                "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # wxyz -> xyzw, as GMR saves it
                "dof_pos": qpos[:, 7:],
                "local_body_pos": None,
                "link_body_list": None,
            },
            fh,
        )
    print(f"[gvhmr_to_g1_headless] {len(qpos)} frames @ {fps} Hz, human height {float(height):.2f} m -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
