#!/usr/bin/env python3
"""Shrink rsl_rl checkpoints in place: drop the exact-resume simulator snapshot, keep the policy.

A 16384-env checkpoint is ~320 MB, of which ~314 MB is ``hope_exact_resume_state`` (every env's
command/buffer state, only needed to continue the SAME run bit-exactly via ``load_exact``). Evaluation,
ONNX export and ``checkpoint_path`` warm starts / ordinary resumes (``runner.load``) only read
``model_state_dict``, ``optimizer_state_dict``, ``iter`` and ``infos``. The slim file keeps those plus
the env step counter under the path ``train.py restore_step_counter=true`` reads, so a resume from it
continues the step-keyed ramps. ``load_exact`` refuses it (no schema_version), which is intended.

    python3 hope_training/whole_body_tracking/scripts/slim_checkpoint.py artifacts/antioch/*/model_*.pt
"""

from __future__ import annotations

import os
import sys

import torch


def slim(path: str) -> tuple[float, float] | None:
    before = os.path.getsize(path) / 1e6
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("hope_exact_resume_state")
    if not isinstance(state, dict) or state.get("slim"):
        return None
    counter = (((state.get("environment_resume_state") or {}).get("env_buffers") or {})
               .get("common_step_counter"))
    ckpt["hope_exact_resume_state"] = {
        "slim": True,
        "environment_resume_state": {"env_buffers": {"common_step_counter": counter}},
    }
    tmp = path + ".slim_tmp"
    torch.save(ckpt, tmp)
    torch.load(tmp, map_location="cpu", weights_only=False)["model_state_dict"]  # readable before replacing
    os.replace(tmp, path)
    return before, os.path.getsize(path) / 1e6


def main(paths: list[str]) -> int:
    saved = 0.0
    for p in paths:
        r = slim(p)
        if r is None:
            print(f"[slim] {p}: already slim / no resume state")
            continue
        saved += r[0] - r[1]
        print(f"[slim] {p}: {r[0]:.0f} -> {r[1]:.1f} MB", flush=True)
    print(f"[slim] freed {saved / 1000:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
