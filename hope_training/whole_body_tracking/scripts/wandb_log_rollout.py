#!/usr/bin/env python3
"""Log a MuJoCo rollout mp4 (+ its success-rate JSON) to a companion W&B run, keyed by iteration.

Used by the training sidecar: the live rsl_rl training process owns its own W&B run, and W&B does
not support two processes writing one run concurrently, so rollouts go to a sibling run in the
same project/group (``<run_name>-rollouts``) with ``iteration`` as the x-axis. Host Python only
(``wandb`` + stdlib); auth from ``~/.netrc`` or ``WANDB_API_KEY``.

Usage:
    python scripts/wandb_log_rollout.py --project hope-g1 --run-name g1_v1 --iteration 500 \
        --video <run>/videos/mujoco_iter500.mp4 --json <run>/videos/mujoco_iter500.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--run-name", required=True, help="Training run name; the companion run is <run-name>-rollouts.")
    p.add_argument("--run-id", default=None, help="Explicit companion run id (default: hope-<run-name>-rollouts).")
    p.add_argument("--entity", default=os.environ.get("WANDB_USERNAME") or None)
    p.add_argument("--iteration", type=int, required=True, help="Training iteration of the checkpoint.")
    p.add_argument("--video", required=True, help="mp4 to upload.")
    p.add_argument("--json", default=None, help="mujoco_eval_onnx --json-out file (success_rate).")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path, recorded as text.")
    p.add_argument("--key", default="Video/mujoco_rollout")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import wandb

    video = pathlib.Path(args.video)
    if not video.is_file():
        raise FileNotFoundError(video)
    run_id = args.run_id or f"hope-{args.run_name}-rollouts"
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        name=f"{args.run_name}-rollouts",
        group=args.run_name,
        job_type="mujoco-rollout",
        resume="allow",
        reinit=True,
    )
    run.define_metric("iteration")
    run.define_metric("*", step_metric="iteration")
    payload: dict = {
        "iteration": int(args.iteration),
        args.key: wandb.Video(str(video), format="mp4", caption=f"iteration {args.iteration}"),
    }
    if args.json and pathlib.Path(args.json).is_file():
        data = json.loads(pathlib.Path(args.json).read_text())
        for k, v in data.items():
            if isinstance(v, (int, float)):
                payload[f"Eval/mujoco_{k}"] = float(v)
    if args.checkpoint:
        payload["checkpoint"] = str(args.checkpoint)
    run.log(payload)
    run.finish()
    print(f"[wandb_log_rollout] logged iteration {args.iteration} to {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
