"""Make a policy.onnx for a checkpoint without Isaac: copy its actor weights into an exported graph.

``export_onnx.py`` starts Isaac on a GPU. When that is not possible (no free GPU, or new CUDA contexts
fail on the node) and a policy.onnx of the SAME task/architecture already exists, this swaps the
actor weights of a checkpoint into that graph on the CPU. The graph is the plain actor MLP
(``actor_critic.actor.<i>.weight/bias``); names and shapes must match exactly, or it refuses.

    .venv/bin/python hope_training/whole_body_tracking/scripts/onnx_from_checkpoint.py \
        --template artifacts/evals/g1_fh_diag_model_5000/policy.onnx \
        --checkpoint model_15400.pt --out artifacts/evals/g1_fh_diag_model_15400/policy.onnx

Check it once against a real export with ``--compare <policy.onnx exported from the same checkpoint>``.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def swap_weights(template: str, checkpoint: str, out: str) -> int:
    import onnx
    import torch
    from onnx import numpy_helper

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    model = onnx.load(template)
    replaced = 0
    for init in model.graph.initializer:
        key = init.name.removeprefix("actor_critic.")
        if key not in state:
            raise KeyError(f"graph weight {init.name} has no checkpoint entry {key}")
        new = state[key].numpy().astype(numpy_helper.to_array(init).dtype)
        if new.shape != tuple(init.dims):
            raise ValueError(f"{init.name}: graph {tuple(init.dims)} vs checkpoint {new.shape}")
        init.CopyFrom(numpy_helper.from_array(new, init.name))
        replaced += 1
    actor_keys = [k for k in state if k.startswith("actor.")]
    if replaced != len(actor_keys):
        raise ValueError(f"replaced {replaced} tensors but the checkpoint has {len(actor_keys)} actor tensors")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    return replaced


def max_output_diff(a: str, b: str, n: int = 256, seed: int = 0) -> float:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    sa, sb = ort.InferenceSession(a), ort.InferenceSession(b)
    name = sa.get_inputs()[0].name
    dim = sa.get_inputs()[0].shape[1]
    worst = 0.0
    for _ in range(n):
        x = rng.normal(size=(1, dim)).astype(np.float32)
        worst = max(worst, float(np.abs(sa.run(None, {name: x})[0] - sb.run(None, {name: x})[0]).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="an exported policy.onnx of the same task")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="a real export of the same checkpoint, to verify")
    args = ap.parse_args()
    n = swap_weights(args.template, args.checkpoint, args.out)
    print(f"[onnx_from_checkpoint] {n} actor tensors -> {args.out}")
    if args.compare:
        print(f"[onnx_from_checkpoint] max |output difference| vs {args.compare}: {max_output_diff(args.out, args.compare):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
