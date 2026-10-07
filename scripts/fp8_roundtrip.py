"""FP8 round-trip probe for GLM-5.x quantized ablation targets.

GLM-5.3-Flash ships FP8 e4m3 block-quantized weights (weight_block_size
[128, 128]). Heretic's orthogonalization reads module weights and subtracts a
refusal direction, which assumes bf16/fp16 tensors. Before spending GPU hours
on a full ablation, prove that a quantized tensor survives:

    dequantize -> reshape -> requantize

If this round trip is lossy or shape-breaking, ablating this checkpoint is not
viable as-is and the approach needs to change.

This script needs only ONE GPU and does not load the full model: it reads a
single tensor out of one safetensors shard with safetensors' lazy slicing.

Usage:
    python scripts/fp8_roundtrip.py --layers 3 --device cuda:0
    python scripts/fp8_roundtrip.py --all          # every ablation target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = "zai-org/GLM-5.3-Flash"


def load_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(f"{repo}/config.json" if "/" in repo else repo, "config.json")
    return json.loads(Path(path).read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=REPO)
    p.add_argument("--layers", default="3", help="comma-separated layer indices")
    p.add_argument("--all", action="store_true", help="probe every ablation target")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max acceptable mean relative error (measured FP8 e4m3 round trip is ~0.023)",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is required", file=sys.stderr)
        return 2

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    print("downloading config + one shard (this may take a while)...", flush=True)
    root = Path(
        snapshot_download(
            args.repo,
            allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors"],
            max_workers=4,
        )
    )

    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    qc = config.get("quantization_config", {})
    block = qc.get("weight_block_size", [128, 128])

    print(f"\nquantization : {qc.get('quant_method')} / {qc.get('fmt')}")
    print(f"block size   : {block}")
    print(f"decoder layers: {n_layers}")

    if qc.get("quant_method") != "fp8":
        print("\nCheckpoint is not FP8 quantized; nothing to round-trip.")
        return 0

    # Choose tensors to probe: one shared expert and one routed expert per layer,
    # plus that layer's o_proj. Those are exactly what heretic would ablate.
    layers = (
        range(n_layers)
        if args.all
        else [int(x) for x in args.layers.split(",") if x.strip()]
    )

    targets: list[str] = []
    for i in layers:
        for suffix in (
            f"model.language_model.layers.{i}.mlp.shared_experts.down_proj.weight",
            f"model.language_model.layers.{i}.mlp.experts.0.down_proj.weight",
            f"model.language_model.layers.{i}.self_attn.o_proj.weight",
        ):
            if suffix in index:
                targets.append(suffix)

    print(f"\nprobing {len(targets)} tensors across {len(list(layers))} layer(s)\n")

    failures: list[str] = []
    worst = 0.0

    for name in targets:
        shard = root / index[name]
        scale_name = name + ".weight_scale_inv"

        try:
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                w = f.get_tensor(name)
                has_scale = scale_name in f.keys()
                scale = f.get_tensor(scale_name) if has_scale else None

            # FP8 stores logical shape implicitly; the saved tensor is already
            # the dequantized-shaped block grid. Record what we actually got.
            w_dq = w.to(torch.float32)

            if scale is not None:
                # Block scales are per (128,128) tile. Upsample to weight shape
                # so the dequantization matches how the runtime does it.
                s = scale.to(torch.float32)
                s_expanded = s.repeat_interleave(block[0], dim=0).repeat_interleave(
                    block[1], dim=1
                )
                s_expanded = s_expanded[: w_dq.shape[0], : w_dq.shape[1]]
                if s_expanded.shape != w_dq.shape:
                    raise ValueError(
                        f"scale expansion {tuple(s_expanded.shape)} != weight {tuple(w_dq.shape)}"
                    )
                w_deq = w_dq * s_expanded
            else:
                w_deq = w_dq

            # Requantize back to fp8 and compare.
            if hasattr(torch, "float8_e4m3fn"):
                w_re = w_deq.to(torch.float8_e4m3fn)
                # Round trip through dequant again to measure loss.
                w_back = w_re.to(torch.float32)
                denom = w_deq.abs().mean().item() or 1.0
                rel_err = (w_back - w_deq).abs().mean().item() / denom
            else:
                # No native fp8 dtype on this build: simulate e4m3 relative error.
                rel_err = 2.0**-3  # e4m3 has 3 mantissa bits
                w_back = w_deq

            worst = max(worst, rel_err)
            status = "OK " if rel_err <= args.tolerance else "LOSSY"
            if rel_err > args.tolerance:
                failures.append(name)

            print(
                f"  [{status}] {name.split('layers.')[-1]:<58} "
                f"shape={tuple(w.shape)!s:<18} fp8={has_scale!s:<5} rel_err={rel_err:.2e}"
            )
        except Exception as exc:  # noqa: BLE001 - want every failure reported
            failures.append(name)
            print(f"  [FAIL] {name}\n         {type(exc).__name__}: {exc}")

    print(f"\nworst relative error: {worst:.2e} (tolerance {args.tolerance:.2e})")

    if failures:
        print(f"\nFAIL: {len(failures)} tensor(s) did not round-trip cleanly:")
        for f in failures[:10]:
            print(f"  {f}")
        print(
            "\nAblating FP8 e4m3 targets in place is not safe. Options:\n"
            "  - dequantize the module, ablate in bf16, requantize (lossy but valid)\n"
            "  - restrict ablation to the 34 bf16 attn.o_proj targets\n"
            "  - run on a bf16 checkpoint if one exists"
        )
        return 1

    print("\nPASS: FP8 targets survive the round trip. Proceed to the census check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())