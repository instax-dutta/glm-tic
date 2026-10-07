"""Index-based structural validator for GLM-5.x abliteration targets.

Walks a GLM-5.x `model.safetensors.index.json` (~200 KB) plus `config.json` and
reports exactly which modules heretic's `get_abliterable_components()` will
discover, per layer -- without downloading the 306 GiB of weights.

heretic walks the *runtime* module tree. Until the model is loaded, the index
is the only faithful picture of that tree, because checkpoint tensor names are
derived from the same module paths.

This catches three classes of problem before an expensive GPU load test:
  1. config/index disagreement (layer counts, expert counts, dense-layer layout)
  2. probes that silently resolve to nothing (a wrong module path)
  3. precision problems -- FP8 block-quantized targets cannot be ablated by
     subtracting a direction in bf16 without a dequantize/requantize round-trip

Usage:
    python scripts/validate_glm_index.py                      # fetch + cache
    python scripts/validate_glm_index.py --config c.json --index i.json
    python scripts/validate_glm_index.py --repo zai-org/GLM-5.3-Flash
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path.home() / ".cache" / "glm-tic"

# The exact tensor suffix each heretic probe resolves to.
PROBE_SUFFIXES = {
    "attn.o_proj": "self_attn.o_proj.weight",
    "mlp.down_proj (dense)": "mlp.down_proj.weight",
    "mlp.down_proj (shared)": "mlp.shared_experts.down_proj.weight",
}
# Matched against the tensor name with a trailing ".weight" already stripped.
ROUTED_RE = re.compile(r"^mlp\.experts\.(\d+)\.down_proj$")

LAYER_PATTERNS = [
    re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$"),
    re.compile(r"^model\.model\.layers\.(\d+)\.(.+)$"),
    re.compile(r"^language_model\.layers\.(\d+)\.(.+)$"),
    re.compile(r"^model\.layers\.(\d+)\.(.+)$"),
]

# FP8 block quantization stores a companion per-block scale tensor.
SCALE_SUFFIX = ".weight_scale_inv"


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "glm-tic-validator"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
    except urllib.error.URLError as exc:
        raise SystemExit(f"failed to fetch {url}: {exc}") from exc
    return dest


@dataclass
class LayerReport:
    index: int
    mlp_kind: str = "unknown"
    routed_experts: int = 0
    has_shared: bool = False
    has_dense: bool = False
    has_attn_o_proj: bool = False
    fp8_targets: int = 0
    total_targets: int = 0


def is_mtp_layer(idx: int, n_layers: int) -> bool:
    """GLM appends num_nextn_predict_layers MTP heads past the decoder stack."""
    return idx >= n_layers


def scan(
    index_map: dict[str, str], n_layers: int | None, n_nextn: int = 0
) -> tuple[dict[int, LayerReport], dict[int, LayerReport]]:
    """Return (decoder_layers, mtp_layers).

    The two are kept apart deliberately: an MTP head is structurally similar
    enough to a decoder layer that folding it in silently would let a wrong
    num_hidden_layers validate cleanly.
    """
    reports: dict[int, LayerReport] = {}
    scale_tensors = {t for t in index_map if t.endswith(SCALE_SUFFIX)}

    for tensor_name in index_map:
        stem = tensor_name
        if stem.endswith(".weight"):
            stem = stem[: -len(".weight")]

        layer_idx = suffix = None
        for pattern in LAYER_PATTERNS:
            m = pattern.match(stem)
            if m:
                layer_idx = int(m.group(1))
                suffix = m.group(2)
                break
        if layer_idx is None:
            continue

        r = reports.setdefault(layer_idx, LayerReport(index=layer_idx))

        matched = False
        if suffix == "self_attn.o_proj":
            r.has_attn_o_proj = True
            matched = True
        elif suffix == "mlp.down_proj":
            r.has_dense = True
            r.mlp_kind = "dense"
            matched = True
        elif suffix == "mlp.shared_experts.down_proj":
            r.has_shared = True
            r.mlp_kind = "moe"
            matched = True
        else:
            m = ROUTED_RE.match(suffix)
            if m:
                r.routed_experts = max(r.routed_experts, int(m.group(1)) + 1)
                r.mlp_kind = "moe"
                matched = True

        if matched:
            r.total_targets += 1
            # stem is the tensor name minus a trailing ".weight"; the companion
            # FP8 block scale is named "<stem>.weight_scale_inv".
            if stem + SCALE_SUFFIX in scale_tensors:
                r.fp8_targets += 1

    mtp: dict[int, LayerReport] = {}
    if n_layers is not None:
        for i in list(reports):
            if is_mtp_layer(i, n_layers):
                mtp[i] = reports.pop(i)

    return reports, mtp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="zai-org/GLM-5.3-Flash")
    p.add_argument("--config", type=Path)
    p.add_argument("--index", type=Path)
    args = p.parse_args()

    if args.config and args.index:
        config_path, index_path = args.config, args.index
    else:
        base = f"https://huggingface.co/{args.repo}/resolve/main"
        config_path = fetch(f"{base}/config.json", CACHE / "config.json")
        index_path = fetch(
            f"{base}/model.safetensors.index.json",
            CACHE / "model.safetensors.index.json",
        )

    config = json.loads(config_path.read_text())
    index_map = json.loads(index_path.read_text())["weight_map"]

    tc = config.get("text_config", config)
    n_layers = tc.get("num_hidden_layers")
    n_experts = tc.get("n_routed_experts")
    n_shared = tc.get("n_shared_experts", 1)
    k_dense = tc.get("first_k_dense_replace")
    n_nextn = tc.get("num_nextn_predict_layers", 0)
    qc = config.get("quantization_config") or {}
    quant_method = qc.get("quant_method")

    print(f"\nconfig: {config_path}")
    print(f"index:  {index_path}  ({len(index_map):,} tensors)\n")

    print("=== config ===")
    print(f"  architectures          : {config.get('architectures')}")
    print(f"  num_hidden_layers      : {n_layers}")
    print(f"  n_routed_experts       : {n_experts}")
    print(f"  n_shared_experts       : {n_shared}")
    print(f"  first_k_dense_replace  : {k_dense}")
    print(f"  num_nextn_predict_layers: {n_nextn}  (MTP heads, excluded from ablation)")
    print(f"  quantization           : {quant_method} / {qc.get('fmt')}")

    reports, mtp = scan(index_map, n_layers)
    moe = [r for r in reports.values() if r.mlp_kind == "moe"]
    dense = [r for r in reports.values() if r.mlp_kind == "dense"]

    census: Counter = Counter()
    fp8_census: Counter = Counter()
    for r in reports.values():
        census["attn.o_proj"] += int(r.has_attn_o_proj)
        census["mlp.down_proj"] += r.routed_experts + int(r.has_shared) + int(r.has_dense)
        fp8_census["fp8"] += r.fp8_targets
        fp8_census["total"] += r.total_targets

    print(f"\n=== index scan ({n_layers} decoder layers, MTP excluded) ===")
    print(f"  layers present in index : {len(reports)}  ({min(reports)}..{max(reports)})")
    print(f"  MoE layers   : {len(moe)}")
    print(f"  dense layers : {len(dense)}  -> {sorted(r.index for r in dense)}")
    print(f"  MoE layers with shared_experts.down_proj: {sum(r.has_shared for r in moe)}/{len(moe)}")
    print(
        f"  layers with self_attn.o_proj           : "
        f"{sum(1 for r in reports.values() if r.has_attn_o_proj)}/{len(reports)}"
    )

    routed_total = sum(r.routed_experts for r in moe)
    print(f"\n=== mlp.down_proj breakdown ===")
    print(f"  routed expert down_proj : {routed_total:,}")
    print(f"  shared expert down_proj : {sum(r.has_shared for r in moe):,}")
    print(f"  dense FFN down_proj     : {sum(r.has_dense for r in dense):,}")
    print(f"  TOTAL mlp.down_proj     : {census['mlp.down_proj']:,}")
    print(f"  attn.o_proj             : {census['attn.o_proj']:,}")
    print(f"  TOTAL abliterable       : {sum(census.values()):,}")

    if fp8_census["total"]:
        pct = 100 * fp8_census["fp8"] / fp8_census["total"]
        print(f"\n=== precision ===")
        print(f"  FP8 targets    : {fp8_census['fp8']:,} / {fp8_census['total']:,} ({pct:.1f}%)")
        print(f"  bf16 targets   : {fp8_census['total'] - fp8_census['fp8']:,}")

    errors: list[str] = []
    warnings: list[str] = []

    if n_layers is not None and len(reports) != n_layers:
        errors.append(f"index has {len(reports)} decoder layers, config says {n_layers}")

    # An MTP head looks structurally like a decoder layer, so if the config's
    # num_hidden_layers is wrong the head gets silently counted as one. Catch it.
    if mtp:
        print(f"\n  MTP heads detected outside the decoder stack: {sorted(mtp)}")
        if n_nextn == 0:
            errors.append(
                f"index contains layers {sorted(mtp)} past num_hidden_layers="
                f"{n_layers}, but config declares num_nextn_predict_layers=0"
            )
        elif len(mtp) != n_nextn:
            errors.append(
                f"index has {len(mtp)} extra layer(s) {sorted(mtp)} past the decoder "
                f"stack, config declares num_nextn_predict_layers={n_nextn}"
            )
    elif n_nextn:
        warnings.append(
            f"config declares num_nextn_predict_layers={n_nextn} but no extra layer "
            f"was found past index {n_layers}"
        )

    if n_experts:
        bad = [r.index for r in moe if r.routed_experts != n_experts]
        if bad:
            errors.append(
                f"{len(bad)} MoE layers have routed expert count != {n_experts}: {bad[:5]}"
            )

    if k_dense is not None:
        expected_dense = set(range(k_dense))
        actual_dense = {r.index for r in dense}
        if expected_dense != actual_dense:
            errors.append(
                f"dense layers {sorted(actual_dense)} != expected {sorted(expected_dense)}"
            )

    if n_layers and n_experts and k_dense is not None:
        n_moe = n_layers - k_dense
        expected_down = n_moe * (n_experts + n_shared) + k_dense
        if census["mlp.down_proj"] != expected_down:
            errors.append(
                f"mlp.down_proj {census['mlp.down_proj']} != "
                f"{n_moe} MoE x ({n_experts}+{n_shared}) + {k_dense} dense "
                f"= {expected_down}"
            )
        else:
            print(f"  arithmetic check: OK ({expected_down})")
        if census["attn.o_proj"] != n_layers:
            errors.append(
                f"attn.o_proj {census['attn.o_proj']} != num_hidden_layers {n_layers}"
            )

    missing_shared = [r.index for r in moe if not r.has_shared]
    if missing_shared:
        errors.append(
            f"{len(missing_shared)} MoE layers lack shared_experts.down_proj -- "
            f"heretic's GLM patch would find nothing on these: {missing_shared[:5]}"
        )

    if quant_method == "fp8":
        warnings.append(
            f"{fp8_census['fp8']:,} of {fp8_census['total']:,} targets are FP8 "
            f"{qc.get('fmt')} with block scales. heretic's orthogonalization "
            "assumes bf16/fp16 tensors: direction extraction on quantized "
            "weights needs a dequantize -> ablate -> requantize path, and the "
            f"block scale ({qc.get('weight_block_size')}) must be applied "
            "consistently. Untested."
        )

    print()
    for w in warnings:
        print(f"  WARN: {w}")
    for e in errors:
        print(f"  ERROR: {e}")

    if errors:
        print("\nFAIL: index and config disagree, or heretic's probes would not resolve.\n")
        return 1

    print("\nPASS: index is consistent with config; heretic's probes resolve modules.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())