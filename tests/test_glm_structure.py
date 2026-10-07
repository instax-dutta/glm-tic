"""Structural test: does heretic's module discovery see a GLM-5.x MoE block layout?

Builds a tiny synthetic model that mirrors GLM-5.3-Flash's *structure* (without
downloading 306 GB of weights), then runs heretic's real get_layers /
get_layer_modules / get_abliterable_components against it.

Run: uv run python tests/test_glm_structure.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heretic.model import Model  # noqa: E402


class SharedExpert(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class MoEBlock(nn.Module):
    """Mirrors GLM: 288-style routed experts + 1 always-active shared expert."""

    def __init__(self, hidden: int, inter: int, n_experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList(
            SharedExpert(hidden, inter) for _ in range(n_experts)
        )
        self.shared_experts = SharedExpert(hidden, inter)


class SelfAttn(nn.Module):
    """GLM exposes o_proj on BOTH linear-attention and sparse-attention layers."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_norm = nn.LayerNorm(hidden)


class DenseBlock(nn.Module):
    def __init__(self, hidden: int, inter: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class GlmLikeLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        inter: int,
        moe_inter: int,
        n_experts: int,
        dense: bool,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = SelfAttn(hidden)
        if dense:
            self.mlp = DenseBlock(hidden, inter)
        else:
            self.mlp = MoEBlock(hidden, moe_inter, n_experts)


class GlmLikeInner(nn.Module):
    def __init__(self, layers: list[GlmLikeLayer], hidden: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(128, hidden)
        self.layers = nn.ModuleList(layers)


class GlmLikeTop(nn.Module):
    def __init__(self, layers: list[GlmLikeLayer], hidden: int) -> None:
        super().__init__()
        self.model = GlmLikeInner(layers, hidden)


def build_glm_like(n_layers: int = 4, n_experts: int = 3, first_dense: int = 1):
    """first_dense layers are dense FFN, the rest are MoE (GLM uses 3 of 45)."""
    hidden, inter, moe_inter = 16, 32, 8
    layers = [
        GlmLikeLayer(hidden, inter, moe_inter, n_experts, dense=i < first_dense)
        for i in range(n_layers)
    ]
    return GlmLikeTop(layers, hidden)


def probe(fake_top) -> Model:
    """Bind heretic's discovery methods to a synthetic model, no HF loading."""
    holder = Model.__new__(Model)
    holder.model = fake_top
    return holder


def main() -> int:
    n_layers, n_experts, first_dense = 4, 3, 1
    holder = probe(build_glm_like(n_layers, n_experts, first_dense))

    layers = holder.get_layers()
    assert len(layers) == n_layers, f"get_layers returned {len(layers)}"

    # Per-layer module discovery.
    dense_layer = holder.get_layer_modules(0)
    moe_layer = holder.get_layer_modules(n_layers - 1)

    assert "mlp.down_proj" in dense_layer, "dense FFN down_proj not found"
    assert "attn.o_proj" in moe_layer, "self_attn.o_proj not found"

    # The GLM-specific patch: shared_experts.down_proj must be collected.
    assert "mlp.down_proj" in moe_layer, "MoE layer lost mlp.down_proj"

    routed = len(moe_layer["mlp.down_proj"])
    expected_routed = n_experts + 1  # routed experts + the shared expert
    assert routed == expected_routed, (
        f"expected {expected_routed} down_proj modules "
        f"({n_experts} routed + 1 shared), got {routed}"
    )

    components = holder.get_abliterable_components()
    assert set(components) == {"attn.o_proj", "mlp.down_proj"}, (
        f"unexpected component set: {components}"
    )

    print(f"get_layers -> {len(layers)} layers")
    print(f"dense layer components -> {sorted(dense_layer)}")
    print(f"MoE layer components -> {sorted(moe_layer)}")
    print(f"MoE mlp.down_proj modules -> {routed} "
          f"({n_experts} routed + 1 shared expert)")
    print(f"get_abliterable_components -> {components}")
    print("PASS: GLM-5.x structure fully discoverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
