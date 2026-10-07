"""Structural test: does heretic's module discovery see a GLM-5.x MoE block layout?

Builds a tiny synthetic model that mirrors GLM-5.3-Flash's *structure* (without
downloading 306 GB of weights), then runs heretic's real get_layers /
get_layer_modules / get_abliterable_components against it.

peft and bitsandbytes are stubbed before importing heretic.model: both ship
native extensions that SIGILL on CPUs without AVX, and neither is exercised by
the discovery methods under test here. Stubbing lets the pure-Python module
discovery logic run on any host.

Run: python tests/test_glm_structure.py
"""

import sys
import types
from pathlib import Path

import torch
from torch import nn


def _stub_native_extensions() -> None:
    """Replace peft/bitsandbytes with inert stand-ins before heretic imports them."""
    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")

        class _Config:  # noqa: D401 - stub
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class _PeftModel:  # noqa: D401 - stub
            pass

        peft.LoraConfig = _Config
        peft.PeftConfig = _Config
        peft.PeftModel = _PeftModel
        peft.get_peft_model = lambda *a, **k: None
        peft.get_peft_config = lambda m: None
        peft.get_peft_model_state_dict = lambda m: {}
        peft.set_peft_model_state_dict = lambda m, s: m
        peft.add_adapter = lambda m, *a, **k: m
        peft.prepare_model_for_kbit_training = lambda m, *a, **k: m
        peft.PeftType = types.SimpleNamespace(LORA="LORA")
        peft.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
        sys.modules["peft"] = peft

    if "bitsandbytes" not in sys.modules:
        bnb = types.ModuleType("bitsandbytes")
        bnb.nn = types.ModuleType("bitsandbytes.nn")
        bnb.nn.Linear4bit = type("Linear4bit", (), {})
        bnb.nn.Params4bit = type("Params4bit", (), {})
        bnb.__version__ = "4bit-stub"
        sys.modules["bitsandbytes"] = bnb
        sys.modules["bitsandbytes.nn"] = bnb.nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_stub_native_extensions()

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
    torch.manual_seed(0)

    n_layers, n_experts, first_dense = 4, 3, 1
    holder = probe(build_glm_like(n_layers, n_experts, first_dense))

    layers = holder.get_layers()
    assert len(layers) == n_layers, f"get_layers returned {len(layers)}"

    # Per-layer module discovery.
    dense_layer = holder.get_layer_modules(0)
    moe_layer = holder.get_layer_modules(n_layers - 1)

    assert "mlp.down_proj" in dense_layer, "dense FFN down_proj not found"
    assert "attn.o_proj" in moe_layer, "self_attn.o_proj not found"

    routed = len(moe_layer["mlp.down_proj"])
    expected_routed = n_experts + 1  # routed experts + the shared expert
    assert routed == expected_routed, (
        f"expected {expected_routed} down_proj modules "
        f"({n_experts} routed + 1 shared), got {routed}"
    )

    # Verify the GLM-specific patch by object identity, not just by count: the
    # shared expert's down_proj must be the exact module the discovery collected.
    shared_down_proj = layers[n_layers - 1].mlp.shared_experts.down_proj
    assert shared_down_proj in moe_layer["mlp.down_proj"], (
        "collected mlp.down_proj set does not include the shared expert"
    )

    # And every routed expert must be collected too.
    routed_mods = [e.down_proj for e in layers[n_layers - 1].mlp.experts]
    missing = [m for m in routed_mods if m not in moe_layer["mlp.down_proj"]]
    assert not missing, f"{len(missing)} routed expert down_proj modules not collected"

    components = holder.get_abliterable_components()
    assert set(components) == {"attn.o_proj", "mlp.down_proj"}, (
        f"unexpected component set: {components}"
    )

    print(f"get_layers -> {len(layers)} layers")
    print(f"dense layer components -> {sorted(dense_layer)}")
    print(f"MoE layer components -> {sorted(moe_layer)}")
    print(
        f"MoE mlp.down_proj modules -> {routed} "
        f"({n_experts} routed + 1 shared expert)"
    )
    print(f"get_abliterable_components -> {components}")
    print("PASS: GLM-5.x structure fully discoverable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
