# glm-tic — heretic fork for GLM-5.x abliteration

A fork of [p-e-w/heretic](https://github.com/p-e-w/heretic) (AGPL-3.0-or-later) that adds the
module-discovery hooks needed to abliterate **GLM-5.x** architectures, specifically
`Glm5NextForConditionalGeneration` (`model_type: glm5_next`).

Target model: `zai-org/GLM-5.3-Flash` — 320B total / 18B active MoE, 45 layers
(34 Kimi Delta Attention linear-attention + 11 DeepSeek Sparse Attention),
288 routed experts + 1 always-active shared expert, MIT license, ~305.8 GiB of
FP8 safetensors across 62 shards.

---

## ⚠️ Status: UNTESTED

**The patches in this fork have not been executed against the real model.** They are
lint-clean and structurally reasoned from the published `config.json` and
`model.safetensors.index.json`, but no run has completed. The structural test in
`tests/test_glm_structure.py` was written but could not run on the authoring host
(no AVX; PyTorch backward passes and the test itself SIGILL).

Validate on a GPU box before trusting any output. See "Validation" below.

---

## What changed vs upstream

### 1. `get_layers()` — GLM-5.x decoder nesting

Upstream tries `model.model.language_model.layers` (multimodal) then
`model.model.layers` (text-only). GLM-5.x nests the decoder at
`model.language_model.layers` — one level shallower than the multimodal path.

The patch adds an intermediate attempt with a non-empty check so a partially
initialised attribute can't silently return an empty layer list.

### 2. `get_layer_modules()` — the always-active shared expert

**This is the substantive fix.** Upstream collects `layer.mlp.experts[*].down_proj`
(Qwen3-style routed experts) but has no probe for `layer.mlp.shared_experts.down_proj`.

GLM keeps one shared expert that fires on **every token regardless of routing**.
Leaving it out means the highest-signal refusal locus in a GLM MoE is never
orthogonalized. The patch collects it into the existing `mlp.down_proj` component
so Optuna's weight distributions apply unchanged.

### What was *not* needed

Earlier analysis suggested GLM's 34 linear-attention layers might lack an
orthogonalization target. **They don't.** Layer 0 of the published index carries
`self_attn.o_proj.weight`, so upstream's existing `layer.self_attn.o_proj` probe
already covers them. No patch required.

---

## Coverage after patching

| GLM tensor | Count | Reachable |
|---|---|---|
| `self_attn.o_proj` | 46 | ✅ upstream |
| `mlp.experts.*.down_proj` | 12,384 | ✅ upstream |
| `mlp.shared_experts.down_proj` | 43 | ✅ **this fork** |
| `mlp.down_proj` (dense, layers 0–2) | 3 | ✅ upstream |
| `hyper_connection` (`hc_*`) | — | not a projection, skipped |

Components returned: `["attn.o_proj", "mlp.down_proj"]`

---

## Known open problem: 306 GiB residency

Upstream's `abliterate()` needs all layers co-resident to compute the refusal
direction across the full model — see [p-e-w/heretic#135][issue135], closed
wontfix 2026-02-13. At FP8, GLM-5.3-Flash is 305.8 GiB, so:

- **Minimum:** a single node with ≥320 GB of aggregate VRAM, with NVLink if the
  model straddles devices.
- **Practical:** `p4de.24xlarge` (8× A100 80 GB, 640 GB, NVLink) or
  `g7e.48xlarge` (8× RTX PRO 6000 Blackwell 96 GB, 768 GB, PCIe only).
- bitsandbytes quantization does not meaningfully help; the weights are already
  native FP8.
- A layer-streaming approach (compute direction from a calibration pass, then
  stream one layer at a time) is the escape hatch if whole-model residency is
  impractical. Not implemented here.

[issue135]: https://github.com/p-e-w/heretic/issues/135

---

## Validation

```bash
git clone https://github.com/instax-dutta/glm-tic.git
cd glm-tic
uv venv
uv pip install -e .
uv run python tests/test_glm_structure.py
```

`tests/test_glm_structure.py` builds a synthetic GLM-shaped MoE block (routed
experts + shared expert + `self_attn.o_proj`) — no 306 GB download — and asserts
that `get_layers`, `get_layer_modules`, and `get_abliterable_components` discover
the expected modules, including the shared expert.

Expected output:

```
get_layers -> 4 layers
dense layer components -> ['attn.o_proj', 'mlp.down_proj']
MoE layer components -> ['attn.o_proj', 'mlp.down_proj']
MoE mlp.down_proj modules -> 4 (3 routed + 1 shared expert)
get_abliterable_components -> ['attn.o_proj', 'mlp.down_proj']
PASS: GLM-5.x structure fully discoverable
```

### Loading the real model

```bash
uv run heretic --model zai-org/GLM-5.3-Flash --trust-remote-code
```

On startup heretic prints the abliterable component census. For GLM-5.3-Flash
you should see roughly:

```
* Transformer model with 45 layers
* Abliterable components:
  * attn.o_proj: 46 modules total
  * mlp.down_proj: 12430 modules total
```

If `mlp.down_proj` reports ~12,387 instead of ~12,430, the shared-expert patch
did not take effect. If the model fails to load at all, that's a
`transformers` version issue — the config declares `transformers_version: 5.16.0`.

---

## Upstream

- Repository: [p-e-w/heretic](https://github.com/p-e-w/heretic)
- License: AGPL-3.0-or-later (inherited, see `LICENSE`)
- Upstream README: `README.upstream.md`
- Method: Arditi et al. 2024, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717),
  plus the projected-abliteration refinement
- Related upstream issues: [#90 (GLM family)][issue90], [#135 (MoE memory)][issue135]

[issue90]: https://github.com/p-e-w/heretic/issues/90

Forked from upstream commit `17f8dff`.

## Disclaimer

For research use on models you have the right to modify. Respect each model's
license — GLM-5.3-Flash is MIT, but that does not extend to whatever you run this on.
