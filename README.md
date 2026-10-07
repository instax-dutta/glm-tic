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

Counts below are **verified** by `scripts/validate_glm_index.py`, which walks the
real `model.safetensors.index.json` from the Hub. They are not estimates.

Decoder stack only — the checkpoint also contains a 46th layer that is an MTP
(multi-token-prediction) head, not a decoder layer. `num_nextn_predict_layers=1`
in `config.json`; it is structurally near-identical to a decoder layer, so the
validator separates and reports it explicitly rather than counting it in.

| GLM tensor | Count | Reachable by |
|---|---|---|
| `self_attn.o_proj` | 45 | ✅ upstream |
| `mlp.experts.*.down_proj` | 12,096 | ✅ upstream |
| `mlp.shared_experts.down_proj` | 42 | ✅ **this fork** |
| `mlp.down_proj` (dense, layers 0–2) | 3 | ✅ upstream |
| `hyper_connection` (`hc_*`) | — | not a projection, skipped |

- **45 decoder layers**: 3 dense (`first_k_dense_replace=3`) + 42 MoE.
- **288 routed experts** per MoE layer (`n_routed_experts`), 8 active per token
  (`num_experts_per_tok`), plus **1 shared expert** (`n_shared_experts`).
- Arithmetic check enforced by the validator:
  `42 × (288 + 1) + 3 = 12,141` ✅
- **Total abliterable modules: 12,186** (`12,141` `mlp.down_proj` + 45
  `attn.o_proj`).

Components returned: `["attn.o_proj", "mlp.down_proj"]`

### ⚠️ The checkpoint is FP8 — this is the bigger problem

`config.json` declares `quantization_config.quant_method = "fp8"`, `fmt = "e4m3"`,
`weight_block_size = [128, 128]`, dynamic activation scheme. The index confirms
it: **12,152 of 12,186 ablation targets (99.7%) are FP8 with companion
`.weight_scale_inv` block-scale tensors.** Only 34 targets are bf16 — exactly the
`self_attn.o_proj` of the 34 `linear_attention` layers, which
`modules_to_not_convert` excludes from quantization.

What this means for heretic:

- Direction extraction reads module weights and orthogonalizes them. FP8
  e4m3 has ~2–3 decimal digits of precision and a 128×128 block scale that must
  be applied consistently. Heretic's math assumes bf16/fp16 tensors.
- Ablating a quantized tensor means dequantize → ablate → requantize, and the
  result may not survive requantization without loss. Orthogonalizing an FP8
  tensor in place risks corrupting the block structure.
- This is **untested**. `p4de.24xlarge` (8× A100 80 GB) is still the right
  hardware, but the first two hours should be a *load-and-ablate-one-layer* test,
  not a full run.

Concretely, the first GPU experiment should dequantize a single MoE layer's
routed + shared `down_proj`, confirm the tensors reshape back cleanly, and only
then attempt direction extraction on one module.

### → [`GPU_TEST.md`](GPU_TEST.md) — exact commands to run when you have a GPU

A four-step gated runbook: structural checks (no GPU), the FP8 round-trip probe
(one GPU, this is the gate), the component census, then ablation and export.
Include the printed output of any failing step when reporting back.

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
the expected modules, including the shared expert. It stubs `peft` and
`bitsandbytes`, whose native extensions require CPU features some hosts lack.

Expected output:

```
get_layers -> 4 layers
dense layer components -> ['attn.o_proj', 'mlp.down_proj']
MoE layer components -> ['attn.o_proj', 'mlp.down_proj']
MoE mlp.down_proj modules -> 4 (3 routed + 1 shared expert)
get_abliterable_components -> ['attn.o_proj', 'mlp.down_proj']
PASS: GLM-5.x structure fully discoverable
```

### Validating against the real index (no weights needed)

`scripts/validate_glm_index.py` fetches only `config.json` and
`model.safetensors.index.json` — a few hundred KB — and predicts heretic's
component census from the checkpoint's own module paths:

```bash
uv run python scripts/validate_glm_index.py --repo zai-org/GLM-5.3-Flash
```

It cross-checks config against index (layer counts, expert counts, dense-layer
layout), confirms each heretic probe resolves to a real module, separates the
MTP head from the decoder stack, and reports the precision of every target.
Exit code is non-zero on any inconsistency, so it is usable as a pre-flight
check.

Actual output for GLM-5.3-Flash:

```
layers present in index : 45  (0..44)
MoE layers   : 42
dense layers : 3  -> [0, 1, 2]
MoE layers with shared_experts.down_proj: 42/42
layers with self_attn.o_proj           : 45/45

routed expert down_proj : 12,096
shared expert down_proj : 42
dense FFN down_proj     : 3
TOTAL mlp.down_proj     : 12,141
attn.o_proj             : 45
TOTAL abliterable       : 12,186

FP8 targets    : 12,152 / 12,186 (99.7%)
arithmetic check: OK (12141)

MTP heads detected outside the decoder stack: [45]
PASS: index is consistent with config; heretic's probes resolve modules.
```

### Loading the real model

```bash
uv run heretic --model zai-org/GLM-5.3-Flash --trust-remote-code
```

On startup heretic prints the abliterable component census. For GLM-5.3-Flash
it should match the validator's prediction exactly:

```
* Transformer model with 45 layers
* Abliterable components:
  * attn.o_proj: 45 modules total
  * mlp.down_proj: 12141 modules total
```

Those two numbers are the cheapest possible proof that both patches took effect:
the shared-expert patch is what turns `mlp.down_proj` from 12,099 into 12,141,
and anything other than 45 layers means the MTP head leaked in or a decoder
layer was missed. If the model fails to load at all, that is a `transformers`
version issue — the config declares `transformers_version: 5.16.0`.

**Before any full run**, load a single MoE layer and round-trip one quantized
`down_proj` through dequantize → reshape → requantize. 99.7% of the targets are
FP8; if that round-trip is lossy or shape-breaking, the ablation approach needs
to change before you spend hours on it.

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
