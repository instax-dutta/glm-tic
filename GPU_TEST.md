# GPU test runbook — glm-tic on GLM-5.3-Flash

Ordered so each step gates the next. **Do not skip step 3.** Steps 0–3 need no
GPU and take minutes; they catch the failures that would otherwise waste a
multi-hour instance.

Reference hardware: `p4de.24xlarge` (8× A100 80 GB, NVLink), `$27.44705/hr`
Linux Shared on-demand in `us-east-1`. With a `$5,000` credit that is ~182
hours total, so budget in hours, not days.

### Quick reference

| Step | Needs | Time | Gate |
|---|---|---|---|
| 0 | nothing | 1 min | both scripts exit 0 |
| 1 | 1 GPU + network | 15 min | FP8 round trip `rel_err < 0.05` |
| 2 | 8 GPUs | 1–2 hr | census prints `45` / `45` / `12141` |
| 3 | 8 GPUs | hours | ablation completes |

If any gate fails, **stop there** and send me the output. Do not proceed to the
next step — each failure mode invalidates everything after it.

---

## Step 0 — Structural checks (no GPU, ~1 min)

```bash
git clone https://github.com/instax-dutta/glm-tic.git
cd glm-tic

uv venv
uv pip install -e .

uv run python tests/test_glm_structure.py
uv run python scripts/validate_glm_index.py --repo zai-org/GLM-5.3-Flash
```

Expected:

```
PASS: GLM-5.x structure fully discoverable
...
TOTAL abliterable       : 12,186
arithmetic check: OK (12141)
MTP heads detected outside the decoder stack: [45]
PASS: index is consistent with config; heretic's probes resolve modules.
```

Both exit `0`. If either exits non-zero, **stop** — the real model will not
load.

---

## Step 1 — FP8 round-trip (needs 1 GPU, ~15 min) ← THE GATE

99.7% of ablation targets (12,152 / 12,186) are FP8 `e4m3` block-quantized with
`weight_block_size = [128, 128]`. Heretic contains **no FP8 handling at all**
(`grep -rn 'fp8\|weight_scale_inv' src/` returns nothing). Its orthogonalization
math assumes bf16/fp16.

So before anything else, prove that a quantized tensor survives a
dequantize → reshape → requantize round trip. If it does not, ablation of this
checkpoint is not viable and every later step is moot.

```bash
uv pip install torch transformers accelerate safetensors huggingface_hub
uv run python scripts/fp8_roundtrip.py --layers 3
```

What it must show: tensor dequantizes to the declared shape, requantizes with
the same shape and block scale, and the relative error stays bounded.

**Measured reference:** on this repo's CPU build (`torch 2.14.1`), a synthetic
FP8 e4m3 round trip gives a mean relative error of **2.25e-02**. The script's
default tolerance is `0.05`, so a real tensor landing near `2e-02` is normal,
not a failure. A result at or near `1e-01` means the block scale is being
applied wrong.

**If this fails, stop and report the output.** Do not proceed to step 2.

---

## Step 2 — Load the model (8 GPUs, ~1–2 hr, mostly download)

306 GiB across 62 shards. Download dominates; start it early.

```bash
# in a tmux/screen session, or with nohup
export HF_HUB_ENABLE_HF_TRANSFER=1
uv pip install hf_transfer accelerate compressed-tensors

uv run python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('zai-org/GLM-5.3-Flash')
print(c.architectures, c.text_config.num_hidden_layers)
"
```

Then a structure-only load — meta device, no weights, no VRAM:

```bash
uv run python -c "
from transformers import AutoModelForImageTextToText, AutoConfig
import torch
m = AutoModelForImageTextToText.from_pretrained(
    'zai-org/GLM-5.3-Flash', device_map='meta', dtype=torch.bfloat16
)
print('class:', type(m).__name__)
print('footprint GB:', m.get_memory_footprint()/1024**3)
"
```

This answers the one question the index cannot: does `transformers` actually
expose the runtime module paths the index implies?

Then run heretic itself, which prints the census on startup:

```bash
cp config.default.toml config.toml
```

Edit `config.toml` — minimum viable:

```toml
model = "zai-org/GLM-5.3-Flash"
device_map = "auto"
dtypes = ["bfloat16", "float16"]
quantization = "none"
offload_outputs_to_cpu = true
n_trials = 2              # smoke test only; default is 100
n_startup_trials = 1
max_response_length = 32
```

```bash
uv run heretic
```

**The census is the pass/fail gate.** Expected exactly:

```
* Transformer model with 45 layers
* Abliterable components:
  * attn.o_proj: 45 modules total
  * mlp.down_proj: 12141 modules total
```

| What you see | What it means |
|---|---|
| `45` / `45` / `12141` | ✅ both patches live |
| `mlp.down_proj: 12099` | ❌ shared-expert patch did not take — `42×288 + 3` means `mlp.shared_experts.down_proj` is not being collected |
| `46` layers | ❌ `get_layers()` is returning the MTP head; the decoder-nesting patch is wrong |
| model fails to load | `transformers` too old — config declares `transformers_version: 5.16.0` |

---

## Step 3 — Ablate and export

Only after step 2 prints the expected census.

```bash
uv run heretic   # with n_trials = 2 first; raise to 100 only if step 2 was clean
```

For the export, expect `MERGE` to be the expensive option — merging a LoRA back
into a 306 GiB FP8 model needs host RAM well above the 1.15 TB on `p4de`. Prefer
`ADAPTER` (export LoRA only) for the first run and merge later if it works.

---

## Known open problems

These are unresolved as of the last commit. Budget for them.

1. **FP8 quantization** — the gate above. Untested.
2. **306 GiB residency** — heretic needs all layers resident to compute a global
   refusal direction. `p-e-w/heretic#135`, closed-wontfix. On `p4de` (8×80 GB =
   640 GB) it fits, but activations for 12,186 hooked modules may not.
3. **Hook scale** — 12,186 forward hooks. Upstream's batching comments anticipate
   MoE experts not firing per prompt, but stability at this scale is unknown.
   Watch for OOM during residual analysis.
4. **`transformers` support** — `Glm5NextForConditionalGeneration` must exist in
   the installed version. If it does not, `--trust-remote-code` will not help
   because heretic only sets `trust_remote_code` for models it has already
   marked trusted, and that happens after load.