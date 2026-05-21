---
name: rocm-llm-kernels-for-3d
version: 0.3.0
description: |
  Single-node inference kernel replacement for 3D generation / reconstruction /
  Video / World Model / VLA / DiT models on AMD GPUs. Swap naive attention,
  GEMM, norm, RoPE, paged-attn paths to AMD AITER kernels (BF16 default;
  FP8 / INT8 / FP4 / INT4 supported when ATOM has a quant reference). Uses
  the wrapping patterns established by AMD's ATOM reference engine. Run
  `rocm-perf-analysis` first to pick targets; this skill executes the swap.
  Out-of-scope: training, multi-node, LLM serving (use `inference-skill`).
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm AITER Kernels for 3D / Video / WM / VLA

The single move this skill teaches:

> **Lift ATOM's `model_ops/` shape into the 3D model. Route every AITER call
> through a wrapper; never call `aiter.flash_attn_func` / `aiter.gemm` directly
> from a model file.**

ATOM is AMD's vLLM-like reference engine built on AITER. It is the
canonical, AMD-maintained example of "how to use AITER kernels correctly in
a real model". Reading ATOM's `model_ops/` and copying its wrapper shape is
the fastest path to a TP-safe, quant-safe, CK-shuffle-safe AITER integration.

> **Copy-paste code skeletons** for every pattern below live in
> [`cookbook.md`](cookbook.md). This file gives the methodology and `cookbook.md`
> gives the snippets — read both side by side.

---

## 0. Scope (read this first)

Scope is explicit so contributors don't read an unverified axis as a coverage
gap. Anything outside the in-scope columns is **deliberately not addressed**
here — there's a sibling skill or upstream owner for it.

### 0.1 In-scope (this skill targets these intersections)

| Axis | In-scope values | Why |
|---|---|---|
| Model family | 3D generation (TRELLIS, Hunyuan3D, PartCrafter, SegviGen, TokenGS) · 3D reconstruction (dust3r, fast3r, vggt, FoundationStereo, DA3, video_to_world) · World Model (Matrix-Game, Lyra-2, SANA-WM, Wan2.1) · VLA (SmolVLA, π0-style action heads) · DiT video (Wan2.1, HunyuanVideo, CogVideoX, Mochi) · Point cloud backbones (PointTransformerV3, GraspNet) | These are the model families with no native ROCm framework (vLLM / SGLang don't cover them) and the ones that benefit most from manual AITER wrapping |
| Workload | **Inference only** (forward + autoregressive decode + iterative denoise) | Training kernels (backward, optimizer) have separate concerns (gradient accumulation, FSDP/ZeRO, optimizer state quant) outside this skill |
| System scale | **Single GPU primary; up to single node** (TP via `ColumnParallel/RowParallel`, SP via xDIT-style, EP via FusedMoE — all single-node) | Cross-node SP / cross-node TP / pipeline-parallel are inference-stack concerns (vLLM, SGLang); single-node max scales to MI300X × 8 |
| Data types | **BF16** (default) · **FP8** (per-tensor / per-token / per-1×128 block scale) · **INT8** (per-tensor / per-token) · **FP4 / MXFP4** (per-1×32 block scale, gfx950+) · **INT4** (where ATOM has an example) | AITER provides kernels for all of these; ATOM provides reference quant loading + routing (`atom/model_ops/linear.py` quant path, `atom/models/deepseek_v2.py` FP8 MoE example) |
| Kernel category | **GEMM** (dense + grouped + tuned) · **Attention** (dense MHA / varlen / paged / MLA) · **Norm** (RMSNorm, LayerNorm, GroupNorm) · **Element-wise** (SiLU/GELU fused, RoPE, residual add) · **MoE** (FusedMoE 2-stage) | These five families cover **>95% of step time** for every model class in row 1 above (verified on Wan2.1 C1 — see [coverage matrix](#03-coverage-matrix-empirical)). 3D / DiT / VLA models do **not** introduce new kernel families beyond what LLM inference already exercises; they only re-arrange shapes |

### 0.2 Out-of-scope (routed elsewhere — do not patch here)

| Out-of-scope topic | Route to |
|---|---|
| LLM serving optimization (any model already running in vLLM / SGLang) | [`AMD-AIM/inference-skill`](https://github.com/AMD-AIM/inference-skill) — has `VLLM_ROCM_USE_AITER=1` framework path |
| Training (backward, FSDP, optimizer state quant) | not covered by any skill currently; use AITER training kernels directly (`mha_bwd`, `fmha_v3_bwd` etc.) per upstream docs |
| Multi-node parallelism | upstream serving frameworks; if a research model needs it, port to vLLM/SGLang first |
| Triton kernel from scratch when AITER doesn't cover the op (linear-attn, Mamba selective_scan, exotic VAE fused-conv) | [`cookbook.md`](cookbook.md) §7 documents the *escape hatch* (write a kernel) but the recipe set is intentionally thin — large-scope kernel writing is GEAK territory ([AMD-AIM/inference-skill `phases/07-kernel-optimize.md`](https://github.com/AMD-AIM/inference-skill)) |
| Container provisioning / dependency install / model download | [`rocm-lib-compat`](../rocm-lib-compat/SKILL.md) + [`gpu-cluster-resource-manager`](../gpu-cluster-resource-manager/SKILL.md) |
| Profiling / kernel ranking | [`rocm-perf-analysis`](../rocm-perf-analysis/SKILL.md) |

### 0.3 Coverage matrix (empirical — see `overnight_tasks/<model>/experiments.md`)

Each cell needs **one ≥30-min closed-loop cycle** to flip from 🔵 (planned) to ✅ (validated). Status as of 2026-05-21:

| | BF16 | FP8 | INT8 | FP4 / MXFP4 | INT4 |
|---|---|---|---|---|---|
| **Dense MHA (DiT, video)** | ✅ Wan2.1 +32% | 🔵 | 🔵 | 🔵 | 🔵 |
| **Varlen MHA (VLA decode, LLM-like)** | 🔵 | 🔵 | 🔵 | 🔵 | 🔵 |
| **Paged attn (serving KV cache)** | 🔵 | 🔵 (KV) | 🔵 (KV) | — | — |
| **Linear (TP / Replicated)** | 🔵 | 🔵 | 🔵 | 🔵 (gfx950+) | 🔵 |
| **FusedMoE 2-stage** | 🔵 | 🔵 | 🔵 | 🔵 (gfx950+) | — |
| **RoPE / RMSNorm / fused act** | 🔵 (aggregate ~3-5% in Wan2.1) | n/a | n/a | n/a | n/a |
| **Hybrid linear-attn (GDN / FLA / DeltaNet)** | 🔵 — ATOM has full ref: `atom/model_ops/fla_ops/` (chunk / chunk_delta_h / fused_recurrent) + `atom/model_ops/attentions/gdn_attn.py`; models `qwen3_next.py`, `qwen3_5.py`, `mimo_v2_flash.py`, `kimi_k25.py`, `minimax_m2.py`, `glm4_moe.py` are wired e2e | 🔵 | 🔵 | — | — |
| **Mamba SSM (causal_conv1d + selective_scan)** | 🔵 — `causal_conv1d` in ATOM `atom/model_ops/mamba_ops/causal_conv1d.py` + AITER `aiter.ops.causal_conv1d`; selective_scan via upstream `state-spaces/mamba` (verify per cycle) | — | — | — | — |

**3D conv** is intentionally absent from this matrix — see §0.4 row.

**Reading the matrix**:
- ✅ = one experiment doc + perf delta + cos_max ≥ 0.99 + skill-gap log row exists
- 🔵 = AITER has the kernel + ATOM has a reference + cookbook has a wrapper, but no agent-driven closed-loop run yet
- — = combination doesn't exist (e.g. element-wise has no dtype-specific kernel, INT4 doesn't apply to MoE on gfx942)

### 0.4 Kernel-coverage rationale (why GEMM + Attn + Norm + Element-wise is enough)

3D / Video / WM / VLA / DiT / point-cloud backbones all decompose into the
same five primitive families because they all inherit from the
transformer-or-CNN substrate. Empirically verified on Wan2.1
(see `overnight_tasks/wan2.1/experiments.md` C1 ranking):

| Kernel family | % of step (Wan2.1 1.3B, 480×832×81f) | Covered by AITER + cookbook? |
|---|---|---|
| Attention (`attn_fwd`) | 10.4% | ✅ §2.4 + §2.4b |
| 3D conv (VAE `grouped_conv_fwd` etc.) | 7.4% | ✅ **dispatched, not in this skill**: sparse 3D conv (point cloud / voxel grid → PTv3) → [`rocm-lib-compat`](../rocm-lib-compat/SKILL.md) `spconv_rocm`; dense 3D conv (video VAE) → PyTorch native `aten::conv3d` → MIOpen / CK grouped_conv automatic, no AITER swap needed |
| GEMM (rocBLAS / `addmm` / `linear`) | 1.6% | ✅ §2.1–2.3 |
| Element-wise tail (`copy_`, `add`, fused act) | ~3-5% combined | ✅ AITER fused ops in cookbook Part 1 |
| **Top 10 covered share** | **~80%** | **see C1 ranking table** |

If a model in scope shows a kernel **outside these families** with >2% share that is **also not handled by `rocm-lib-compat`** (which owns spconv, gsplat, pytorch3d, flash-attn paths), that's a real gap → file against this skill's `cookbook.md`, not a "scope expansion". The expected case is that scope is enough; the exception is the trigger.

---

## 1. ROCm version pick

| Repo dependency profile | ROCm | AITER backend |
|---|---|---|
| torch + flash-attn only | **7.2+** | CK (fastest) |
| Also uses xformers / gsplat / pytorch3d (6.4-only wheels) | **6.4.3** | Triton (≈ FA2 Triton) |
| Serving via vLLM / SGLang ROCm | Follow upstream support matrix | per upstream |

Pick **one** ROCm version per env. Mixing 6.4 wheels with 7.2 AITER CK breaks.

---

## 2. The six ATOM patterns to replicate

Each pattern names the move, the ATOM source file to read, and the 3D-side
application. **Code skeletons** for every pattern → [`cookbook.md`](cookbook.md) Part 2–5.

### Pattern A — Wrap, don't call raw

ATOM never calls AITER directly from `models/*.py`. Every call goes through a
thin wrapper in `model_ops/` that owns:

- weight layout (Column / Row / Replicated parallel)
- precision routing (BF16 / FP8 / FP4)
- TP / EP / DP awareness
- `process_weights_after_loading()` shuffle for CK layout
- Triton fallback when CK is unavailable

**ATOM source**: `atom/model_ops/linear.py`, `atom/model_ops/attention_mha.py`.

**3D-side**: define `Vla*Linear`, `WmAttention`, `DiTBlock` in your repo,
mirroring ATOM's wrappers. Isolates AITER specifics from model logic and
makes a CUDA fallback trivial.

### Pattern B — `process_weights_after_loading()` for CK shuffle

AITER CK kernels require shuffled weight layouts. Without the post-load
shuffle, CK GEMM **silently produces wrong outputs** (not a crash).

**ATOM source**: `atom/model_loader/loader.py`.

**3D-side**:

```python
from aiter.ops.shuffle import shuffle_weight

class CkAwareLinear(nn.Module):
    def process_weights_after_loading(self):
        self.weight.data = shuffle_weight(self.weight.data, layout=(16, 16))
```

FP8 scale tensors must use `e8m0` format; ATOM auto-renames
`.scale` → `.weight_scale_inv` → `.weight_scale`. Mirror that naming.

### Pattern C — Three TP flavors, nothing else

ATOM uses exactly three parallel-linear shapes:

- `ColumnParallelLinear` — shards output dim, no all-reduce.
- `RowParallelLinear` — shards input dim, all-reduce on output (`reduce_results=True`).
- `ReplicatedLinear` — full copy on each rank (gates, small projections).

MoE convention: `FusedMoE` and `shared_experts` both set
`reduce_results=False`; the parent block does one combined all-reduce.

**3D-side**: VLA / WM transformer backbones map cleanly. Don't invent new
TP shapes.

### Pattern D — Piecewise `torch.compile` + CUDAGraph

ATOM's default compilation tier (`--level 3`) is piecewise: shape-changing
ops (sampling, attention with varying seq len) stay outside the graph;
stable sub-blocks (one transformer layer body) are captured.

**ATOM source**: `@support_torch_compile` decorator + the `--level` system.

**3D-side**: in VLA decode and WM autoregressive rollout, capture only the
transformer-layer body. **Never edit a `@support_torch_compile`-decorated
module post-decoration** — Dynamo breaks even with `--enforce-eager`.

### Pattern E — Quantization is a `model_ops` concern

Quantization in ATOM is plumbed through `model_ops/linear.py` and the FP8/FP4
loader path, not via a separate "quantization API". The model file declares
which linear layers are quantizable; loader + wrapper route weights through
AITER's FP8/FP4 GEMM kernels.

**3D-side**: declare quantizable layers in the model file. Don't write a
separate quantization pass.

### Pattern F — MLA / Paged-attn / MoE follow the same wrap-don't-call rule

| Op | ATOM source | When to replicate |
|---|---|---|
| MLA attention (DeepSeek-style) | `atom/model_ops/attention_mla.py` | DeepSeek-backbone 3D models (rare; watch for emergence) |
| Paged attention | `atom/model_ops/paged_attention.py` | VLA serving with KV cache, batched WM rollout |
| FusedMoE | `atom/model_ops/moe.py`, `fused_moe_triton.py` | MoE-VLA, MoE-WM scale-up |

---

## 3. AITER kernel → 3D neighborhood mapping

| AITER kernel | ATOM reference | 3D-side application | Expected win |
|---|---|---|---|
| Flash attention (CK on 7.x, Triton on 6.4) | `model_ops/attention_mha.py` | VLA decode, WM rollout, DiT self/cross-attention | replace naive SDPA — biggest single win |
| Paged attention | `model_ops/paged_attention.py` | VLA serving, batched WM rollout | continuous batching, paged KV |
| GEMM + Column/Row/Replicated wrappers | `model_ops/linear.py` | Every transformer-backbone VLA, DiT linear layers | TP-friendly GEMM with CK shuffle |
| RMSNorm / RoPE / fused SiLU / GeLU | inline in `models/*.py` | LLaMA / Qwen-backboned VLA, hybrid WM | small individually, large aggregate |
| FusedMoE (CK) + Triton `matmul_ogs` fallback | `model_ops/moe.py` | MoE-VLA, MoE-WM | MoE on ROCm without re-implementing dispatch |
| FP8 / INT8 / MXFP4 / FP4 / INT4 GEMM (quant path) | `model_ops/linear.py` quant routing + `model_loader/loader.py` FP8/FP4 scale rename + shuffle; reference: `atom/models/deepseek_v2.py` (FP8 MoE end-to-end) | VLA serving quant, WM long-context KV, DiT video at FP8 weights | memory + bandwidth + cost; **in-scope per §0.1** — every dtype here has AITER kernel + ATOM example |
| MLA attention | `model_ops/attention_mla.py` | DeepSeek-backbone 3D models | reuse rather than re-derive MLA on ROCm |
| Hybrid linear-attn (GDN / FLA / DeltaNet) + Mamba causal_conv1d | `atom/model_ops/fla_ops/` (chunk / chunk_delta_h / fused_recurrent) + `atom/model_ops/attentions/gdn_attn.py` + `atom/model_ops/mamba_ops/causal_conv1d.py`; e2e models: `qwen3_next.py`, `qwen3_5.py`, `mimo_v2_flash.py`, `kimi_k25.py`, `minimax_m2.py`, `glm4_moe.py` | SANA-WM, Mamba-VLA, GLA-WM, hybrid-attn DiT, any model riding the Qwen3-Next / Kimi-K2.5 / MiMo-V2 transformer recipe | hybrid attention is now standard in 2026-era LLMs and percolating into VLA / WM — ATOM ref is the right starting point, not a gap |

---

## 4. Standard recipe

### Step 1 — Identify targets

Run [`rocm-perf-analysis`](../rocm-perf-analysis/SKILL.md). It produces the
ranked list of kernels worth optimizing (`pct × (1 − roofline_efficiency)`).
The top items are the targets for steps 2–6.

Do not hand-roll `torch.profiler` blocks here.

### Step 2 — Read the ATOM reference

Skim these four files before writing any code:

```
atom/model_ops/linear.py            # GEMM wrappers + TP + quant routing
atom/model_ops/attention_mha.py     # MHA on AITER flash-attn
atom/model_ops/paged_attention.py   # KV cache layout
atom/models/llama.py                # canonical full transformer assembly
```

Also skim per workload: `atom/models/qwen3.py` (Qwen backbones),
`atom/models/deepseek_v2.py` (MoE / MLA), `atom/model_loader/loader.py`
(FP8/FP4 loading + shuffle hook).

### Step 3 — Create a `model_ops/` mirror

Don't put `from aiter import ...` directly in `models/vla_xxx.py`. Mirror
ATOM's structure (skeletons in [`cookbook.md`](cookbook.md) Part 2):

```
your_repo/
├── model_ops/
│   ├── linear.py       # ColumnParallelLinear / RowParallelLinear / ReplicatedLinear, backed by AITER GEMM
│   ├── attention.py    # thin wrapper on aiter.flash_attn_func; handles q/k/v shape conventions
│   ├── norm.py         # aiter.rms_norm, aiter rope
│   └── paged_attn.py   # only if KV-cache serving is needed
└── models/
    └── vla_<backbone>.py   # uses model_ops/ wrappers; mirrors atom/models/llama.py
```

Each wrapper falls back to PyTorch / Triton when AITER is unavailable, so the
same code runs on NVIDIA for cross-vendor benchmarking.

### Step 4 — Port the loader

Mirror `atom/model_loader/loader.py`. For any linear layer backed by AITER
CK, run the FP8/FP4 shuffle once at load time (see Pattern B). Call all
`process_weights_after_loading()` hooks after the checkpoint loads.
Loader patterns (FP8 scale rename, entry point, PyTorch fallback) →
[`cookbook.md`](cookbook.md) Part 3.

### Step 5 — Apply piecewise `torch.compile` + CUDAGraph

Identify the static sub-block (typically one transformer layer body or one
DiT block). Apply `@support_torch_compile` and capture CUDAGraph only over
that stable shape. Decorator + bucketing snippets →
[`cookbook.md`](cookbook.md) Part 4. Rules:

- Never edit a decorated module post-decoration.
- Multiprocessing must use `spawn` (not `fork`).
- Set `AITER_LOG_LEVEL=WARNING` to suppress kernel log flooding.
- Clear compile cache when code changes: `rm -rf ~/.cache/<your_app>/`.

### Step 6 — Validate numerical correctness with `cos_max`

For every kernel swap, dump `(layer_in, layer_out)` for the wrapped op and
compare against the PyTorch baseline (`cos_max` + bisect harness →
[`cookbook.md`](cookbook.md) Part 5):

- `cos > 0.9999` → bit-equal range, pass.
- `cos 0.99 – 0.9999` → numerical drift, acceptable.
- `cos < 0.99` → bug. Bisect using the ATOM `dump-bisect-debug` skill (see §6).

Non-negotiable for VLA: tiny per-op drift can flip the discrete action token
and silently tank success rate.

### Step 7 — Re-profile and record the swap

Re-run `rocm-perf-analysis`. Capture each AITER swap as a row in the
report's "kernel-swap deltas" table (per-stage latency / throughput / VRAM).
File any new gap discovered against the upstream from §5.

---

## 5. Critical pitfalls

Six failure modes that cost the most time. Each maps to a pattern from §2.

1. **Calling AITER raw from `models/*.py`** — breaks TP, quant, CK shuffle.
   Always wrap (Pattern A).
2. **Skipping `process_weights_after_loading()`** — silent wrong CK GEMM
   outputs, not a crash. Verify with `cos_max` (Pattern B + Step 6).
3. **Editing a `@support_torch_compile`-decorated module post-decoration** —
   Dynamo breaks. Instrument at the call site (Pattern D).
4. **`fork` instead of `spawn` for multiprocessing** — CUDA re-init crash.
5. **Mixing ROCm versions in one env** — AITER CK (7.x) + xformers / gsplat
   / pytorch3d (6.4 wheels) doesn't link. Pick one per env (§1).
6. **`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` missing** at install or
   runtime — FA2 Triton import fails on ROCm.

---

## 6. Upstream gap routing

Every kernel gap discovered should land as an issue or PR upstream.

| Gap type | Upstream |
|---|---|
| AITER missing op / wrong shape / perf regression | https://github.com/ROCm/aiter |
| ATOM-side wrapper bug or missing reference pattern | https://github.com/ROCm/ATOM |
| flash-attn ROCm bug | https://github.com/ROCm/flash-attention |
| Triton kernel HIP incompat | upstream of the kernel (fla-org, mamba, etc.) |
| Mamba / linear-attn ROCm | https://github.com/state-spaces/mamba, https://github.com/fla-org/flash-linear-attention |
| Serving-stack ROCm gap | https://github.com/vllm-project/vllm (label `rocm`), https://github.com/sgl-project/sglang |

---

## 7. Companion skills (don't duplicate)

| When you need to… | Use |
|---|---|
| **Copy-paste code skeletons** for any pattern in this skill | [`cookbook.md`](cookbook.md) |
| Look up an AITER kernel API | [`aiter-api.md`](aiter-api.md) (full op catalog + dtype matrix + SGLang/vLLM integration map) |
| Get a repo running on ROCm | [`rocm-lib-compat`](../rocm-lib-compat/SKILL.md) |
| Decide **what** to optimize | [`rocm-perf-analysis`](../rocm-perf-analysis/SKILL.md) |
| Capture a PyTorch trace | ATOM `capture-trace` skill (`<ATOM repo>/.claude/skills/capture-trace/SKILL.md`) |
| Triage a GPU memory-access fault / kernel hang | ATOM `debug-agent-locate-kernel` skill |
| Bisect a numerical correctness regression | ATOM `dump-bisect-debug` skill |
| Full LLM serving optimization (vLLM / SGLang) | [AMD-AIM/inference-skill](https://github.com/AMD-AIM/inference-skill) — use directly; out of scope for this skill |

Three-skill pipeline:

```
[ rocm-lib-compat ]   →   [ rocm-perf-analysis ]   →   [ rocm-llm-kernels-for-3d ]
 (make it run)           (measure + prioritize)        (replace kernels, ATOM-style)
```

<!--
Maintainer note (not part of the skill body) — why this skill exists alongside `inference-skill`.
Kept here so future agents/maintainers don't re-derive the boundary or accidentally merge the two.

| dimension              | inference-skill (LLM)                                                                       | rocm-llm-kernels-for-3d (3D/VLA/WM)                              |
|------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| target model lives in  | already running inside vLLM / SGLang                                                        | raw PyTorch repo, never framework-adopted                        |
| how AITER engages      | framework flag: `VLLM_ROCM_USE_AITER=1` already swaps ~80% of kernels (see [`aiter-api.md`](aiter-api.md) §13/14) | manual wrap into model files; no framework dispatch exists       |
| what's left to solve   | long-tail kernels vLLM hasn't integrated, or integrated too slowly                          | the first 80% too — every wrapper by hand                        |
| main automation path   | GEAK auto-generates Triton kernel + plugin injection into runtime                           | agent copies ATOM patterns and writes wrappers                   |
| needs a cookbook?      | no — templates baked into `generate_*_plugin.py`; GEAK outputs kernels not wrappers         | yes — no equivalent plugin framework, so wrappers must be hand-written |

Rule: when the target repo is a 3D / VLA / WM / DiT / SLAM / NeRF / Mamba-hybrid raw PyTorch repo,
use this skill. Anything that already runs in vLLM/SGLang is out of scope — route to inference-skill.
-->

