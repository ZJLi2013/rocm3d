---
name: rocm-kernels-for-3d
version: 0.5.0
description: |
  Single-node inference kernel replacement for 3D generation / reconstruction /
  Video / World Model / VLA / DiT models on AMD GPUs. Optimizes the kernel
  families that actually dominate the profile: attention, GEMM, norm, RoPE,
  conv2d/conv3d/depthwise/grouped conv, sparse conv, collective comm,
  and fused elementwise/activation paths. Uses AITER/ATOM
  wrappers where they fit, and routes non-AITER kernels to family-specific
  recipes or TODOs. Run `rocm-perf-analysis` first to pick targets.
  Out-of-scope: training, multi-node, LLM serving (use `inference-skill`).
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm Kernels for 3D / Video / WM / VLA

The single rule this skill teaches:

> **Optimize the kernels that dominate the measured profile. Do not assume
> attention is the target.**

For attention/GEMM/norm/quant/collective-comm paths, ATOM/AITER provide
concrete wrapper or kernel references. For conv and sparse conv, use the family
routing below. For `copy_`, layout conversion, and tensor indexing
`scatter/gather`, this skill only documents why they are **not** AITER kernel
replacement targets; route them to model-side graph/dataflow cleanup. Dedicated
renderers such as `gsplat` and `nvdiffrast` are not optimized here; route those
to `rocm-lib-compat` for backend verification and upstream/library follow-up.

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
| Model family | 3D generation (TRELLIS, Hunyuan3D, PartCrafter, SegviGen, TokenGS) · 3D reconstruction (dust3r, fast3r, vggt, FoundationStereo, DA3, video_to_world) · World Model (Matrix-Game, Lyra-2, SANA-WM, Wan2.1) · VLA (SmolVLA, π0-style action heads) · DiT video (Wan2.1, HunyuanVideo, CogVideoX, Mochi) · Point cloud backbones (PointTransformerV3, GraspNet) | These are the model families with no native ROCm framework (vLLM / SGLang don't cover them) and the ones that need direct kernel-family optimization after profiling |
| Workload | **Inference only** (forward + autoregressive decode + iterative denoise) | Training kernels (backward, optimizer) have separate concerns (gradient accumulation, FSDP/ZeRO, optimizer state quant) outside this skill |
| System scale | **Single GPU primary; up to single node** (TP via `ColumnParallel/RowParallel`, SP via xDIT-style, EP via FusedMoE — all single-node) | Cross-node SP / cross-node TP / pipeline-parallel are inference-stack concerns (vLLM, SGLang); single-node max scales to MI300X × 8 |
| Data types | **BF16** (default) · **FP8** (per-tensor / per-token / per-1×128 block scale) · **INT8** (per-tensor / per-token) · **FP4 / MXFP4** (per-1×32 block scale, gfx950+) · **INT4** (where ATOM has an example) | AITER/ATOM have reference paths for these dtypes where applicable; verify exact op + dtype coverage per workload (`atom/model_ops/linear.py` quant path, `atom/models/deepseek_v2.py` FP8 MoE example) |
| Kernel category | **Attention** · **GEMM / quant GEMM** · **Norm / RoPE / activation** · **MoE** · **collective comm** · **conv2d / conv3d / depthwise / grouped conv** · **sparse conv** | Only families with a plausible kernel owner or concrete next step are optimization targets here. `copy_` / layout conversion / tensor indexing are reported by perf-analysis but are model/dataflow cleanup, not generic kernel replacement. |

### 0.2 Out-of-scope (routed elsewhere — do not patch here)

| Out-of-scope topic | Route to |
|---|---|
| LLM serving optimization (any model already running in vLLM / SGLang) | [`AMD-AIM/inference-skill`](https://github.com/AMD-AIM/inference-skill) — has `VLLM_ROCM_USE_AITER=1` framework path |
| Training (backward, FSDP, optimizer state quant) | not covered by any skill currently; use AITER training kernels directly (`mha_bwd`, `fmha_v3_bwd` etc.) per upstream docs |
| Multi-node parallelism | upstream serving frameworks; if a research model needs it, port to vLLM/SGLang first |
| Triton / HIP kernel from scratch when no maintained ROCm path exists | [`cookbook.md`](cookbook.md) §7 documents the escape hatch. Prefer existing maintained paths first (`spconv_rocm`, AITER/ATOM, torch/ROCm backends) before writing a kernel. |
| Library-specific rasterizers (`gsplat`, `nvdiffrast`, pytorch3d raster ops) | `rocm-lib-compat` verifies the correct ROCm backend and routes to the owning library/upstream. This skill does not hand-optimize those renderer kernels. |
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
| **Dense video/VAE conv** | 🔵 SANA-WM surfaced this as top bottleneck; TODO recipe covers phase split + layout diagnosis, not MIOpen tuning | — | — | — | — |
| **Sparse conv / voxel conv** | 🔵 compatibility validated via `spconv_rocm`; perf recipe TODO lives here | — | — | — | — |
| **Collective comm (all-reduce / all-gather / reduce-scatter)** | 🔵 AITER has `ops/triton/comms/{all_gather,reduce_scatter}` and fused reduce-scatter + RMSNorm + quant + all-gather; use only for distributed TP/EP/SP paths | — | — | — | — |
| **Fused activation / elementwise** | 🔵 AITER has fused SiLU-mul and activation+FP8/FP4 quant paths; generic add/mul/copy is not covered | n/a | n/a | n/a | n/a |

**Reading the matrix**:
- ✅ = one experiment doc + perf delta + cos_max ≥ 0.99 + skill-gap log row exists
- 🔵 = AITER has the kernel + ATOM has a reference + cookbook has a wrapper, but no agent-driven closed-loop run yet
- 🔵 compatibility validated = backend import/smoke works, but this skill has not completed a perf optimization loop
- — = combination doesn't exist (e.g. element-wise has no dtype-specific kernel, INT4 doesn't apply to MoE on gfx942)

### 0.4 Kernel-coverage rationale (why all kernel families stay in scope)

3D / Video / WM / VLA / DiT / point-cloud workloads share transformer kernels
with LLMs, but they also surface conv, sparse, collective comm, and fused
activation bottlenecks. `rocm-perf-analysis` decides the target; this skill
must not narrow the action space to attention/GEMM.

| Kernel family | Example signal | Owner / next action |
|---|---|---|
| Attention (`attn_fwd`, `sdpa`, `flash_attn`) | Wan2.1 C2: attention swap helped when attention was top target | AITER/ATOM wrapper (§2-3) |
| Dense video/VAE conv (`conv3d`, grouped/depthwise conv, `Im3d2Col`) | SANA-WM P1: conv dominated, attention only ~3.7% | Conv/layout TODO (§3.1); do not stop at "not AITER" |
| Sparse conv (`spconv*`, indice pairs) | PointTransformerV3 / voxel backbones | `spconv_rocm` + sparse-conv TODO (§3.1) |
| Collective comm (`all_reduce`, `all_gather`, `reduce_scatter`) | TP/EP/SP distributed inference | AITER comm / Iris or framework comm path (§3.1) |
| Fused activation / elementwise (`silu`, `gelu`, `swiglu`, activation+quant) | MLP / MoE / quant paths | AITER fused activation or quant-fusion path (§3.1) |
| GEMM / linear | rocBLAS / hipBLASLt / AITER GEMM | AITER/ATOM wrapper (§2-3) |

External library kernels (`gsplat`, `nvdiffrast`, pytorch3d raster ops) should
still appear in perf reports, but they route to `rocm-lib-compat` backend
sanity checks and the owning upstream library, not to this generic kernel skill.
Generic `copy_`, layout conversion, and tensor indexing `scatter/gather` should
also appear in perf reports, but without a concrete kernel owner they route to
model-side cleanup rather than this skill.

If a kernel family is >2% of runtime and has no recipe here, that is a skill
gap. Record it as a TODO or upstream issue; do not silently pivot back to
attention/GEMM just because those recipes are better documented.

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
makes a CUDA/HIP fallback trivial.

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

### 3.1 Non-AITER kernel families (first-class targets)

These are not second-class "compatibility" problems. If they top the profile,
they are the optimization target.

| Kernel family | First actions | Current status |
|---|---|---|
| Dense video/VAE conv (`conv2d`, `conv3d`, depthwise/grouped conv, `Im3d2Col`) | Split denoise vs VAE encode/decode; inspect layout churn around conv; try `channels_last` / `channels_last_3d` only with correctness + latency checks; try `torch.compile` on stable VAE blocks; record CK/MIOpen-dispatched kernel names but do not rely on old MIOpen tuning as the main plan | TODO recipe, motivated by SANA-WM P1 |
| Sparse conv / voxel conv (`spconv*`, indice pairs) | Confirm `spconv_rocm` backend from `rocm-lib-compat`; profile indice generation vs GEMM vs gather/scatter; compare against dense fallback only for sanity | Validated compatibility, perf TODO |
| Collective comm (`all_reduce`, `all_gather`, `reduce_scatter`) | Only for distributed TP/EP/SP inference. Check whether framework comm is NCCL/RCCL/custom, then evaluate AITER/Iris comm kernels or fused comm+norm+quant only if topology and tensor shapes match | AITER has concrete comm kernels; not related to tensor indexing `scatter/gather` |
| Fused activation / elementwise (`silu`, `gelu`, `swiglu`, activation+quant) | If activation or activation+quant is a top op, map to AITER fused SiLU-mul / activation+FP8/FP4 quant / fused MoE paths; generic `aten::add`/`aten::mul` should first try graph fusion, not a hand-written kernel | AITER covers specific fused patterns, not arbitrary elementwise |

When this table says TODO, the correct agent behavior is to document the gap
and run the smallest diagnostic experiment, not to declare the perf loop done.

---

## 4. Standard recipe

### 4.0 KernelPilot-style optimization loop

This skill should use a KernelPilot-like loop structure for serious kernel
optimization runs, while replacing CUDA/Nsight-specific pieces with the ROCm
toolchain. KernelPilot is a peer-level reference for loop hygiene: standalone
workspace, evidence ledgers, prior-art lookup, profiling only when it changes
the next edit, and review-gated iterations.

Adopt the structure, not the CUDA assumptions:

| KernelPilot idea | ROCm/3D adaptation |
|---|---|
| Standalone candidate workspace | Create a small harness outside the model repo when testing a new HIP/Triton/AITER wrapper; only port back after correctness + latency are proven |
| Prior-art lookup | Check AITER, ATOM, ROCm library forks, PyTorch ROCm issues, and relevant upstream PRs before writing a new kernel |
| Profiling evidence | Use `rocm-perf-analysis` / TraceLens / rocprof evidence instead of Nsight Compute |
| Benchmark ledger | Record shapes, dtype, baseline latency, candidate latency, correctness metric, and regression notes per iteration |
| Review-gated iteration | Do not declare victory from one fast microbench; require correctness, workload-representative latency, and no downstream pipeline regression |
| Upstream placement decision | Decide whether the fix belongs in the model repo, a ROCm library fork, AITER/ATOM, or an isolated experiment |

For each optimization target, keep a minimal ledger:

```markdown
| Iter | Candidate | Shape / dtype | Correctness | Latency delta | Evidence | Decision |
|---|---|---|---|---|---|---|
| 0 | baseline | M,N,K / bf16 | reference | 1.00x | trace/profile path | keep |
| 1 | <wrapper/kernel> | same | cos=... / max_diff=... | ...x | bench artifact | accept/reject |
```

Use this loop when an optimization may take multiple edits or when a kernel
swap could silently change model behavior. For one-line backend routing fixes,
the normal standard recipe below is enough.

### Step 1 — Identify targets

Run [`rocm-perf-analysis`](../rocm-perf-analysis/SKILL.md). It produces the
ranked list of kernels worth optimizing (`pct × (1 − roofline_efficiency)`).
The top items are the targets for steps 2–7. If the top target is conv, sparse,
collective comm, or fused activation, use §3.1 before any attention/GEMM work.
If the top target is generic `copy_`, layout conversion, tensor indexing
`scatter/gather`, `gsplat`, `nvdiffrast`, or pytorch3d raster, route away from
this skill to the owner named by `rocm-perf-analysis`.

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

### Step 5.1 — Combine `torch.compile` with targeted dtype conversion

Use this path when profiling shows **GEMM/linear is hot**, standalone
BF16/FP16 GEMM is faster, but global autocast regresses e2e because of
cast/layout/BMM churn. GroundingDINO GDINO-15/16/17 is the reference failure
mode: transformer compile helped, but ad hoc low-precision monkey patches
failed before producing a valid e2e result.

Default order:

1. **Compile first, dtype second.** Establish an FP32 `torch.compile` baseline
   on the largest stable block (`transformer`, encoder block, DiT block). Record
   latency, graph breaks, detections/actions, and output drift.
2. **Localize graph breaks before dtype work.** If Dynamo reports scalar
   extraction or dynamic-shape breaks, try a no-code diagnostic such as
   `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1`. If the message turns into
   "could not extract specialized integer" around `torch.linspace(H_)`,
   slicing, or shape loops, fix/capture/cache the shape logic before adding
   dtype changes.
3. **Convert modules, not bound methods.** Replace a submodule with a real
   `nn.Module`, or add a small wrapper whose tensors are registered as
   `nn.Parameter(requires_grad=False)` or `register_buffer(...)`. Do not attach
   raw tensors or child modules that reference their parent layer.
4. **Keep dtype boundaries explicit.** Cast inputs once at the module boundary,
   run the hot linear/FFN in BF16/FP16, then cast back once before FP32 residual
   add / LayerNorm if the surrounding block stays FP32.
5. **Benchmark the combination, not only the GEMM.** A low-precision `addmm`
   microbench can be 2-5x faster while e2e is flat or slower because of casts,
   layout copies, BMM, or graph breaks.

Promotion checklist:

| Check | Pass condition |
|---|---|
| Correctness | same user-visible result, plus logits/actions close enough for the task |
| Device registration | no CPU constants in compiled graph; weights/scales are parameters or buffers on the target device |
| Graph breaks | count and source are no worse than FP32 compile baseline |
| Latency | beats FP32 compile baseline, not just eager FP32 |
| Fallback | can disable the dtype wrapper and run the original PyTorch path |

Avoid these patterns:

- **Global autocast as the first fix** — it often expands cast/layout work and
  can make BMM or custom ops slower.
- **Monkey-patching `layer.forward_ffn` with a closure over raw tensors** —
  Dynamo may treat the tensors as CPU constants or fail fake-tensor device
  propagation.
- **Attaching a wrapper module that stores `self.layer = parent_layer`** — this
  creates module traversal recursion.
- **Leaving dynamic shape construction inside the compiled block** when the
  shapes are fixed for inference. Cache reference grids/proposals or compute
  Python ints outside the compiled region.

### Step 6 — Validate numerical correctness with `cos_max`

For every kernel swap, dump `(layer_in, layer_out)` for the wrapped op and
compare against the PyTorch baseline (`cos_max` + bisect harness →
[`cookbook.md`](cookbook.md) Part 5):

- `cos > 0.9999` → bit-equal range, pass.
- `cos 0.99 – 0.9999` → numerical drift, acceptable.
- `cos < 0.99` → bug. Bisect using the ATOM `dump-bisect-debug` skill (see §7).

Non-negotiable for VLA: tiny per-op drift can flip the discrete action token
and silently tank success rate.

### Step 7 — Re-profile and record the swap

Re-run `rocm-perf-analysis`. Capture each AITER swap as a row in the
report's "kernel-swap deltas" table (per-stage latency / throughput / VRAM).
File any new gap discovered against the upstream from §6.

---

## 5. Critical pitfalls

Eight failure modes that cost the most time. Each maps to a pattern from §2
or GDINO-15/16/17.

1. **Calling AITER raw from `models/*.py`** — breaks TP, quant, CK shuffle.
   Always wrap (Pattern A).
2. **Skipping `process_weights_after_loading()`** — silent wrong CK GEMM
   outputs, not a crash. Verify with `cos_max` (Pattern B + Step 6).
3. **Editing a `@support_torch_compile`-decorated module post-decoration** —
   Dynamo breaks. Instrument at the call site (Pattern D).
4. **`fork` instead of `spawn` for multiprocessing** — CUDA/HIP runtime re-init crash.
5. **Mixing ROCm versions in one env** — AITER CK (7.x) + xformers / gsplat
   / pytorch3d (6.4 wheels) doesn't link. Pick one per env (§1).
6. **`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` missing** at install or
   runtime — FA2 Triton import fails on ROCm.
7. **Targeted dtype monkey patches using raw tensors** — Dynamo/fake tensors can
   see CPU constants (`cuda:0` input vs CPU weight) or fail device propagation.
   Use real module parameters/buffers and move the wrapper to device.
8. **Treating `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` as a fix** — it is a
   diagnostic. If breaks become dynamic `torch.linspace(H_)` / slice-shape
   errors, make shapes static/cached or move that logic outside compile.

---

## 6. Upstream gap routing

Every kernel gap discovered should land as an issue or PR upstream.

| Gap type | Upstream / owner |
|---|---|
| AITER missing op / wrong shape / perf regression | https://github.com/ROCm/aiter |
| ATOM-side wrapper bug or missing reference pattern | https://github.com/ROCm/ATOM |
| flash-attn ROCm bug | https://github.com/ROCm/flash-attention |
| Triton kernel HIP incompat | upstream of the kernel (fla-org, mamba, etc.) |
| Mamba / linear-attn ROCm | https://github.com/state-spaces/mamba, https://github.com/fla-org/flash-linear-attention |
| Dense conv / video VAE perf gap | this skill's conv TODO until a maintained CK/Triton/HIP path exists |
| Sparse conv perf gap | `spconv_rocm` issue / PR, plus this skill's sparse-conv TODO |
| gsplat / nvdiffrast perf gap | `rocm-lib-compat` backend sanity check, then `amd_gsplat` / ROCm nvdiffrast fork or owning repo issue |
| Layout/copy regression | model repo first; remove format churn before filing kernel issues |
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
| Consult prior-art for kernel-loop structure or CUDA/Triton implementation ideas | [KernelPilot](https://github.com/BBuf/kernel-pilot) / KernelWiki, adapted to ROCm evidence and APIs |
| Full LLM serving optimization (vLLM / SGLang) | [AMD-AIM/inference-skill](https://github.com/AMD-AIM/inference-skill) — use directly; out of scope for this skill |

Three-skill pipeline:

```
[ rocm-lib-compat ]   →   [ rocm-perf-analysis ]   →   [ rocm-kernels-for-3d ]
 (make it run)           (measure + classify)           (optimize selected family)
```

