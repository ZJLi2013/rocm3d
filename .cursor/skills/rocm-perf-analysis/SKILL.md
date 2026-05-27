---
name: rocm-perf-analysis
version: 0.1.0-draft
author: ZJLi2013
description: |
  Performance analysis skill for 3D / Video / World Model / VLA inference on
  AMD ROCm. Provides a phase-split roofline workflow built on AMD's TraceLens
  tool, GPU peak-TFLOPS reference data, optional same-config NVIDIA baselines,
  and kernel bottleneck classification.

  Use when:
    - profiling a VLA / WM / DiT / video diffusion model on MI300X / MI350X
    - producing a roofline report (FLOPS/byte, TFLOPS/s, Pct Roofline) per op
    - splitting a trace into compute-heavy and memory-heavy phases for fair
      bottleneck attribution
    - deciding which kernels are worth optimizing (priority = pct × gap)
    - generating a comparable AMD-vs-NV perf report for a 3D-adjacent model
    - checking whether a ROCm hotspot is a real backend gap or just model cost

  Companion to:
    - `rocm-lib-compat` (compatibility — get the repo running)
    - `rocm-kernels-for-3d` (kernel-family optimization after profiling)

  This skill is the **measurement and prioritization** half of the perf loop.
  Use this skill *first* to decide what to optimize, then use
  `rocm-kernels-for-3d` to actually replace or reshape kernels.
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm Perf Analysis for 3D / VLA / WM

> **Status**: v0.1 draft. Built by lifting the highest-value pieces from
> [AMD-AIM/inference-skill](https://github.com/AMD-AIM/inference-skill)
> (`inferencex-optimize` skill) and re-targeting them at the 3D / Physical-AI
> neighborhood.

## 0. TL;DR

Four things this skill does, and nothing else:

1. **Run TraceLens** on a PyTorch profiler trace to get structured CSV output
   (`GEMM.csv`, `SDPA_fwd.csv`, `kernel_summary.csv`, `unified_perf_summary.csv`).
2. **Phase-split** the trace into compute-heavy and memory-heavy phases, then
   run **per-phase roofline** against the GPU's peak TFLOPS. (Aggregate-only
   roofline hides bottlenecks.)
3. **Rank all kernels** by `priority_score = pct_of_runtime × (1 − roofline_efficiency)`,
   classify each row by kernel family, and produce a short list of optimization
   targets to hand off to `rocm-kernels-for-3d` or `rocm-lib-compat`.
4. **Add a same-config NVIDIA baseline when available** (H100/H200/B200 vs
   MI300X/MI325X/MI350X). This is recommended, not a hard blocker: if the repo
   has no NV reference and no NV node is available, mark the baseline as missing
   and avoid claiming that a ROCm hotspot is "abnormally slow" solely from its
   ROCm runtime share.

Everything else — kernel replacement, AITER wrapping, conv/sparse/comm
optimization — is the job of `rocm-kernels-for-3d`. This skill stops at
producing the classified target list and the perf report.

---

## 1. Why this skill exists

The standard "latency / throughput / VRAM" report is **not enough** for 3D /
VLA / WM models, because:

- A single aggregate number cannot tell whether the bottleneck is GEMM, attention,
  norm, comm, or kernel launch overhead.
- VLA / WM / DiT / video diffusion all have **two distinct phases** with very
  different compute / memory profiles. Aggregate roofline analysis flattens this
  and misleads the optimization decision.
- Without a peak TFLOPS reference, "this op is slow" is opinion. Roofline (`Pct
  Roofline = achieved TFLOPS/s ÷ peak TFLOPS/s for the op's precision`) makes
  it a number.
- Without a same-config NVIDIA baseline, "this ROCm hotspot is a backend gap"
  is also opinion. A top ROCm op may be model-intrinsic cost. Use NV baselines
  when the repo provides them or when an H100/H200/B200 run is practical.

AMD already shipped a battle-tested LLM-side toolchain
([inferencex-optimize](https://github.com/AMD-AIM/inference-skill)) that solves
this for LLM serving. This skill is a **3D-neighborhood port** of that toolchain:
same TraceLens, same GPU peak data, same phase-split idea — but the phase
definitions and the model coverage target VLA / WM / DiT / video, not LLM
prefill/decode.

---

## 2. Tool Stack (external — don't reinvent)

| Tool | Source | Role |
|---|---|---|
| **TraceLens** | [AMD-AGI/TraceLens-internal](https://github.com/AMD-AGI/TraceLens-internal) | Parses PyTorch profiler trace → structured CSV (GEMM, SDPA, kernels, roofline) |
| **TraceLens phase splitter** | `TraceLens-internal/examples/custom_workflows/split_vllm_trace_annotation.py` | Splits a single trace into `prefilldecode_*.json.gz` and `decode_*.json.gz`. Designed for vLLM but works for any annotated trace. |
| **GPU peak TFLOPS table** | [`gpu-specs.md`](gpu-specs.md) | Authoritative per-GPU dense TFLOPS for matrix_fp16 / bf16 / fp8 / fp4 / vector_*; required input for roofline |
| **rocm-smi / rocminfo** | ROCm install | GPU model + arch detection; powers `gpu-specs.md` auto-pick |

**Rule**: this skill calls TraceLens. It does **not** wrap, re-implement, or
extend TraceLens. If TraceLens output is missing a field, file an issue
upstream; do not hack a substitute here.

---

## 3. The Phase-Split Recipe

Six steps. Each step is a copy-paste shell block. No new scripts to write.

### Step 1 — Capture a PyTorch profiler trace

Use whatever harness your model already has. Minimum requirement: profiler
must emit per-rank trace files named `*-rank-N*.json.gz` (or `*-rank-N*.json`).
If your harness emits a single merged trace, name it `merged-<anything>.json.gz`.

```python
import torch.profiler as P
with P.profile(
    activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.CUDA],
    schedule=P.schedule(wait=2, warmup=2, active=8, repeat=1),
    on_trace_ready=P.tensorboard_trace_handler("./profiles/"),
    record_shapes=True, with_stack=False, with_flops=True,
) as prof:
    for step in range(12):
        run_one_step_of_your_model()   # one VLA decode / WM step / DiT step
        prof.step()
```

Annotate the two phases of your model with `torch.profiler.record_function`
so the splitter can find boundaries:

```python
with torch.profiler.record_function("phase_compute_heavy"):
    ...   # image encoding / condition encoding / prefill-equivalent
with torch.profiler.record_function("phase_memory_bound"):
    ...   # action decode / autoregressive rollout / iterative denoise step
```

See §4 for the exact phase definitions per model family.

### Step 2 — Install TraceLens (one-time)

```bash
export PATH="$HOME/.local/bin:$PATH"

if ! command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
    [ -d "$HOME/TraceLens-internal" ] || \
        git clone git@github.com:AMD-AGI/TraceLens-internal.git "$HOME/TraceLens-internal"
    pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
    hash -r
fi

command -v TraceLens_generate_perf_report_pytorch && echo "TraceLens ready"
```

> If clone fails (no GitHub access from the node), the inferencex-skill repo
> bundles a tarball at `inference-skill/skills/inferencex-optimize/scripts/TraceLens-internal.tar.gz`
> — fall back to `tar xzf` + `pip install`.

### Step 3 — Detect GPU and write `gpu_arch.json`

This file is required by the roofline step (it carries peak TFLOPS for the
ops's precision). The full per-GPU table lives in [`gpu-specs.md`](gpu-specs.md);
this block auto-picks one entry and writes it out.

```bash
mkdir -p ./results
python3 - <<'PY'
import json, subprocess, re, os

SPECS = json.load(open(os.path.expanduser("./.cursor/skills/rocm-perf-analysis/gpu-specs.json"))) \
    if os.path.isfile("./.cursor/skills/rocm-perf-analysis/gpu-specs.json") else None

# If gpu-specs.json is not present (we ship gpu-specs.md instead), inline a minimal lookup.
# Full table + sources in gpu-specs.md.
if SPECS is None:
    SPECS = {
      "MI300X": {"arch": "gfx942", "mem_bw_gbps": 5325, "memory_gb": 192,
                 "max_achievable_tflops": {"matrix_bf16": 1307, "matrix_fp16": 1307, "matrix_fp8": 2615,
                                           "vector_bf16": 163, "vector_fp16": 163, "vector_fp32": 163}},
      "MI325X": {"arch": "gfx942", "mem_bw_gbps": 6000, "memory_gb": 256,
                 "max_achievable_tflops": {"matrix_bf16": 1307, "matrix_fp16": 1307, "matrix_fp8": 2615,
                                           "vector_bf16": 163, "vector_fp16": 163, "vector_fp32": 163}},
      "MI350X": {"arch": "gfx950", "mem_bw_gbps": 8000, "memory_gb": 288,
                 "max_achievable_tflops": {"matrix_bf16": 2310, "matrix_fp16": 2307, "matrix_fp8": 4614,
                                           "matrix_fp4": 9228, "vector_bf16": 144, "vector_fp16": 144}},
      "MI355X": {"arch": "gfx950", "mem_bw_gbps": 8000, "memory_gb": 288,
                 "max_achievable_tflops": {"matrix_bf16": 2500, "matrix_fp16": 2500, "matrix_fp8": 5000,
                                           "matrix_fp4": 10000, "vector_bf16": 158, "vector_fp16": 158}},
      "H100":   {"arch": "sm_90",  "mem_bw_gbps": 3350, "memory_gb": 80,
                 "max_achievable_tflops": {"matrix_bf16": 990, "matrix_fp16": 990, "matrix_fp8": 1979,
                                           "vector_bf16": 134, "vector_fp16": 134, "vector_fp32": 67}},
      "H200":   {"arch": "sm_90",  "mem_bw_gbps": 4800, "memory_gb": 141,
                 "max_achievable_tflops": {"matrix_bf16": 990, "matrix_fp16": 990, "matrix_fp8": 1979,
                                           "vector_bf16": 134, "vector_fp16": 134, "vector_fp32": 67}},
      "B200":   {"arch": "sm_100", "mem_bw_gbps": 8000, "memory_gb": 192,
                 "max_achievable_tflops": {"matrix_bf16": 2250, "matrix_fp16": 2250, "matrix_fp8": 4500,
                                           "matrix_fp4": 9000, "vector_bf16": 160, "vector_fp16": 160}},
    }

gpu = None
try:
    out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=10).stdout.lower()
    for k in sorted(SPECS, key=len, reverse=True):
        if k.lower() in out:
            gpu = k; break
except Exception: pass
if not gpu:
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10).stdout.lower()
        if "gfx942" in out: gpu = "MI300X"
        elif "gfx950" in out: gpu = "MI355X"
    except Exception: pass
if not gpu:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.upper()
        for k in ("B200", "H200", "H100"):
            if k in out: gpu = k; break
    except Exception: pass

if gpu:
    spec = {"name": gpu, **SPECS[gpu]}
    json.dump(spec, open("./results/gpu_arch.json", "w"), indent=2)
    print(f"GPU_ARCH_DETECTED={gpu}; wrote results/gpu_arch.json")
else:
    print("WARNING: GPU not detected; roofline data will be missing.")
PY
```

### Step 4 — Phase-split the trace

```bash
SPLIT=$HOME/TraceLens-internal/examples/custom_workflows/split_vllm_trace_annotation.py
TRACE=./profiles/<your-merged-or-rank0-trace>.json.gz

mkdir -p ./results/phase_split
python3 "$SPLIT" "$TRACE" \
    -o ./results/phase_split \
    --find-steady-state --num-steps 32 \
    2>&1 | tail -30

ls -lh ./results/phase_split/
# Output: prefilldecode_*.json.gz  and  decode_*.json.gz
```

The splitter looks for `record_function` annotations to find phase
boundaries. The names it recognizes are vLLM-style (`prefill[...]`,
`decode[...]`). If your annotations use different names (see §4), pass
`--annotation-prefix-prefill <your_prefix>` and similar.

> If the splitter does not work on your trace, **first** try renaming your
> annotations to `prefill[step_i]` and `decode[step_i]` before patching the
> splitter — the rest of the pipeline assumes this naming.

#### Fallback for non-vLLM workloads

Many 3D-adjacent models don't have a clean "prefill then decode" boundary
(e.g. iterative diffusion sampling, single-shot feed-forward
reconstruction). When the splitter exits with no output, choose one of:

1. **Manual step-bucket split** — if your model has a clear step index
   (DiT denoising step, autoregressive WM step), name annotations
   `prefill[step_0]` and `decode[step_{1..N-1}]`. Step 0 carries the
   condition encode + first compute-heavy step; remaining steps are
   typically more memory-bound. Re-run the splitter.

2. **Aggregate-roofline-only mode** — skip the splitter; run Step 5
   directly on the un-split trace into a single output dir
   (`tracelens_all_csvs/`). Step 6 ranking still works; only the
   per-phase comparison in the report template (§7) is omitted. This is
   the correct mode for single-shot feed-forward models (VGGT / DUSt3R /
   FAST3R / single-image stereo).

3. **Skip TraceLens splitter entirely; custom split** — if neither helps,
   slice the trace by wall-clock window (e.g. seconds 2–4 = compute-heavy
   warm-up, seconds 4–10 = steady state) using a small preprocessing
   script. Tradeoff: loses op-level phase attribution but recovers
   per-window roofline.

Document which fallback was used in the report's "Phase split" line so
results are reproducible.

### Step 5 — Run TraceLens roofline per phase

```bash
INFREP=$HOME/TraceLens-internal/TraceLens/Reporting/generate_perf_report_pytorch_inference.py
GPU_ARCH=./results/gpu_arch.json

for PHASE in prefilldecode decode; do
    PHASE_TRACE=$(ls ./results/phase_split/${PHASE}_*.json.gz 2>/dev/null | head -1)
    [ -z "$PHASE_TRACE" ] && continue

    OUT=./results/tracelens_${PHASE}_csvs
    mkdir -p "$OUT"
    python3 "$INFREP" \
        --profile_json_path "$PHASE_TRACE" \
        --output_csvs_dir "$OUT" \
        --enable_pseudo_ops --group_by_parent_module --enable_kernel_summary \
        $([ -f "$GPU_ARCH" ] && echo "--gpu_arch_json_path $GPU_ARCH") \
        2>&1 | tail -10
done
```

This produces, per phase:

- `GEMM.csv` — one row per unique GEMM shape (`M`, `N`, `K`, `FLOPS/Byte`, `TFLOPS/s`, `Pct Roofline`)
- `SDPA_fwd.csv` / `FLASH_ATTN_fwd.csv` — attention roofline
- `unified_perf_summary.csv` — every op with FLOPS/byte, TFLOPS/s, bound type
- `kernel_summary.csv` — kernel-level rollup
- `ops_summary_by_category.csv` — time grouped by GEMM / attention / comm / norm / etc.
- `gpu_timeline.csv` — % computation / communication / idle

### Step 6 — Rank and classify kernels for optimization

```bash
python3 - <<'PY'
import csv, glob, os
PHASES = ["prefilldecode", "decode"]
rows = []

def classify(name, cat):
    s = f"{name} {cat}".lower()
    if any(x in s for x in ("nccl", "rccl", "allreduce", "all_reduce", "allgather", "all_gather", "reduce_scatter")):
        return "collective_comm", "distributed comm: AITER/Iris comm path if topology+shape match; otherwise framework/RCCL"
    if any(x in s for x in ("flash_attn", "paged_attn", "sdpa", "attn_fwd", "ck_fmha", "mha_fwd", "mla")):
        return "attention", "rocm-kernels-for-3d attention/AITER section"
    if any(x in s for x in ("hipblas", "rocblas", "cijk_", "aten::mm", "aten::addmm", "aten::matmul", "aten::linear")):
        if any(q in s for q in ("fp8", "mxfp4", "dequant", "quant")):
            return "gemm_quant", "rocm-kernels-for-3d AITER quant GEMM path"
        return "gemm", "rocm-kernels-for-3d GEMM/linear path"
    if any(x in s for x in ("conv3d", "conv2d", "convolution", "miopen_convolution", "grouped_conv", "depthwise", "im3d2col")):
        return "conv", "rocm-kernels-for-3d dense/video conv + layout checklist"
    if any(x in s for x in ("spconv", "indice", "sparseconv")):
        return "sparse_conv", "rocm-lib-compat backend check, then rocm-kernels-for-3d sparse-conv TODO"
    if any(x in s for x in ("gsplat", "raster", "nvdiffrast", "interpolate", "antialias", "texture")):
        return "external_raster", "rocm-lib-compat backend check; route to owning library/upstream"
    if any(x in s for x in ("copy_", "contiguous", "transpose", "permute", "view_as", "reshape")):
        return "copy_layout", "model-side layout cleanup; no generic AITER kernel owner"
    if any(x in s for x in ("scatter", "gather", "index_select", "index_add", "segment", "sort", "topk")):
        return "indexing_scatter_gather", "model/library indexing path; not AITER comm unless it is all_gather/reduce_scatter"
    if any(x in s for x in ("rms_norm", "layer_norm", "add_rmsnorm", "group_norm")):
        return "norm", "rocm-kernels-for-3d norm/fused elementwise path"
    if any(x in s for x in ("rope", "silu", "gelu", "elementwise", "mul", "add")):
        return "elementwise", "AITER only for known fused activation/quant patterns; generic add/mul -> graph fusion"
    if any(x in s for x in ("chunk_state", "ssd_", "mamba", "gla_", "selective_scan")):
        return "linear_attn_mamba", "rocm-kernels-for-3d linear-attn/Mamba TODO or upstream"
    return "other", "manual triage; add a rocm-kernels-for-3d TODO if >2%"

for ph in PHASES:
    f = f"./results/tracelens_{ph}_csvs/unified_perf_summary.csv"
    if not os.path.isfile(f): continue
    for r in csv.DictReader(open(f)):
        try:
            pct  = float(r.get("Percentage (%)", 0) or 0)
            roof = float(r.get("Pct Roofline_mean", 0) or 0)
            score = pct * (1.0 - roof / 100.0)
            name = r.get("name", "")
            cat = r.get("op category", "")
            ktype, handoff = classify(name, cat)
            rows.append((score, ph, ktype, handoff, name, cat, pct, roof))
        except ValueError: continue

rows.sort(reverse=True)
print(f"{'score':>7} {'phase':>13} {'pct':>6} {'roof%':>6} {'type':>16}  op -> handoff")
print("-" * 120)
for score, ph, ktype, handoff, name, cat, pct, roof in rows[:20]:
    print(f"{score:7.2f} {ph:>13} {pct:5.1f}% {roof:5.1f}% {ktype:>16}  {name[:48]} -> {handoff}")
PY
```

Output is the hand-off table: top 20 ops ranked by
`pct × (1 − roofline_efficiency)`, each with a forced `kernel_type` and next
owner. Items with high score are both time-consuming **and** far from peak.
Do not drop conv/sparse/comm/elementwise rows just because attention is small.
Route only rows with a concrete owner to `rocm-kernels-for-3d`. Route external
library raster kernels (`gsplat`, `nvdiffrast`, pytorch3d raster), generic
layout/copy, and indexing scatter/gather away from kernel replacement unless a
minimal reproducer identifies a real kernel owner.

---

## 4. Phase-Split for 3D-Adjacent Models

The vLLM "prefill vs decode" split was the original use-case. For 3D-adjacent
models, the equivalent split is:

| Model family | Compute-heavy phase (rename to `prefill`) | Memory / iter-heavy phase (rename to `decode`) | Annotation example |
|---|---|---|---|
| **VLA** (SmolVLA / Pi0 / OpenVLA / StarVLA) | image / state encoding + initial context build | per-token action decoding | `record_function("prefill[obs_encode]")`, `record_function("decode[action_step_{i}]")` |
| **World Model** (SANA-WM / Diffusion Forcing / Cosmos / Genie / le-wm) | condition encoding + first denoising / first rollout step | subsequent autoregressive rollout steps | `prefill[condition+step0]`, `decode[step_{i}]` |
| **DiT / video diffusion** (HunyuanVideo / CogVideoX / Mochi / Wan) | text/image encoding + first sampling step | remaining sampling steps (typically 20–50) | `prefill[encode+step0]`, `decode[step_{i}]` |
| **3D reconstruction (feed-forward)** (VGGT / DUSt3R / FAST3R / PAGE4D) | multi-view feature extraction + cross-view attention | output head (per-pixel depth / per-vertex offset) | `prefill[feat+xattn]`, `decode[head]` |
| **3DGS forward optimizer** (gsplat-based) | rasterizer forward / backward | densification / pruning / opt step | `prefill[raster]`, `decode[opt+densify]` |
| **Stereo / depth model** (FoundationStereo / Depth-Anything-3) | feature backbone + cost-volume / disparity refinement | (often no decode-equivalent — single forward) | `prefill[backbone+refine]` only |

**Rule**: pick a split that puts large-batch tensor ops in `prefill` and
small-batch / iterative ops in `decode`. The two will land on different
sides of the roofline; that's the whole point.

**Single-shot models** (feed-forward 3D reconstruction, stereo): use `prefill`
only. Don't fake a `decode` phase.

---

## 5. GPU Peak TFLOPS Reference

The phase-split roofline needs **per-precision peak TFLOPS**. Full per-GPU
specs (MI300X / MI325X / MI350X / MI355X / H100 / H200 / B200) with sources
live in [`gpu-specs.md`](gpu-specs.md).

Key columns to know:

- `matrix_bf16` / `matrix_fp16` — for AITER CK / flash-attn / GEMM bound to bf16/fp16
- `matrix_fp8` — for FP8 quantized GEMM (AITER quant path)
- `matrix_fp4` — for MXFP4 / FP4 GEMM (MI350X+ / B200 only)
- `vector_bf16` / `vector_fp16` / `vector_fp32` — for norm / RoPE / activation (memory-bound, but still need a vector peak)
- `mem_bw_gbps` — for memory-bound ops; ratio with peak TFLOPS gives the
  "compute / memory boundary": `arith_intensity_crossover = peak_tflops / mem_bw`

**Read [`gpu-specs.md`](gpu-specs.md)** before reporting roofline numbers,
to pick the right column for the op's precision.

---

## 6. Bottleneck Classification (lightweight)

The executable classifier is in §3 step 6; keep that as the single source of
truth. When editing it, preserve these routing rules:

- `attention`, `gemm`, `conv`, `norm`, fused activation, and high-share
  `linear_attn_mamba` rows can become `rocm-kernels-for-3d` work.
- `sparse_conv` first needs `rocm-lib-compat` backend verification, then a
  sparse-conv perf recipe if the backend is correct.
- `external_raster`, `copy_layout`, and `indexing_scatter_gather` are reported
  but not generic kernel-replacement targets.
- Keep `collective_comm` separate from compute kernels; it is actionable only
  for distributed workloads with a plausible AITER/Iris or framework/RCCL owner.

---

## 7. Evidence Contract

This skill adopts the evidence discipline from
[AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS),
but keeps the implementation 3D/ROCm-specific. A perf claim is complete only
when the report can answer these questions without re-running the experiment:

| Evidence | Required contents |
|---|---|
| Run identity | repo commit, ROCm/PyTorch/AITER versions, Docker image, GPU model/count, precision |
| Workload | exact input assets/prompts/videos, shapes, batch, steps, warmup, scheduler/cache state, random seed when relevant |
| Bounded candidate | the one install/backend/kernel change being evaluated; failed candidates stay in the notes |
| Stage split | compute-heavy vs memory/iter-heavy phase names, or an explicit statement that aggregate-only profiling was used |
| Raw artifacts | original trace(s), TraceLens CSVs, `gpu_arch.json`, logs, and any same-config NVIDIA reference |
| Action table | top kernels by priority, `kernel_type`, handoff owner, and whether the next step is compat, model cleanup, kernel work, or upstream issue |

Rules:
- Keep successful and failed candidates visible. A failed AITER/ROCm backend
  attempt is evidence, not noise.
- Report external raster / point / sparse libraries by measured time unless a
  valid roofline reference exists for that kernel family.
- Do not call a ROCm hotspot a backend gap from runtime share alone. Use a
  same-config NVIDIA baseline when practical; otherwise mark it unavailable.
- Preserve the raw artifacts path in the report so the next agent can replay or
  re-rank without guessing.

---

## 8. Standard Perf Report Template

When a profile run completes, produce this report (drop into the model's repo
or `overnight_tasks/<repo>/perf.md`):

```markdown
# <Model> on <GPU> — Perf Report

## Environment
- Run ID: <repo>-<date>-<candidate>
- Repo commit: <sha>
- ROCm: <version>      PyTorch: <version>      AITER: <version>
- Docker base: <image>
- GPU: <MI300X / MI355X / H100 / H200 / B200>, count <N>, arch <gfx942 / gfx950 / sm_90 / sm_100>
- Peak TFLOPS for op precision: matrix_bf16=<X>, matrix_fp8=<Y>, vector_bf16=<Z>
- HBM bandwidth: <X> GB/s

## Workload
- Input artifacts: <image/video/assets/prompts path or dataset slice>
- Input shape: <…>, batch=<…>, seq_len / n_steps=<…>
- Warmup / active steps: <…>
- Precision: <bf16 / fp8 / fp4>
- Phase split: prefill = <annotation>, decode = <annotation>

## Candidate
- Baseline command: `<...>`
- Candidate change: <install/backend/kernel/model-side cleanup>
- Candidate command: `<...>`
- Failed candidates kept in notes: <yes/no, path>

## NVIDIA Baseline (recommended, not blocking)
- Status: <same-config H100/H200/B200 measured / official same-config reference / unavailable>
- If unavailable: <reason: no NV node, repo has no published baseline, workload cannot run on CUDA without porting, etc.>
- Same-config guardrails: same model weights, shape, batch, precision, steps, scheduler, warmup/cache state, and timing boundary.

| Metric | ROCm <GPU> | NVIDIA <GPU> | Notes |
|-------:|-----------:|-------------:|------|
| End-to-end wall time | … | … | … |
| Prefill / compute-heavy phase | … | … | … |
| Decode / iter-heavy phase | … | … | … |
| Top target phase/op | … | … | e.g. MIOpen/CK Conv3D vs cuDNN Conv3D |

Use this section to decide whether a ROCm hotspot is a real cross-vendor gap.
If the section is unavailable, still report the ROCm profile, but do not use
ROCm runtime share alone as proof that a kernel family needs replacement.

## GPU Utilization (per phase)

| Metric                | Full Trace | Prefill (compute-heavy) | Decode (mem/iter-heavy) |
|-----------------------|-----------:|------------------------:|------------------------:|
| Computation Time (%)  | …          | …                       | …                       |
| Communication Time (%)| …          | …                       | …                       |
| GPU Idle Time (%)     | …          | …                       | …                       |
| Dominant Bound        | …          | compute                 | memory                  |

## Top Ops by Optimization Priority

(from §3 step 6 — score = pct × (1 − roofline_efficiency); comm excluded)

| Score | Phase   | Op (category)               | Pct of phase | Pct Roofline | Kernel type | Handoff |
|------:|---------|-----------------------------|-------------:|-------------:|-------------|---------|
| ...   | decode  | ck_fmha_… (attention)       | 32%          | 18%          | attention   | `rocm-kernels-for-3d` attention/AITER |
| ...   | prefill | `miopen_convolution`        | 45%          | n/a          | conv        | `rocm-kernels-for-3d` conv |
| ...   | decode  | `copy_` / transpose         |  8%          | n/a          | copy_layout | layout elimination |

## GEMM Roofline (per phase)

(from `GEMM.csv`; one row per unique M×N×K)

| Phase   | M×N×K        | FLOPS/Byte | TFLOPS/s | Bound  | Pct Roofline |
|---------|--------------|-----------:|---------:|--------|-------------:|
| prefill | 4096×4096×… |    …       | …        | …      | …            |
| decode  | 1×4096×…    |    …       | …        | memory | …            |

## Attention Roofline (per phase)

(from `SDPA_fwd.csv` or `FLASH_ATTN_fwd.csv`)

…

## Optimization Plan

(hand-off to `rocm-kernels-for-3d` / `rocm-lib-compat`)

1. <op A> — kernel_type=<...>, selected owner=<...>, expected speedup …
2. <op B> — if conv/sparse/comm/fused-activation, run that family checklist before attention/GEMM work; if copy/layout/indexing/external raster, route to the named owner instead of inventing a kernel recipe
3. <op C> — kernel gap: file upstream issue or add a `rocm-kernels-for-3d` TODO

## Raw artifacts

- Run log:                  `results/run.log`
- Candidate notes:          `results/candidates.md`
- TraceLens CSVs (prefill): `results/tracelens_prefilldecode_csvs/`
- TraceLens CSVs (decode):  `results/tracelens_decode_csvs/`
- Phase-split traces:       `results/phase_split/`
- GPU arch:                 `results/gpu_arch.json`
```

---

## 9. Cross-references

| When you need to… | Use this | Not this |
|---|---|---|
| Get a repo running on ROCm | [`rocm-lib-compat`](../rocm-lib-compat/SKILL.md) | this skill |
| Decide **what** to optimize | **this skill** | rocm-kernels-for-3d |
| Actually optimize a classified kernel family | [`rocm-kernels-for-3d`](../rocm-kernels-for-3d/SKILL.md) | this skill |
| Capture a trace / triage GPU fault / numerical bisect | ATOM repo `.claude/skills/` (`capture-trace`, `debug-agent-locate-kernel`, `dump-bisect-debug`) | reinvent here |
| Full LLM serving optimization (vLLM / SGLang) | [AMD-AIM/inference-skill](https://github.com/AMD-AIM/inference-skill) `inferencex-optimize` | this skill (use AMD's full pipeline) |

**Mental model**:

```
[ rocm-lib-compat ]   →   [ rocm-perf-analysis ]   →   [ rocm-kernels-for-3d ]
 (make it run)           (measure + classify)           (optimize selected family)
```

---

## 10. Out of Scope

- **LLM serving optimization** — use AMD's full
  [inferencex-optimize](https://github.com/AMD-AIM/inference-skill) pipeline; it
  already automates the 9-phase end-to-end loop including kernel optimization
  via GEAK. This skill stops at "measure & prioritize" for 3D workloads.
- **Writing new Triton/HIP kernels** — hand off to `rocm-kernels-for-3d` only
  after this skill has ranked and classified the target.
- **vLLM / SGLang server lifecycle** — `inferencex-optimize` and
  `vllm-optimize` handle this for LLM; we don't recreate it.
- **Training-side profiling** — this skill is inference / eval / rollout
  focused.
- **Compatibility (CUDA → ROCm) issues** — that's `rocm-lib-compat`.

---

## 11. Maintenance Notes

Keep roadmap and unresolved research questions out of this skill body. If a
profile exposes a new recurring gap, either add a concise routing rule above or
record it in the relevant `overnight_tasks/<repo>/experiments.md`.
