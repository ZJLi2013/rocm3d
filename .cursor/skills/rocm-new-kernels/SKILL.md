---
name: rocm-new-kernels
version: 0.1.0
description: |
  Design, implement, validate, and iterate missing forward-only AMD GPU kernels
  for 3D / Video / World Model / VLA / DiT inference when no maintained
  AITER/ATOM/ROCm-library kernel exists. Use after `rocm-perf-analysis` ranks a
  hotspot and `rocm-kernels-for-3d` / `rocm-lib-compat` cannot route it to an
  existing backend. Covers HIP, Triton AMD, CK/Composable Kernel, standalone
  harnesses, correctness oracles, microbenchmarks, profiling-guided iteration,
  shape specialization, AutoKernel-style tiered tuning, ledgers, and upstream
  placement decisions.
  Example trigger: GroundingDINO MsDeformAttn forward on ROCm.
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm New Kernels

This skill is for **missing kernels**, not wrapper migration.

Use it when the profile says a kernel family matters, but:

- AITER/ATOM has no matching op or shape path.
- ROCm library compatibility has been checked and there is no intended backend.
- Model-side cleanup cannot remove the hotspot.
- A minimal custom HIP / Triton AMD / CK kernel is justified.

If an AITER/ATOM or ROCm-library path exists, use
[`rocm-kernels-for-3d`](../rocm-kernels-for-3d/SKILL.md) instead.

---

## 0. Scope

| Axis | In scope |
|---|---|
| Workload | Inference only; forward kernels first. |
| Model area | 3D generation/reconstruction, video, WM, VLA, DiT, visual detection/segmentation, point clouds. |
| Kernel types | Custom attention variants, deformable attention, gather/scatter-heavy ops, small reductions, indexing transforms, fused elementwise, shape-specific glue kernels, missing sparse/conv/raster-adjacent kernels. |
| Implementation | HIP C++/PyTorch extension, Triton AMD, CK/Composable Kernel wrappers, or a library fork patch. |
| Scale | Single GPU or single node only. |

Out of scope:

- Training backward kernels unless the user explicitly narrows the task.
- LLM serving kernels already owned by vLLM/SGLang/inference-skill.
- Blindly rewriting a library-owned kernel before `rocm-lib-compat` verifies the backend.
- Long benchmark campaigns without a correctness oracle.

---

## 0.5 MI300 / MI350 Hardware Knowhow

Read this before choosing HIP vs Triton AMD vs CK. Peak numbers live in
[`../rocm-perf-analysis/gpu-specs.md`](../rocm-perf-analysis/gpu-specs.md);
use them for roofline expectations, not as guaranteed achieved throughput.

| GPU | Arch | Memory | Bandwidth | Matrix peak to keep in mind |
|---|---|---:|---:|---|
| MI300X | CDNA3 `gfx942` | 192 GB | ~5.3 TB/s | BF16/FP16 1307 TFLOPS, FP8/INT8 2615 TFLOPS |
| MI325X | CDNA3 `gfx942` | 256 GB | ~6.0 TB/s | same compute class as MI300X |
| MI350X | CDNA4 `gfx950` | 288 GB | ~8.0 TB/s | BF16 2310 TFLOPS, FP8/INT8 4614 TFLOPS, FP4/FP6 9228 TFLOPS |
| MI355X | CDNA4 `gfx950` | 288 GB | ~8.0 TB/s | higher-clock MI350 class |

Kernel-design implications:

- **Target arch explicitly**: build extensions with `PYTORCH_ROCM_ARCH=gfx942`
  for MI300/MI325 and `gfx950` for MI350/MI355. Do not assume a kernel tuned on
  `gfx942` is optimal on `gfx950`.
- **Wavefront is 64 lanes on CDNA**: CUDA warp-32 assumptions often break
  indexing, reductions, and lane masks. Port warp-level logic deliberately.
- **BF16 is the default tensor dtype** on MI300/MI350 inference. FP8 is viable
  for bandwidth-bound GEMM-like paths when scaling is already part of the model.
  FP4/FP6 are `gfx950+` paths; do not design around them for MI300X.
- **Matrix vs vector roofline matters**: GEMM/attention-like kernels must use
  MFMA/CK/AITER-style matrix paths to approach matrix peak. Elementwise,
  indexing, deformable attention, and gather/scatter kernels usually live near
  vector or memory rooflines.
- **LDS is useful only with reuse**: staging irregular data into LDS can hurt if
  it adds sync/VGPR pressure without enough reuse. For deformable attention and
  sparse sampling, first fix coalescing and work partitioning.
- **VGPR pressure can dominate occupancy**: shape-specialized kernels that look
  faster per work item may lose on occupancy. Record VGPR/occupancy signals when
  profiling explains a plateau.
- **Memory coalescing is often the first win** for gather/scatter-heavy 3D ops.
  Describe lane-to-address mapping in the ledger before changing tile shapes.
- **Shape specialization is acceptable** for 3D workloads. If `W` has a few hot
  regimes, build a dispatcher instead of forcing one universal kernel.

### GEAK-Inspired Iteration

AMD GEAK (Generating Efficient AI-centric Kernels) is an agentic Triton-kernel
generation framework evaluated on AMD Instinct GPUs such as MI250X and MI300X.
Its useful lesson for this skill is the loop shape, not a requirement to use the
GEAK codebase:

```text
generate candidates -> run correctness/latency -> reflect on failures
-> optimize the next candidate -> keep best correct result
```

When using this pattern manually:

- Run multiple small candidates instead of one large rewrite.
- Keep each candidate tied to a single hypothesis, e.g. "vectorize loads",
  "change program id mapping", "specialize head_dim", "reduce temporary writes".
- Reject candidates on correctness before benchmarking.
- Promote only correct candidates with reproducible commands.
- Use parallel candidate ideas only when the harness is stable enough that
  failures are cheap to classify.
- Add profiling evidence only when correctness + latency logs cannot explain
  the next edit. GEAK-style loops scale by cheap candidate evaluation first,
  profiler-guided diagnosis second.

References for context: AMD's GEAK ROCm blog and paper describe inference-time
scaling, generator/reflector/optimizer roles, TritonBench-style evaluation, and
MI300-focused speed/correctness measurements.

---

## 0.6 AutoKernel / Kernel-Skills Playbook

This skill absorbs the useful execution discipline from
[`awesome-kernel-skills`](https://github.com/ZJLi2013/awesome-kernel-skills):
use a small, evidence-driven loop rather than a large rewrite.

Per candidate:

1. Establish baseline correctness and latency.
2. Classify bottleneck: memory, compute, latency, or mixed/unclear.
3. Pick one tiered edit direction from the table below.
4. Verify correctness before benchmarking.
5. Benchmark only correct candidates and record the decision.

Default stop conditions: target met, no correct candidate improves the best
baseline, max useful iterations reached, or profiling says the bottleneck moved
outside the custom kernel.

### Tiered Edit Priority

| Bottleneck | Prefer first | Consider | Deprioritize |
|---|---|---|---|
| Memory-bound | Tier 1 tile/occupancy, Tier 2 coalescing/vector loads/cache/LDS only with reuse | Tier 3 fusion to remove round-trips | Instruction shaving while bytes dominate |
| Compute-bound | Tier 1 MMA-friendly tile shapes, Tier 3 inner-loop math/accumulator/fused epilogue | Tier 5 AMD MFMA/gfx-specific knobs | Fusion that raises VGPR pressure without traffic savings |
| Latency-bound | Tier 3 fuse small glue kernels, remove host/device sync | Tier 4 persistent scheduling or reduced launch count | Split-K/complex schedules before launch granularity is fixed |
| Irregular gather/scatter | Lane-to-address mapping, coalesced partial tiles, shape-specialized dispatch | Tail splitting, prefetch, LDS only with clear reuse | Universal kernels that hide the hot shape behind slow generality |

Keep every candidate tied to one hypothesis. If two changes are needed, run them
as two ledger rows so improvement or regression remains attributable.

### Correctness Gate

Use the strongest practical subset of the 5-stage kernel verification protocol:

1. Basic comparison against PyTorch/CPU/CUDA fallback on standard shapes.
2. Target dtype coverage: usually BF16/FP16 plus FP32 oracle accumulation.
3. Edge shapes: non-divisible dimensions, tiny tensors, largest hot shapes,
   masks, empty/degenerate ranges when the op supports them.
4. Determinism only when atomics, races, reductions, graph capture, or training
   reuse make it relevant; do not over-test simple inference glue.
5. Stress loop before promotion: repeated calls, memory stability, and rare-tail
   shapes from `W`.

Suggested tolerances must be justified by dtype and reduction order. Never relax
the oracle just to pass a candidate.

### Benchmark Contract

For op-level timing:

- warm up enough to exclude JIT/autotune/cold cache effects
- synchronize before and after timed regions
- use enough repeats for a stable median; record p10/p90 if variance matters
- report latency plus TFLOPS for compute-bound or GB/s for memory-bound kernels
- compare against fallback/PyTorch and, when available, the previous best custom
  candidate
- save JSON or ledger rows with GPU, gfx target, ROCm, PyTorch, compiler flags,
  dtype, shape, warmup/repeat counts, and exact command

Do not promote a microbenchmark win until the model-level or phase-level profile
shows the intended end-to-end movement.

---

## 1. Start With K/R/W

Before writing code, recover the kernel contract:

```text
K: kernel semantics
   op formula, tensor shapes, dtype, layout, masks/indices, boundary behavior

R: correctness reference / oracle
   PyTorch fallback, CPU reference, existing CUDA output, or saved golden tensors

W: workload distribution
   exact shapes from the model, representative batch/step cases, hot-path dtype
```

Ask the user only if `K`, `R`, `W`, target GPU, or the promotion criterion is
missing and cannot be inferred safely.

For multiple shape regimes, record them as a distribution. For a single hot
shape, keep the implementation focused.

---

## 2. Required Evidence Files

Use the model repo's `experiments.md` when the repo already has one. For a
standalone kernel workspace, create lightweight ledgers:

```text
README.md
src/
tests/
benchmarks/
profile/
ledgers/attempt-ledger.md
ledgers/optimization-ledger.md
ledgers/lineage.jsonl
```

Minimum ledger row:

```markdown
| Iter | Candidate | Shape/dtype | Correctness | Latency | Evidence | Decision |
|---|---|---|---|---|---|---|
| 0 | fallback | ... | oracle | 1.00x | command/log | baseline |
| 1 | hip_v0 | ... | max_diff=... | ...x | bench path | keep/reject |
```

Failed candidates are evidence. Keep the reason.

---

## 3. Implementation Path Decision

Choose the simplest path that can plausibly hit the target:

| Situation | Default path |
|---|---|
| Needs custom indexing / irregular gather / small reductions | HIP C++ PyTorch extension. |
| Mostly elementwise / regular tensor program / fast iteration needed | Triton AMD. |
| GEMM/conv-like tiling with reusable CK templates | CK / Composable Kernel wrapper or library fork. |
| Existing CUDA extension has simple kernels | Hipify and repair CUDAisms first. |
| Existing CUDA kernel uses deep NVIDIA-specific intrinsics | Rebuild a minimal ROCm kernel from `K/R/W`; do not line-by-line port blindly. |

Start with a **minimal correct kernel**. Only optimize after it passes the
oracle on representative shapes.

---

## 4. Standard Workflow

### Step 1 — Confirm No Existing Owner

Check these before writing a new kernel:

1. `rocm-kernels-for-3d`: AITER/ATOM wrapper or known family recipe.
2. `rocm-lib-compat`: intended ROCm library backend or fork.
3. Upstream repo: existing CPU/PyTorch/CUDA fallback and semantics.
4. AITER / ATOM / CK / Triton AMD / PyTorch ROCm issues for nearby prior art.

If an existing maintained path exists, stop and route there.

### Step 2 — Build the Smallest Reproducer

Create a harness that can run:

- fallback/oracle output
- candidate output
- correctness comparison
- timing for `W`

Prefer op-level tests before e2e model runs. Keep all shapes explicit.

### Step 3 — Implement `v0` for Correctness

`v0` may be slow. It must be easy to inspect.

Rules:

- Preserve exact indexing semantics before optimizing memory layout.
- Guard all bounds explicitly for irregular ops.
- Use the model's dtype path, but compare against a higher-precision or
  trusted fallback when practical.
- For PyTorch extensions, provide a CPU/PyTorch fallback path for unsupported
  devices or debugging.

### Step 4 — Validate Numerics

Use the strictest oracle that is practical:

| Output type | Suggested check |
|---|---|
| Floating tensor | `max_abs`, `max_rel`, `cos_max`; include dtype and tolerance. |
| Index / mask | exact equality. |
| Detection / segmentation downstream | op-level numeric check first, then e2e output sanity. |

Never weaken the oracle to make a candidate pass. If reduction order causes
expected drift, document the tolerance and compare against the fallback scale.

### Step 5 — Benchmark Correct Candidates Only

Measure:

- warmup policy
- repeated timing count
- median / mean / p10 / p90 when useful
- input shapes
- GPU model, ROCm, PyTorch, compiler flags

Report op-level speed and the e2e effect. A fast microbench that does not move
the model is not enough for promotion.

### Step 6 — Profile Only When It Changes the Next Edit

Use the lightest tool that can answer the decision:

| Question | Tool |
|---|---|
| Which model phase/op/kernel family should be optimized? | `rocm-perf-analysis` + TraceLens. |
| Did the new op move end-to-end runtime? | `rocm-perf-analysis` or the model's e2e profiler. |
| Why is this custom kernel slow at op level? | `rocprof v3` counters/traces; use omniperf when deeper occupancy/memory analysis is needed. |
| Is this a shape-regime issue? | microbench performance map + targeted `rocprof v3` on representative cases. |

Good profiling triggers:

- correct candidate is slower than fallback
- candidate plateaus far from target
- performance differs by shape regime
- suspected memory coalescing, LDS pressure, VGPR/occupancy, launch overhead, or
  layout churn

Record the profile path and the one decision it changed. If `rocprof v3` is
used, keep the exact command, selected counters/options, ROCm version, GPU arch,
and the candidate commit/file version in `profile/` or the attempt ledger.

### Step 7 — Optimize One Cause at a Time

Common AMD kernel edit directions:

- improve global load/store coalescing
- vectorize loads/stores where alignment allows
- reduce temporary tensor materialization
- move reused data to LDS only when reuse pays for synchronization
- reduce VGPR pressure to improve occupancy
- specialize for hot shapes instead of over-generalizing
- split irregular work to reduce tail effects
- fuse small elementwise/reduction glue around the custom op
- remove host/device sync and unnecessary `contiguous()` churn

Do not mix several changes in one candidate unless each can be isolated by the
ledger.

### Step 8 — Decide Placement

At the end, decide where the kernel belongs:

| Result | Placement |
|---|---|
| Model-specific op, narrow shapes | model repo extension. |
| General op useful across 3D/VLA/WM | ROCm library fork or upstream issue/PR. |
| AITER/ATOM missing reusable op | AITER/ATOM issue or PR with reproducer. |
| Experimental but not production-ready | keep in `overnight_tasks` / experiment workspace. |

---

## 5. AMD Kernel Design Checklist

Use this as a compact design review before promotion:

- `K/R/W` is written down.
- Baseline fallback command and candidate command are reproducible.
- Correctness covers all hot shapes in `W`.
- The candidate does not rely on accidental tensor contiguity unless checked.
- Dtype behavior is explicit: fp32 accumulation, bf16/fp16 output, or integer exactness.
- Memory access pattern is described: contiguous, strided, gather/scatter, tiled.
- Bounds and masks match the fallback exactly.
- Benchmark excludes one-time compile/JIT cost unless reporting cold-start.
- Op-level speedup and e2e model impact are both recorded.
- Failed candidates and regressions are preserved.

---

## 6. MsDeformAttn-Style Pattern

For deformable attention or similar irregular sampling kernels:

1. Treat the PyTorch fallback as the semantic oracle.
2. Start with a forward-only kernel.
3. Preserve index math exactly before optimizing.
4. Benchmark per realistic `(batch, query, heads, levels, points, channels)`.
5. Watch gather-heavy memory behavior and branch divergence.
6. Prefer shape-specialized fast paths only after the generic path is correct.
7. Register the op through PyTorch dispatcher if graph capture / `torch.compile`
   matters for the model.

---

## 7. Handoff With Other ROCm Skills

```text
[ rocm-lib-compat ]      make sure no intended ROCm backend already exists
[ rocm-perf-analysis ]   rank and classify hotspots; profile after candidates
[ rocm-kernels-for-3d ]  use AITER/ATOM/ROCm library wrappers when available
[ rocm-new-kernels ]     create missing kernels with K/R/W + harness + ledger
```

Rule: if the task becomes "wire an existing AITER/ATOM op", leave this skill
and use `rocm-kernels-for-3d`. If the task becomes "find the hotspot", leave
this skill and use `rocm-perf-analysis`.
