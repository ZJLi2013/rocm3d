# GPU Peak TFLOPS Reference

Dense peak TFLOPS (no sparsity) and HBM bandwidth for the GPUs we run.
Data lifted from [AMD-AIM/inference-skill `phases/05-profile-analyze.md`](https://github.com/AMD-AIM/inference-skill/blob/main/skills/inferencex-optimize/phases/05-profile-analyze.md);
sources are AMD/NVIDIA datasheets and Hot Chips disclosures.

Used by:

- [`SKILL.md` § 3 step 3](SKILL.md#3-the-phase-split-recipe) — auto-pick GPU and write `gpu_arch.json`
- [`SKILL.md` § 5](SKILL.md#5-gpu-peak-tflops-reference) — pick the right precision column when reading roofline CSVs
- TraceLens `generate_perf_report_pytorch_inference.py --gpu_arch_json_path` — roofline computation

---

## How to choose the right column for roofline

For each profiled op, look at its precision and pick the matching peak:

| If the op is… | Use this peak column |
|---|---|
| GEMM in BF16 / FP16 (most AITER CK, flash-attn) | `matrix_bf16` or `matrix_fp16` |
| GEMM in FP8 (AITER quant path) | `matrix_fp8` |
| GEMM in MXFP4 / FP4 (MI350X+ / B200 only) | `matrix_fp4` |
| GEMM in FP6 (MI350X+ only) | `matrix_fp6` |
| RMSNorm / RoPE / activation / elementwise | `vector_bf16` or `vector_fp16` (memory-bound; vector peak still bounds) |
| Communication (NCCL/RCCL) | **don't roofline** — report as time only |
| spconv / gsplat / nvdiffrast | **no peak available** — report time only |

**Compute-vs-memory crossover** (per precision):

```
arith_intensity_crossover = peak_tflops × 1e12  /  (mem_bw_gbps × 1e9)
                          = peak_tflops × 1000  /  mem_bw_gbps   (FLOPs/byte)
```

If op's `FLOPS/Byte < crossover` → memory-bound; if `>=` → compute-bound.

Example (MI300X bf16): `1307 × 1000 / 5325 ≈ 245 FLOP/byte`. A GEMM with
arith intensity below 245 is memory-bound on this GPU regardless of TFLOPS/s.

---

## AMD CDNA3 (gfx942)

### MI300X

```json
{
  "name": "MI300X",
  "arch": "gfx942",
  "mem_bw_gbps": 5325,
  "memory_gb": 192,
  "tdp_w": 750,
  "max_achievable_tflops": {
    "matrix_fp16":  1307,
    "matrix_bf16":  1307,
    "matrix_tf32":   654,
    "matrix_fp32":   163,
    "matrix_fp64":   163,
    "matrix_fp8":   2615,
    "matrix_int8":  2615,
    "vector_fp16":   163,
    "vector_bf16":   163,
    "vector_fp32":   163,
    "vector_fp64":    82
  }
}
```

Source: AMD MI300X datasheet + Hot Chips 2024.

### MI325X

Same CDNA3 die as MI300X at higher clocks; 256 GB HBM3E. Same peak TFLOPS,
higher memory bandwidth.

```json
{
  "name": "MI325X",
  "arch": "gfx942",
  "mem_bw_gbps": 6000,
  "memory_gb": 256,
  "tdp_w": 1000,
  "max_achievable_tflops": {
    "matrix_fp16":  1307,
    "matrix_bf16":  1307,
    "matrix_tf32":   654,
    "matrix_fp32":   163,
    "matrix_fp64":   163,
    "matrix_fp8":   2615,
    "matrix_int8":  2615,
    "vector_fp16":   163,
    "vector_bf16":   163,
    "vector_fp32":   163,
    "vector_fp64":    82
  }
}
```

Source: AMD MI325X press materials.

---

## AMD CDNA4 (gfx950)

### MI350X (air-cooled)

```json
{
  "name": "MI350X",
  "arch": "gfx950",
  "mem_bw_gbps": 8000,
  "memory_gb": 288,
  "tdp_w": 1000,
  "max_achievable_tflops": {
    "matrix_fp16":  2307,
    "matrix_bf16":  2310,
    "matrix_fp32":   144,
    "matrix_fp64":    72,
    "matrix_fp8":   4614,
    "matrix_fp6":   9228,
    "matrix_fp4":   9228,
    "matrix_int8":  4614,
    "vector_fp16":   144,
    "vector_bf16":   144,
    "vector_fp32":   144,
    "vector_fp64":    72
  }
}
```

Source: AMD MI350X GPU datasheet (June 2025).

### MI355X (liquid-cooled, ~8% higher clocks than MI350X)

```json
{
  "name": "MI355X",
  "arch": "gfx950",
  "mem_bw_gbps": 8000,
  "memory_gb": 288,
  "tdp_w": 1400,
  "max_achievable_tflops": {
    "matrix_fp16":  2500,
    "matrix_bf16":  2500,
    "matrix_fp32":   158,
    "matrix_fp64":    79,
    "matrix_fp8":   5000,
    "matrix_fp6":  10000,
    "matrix_fp4":  10000,
    "matrix_int8":  5000,
    "vector_fp16":   158,
    "vector_bf16":   158,
    "vector_fp32":   158,
    "vector_fp64":    79
  }
}
```

Source: AMD press specs for MI355X.

---

## NVIDIA Hopper (sm_90)

### H100 (SXM5, 700W)

```json
{
  "name": "H100",
  "arch": "sm_90",
  "mem_bw_gbps": 3350,
  "memory_gb": 80,
  "tdp_w": 700,
  "max_achievable_tflops": {
    "matrix_fp16":   990,
    "matrix_bf16":   990,
    "matrix_tf32":   495,
    "matrix_fp32":    67,
    "matrix_fp64":    34,
    "matrix_fp8":   1979,
    "matrix_int8":  1979,
    "vector_fp16":   134,
    "vector_bf16":   134,
    "vector_fp32":    67,
    "vector_fp64":    34
  }
}
```

Source: NVIDIA H100 datasheet.

### H200 (same die as H100, 141 GB HBM3e)

```json
{
  "name": "H200",
  "arch": "sm_90",
  "mem_bw_gbps": 4800,
  "memory_gb": 141,
  "tdp_w": 700,
  "max_achievable_tflops": {
    "matrix_fp16":   990,
    "matrix_bf16":   990,
    "matrix_tf32":   495,
    "matrix_fp32":    67,
    "matrix_fp64":    34,
    "matrix_fp8":   1979,
    "matrix_int8":  1979,
    "vector_fp16":   134,
    "vector_bf16":   134,
    "vector_fp32":    67,
    "vector_fp64":    34
  }
}
```

Source: NVIDIA H200 datasheet.

---

## NVIDIA Blackwell (sm_100)

### B200

```json
{
  "name": "B200",
  "arch": "sm_100",
  "mem_bw_gbps": 8000,
  "memory_gb": 192,
  "tdp_w": 1000,
  "max_achievable_tflops": {
    "matrix_fp16":  2250,
    "matrix_bf16":  2250,
    "matrix_tf32":  1200,
    "matrix_fp32":    80,
    "matrix_fp64":    40,
    "matrix_fp8":   4500,
    "matrix_fp4":   9000,
    "matrix_int8":  4500,
    "vector_fp16":   160,
    "vector_bf16":   160,
    "vector_fp32":    80,
    "vector_fp64":    40
  }
}
```

Source: NVIDIA B200 datasheet.

---

## At-a-glance comparison (matrix_bf16 / matrix_fp8 / mem_bw)

| GPU | matrix_bf16 (TFLOPS) | matrix_fp8 (TFLOPS) | matrix_fp4 (TFLOPS) | HBM BW (GB/s) | bf16 crossover (FLOPs/byte) |
|---|---:|---:|---:|---:|---:|
| MI300X  | 1307 | 2615 | —     | 5325 | 245 |
| MI325X  | 1307 | 2615 | —     | 6000 | 218 |
| MI350X  | 2310 | 4614 |  9228 | 8000 | 289 |
| MI355X  | 2500 | 5000 | 10000 | 8000 | 313 |
| H100    |  990 | 1979 | —     | 3350 | 296 |
| H200    |  990 | 1979 | —     | 4800 | 206 |
| B200    | 2250 | 4500 |  9000 | 8000 | 281 |

A GEMM with arith intensity below the crossover is memory-bound on that GPU.
Most decode-phase ops (small batch, small M) fall below this on every GPU
in this table — that's why phase-split roofline matters.

---

## Notes

- These are **achievable dense peak** (no sparsity, no over-claim). The
  `inferencex-optimize` repo specifically labels them "max_achievable" rather
  than "marketing peak" — we keep the same convention.
- Vector peaks are roughly `peak_tflops / 16` for matrix-capable arches; they
  bound memory-bound ops like RMSNorm / RoPE / activation.
- When a new GPU SKU appears (MI400 / B300 / etc.), add a JSON block above
  in the same format, then re-run the auto-detect block in
  [`SKILL.md` § 3 step 3](SKILL.md#3-the-phase-split-recipe).
