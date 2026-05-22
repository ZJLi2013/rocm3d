
# AITER (AI Tensor Engine for ROCm) — Kernel 全景梳理

> 最后更新: 2026-04-21
> 版本: v0.1.10.post2 (latest in sglang CI)
> 仓库: [ROCm/aiter](https://github.com/ROCm/aiter)
> 目标架构: MI300X/MI308X (gfx942), MI350X (gfx950)

---

## 目录

1. [仓库结构](#1-仓库结构)
2. [GEMM 类 Kernels](#2-gemm-类-kernels)
3. [Attention 类 Kernels](#3-attention-类-kernels)
4. [MoE 类 Kernels](#4-moe-类-kernels)
5. [Element-wise / Activation / Norm 类](#5-element-wise--activation--norm-类)
6. [Quantization 类 Kernels](#6-quantization-类-kernels)
7. [Communication 类 Kernels (含 Iris/MORI)](#7-communication-类-kernels-含-irismori)
8. [KV Cache 类 Kernels](#8-kv-cache-类-kernels)
9. [Position Encoding / RoPE](#9-position-encoding--rope)
10. [Sampling 类 Kernels](#10-sampling-类-kernels)
11. [其他 Kernels](#11-其他-kernels)
12. [Data Type 支持总览](#12-data-type-支持总览)
13. [在 SGLang 中的集成](#13-在-sglang-中的集成)
14. [在 vLLM 中的集成](#14-在-vllm-中的集成)
15. [参考链接](#15-参考链接)

---

## 1. 仓库结构

```
aiter/
├── aiter/
│   ├── __init__.py              # 统一导出入口 (40+ ops modules)
│   ├── configs/                 # tuned GEMM configs (CSV per arch)
│   ├── dist/                    # 分布式: custom_allreduce, parallel_state
│   ├── jit/                     # FlyDSL JIT 编译框架 + flydsl_cache
│   │   └── core.py              # compile_ops 装饰器 (核心 JIT 基础设施)
│   ├── ops/                     # 所有 kernel Python 接口
│   │   ├── triton/              # Triton 实现的 kernels
│   │   │   ├── comms/           # Iris 通信原语 (reduce_scatter, all_gather)
│   │   │   ├── quant.py         # Triton 量化 kernels
│   │   │   └── gluon/           # Gluon PA decode kernel
│   │   ├── attention.py         # PA (paged attention) 全系列
│   │   ├── mha.py               # MHA / Flash Attention / MLA (~3100 行)
│   │   ├── gemm_op_a8w8.py      # INT8/FP8 GEMM
│   │   ├── gemm_op_a16w16.py    # FP16/BF16 GEMM
│   │   ├── gemm_op_a4w4.py      # MXFP4 GEMM
│   │   ├── batched_gemm_op_a8w8.py
│   │   ├── batched_gemm_op_bf16.py
│   │   ├── deepgemm.py          # DeepGEMM (CK-based grouped GEMM)
│   │   ├── moe_op.py            # MoE GEMM (CK 2-stage)
│   │   ├── moe_sorting.py       # MoE token sorting
│   │   ├── moe_sorting_opus.py  # MoE sorting (opus variant)
│   │   ├── activation.py        # SiLU, GELU, fused act
│   │   ├── rmsnorm.py           # RMSNorm (fused variants)
│   │   ├── groupnorm.py         # GroupNorm (HIP)
│   │   ├── quant.py             # Quantization (per-token, per-tensor, per-group, MXFP4)
│   │   ├── pos_encoding.py      # RoPE (rotary_embedding_fwd, batched)
│   │   ├── rope.py              # RoPE extended variants
│   │   ├── cache.py             # KV cache ops (reshape_and_cache, concat_mla)
│   │   ├── custom_all_reduce.py # HIP custom allreduce
│   │   ├── quick_all_reduce.py  # Quick allreduce
│   │   ├── communication.py     # Distributed env init
│   │   ├── custom.py            # Legacy GEMM (wvSpltK, LLMM1)
│   │   ├── topk.py              # TopK / biased_grouped_topk / moe_fused_gate
│   │   ├── topk_plain.py        # TopK plain variant
│   │   ├── sample.py            # Sampling (greedy, random, mixed)
│   │   ├── gradlib.py           # hipBLASLt / rocBLAS solution GEMM
│   │   ├── mhc.py               # Multi-head cache ops
│   │   ├── causal_conv1d.py     # Causal Conv1D (Mamba/SSM)
│   │   ├── trans_ragged_layout.py
│   │   ├── fused_qk_norm_rope_cache_quant.py
│   │   ├── fused_qk_norm_mrope_cache_quant.py
│   │   ├── fused_qk_rmsnorm_group_quant.py
│   │   ├── fused_split_gdr_update.py
│   │   └── enum.py              # ActivationType, QuantType enums
│   ├── mla.py                   # MLA (Multi-Latent Attention) 高层接口
│   ├── fused_moe.py             # Fused MoE 高层接口
│   ├── paged_attn.py            # Paged Attention 高层接口
│   ├── rot_emb.py               # Rotary Embedding 高层接口
│   └── tuned_gemm.py            # Tuned GEMM 路由 (CSV config lookup)
├── csrc/                        # HIP/CK C++ kernel 源码
│   ├── ck_gemm_a8w8*/           # CK GEMM codegen
│   ├── ck_gemm_moe_2stages*/    # CK MoE 2-stage codegen
│   ├── py_itfs_*/               # pybind11 接口
│   └── ...
└── docs/
    └── triton_comms.md          # Iris 通信文档
```

**JIT 编译机制**: 所有 `@compile_ops("module_xxx")` 装饰的函数在首次调用时通过 FlyDSL JIT 编译 HIP kernel，编译产物缓存在 `flydsl_cache/` 中。

---

## 2. GEMM 类 Kernels

### 2.1 Standard GEMM

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `gemm_a16w16_asm` | A:FP16/BF16, B:FP16/BF16 → BF16 | `gemm_op_a16w16.py` | ASM GEMM，支持 splitK、bias |
| `gemm_a8w8` | A:INT8/FP8, B:INT8/FP8 → BF16/FP16 | `gemm_op_a8w8.py` | CK GEMM，per-tensor/per-token/per-128x128 scale |
| `gemm_a4w4` | A:MXFP4, B:MXFP4 → BF16 | `gemm_op_a4w4.py` | MXFP4 blockscale GEMM, e8m0 scale (gfx950+) |
| `gemm_a4w4_blockscale` | A:FP4x2, B:FP4x2, scale:e8m0 → BF16 | `gemm_op_a4w4.py` | CKTile blockscale variant |

### 2.2 Batched GEMM

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `batched_gemm_a8w8` | A:INT8/FP8, B:INT8/FP8 → BF16 | `batched_gemm_op_a8w8.py` | Batched INT8/FP8 GEMM |
| `batched_gemm_bf16` | A:BF16, B:BF16 → BF16 | `batched_gemm_op_bf16.py` | Batched BF16 GEMM |

### 2.3 DeepGEMM

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `deepgemm` | FP8 + blockscale | `deepgemm.py` | CK-based grouped layout GEMM (DeepSeek 风格) |

### 2.4 hipBLASLt / rocBLAS Solution GEMM

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `hipb_mm` | FP16/BF16/FP8/INT8 + scale | `gradlib.py` | hipBLASLt solution-indexed GEMM |
| `hipb_findallsols` | 同上 | `gradlib.py` | 枚举所有可用 solution |
| `rocb_mm` | FP16/BF16 | `gradlib.py` | rocBLAS solution GEMM |

### 2.5 Legacy GEMM

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `wvSpltK` | FP16/BF16 | `custom.py` | Wave-level split-K GEMM |
| `wvSplitKQ` | FP8 + scale | `custom.py` | Quantized wave split-K |
| `LLMM1` | FP16/BF16 | `custom.py` | LLM matrix multiply |

### 2.6 Tuned GEMM 路由

`tuned_gemm.py` 提供 CSV 配置文件查找机制，根据 `(gfx_arch, cu_num, M, N, K)` 自动选择最优 kernel name 和 splitK 参数。

---

## 3. Attention 类 Kernels

### 3.1 Flash Attention — 完整 callable 表

`aiter.ops.mha` 在 amd-aiter ≥ 0.1.7 暴露 **43 个 callable**（实测 `dir(aiter.ops.mha)` on `amd-aiter==0.1.7.post2.dev18`, 2026-05-21）。按 shape 风格 + dtype 分组如下；前一版本仅列 `flash_attn_varlen_func` + `mha_batch_prefill_func`，导致 diffusion / video-DiT 这类**密集 fixed-shape** 调用方挑错 callable。

#### 3.1.1 Dense (fixed-shape, 非 varlen) — diffusers / DiT / video-WM 主用

| Kernel | Q/K/V layout | Data Types | 何时用 | 说明 |
|--------|---|------------|---|------|
| `flash_attn_func` | `[B, S, H, D]` | BF16/FP16 → 同 dtype | **diffusers DiT、密集 attention（无 packing）** | 标准 dense MHA。**这是 functional SDPA monkeypatch 的对应 callable**（cookbook §2.4b）|
| `flash_attn_fp8_pertensor_func` | 同上 | FP8 + per-tensor scale | FP8 inference 的 dense MHA | per-tensor scaling |
| `fmha_v3_fwd` | `[B, S, H, D]` | BF16/FP16 | gfx942+，amd-aiter JIT 时优先路由 | "fmha_v3" 是 amd-aiter 在 MI300X 上的新一代 dense fwd kernel，JIT 编译 ~48 s 首次，稳态比 AOTriton 快 ~30% |
| `mha_fwd` | `[B, S, H, D]` | BF16/FP16 | low-level / 自定义路径 | 通常不直接调用；通过 `flash_attn_func` 间接路由 |
| `mha_batch_prefill` / `mha_batch_prefill_func` | `[total_tokens, H, D]` + `seqlens` | BF16/FP16 | **LLM serving prefill (batched, fixed-tokens-per-batch)** | 多请求 prefill |

#### 3.1.2 Variable-length (varlen, packed sequences) — LLM serving 主用

| Kernel | Q/K/V layout | Data Types | 何时用 |
|--------|---|------------|---|
| `flash_attn_varlen_func` | `[total_tokens, H, D]` + `cu_seqlens` | BF16/FP16 | LLM 训练/serving 任意 packed seq |
| `flash_attn_varlen_fp8_pertensor_func` | 同上 | FP8 + per-tensor scale | FP8 LLM serving 的 varlen 路径 |
| `fmha_v3_varlen_fwd` | 同上 | BF16/FP16 | varlen 在 gfx942+ 的新一代 kernel |
| `mha_varlen_fwd` | 同上 | BF16/FP16 | low-level |

#### 3.1.3 Backward / Tune / Internal (一般 inference 不用)

| Kernel | 用途 |
|--------|------|
| `mha_bwd`, `mha_varlen_bwd`, `fmha_v3_bwd`, `fmha_v3_varlen_bwd`, `can_impl_fmha_v3_bwd` | 训练 backward |
| `cmdGenFunc_mha_*`, `gen_mha_*_fake_tensors`, `gen_fmha_v3_*_fake_tensor` | torch.compile / fake-tensor / dynamo 集成 |
| `compile_ops`, `torch_compile_guard`, `maybe_contiguous`, `get_gfx` | 工具 / decorator |
| `FlashAttnFunc`, `FlashAttnVarlenFunc`, `mha_batch_prefill_fake_tensors`, `common_mha_*_fake_tensors` | autograd.Function 封装 + fake-tensor |

**完整 43 个 callable 一次性枚举**（用于 inspect / debug）：

```python
import aiter.ops.mha as m
print([n for n in dir(m) if not n.startswith("_") and callable(getattr(m, n))])
# 43 items: FlashAttnFunc, FlashAttnVarlenFunc, can_impl_fmha_v3_bwd,
# cmdGenFunc_mha_batch_prefill, cmdGenFunc_mha_bwd, cmdGenFunc_mha_fwd,
# cmdGenFunc_mha_varlen_bwd, cmdGenFunc_mha_varlen_fwd, common_mha_bwd_fake_tensors,
# common_mha_fwd_fake_tensors, compile_ops, flash_attn_fp8_pertensor_func,
# flash_attn_func, flash_attn_varlen_fp8_pertensor_func, flash_attn_varlen_func,
# fmha_v3_bwd, fmha_v3_fwd, fmha_v3_varlen_bwd, fmha_v3_varlen_fwd,
# gen_fmha_v3_bwd_fake_tensors, gen_fmha_v3_fwd_fake_tensors,
# gen_fmha_v3_varlen_bwd_fake_tensor, gen_fmha_v3_varlen_fwd_fake_tensor,
# gen_mha_bwd_fake_tensors, gen_mha_fwd_fake_tensors,
# gen_mha_varlen_bwd_fake_tensors, gen_mha_varlen_bwd_fake_tensors_common,
# gen_mha_varlen_fwd_fake_tensor, get_gfx, maybe_contiguous, mha_batch_prefill,
# mha_batch_prefill_fake_tensors, mha_batch_prefill_func, mha_bwd, mha_fwd,
# mha_varlen_bwd, mha_varlen_fwd, torch_compile_guard
```

#### 3.1.4 Shape-style → callable 决策表

| Shape style 来源 | 输入 layout | 推荐 callable | cookbook wrapper |
|---|---|---|---|
| `torch.nn.functional.scaled_dot_product_attention` (diffusers, ControlNet, IP-Adapter, 大部分 video DiT) | `[B, H, S, D]` ⇒ transpose 到 `[B, S, H, D]` | **`flash_attn_func`** | cookbook §2.4b "Functional SDPA replacement" (monkeypatch) |
| `flash_attn.flash_attn_func` (HF LLaMA, 多数 LLM) | `[B, S, H, D]` | `flash_attn_func` | cookbook §2.4 `AttentionMHA`（去掉 cu_seqlens 参数）|
| `flash_attn.flash_attn_varlen_func` (vLLM/SGLang prefill, packed) | `[total_tokens, H, D]` + `cu_seqlens` | `flash_attn_varlen_func` | cookbook §2.4 `AttentionMHA` (原样) |
| vLLM batched prefill | `[total_tokens, H, D]` + `seqlens` | `mha_batch_prefill_func` | cookbook §2.4 末尾 note |
| FP8 dense | `[B, S, H, D]` + scale | `flash_attn_fp8_pertensor_func` | 暂无 wrapper（FP8 一般在 LLM serving stack 内） |
| FP8 varlen | `[total_tokens, H, D]` + scale | `flash_attn_varlen_fp8_pertensor_func` | 同上 |

**等价映射**: NVIDIA `flash_attn_2_cuda.varlen_fwd` → AITER `flash_attn_varlen_func`；PyTorch `F.scaled_dot_product_attention` → AITER `flash_attn_func`（要注意 layout 是 `[B,S,H,D]` 不是 `[B,H,S,D]`，wrapper 内 transpose）。

**实测验证** (`overnight_tasks/wan2.1/experiments.md` C2, MI300X gfx942, amd-aiter 0.1.7.post2.dev18)：`flash_attn_func` 在 diffusers Wan2.1 (head_dim=128, seq ≈ 5,200) 上 cos_max=0.998949 vs AOTriton SDPA，稳态 +32% speedup，首次 JIT compile `module_fmha_v3_fwd` 48.4 s。

### 3.2 Paged Attention (Decode)

| Kernel | Q dtype | KV Cache dtype | 说明 |
|--------|---------|---------------|------|
| `paged_attention_ragged` | FP16/BF16 | FP16/BF16/FP8/INT8 | Ragged layout PA (sglang 主用) |
| `pa_fwd_asm` | FP16/BF16 | FP16/BF16/FP8/INT8 | ASM optimized PA, head_size=128 |
| `pa_fwd_naive` | FP16/BF16 | FP16/BF16 + quant | Naive HIP PA, 任意 head_size |
| `paged_attention_v1` | FP16/BF16 | FP16/BF16/FP8 | vLLM 兼容 PA v1 |
| `paged_attention_rocm` | FP16/BF16 | FP16/BF16/FP8/INT8 | ROCm native PA |
| `pa_decode_gluon` | FP16/BF16 | FP16/BF16 | Triton Gluon PA decode |

**ASM vs HIP 自动选择**: 当 `head_size=128` 且 `total_heads > 2*CU_NUM` 时自动选 ASM kernel。

### 3.3 MLA (Multi-Latent Attention)

| Kernel | Data Types | 说明 |
|--------|-----------|------|
| `mla_decode_fwd` | BF16/FP16, KV:FP8 可选 | MLA decode (DeepSeek V2/V3) |
| `mla_prefill_fwd` | BF16/FP16 | MLA prefill |
| `mla_prefill_ps_asm_fwd` | BF16/FP16 | MLA prefill with pre-shuffle ASM |
| `mla_reduce_v1` | FP32 → BF16/FP16 | MLA reduce (multi-step) |
| `get_mla_metadata_v1` / `get_mla_metadata_info_v1` | — | MLA metadata 准备 |

### 3.4 NSA (Native Sparse Attention)

| Kernel | Data Types | 说明 |
|--------|-----------|------|
| Triton NSA indexer kernels | FP16/BF16 | SGLang NSA backend (Triton 实现) |

---

## 4. MoE 类 Kernels

### 4.1 MoE Routing / Gating

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `topk_softmax` | FP32 gating → FP32 weights, INT32 ids | `moe_op.py` | MoE TopK + Softmax |
| `topk_softmax_asm` | 同上 | `moe_op.py` | ASM 优化 variant |
| `topk_sigmoid` | FP32 → FP32/INT32 | `moe_op.py` | TopK + Sigmoid (DeepSeek) |
| `biased_grouped_topk` | FP32 + bias → FP32/INT32 | `topk.py` | Grouped TopK with bias |
| `grouped_topk` | FP32 → FP32/INT32 | `topk.py` | Grouped TopK (softmax/sigmoid) |
| `moe_fused_gate` | FP32 + bias → FP32/INT32 | `topk.py` | Fused gate (大 token 数) |
| `moe_align_block_size` | INT32 | `moe_op.py` | Token-Expert 对齐 |
| `moe_sum` | FP16/BF16/FP32 | `moe_op.py` | MoE output 求和 |

### 4.2 MoE Sorting

| Kernel | 文件 | 说明 |
|--------|------|------|
| `moe_sorting_fwd` | `moe_sorting.py` | Token sorting by expert (支持 local_expert_mask, dispatch_policy) |
| `moe_sorting_opus_fwd` | `moe_sorting_opus.py` | Opus variant sorting |

### 4.3 MoE Fused GEMM (CK 2-Stage)

核心: 将 MoE 的 Gate-Up + Down 投影分解为两个 CK GEMM stage。

| Kernel | A dtype | W dtype | 说明 |
|--------|---------|---------|------|
| `ck_moe_stage1` | BF16/FP16/FP8/INT8/FP4x2 | BF16/FP16/FP8/INT8/FP4x2 | Stage1: Gate·Up GEMM |
| `ck_moe_stage2` | BF16/FP16/FP8/INT8/FP4x2 | BF16/FP16/FP8/INT8/FP4x2 | Stage2: Down GEMM |
| `moe_cktile2stages_gemm1` | FP8/INT8 + blockscale | FP8/INT8 | CKTile stage1 |
| `moe_cktile2stages_gemm2` | FP8/INT8 + blockscale | FP8/INT8 | CKTile stage2 |

**CK 2-Stage codegen** 支持的组合 (通过 `gen_instances.py` 自动生成):

| A dtype | B dtype | Output | Activation | Quant | preshuffle |
|---------|---------|--------|------------|-------|------------|
| BF16 | BF16 | BF16 | Silu/Gelu/No | No | off |
| FP8 | FP8 | BF16 | Silu/Gelu/No | per_Token/per_1x128 | off |
| INT8 | INT8 | BF16 | Silu/Gelu/No | per_Token/per_1x128 | off |
| FP4x2 | FP4x2 | BF16 | Silu/Gelu/No | per_1x32 | on/off |

### 4.4 MoE Fused ASM (单 kernel)

| Kernel | A dtype | W dtype | 说明 |
|--------|---------|---------|------|
| `fmoe` | BF16 | BF16 | 单 kernel fused MoE (gate+down) |
| `fmoe_g1u1` | FP8 + scale | FP8 | FP8 fused MoE w/ gate-up-1 |
| `fmoe_g1u1_tkw1` | FP8 + scale | FP8 | FP8 fused MoE (topk_weight in stage1) |
| `fmoe_int8_g1u0` | INT8 + scale | INT8 | INT8 fused MoE |
| `fmoe_int8_g1u0_a16` | BF16 input, INT8 weight | INT8 | Mixed-precision MoE |
| `fmoe_g1u1_a16` | BF16 input, FP8 weight | FP8 | Mixed-precision MoE |
| `fmoe_fp8_blockscale_g1u1` | FP8 + blockscale | FP8 | FP8 block-scale MoE |
| `moe_stage1_g1u1` | configurable | configurable | 通用 stage1 with ksplit |

---

## 5. Element-wise / Activation / Norm 类

### 5.1 Activation

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `silu_and_mul` | FP16/BF16/FP32 | `activation.py` | SiLU(x) * y, fused |
| `gelu_and_mul` | FP16/BF16/FP32 | `activation.py` | GELU(x) * y, fused |
| `gelu_tanh_and_mul` | FP16/BF16/FP32 | `activation.py` | GELU_tanh(x) * y |

### 5.2 RMSNorm

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `rmsnorm2d_fwd` | FP16/BF16 input, FP16/BF16 weight → FP16/BF16 | `rmsnorm.py` | Standard RMSNorm |
| `fused_add_rmsnorm` | FP16/BF16 | `rmsnorm.py` | Residual add + RMSNorm fused |
| `rmsnorm2d_fwd_with_quant` | FP16/BF16 → INT8/FP8 + scale | `rmsnorm.py` | RMSNorm + quantization fused |
| `fused_add_rmsnorm_with_quant` | FP16/BF16 → INT8/FP8 + scale | `rmsnorm.py` | Residual add + RMSNorm + quant |

### 5.3 GroupNorm

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `GroupNorm` / `_groupnorm_run` | FP16/BF16/FP32 | `groupnorm.py` | HIP GroupNorm with affine |

### 5.4 Fused Ops

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `fused_qk_norm_rope_cache_quant` | BF16 | 同名 `.py` | QK-Norm + RoPE + Cache + Quant 四合一 |
| `fused_qk_norm_mrope_cache_quant` | BF16 | 同名 `.py` | QK-Norm + M-RoPE + Cache + Quant |
| `fused_qk_rmsnorm_group_quant` | BF16 | 同名 `.py` | QK-RMSNorm + Group Quant |
| `fused_qk_rope_concat_and_cache_mla` | BF16 | `cache.py` | QK-RoPE + KV Concat + Cache (MLA) |

---

## 6. Quantization 类 Kernels

### 6.1 Dynamic Quantization (HIP)

| Kernel | Input → Output | 说明 |
|--------|---------------|------|
| `dynamic_per_token_scaled_quant` | FP16/BF16 → INT8/FP8 + per-token scale | 支持 shuffle_scale、num_rows |
| `dynamic_per_tensor_quant` | FP16/BF16 → INT8/FP8 + scalar scale | 全 tensor 量化 |
| `dynamic_per_group_scaled_quant_fp4` | FP16/BF16 → FP4x2 + e8m0 scale | Group size: 32/64/128 |
| `static_per_tensor_quant` | FP16/BF16 → INT8/FP8 | 静态 scale |

### 6.2 SmoothQuant

| Kernel | Input → Output | 说明 |
|--------|---------------|------|
| `smoothquant_fwd` | FP16/BF16 → INT8 | x_scale * input / y_scale |
| `moe_smoothquant_fwd` | FP16/BF16 → INT8 | MoE-aware smoothquant (per topk_ids) |
| `smooth_per_token_scaled_quant` | FP16/BF16 → INT8/FP8 | Smooth + per-token scale |
| `moe_smooth_per_token_scaled_quant` | FP16/BF16 → INT8/FP8 | MoE smooth quant (v1/v2 variants) |

### 6.3 MXFP4 Quantization

| Kernel | Input → Output | 说明 |
|--------|---------------|------|
| `per_1x32_f4_quant` | FP32 → FP4x2 + e8m0 scale | Per-1x32 block, torch 实现 |
| `per_1x32_f4_quant_hip` | FP16/BF16 → FP4x2 + e8m0 | HIP 优化, 支持 shuffle |
| `per_1x32_f4_quant_triton` | FP16/BF16 → FP4x2 + e8m0 | Triton 实现 |
| `fused_dynamic_mxfp4_quant_moe_sort` | FP16/BF16 → FP4x2 + sort | Fused MXFP4 quant + MoE sort |
| `mxfp4_moe_sort_hip` | e8m0 scale reorder | MoE scale sorting with MXFP4 layout |

### 6.4 Triton Quantization

| Kernel | Input → Output | 说明 |
|--------|---------------|------|
| `dynamic_per_token_quant_fp8_i8` | FP16/BF16 → FP8/INT8 | Triton per-token |
| `dynamic_per_tensor_quant_fp8_i8` | FP16/BF16 → FP8/INT8 | Triton per-tensor |
| `static_per_tensor_quant_fp8_i8` | FP16/BF16 → FP8/INT8 | Triton static |

### 6.5 QuantType 枚举

```python
class QuantType(Enum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x32 = 3        # MXFP4 block
    per_1x128 = 4       # FP8 block
    per_128x128 = 5     # 2D block
    per_256x128 = 6
    per_1024x128 = 7
```

---

## 7. Communication 类 Kernels (含 Iris/MORI)

### 7.1 Custom AllReduce (HIP)

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `all_reduce` | FP16/BF16/FP8 | `custom_all_reduce.py` | IPC-based custom allreduce, 支持 FP8 quant |
| `reduce_scatter` | FP16/BF16 | `custom_all_reduce.py` | Custom reduce-scatter |
| `all_gather_reg` / `all_gather_unreg` | FP16/BF16 | `custom_all_reduce.py` | Registered/unregistered all-gather |
| `fused_allreduce_rmsnorm` | FP16/BF16 | `custom_all_reduce.py` | **AllReduce + RMSNorm fused** |
| `fused_allreduce_rmsnorm_quant` | FP16/BF16 → FP8/INT8 | `custom_all_reduce.py` | **AllReduce + RMSNorm + Quant 三合一** |

### 7.2 Quick AllReduce (QR)

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `qr_all_reduce` | FP16/BF16/FP8 | `quick_all_reduce.py` | 快速 allreduce, 支持 quant_level |

### 7.3 Iris Triton Communication (GPU-Initiated)

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `reduce_scatter` | FP32/BF16/FP16/FP8 | `ops/triton/comms/` | Iris GPU-initiated reduce-scatter |
| `all_gather` | FP32/BF16/FP16/FP8 | `ops/triton/comms/` | Iris GPU-initiated all-gather |
| `reduce_scatter_rmsnorm_quant_all_gather` | BF16 → FP8 | `ops/triton/comms/` | **RS + RMSNorm + Quant + AG 四合一** |

**Iris** ([ROCm/iris](https://github.com/ROCm/iris)) 是 AMD 的多 GPU Triton 通信框架:
- GPU-initiated: 无需 CPU 参与，直接 GPU 间通信
- 需要安装: `pip install -e ".[triton_comms]"`
- Heap size 按 `(M, N, dtype, world_size, quant_mode)` 自动计算

### 7.4 Distributed Environment

| API | 文件 | 说明 |
|-----|------|------|
| `init_dist_env` | `communication.py` | 初始化 TP/PP 环境 (NCCL/RCCL backend) |
| `destroy_dist_env` | `communication.py` | 清理 |
| `set_custom_all_reduce` | `dist/parallel_state.py` | 启用 custom allreduce |

---

## 8. KV Cache 类 Kernels

| Kernel | KV dtype | 文件 | 说明 |
|--------|---------|------|------|
| `reshape_and_cache` | FP16/BF16/FP8/INT8 | `cache.py` | 标准 KV cache 写入 |
| `reshape_and_cache_flash` | FP16/BF16 + k/v_scale | `cache.py` | Flash layout cache 写入 |
| `reshape_and_cache_with_pertoken_quant` | FP16/BF16 → FP8/INT8 | `cache.py` | Per-token quant cache |
| `reshape_and_cache_with_block_quant` | FP16/BF16 → FP8/INT8 | `cache.py` | Block quant cache |
| `reshape_and_cache_with_block_quant_for_asm_pa` | FP16/BF16 → INT8 | `cache.py` | ASM PA 专用 block quant layout |
| `concat_and_cache_mla` | BF16 + FP8 scale 可选 | `cache.py` | MLA: kv_c + k_pe concat & cache |
| `fused_qk_rope_concat_and_cache_mla` | BF16 | `cache.py` | **QK-RoPE + Concat + Cache MLA fused** |
| `indexer_k_quant_and_cache` | FP16/BF16 → quant | `cache.py` | NSA K quant and cache |
| `cp_gather_indexer_k_quant_cache` | quant → BF16 | `cache.py` | NSA CP gather |
| `swap_blocks` | any | `cache.py` | Block swap (CPU↔GPU) |
| `copy_blocks` | any | `cache.py` | Block copy |

---

## 9. Position Encoding / RoPE

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `rotary_embedding_fwd` | FP16/BF16, Q/K in-place | `pos_encoding.py` | 标准 RoPE, neox/nope_first |
| `batched_rotary_embedding` | FP16/BF16, Q/K in-place | `pos_encoding.py` | Batched RoPE with cos_sin_cache_offsets |
| RoPE variants | BF16 | `rope.py` | 扩展 RoPE variants |

---

## 10. Sampling 类 Kernels

| Kernel | Input dtype | 文件 | 说明 |
|--------|-----------|------|------|
| `greedy_sample` | FP32/FP16/BF16 logits → INT64 | `sample.py` | Argmax |
| `random_sample` | FP32/FP16/BF16 → INT64 | `sample.py` | Temperature + Gumbel |
| `random_sample_outer_exponential` | FP32/FP16/BF16 → INT64 | `sample.py` | Exponential distribution |
| `mixed_sample` | FP32/FP16/BF16 → INT64 | `sample.py` | Mixed greedy/random (per-token temp) |
| `mixed_sample_outer_exponential` | FP32/FP16/BF16 → INT64 | `sample.py` | Mixed with exponential |
| `exponential` | → FP32 | `sample.py` | Exponential random number gen |
| `top_k_per_row_prefill` / `_decode` | FP32 logits | `topk.py` | Top-K per row (prefill/decode) |
| `top_k_per_row_prefill_fast` / `_decode_fast` | FP32 logits | `topk.py` | Fast ctypes variant |

---

## 11. 其他 Kernels

| Kernel | Data Types | 文件 | 说明 |
|--------|-----------|------|------|
| `causal_conv1d_update` | FP16/BF16/FP32 | `causal_conv1d.py` | Causal Conv1D (Mamba/SSM), width 2-4 |
| `trans_ragged_layout` | various | `trans_ragged_layout.py` | Ragged tensor layout 转换 |
| `fused_split_gdr_update` | BF16 | `fused_split_gdr_update.py` | Split + GDR update |
| `partial_transpose` | FP8/INT8 | `quant.py` | Partial transpose for quant |

---

## 12. Data Type 支持总览

### 按 Kernel 类别

| 类别 | FP32 | BF16 | FP16 | FP8 (e4m3) | INT8 | MXFP4 (e2m1) | FP4x2 |
|------|:----:|:----:|:----:|:----------:|:----:|:------------:|:-----:|
| **GEMM (standard)** | — | ✅ | ✅ | ✅ | ✅ | ✅ (gfx950) | ✅ |
| **GEMM (batched)** | — | ✅ | — | ✅ | ✅ | — | — |
| **DeepGEMM** | — | — | — | ✅ | — | — | — |
| **Flash Attention** | — | ✅ | ✅ | — | — | — | — |
| **Paged Attention** | — | ✅ Q | ✅ Q | ✅ KV | ✅ KV | — | — |
| **MLA** | — | ✅ | ✅ | ✅ KV | — | — | — |
| **MoE GEMM** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MoE Fused ASM** | — | ✅ | — | ✅ | ✅ | — | — |
| **Activation** | ✅ | ✅ | ✅ | — | — | — | — |
| **RMSNorm** | — | ✅ | ✅ | — | — | — | — |
| **GroupNorm** | ✅ | ✅ | ✅ | — | — | — | — |
| **Quantization** | ✅ in | ✅ in | ✅ in | ✅ out | ✅ out | ✅ out | ✅ out |
| **Communication** | ✅ | ✅ | ✅ | ✅ | — | — | — |
| **KV Cache** | — | ✅ | ✅ | ✅ | ✅ | — | — |
| **RoPE** | — | ✅ | ✅ | — | — | — | — |
| **Sampling** | ✅ | ✅ | ✅ | — | — | — | — |
| **Causal Conv1D** | ✅ | ✅ | ✅ | — | — | — | — |

### AITER dtypes 常量

```python
from aiter.utility import dtypes
dtypes.fp16     # torch.float16
dtypes.bf16     # torch.bfloat16
dtypes.fp32     # torch.float32
dtypes.fp8      # torch.float8_e4m3fnuz (ROCm native)
dtypes.i8       # torch.int8
dtypes.fp4x2    # torch.uint8 (packed 2xFP4)
dtypes.fp8_e8m0 # torch.float8_e8m0fnu (MXFP4 scale)
```

---

## 13. 在 SGLang 中的集成

SGLang 通过 `SGLANG_USE_AITER=1` 环境变量启用 AITER 路径。

### 核心集成点

| SGLang 模块 | AITER Kernel | 替代的 NVIDIA Kernel |
|-------------|-------------|---------------------|
| `aiter_backend.py` (attention) | `flash_attn_varlen_func` | FlashAttention-2/3 |
| `aiter_backend.py` (attention) | `paged_attention_ragged` | FlashInfer PA |
| `aiter_backend.py` (MLA) | `mla_decode_fwd`, `mla_prefill_fwd` | Triton MLA |
| `activation.py` | `silu_and_mul`, `gelu_and_mul` | `sgl_kernel.silu_and_mul` |
| `layernorm.py` | `rmsnorm2d_fwd`, `fused_add_rmsnorm` | `sgl_kernel.rmsnorm` |
| `fused_moe_triton/layer.py` | `ck_moe_stage1_fwd`, `ck_moe_stage2_fwd` | Triton fused_moe |
| `fused_moe_triton/layer.py` | `moe_sorting_fwd`, `topk_softmax` | Triton MoE sorting |
| `fused_moe_triton/layer.py` | `biased_grouped_topk` | Python grouped_topk |
| `quantization/` | `smoothquant_fwd`, `per_token_quant` | sgl_kernel quant |
| `custom_all_reduce.py` | `all_reduce`, `fused_allreduce_rmsnorm` | NCCL allreduce |
| `pos_encoding.py` | `rotary_embedding_fwd`, `batched_rotary_embedding` | sgl_kernel RoPE |
| `radix_attention.py` | `reshape_and_cache` | FlashInfer cache ops |
| `sampling/` | `top_k_per_row_*` | PyTorch topk |

### SGLang AITER 导入清单 (aiter_backend.py)

```python
from aiter import (
    flash_attn_varlen_func,
    get_mla_metadata_info_v1,
    get_mla_metadata_v1,
    get_ps_metadata_info_v1,
    get_ps_metadata_v1,
    mha_batch_prefill_func,
    mla_prefill_ps_asm_fwd,
    mla_reduce_v1,
    paged_attention_ragged,
)
from aiter.mla import mla_decode_fwd, mla_prefill_fwd
```

---

## 14. 在 vLLM 中的集成

vLLM 通过 `VLLM_ROCM_USE_AITER=1` 启用，`_aiter_ops.py` 统一管理。

### 核心集成点

| vLLM 模块 | AITER Kernel | Feature Flag |
|-----------|-------------|--------------|
| Linear layers | `tgemm` (tuned GEMM) | `VLLM_ROCM_USE_AITER` |
| RMS Norm | `rmsnorm2d_fwd`, `fused_add_rmsnorm` | `VLLM_ROCM_USE_AITER` |
| Fused MoE | `ck_moe_stage1/2`, `moe_sorting_fwd` | `VLLM_ROCM_USE_AITER` |
| Paged Attention | `paged_attention_rocm` | `VLLM_ROCM_USE_AITER` |
| W8A8 GEMM | `gemm_a8w8` (PTPC) | 独立 flag |
| FP8 Quantization | `per_token_quant_hip` | `VLLM_ROCM_USE_AITER` |
| Custom AllReduce | `all_reduce`, `fused_allreduce_rmsnorm` | `VLLM_ROCM_USE_AITER` |
| Activation | `silu_and_mul`, `gelu_and_mul` | `VLLM_ROCM_USE_AITER` |
| RoPE | `rotary_embedding_fwd` | `VLLM_ROCM_USE_AITER` |

### vLLM AITER Feature Tracking

参考: [vllm#14964 — AITER Kernel Integration tracking](https://github.com/vllm-project/vllm/issues/14964)

---

## 15. 参考链接

| 资源 | 链接 |
|------|------|
| AITER Repo | https://github.com/ROCm/aiter |
| Iris (Triton Comms) | https://github.com/ROCm/iris |
| AITER → SGLang PR (初始) | https://github.com/sgl-project/sglang/pull/4344 |
| AITER → SGLang (attention backend) | https://github.com/sgl-project/sglang/pull/6381 |
| AITER → SGLang (扩展) | https://github.com/sgl-project/sglang/pull/6838 |
| SGLang AITER v0.1.10.post2 | https://github.com/sgl-project/sglang/pull/18423 |
| AITER → vLLM (初始) | https://github.com/vllm-project/vllm/pull/14007 |
| AITER → vLLM (W8A8 PTPC) | https://github.com/vllm-project/vllm/pull/33773 |
| vLLM AITER tracking | https://github.com/vllm-project/vllm/issues/14964 |
| AITER Triton Comms doc | https://github.com/ROCm/aiter/blob/main/docs/triton_comms.md |
| CK (Composable Kernel) | https://github.com/ROCm/composable_kernel |
