# AITER + ATOM Cookbook

Executable code skeletons that turn [`SKILL.md`](SKILL.md)'s patterns into
copy-paste-able snippets. Read this **after** picking a target from
`rocm-perf-analysis`, and **before** wiring AITER into the model.

> **Scope rule**: every snippet is model-agnostic — `[B, S, H, D]` shapes,
> not "SmolVLA decoder layer". Apply by changing the shapes, not by changing
> the structure.

> **AITER kernel API full reference**: see [`aiter-api.md`](aiter-api.md)
> (sibling file). Lists every AITER op + dtype matrix + SGLang/vLLM
> integration map. **This cookbook does not duplicate it** — only lists
> the imports used by the skeletons below.

---

## Verified-against pins

| Component | Pinned version | Notes |
|---|---|---|
| ATOM | `d41076e5` (2026-05-20) | Imports & decorator signatures in Parts 2–4 verified against this commit on a local clone. |
| AITER | `3b2e6f48c` (2026-05-20) | Module names in Part 1 verified against this commit. Full op catalog in [`aiter-api.md`](aiter-api.md). |
| ROCm | 6.4.3 (Triton path) or 7.2+ (CK path) | One per env |
| PyTorch | 2.6+ on 6.4, 2.8+ on 7.x | |

When ATOM/AITER bumps a minor version, re-verify Parts 2–4 first (API drift risk is highest there). To re-verify: `cd $ATOM_REPO && grep -rn "class ColumnParallelLinear" atom/model_ops/linear.py` (and similar for other imports listed below).

---

## Part 0 — Pre-flight

### 0.1 Local clone layout

```bash
# Convention (override via env if needed)
export AITER_REPO="${AITER_REPO:-$HOME/github/aiter}"
export ATOM_REPO="${ATOM_REPO:-$HOME/github/atom}"
```

If `$ATOM_REPO` is unset and the repo isn't cloned, do:

```bash
git clone https://github.com/ROCm/aiter.git "$AITER_REPO"
git clone https://github.com/ROCm/ATOM.git  "$ATOM_REPO"
```

Add the convention to your shell rc so every skill invocation can resolve them.

### 0.2 ROCm version → AITER backend (auto-selected)

| ROCm | AITER backend | flash-attn under the hood |
|---|---|---|
| 6.4.x | Triton v3 (auto) | Triton |
| 7.2+  | CK (auto)        | CK (~25% faster than Triton on MI300X) |

Don't override — AITER picks the right one based on what builds it links against.

### 0.3 Install (PyPI) — quick path

```bash
pip install amd-aiter      # PyPI package name is `amd-aiter`, NOT `aiter`
                           # (`pip install aiter` installs an unrelated 2019 async-iter stub)
```

Default install ships pre-built Python wrapper + JIT-compiles each kernel lazily on first call (~10-60 s/kernel). The "30-60 min CK pre-compile" mode only activates when explicitly setting `PREBUILD_KERNELS=1/2/3` (opt-in). Source: [`aiter/setup.py`](https://github.com/ROCm/aiter/blob/main/setup.py).

### 0.4 Install troubleshooting (read before debugging your own code)

If `import aiter` or `from aiter.ops.mha import flash_attn_func` fails, check in this order — each row is a real failure mode seen on `rocm/pytorch:rocm6.4.3` images, and each fix is one shell line:

| Symptom | Root cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'aiter.ops'` AND `aiter.__version__ == '0.13.20191203'` | `pip install aiter` installed the 2019 async-iter PyPI stub | `pip uninstall -y aiter && pip install amd-aiter` |
| `AttributeError: module 'triton' has no attribute 'jit'` while importing aiter | `rocm/pytorch:rocm6.4.3` ships a broken editable Triton 3.2.0 placeholder | `pip install --force-reinstall triton --index-url https://download.pytorch.org/whl/rocm6.4` → triton 3.7.0. **This fix is the same one documented in [`rocm-lib-compat/SKILL.md`](../rocm-lib-compat/SKILL.md) §"flash-attn: Triton 路径 vs CK 路径" foot-note** — apply it before retrying aiter import. |
| `AttributeError: module 'aiter' has no attribute '__version__'` | amd-aiter doesn't expose `__version__`; not a real failure | use `from importlib.metadata import version; version("amd-aiter")` |
| `[aiter] type hints mismatch, override to --> fmha_v3_fwd(...)` warning at first call | benign — amd-aiter's `compile_ops` overrides Python type hints after JIT compile | ignore; computation proceeds correctly |
| First step takes 30-60 s longer than subsequent steps | one-time JIT compile of an AITER kernel (e.g. `module_fmha_v3_fwd`) | not a bug; see §5.4 measurement protocol |

### 0.5 One-line verify

```python
python3 -c "
import aiter, torch
from importlib.metadata import version
print('amd-aiter:', version('amd-aiter'))
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
# CK availability heuristic — import a CK-only entry point
try:
    from aiter import flash_attn_varlen_func, paged_attention_ragged
    print('CK entry points: OK')
except Exception as e:
    print('CK entry points unavailable; on Triton path:', e)
"
```

---

## Part 1 — AITER imports used in this cookbook

Full op catalog → [`aiter-api.md`](aiter-api.md).
This cookbook's skeletons use the following imports only:

```python
# Attention
from aiter import (
    flash_attn_varlen_func,     # variable-length MHA (most workloads)
    mha_batch_prefill_func,     # batched prefill MHA
    paged_attention_ragged,     # paged KV cache attention
)
from aiter.mla import mla_decode_fwd, mla_prefill_fwd   # MLA (DeepSeek-style)

# GEMM
from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm    # BF16/FP16
from aiter.ops.gemm_op_a8w8  import gemm_a8w8           # FP8 / INT8
from aiter.ops.gemm_op_a4w4  import gemm_a4w4           # MXFP4 (gfx950+)
from aiter.tuned_gemm import tgemm                      # tuned GEMM router

# Norm / activation / RoPE
from aiter.ops.rmsnorm    import rmsnorm2d_fwd, fused_add_rmsnorm
from aiter.ops.activation import silu_and_mul, gelu_and_mul
from aiter.ops.pos_encoding import rotary_embedding_fwd

# Layout / quant
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.quant   import per_token_quant_hip, smoothquant_fwd

# Dtypes
from aiter.utility import dtypes   # dtypes.bf16 / fp8 / fp4x2 / fp8_e8m0 ...

# MoE
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, moe_sorting_fwd
```

If any import fails, the env is broken — go back to `rocm-lib-compat` skill.

---

## Part 2 — ATOM `model_ops/` skeletons

These mirror ATOM's `model_ops/` shape. Each is **minimum viable**; production
code adds quant routing, bias, weight tying, etc. Add those by reading the
matching ATOM file (`$ATOM_REPO/atom/model_ops/<file>.py`).

### 2.1 `ColumnParallelLinear`

```python
# model_ops/linear.py — column-parallel: shards output dim, no all-reduce
import torch
import torch.nn as nn
import torch.distributed as dist
from aiter.tuned_gemm import tgemm
from aiter.ops.shuffle import shuffle_weight


class ColumnParallelLinear(nn.Module):
    """Shards weight along output dim. Output is per-rank; consumer must
    `gather` if it needs the full output (rare in TP transformer blocks)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = False,
        tp_group: "dist.ProcessGroup | None" = None,
        dtype: torch.dtype = torch.bfloat16,
        use_ck_shuffle: bool = True,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        assert out_features % self.tp_size == 0, "out_features must be divisible by TP size"
        self.in_features  = in_features
        self.out_features_per_rank = out_features // self.tp_size
        self.gather_output = gather_output
        self.tp_group = tp_group
        self.use_ck_shuffle = use_ck_shuffle

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features, dtype=dtype)
        )
        self.bias = nn.Parameter(
            torch.empty(self.out_features_per_rank, dtype=dtype)
        ) if bias else None
        self._weights_processed = False

    @torch.no_grad()
    def process_weights_after_loading(self):
        """Called once after checkpoint load, before first forward.
        AITER CK kernels require shuffled layout — skipping this gives
        silently wrong outputs (no crash)."""
        if self._weights_processed or not self.use_ck_shuffle:
            return
        self.weight.data = shuffle_weight(self.weight.data, layout=(16, 16))
        self._weights_processed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]  ->  y: [..., out_features_per_rank]
        y = tgemm.mm(x, self.weight.t())              # x @ W^T
        if self.bias is not None:
            y = y + self.bias
        if self.gather_output and self.tp_size > 1:
            gathered = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(gathered, y, group=self.tp_group)
            y = torch.cat(gathered, dim=-1)
        return y
```

### 2.2 `RowParallelLinear`

```python
# model_ops/linear.py — row-parallel: shards input dim, all-reduce on output
class RowParallelLinear(nn.Module):
    """Shards weight along input dim. Output is summed across ranks via
    all-reduce when `reduce_results=True`. For MoE: set `reduce_results=False`
    on FusedMoE and shared_experts; the parent block does one combined reduce."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        reduce_results: bool = True,
        tp_group: "dist.ProcessGroup | None" = None,
        dtype: torch.dtype = torch.bfloat16,
        use_ck_shuffle: bool = True,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
        assert in_features % self.tp_size == 0, "in_features must be divisible by TP size"
        self.in_features_per_rank = in_features // self.tp_size
        self.out_features = out_features
        self.reduce_results = reduce_results
        self.tp_group = tp_group
        self.use_ck_shuffle = use_ck_shuffle

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank, dtype=dtype)
        )
        # Bias is only added on rank 0 to avoid double-counting after reduce
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self._weights_processed = False

    @torch.no_grad()
    def process_weights_after_loading(self):
        if self._weights_processed or not self.use_ck_shuffle:
            return
        self.weight.data = shuffle_weight(self.weight.data, layout=(16, 16))
        self._weights_processed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x assumed pre-sharded: [..., in_features_per_rank]
        y = tgemm.mm(x, self.weight.t())
        if self.reduce_results and self.tp_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM, group=self.tp_group)
        if self.bias is not None:
            rank = dist.get_rank(self.tp_group) if self.tp_group is not None else 0
            if rank == 0 or not self.reduce_results:
                y = y + self.bias
        return y
```

### 2.3 `ReplicatedLinear`

```python
# model_ops/linear.py — full copy on every rank (gates, embeddings, small heads)
class ReplicatedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.bfloat16,
                 use_ck_shuffle=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.use_ck_shuffle = use_ck_shuffle
        self._weights_processed = False

    @torch.no_grad()
    def process_weights_after_loading(self):
        if self._weights_processed or not self.use_ck_shuffle:
            return
        self.weight.data = shuffle_weight(self.weight.data, layout=(16, 16))
        self._weights_processed = True

    def forward(self, x):
        y = tgemm.mm(x, self.weight.t())
        return y + self.bias if self.bias is not None else y
```

### 2.4 `AttentionMHA` (variable-length, the most common shape)

```python
# model_ops/attention.py — thin wrapper on aiter.flash_attn_varlen_func
import torch, torch.nn as nn
from aiter import flash_attn_varlen_func


class AttentionMHA(nn.Module):
    """Wraps aiter flash-attn varlen. Use this for any per-token / per-sample
    variable-length attention (LLM decode, VLA action decode, DiT block with
    text condition, etc.).

    AITER varlen convention (verified against aiter v0.1.10.post2):
      q, k, v: [total_tokens, num_heads, head_dim]   (bf16/fp16)
      cu_seqlens_q / cu_seqlens_k: [batch + 1] int32, exclusive prefix sum
      max_seqlen_q / max_seqlen_k: python ints
      output: [total_tokens, num_heads, head_dim]
    """

    def __init__(self, num_heads: int, head_dim: int, causal: bool = True,
                 softmax_scale: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal = causal
        self.softmax_scale = softmax_scale or head_dim ** -0.5

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q: int, max_seqlen_k: int):
        # No reshape if caller already produces the right layout.
        # If caller has [B, S, H, D]: reshape upstream via q.view(-1, H, D).
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
```

For **fixed-shape batched** attention (e.g. DiT self-attention with same
`(B, S, H, D)` every block), swap to `mha_batch_prefill_func` — same shape
conventions but no cu_seqlens needed.

### 2.4b Functional SDPA replacement (diffusers / ControlNet / IP-Adapter / 多数 video-DiT)

**When to read this section instead of §2.4**: target repo's attention call site is `torch.nn.functional.scaled_dot_product_attention(q, k, v, ...)` (no module subclass), as in `diffusers.models.attention_dispatch`, ControlNet, IP-Adapter, Mochi, CogVideoX, HunyuanVideo. §2.4's `AttentionMHA` module wrapper doesn't apply — there is no module to subclass.

Three replacement patterns, pick the leftmost feasible one:

| Pattern | Scope | Risk | When to pick |
|---|---|---|---|
| **(a) global monkeypatch** `F.scaled_dot_product_attention` | process-wide | high — affects every SDPA call in the process (text encoder, VAE attention, etc.) | quickest A/B; throwaway perf probe; isolated benchmark |
| **(b) `Attention.set_processor(AITERAttnProcessor())`** | per-`diffusers.models.Attention` instance | low — opt-in per layer, vanilla SDPA still available | production / partial swap (e.g. only DiT, not VAE) |
| **(c) class swap** — re-instantiate `Attention` with custom `processor` at model build | per-module-class | medium — requires model rebuild | downstream training / fine-tune integration |

**Common gotcha**: `F.scaled_dot_product_attention` takes `[B, H, S, D]` but AITER `flash_attn_func` takes `[B, S, H, D]` — the wrapper must transpose `H ↔ S` on both inputs and the output.

#### Pattern (a) — monkeypatch (the quickest A/B; verified on Wan2.1, see [`overnight_tasks/wan2.1/experiments.md`](../../../../overnight_tasks/wan2.1/experiments.md) C2)

```python
# functional_sdpa_to_aiter.py
import torch
import torch.nn.functional as F
from aiter.ops.mha import flash_attn_func   # cookbook §Part 1 import; amd-aiter ≥ 0.1.7

_ORIG_SDPA = F.scaled_dot_product_attention


def _aiter_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
    """Drop-in for torch.nn.functional.scaled_dot_product_attention.

    SDPA layout:  q, k, v: [B, H, S, D]   (PyTorch convention)
    AITER layout: q, k, v: [B, S, H, D]   (transpose H <-> S)
    """
    # AITER flash_attn_func only covers a subset of SDPA's surface;
    # fall back when caller uses features AITER doesn't support.
    if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                          is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    qa = q.transpose(1, 2).contiguous()  # [B, S, H, D]
    ka = k.transpose(1, 2).contiguous()
    va = v.transpose(1, 2).contiguous()
    softmax_scale = scale if scale is not None else q.shape[-1] ** -0.5
    out = flash_attn_func(qa, ka, va, dropout_p=0.0,
                          softmax_scale=softmax_scale, causal=is_causal)
    # flash_attn_func may return tuple (out, lse, ...) — take first
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.transpose(1, 2).contiguous()  # back to [B, H, S, D]


def install():
    F.scaled_dot_product_attention = _aiter_sdpa

def uninstall():
    F.scaled_dot_product_attention = _ORIG_SDPA
```

Usage:

```python
import functional_sdpa_to_aiter as patch
patch.install()
# ... build pipeline, run inference ...
patch.uninstall()
```

#### Pattern (b) — diffusers `AttentionProcessor`

```python
# aiter_attn_processor.py — works for diffusers ≥ 0.30 Attention class
import torch
from aiter.ops.mha import flash_attn_func


class AITERAttnProcessor:
    """Set via attn.set_processor(AITERAttnProcessor()). Layer-scoped."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        residual = hidden_states
        q = attn.to_q(hidden_states)
        kv_in = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = attn.to_k(kv_in)
        v = attn.to_v(kv_in)

        B, S, _ = q.shape
        H = attn.heads
        D = q.shape[-1] // H
        q = q.view(B, S, H, D)
        k = k.view(B, k.shape[1], H, D)
        v = v.view(B, v.shape[1], H, D)

        if attention_mask is not None:
            # fall back; AITER flash_attn_func has no mask arg
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=attention_mask
            ).transpose(1, 2)
        else:
            out = flash_attn_func(q, k, v, dropout_p=0.0,
                                  softmax_scale=D ** -0.5, causal=False)
            if isinstance(out, (tuple, list)):
                out = out[0]

        out = out.reshape(B, S, H * D)
        return attn.to_out[0](out)


def install_aiter_processor(pipe, only_transformer: bool = True):
    """Walk pipe modules; replace processor only on DiT/transformer Attention
    layers by default (leave VAE / text encoder alone to avoid surprises)."""
    target = pipe.transformer if only_transformer and hasattr(pipe, "transformer") else pipe
    from diffusers.models.attention import Attention as DiffusersAttention
    for m in target.modules():
        if isinstance(m, DiffusersAttention):
            m.set_processor(AITERAttnProcessor())
```

#### Decision rule

```
Is the perf measurement throwaway (≤ 1-day investigation)?
  └─ YES  → Pattern (a) monkeypatch
  └─ NO   → Are you only swapping DiT attention (not VAE / text encoder)?
              └─ YES  → Pattern (b) AITERAttnProcessor (install_aiter_processor with only_transformer=True)
              └─ NO   → Pattern (c) class swap at model build time
```

**Measurement contract for Pattern (a)/(b)**: see §5.4 — AITER's first attention call triggers a JIT compile (~30-60 s for `module_fmha_v3_fwd`). Never report wall-clock with step 1 included.

### 2.5 `PagedAttention` (only if serving with KV cache)

```python
# model_ops/paged_attn.py
from aiter import paged_attention_ragged


class PagedAttention(nn.Module):
    """KV cache layout: [num_blocks, block_size, num_kv_heads, head_dim].
    block_table: [batch, max_num_blocks_per_seq] int32 — maps logical
    sequence positions to physical KV-cache block ids.

    Decode-only call (single token per sample):
      q: [batch, num_heads, head_dim]
      output: [batch, num_heads, head_dim]
    """

    def __init__(self, num_heads, num_kv_heads, head_dim, block_size,
                 softmax_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.softmax_scale = softmax_scale or head_dim ** -0.5

    def forward(self, q, k_cache, v_cache, block_table, context_lens,
                max_context_len: int):
        return paged_attention_ragged(
            q, k_cache, v_cache, block_table, context_lens,
            max_context_len=max_context_len,
            num_kv_heads=self.num_kv_heads,
            scale=self.softmax_scale,
        )
```

### 2.6 `FusedMoE` wrapper (only if model has MoE)

```python
# model_ops/moe.py
import torch, torch.nn as nn
from aiter import ck_moe_stage1_fwd, ck_moe_stage2_fwd, moe_sorting_fwd


class FusedMoE(nn.Module):
    """Wraps AITER's 2-stage CK MoE. Stage 1 = topk routing + token sort;
    stage 2 = grouped GEMM + reduce. ATOM convention: set
    `reduce_results=False` on FusedMoE; the parent transformer block does
    one combined all-reduce that also covers shared_experts."""

    def __init__(self, num_experts, top_k, hidden_size, intermediate_size,
                 tp_group=None, reduce_results=False, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Expert weights: usually 2-tuple (gate_up_proj, down_proj) per expert
        self.gate_up = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * intermediate_size, dtype=dtype)
        )
        self.down = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size, dtype=dtype)
        )
        self.tp_group = tp_group
        self.reduce_results = reduce_results

    def forward(self, hidden_states, router_logits):
        # Routing
        sorted_tokens, expert_offsets, topk_weights = moe_sorting_fwd(
            router_logits, self.top_k
        )
        # Stage 1: gate_up GEMM + SiLU·mul activation
        h1 = ck_moe_stage1_fwd(sorted_tokens, self.gate_up, expert_offsets)
        # Stage 2: down GEMM + topk reduce
        out = ck_moe_stage2_fwd(h1, self.down, expert_offsets, topk_weights)
        if self.reduce_results and self.tp_group is not None:
            torch.distributed.all_reduce(out, group=self.tp_group)
        return out
```

---

## Part 3 — Loader patterns

### 3.1 `process_weights_after_loading()` entry point

After `model.load_state_dict(...)`, walk the module tree once:

```python
def process_all_weights_after_loading(model):
    for module in model.modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
```

Call this **exactly once** after checkpoint load, before first forward.
Idempotency-guard each wrapper with `_weights_processed` flag (already done
in Part 2 skeletons).

### 3.2 FP8 scale name reconciliation

Different checkpoint formats use different scale tensor names:

| Checkpoint origin | Scale tensor name | Required by AITER |
|---|---|---|
| TransformerEngine FP8 | `.scale` | rename → `.weight_scale_inv` |
| vLLM FP8 quantized export | `.weight_scale_inv` | rename → `.weight_scale` |
| AITER native | `.weight_scale` | no change |

Reconciliation utility:

```python
import torch

_FP8_SCALE_ALIASES = {".scale": ".weight_scale_inv", ".weight_scale_inv": ".weight_scale"}

def normalize_fp8_scale_names(state_dict: dict) -> dict:
    """Rename FP8 scale tensors to AITER's expected `.weight_scale`.
    Idempotent — running on already-normalized state_dict is a no-op."""
    renamed = {}
    for k, v in state_dict.items():
        new_k = k
        for old_suffix, new_suffix in _FP8_SCALE_ALIASES.items():
            if k.endswith(old_suffix):
                new_k = k[:-len(old_suffix)] + new_suffix
                break
        renamed[new_k] = v
    return renamed

# Also: AITER FP8 scale tensor must be `dtype=torch.float8_e8m0fnu` (= dtypes.fp8_e8m0)
# Cast on load if checkpoint stores it as float32:
def ensure_e8m0_scales(state_dict: dict) -> dict:
    from aiter.utility import dtypes
    for k in list(state_dict.keys()):
        if k.endswith(".weight_scale") and state_dict[k].dtype == torch.float32:
            state_dict[k] = state_dict[k].to(dtypes.fp8_e8m0)
    return state_dict
```

### 3.3 PyTorch fallback when AITER is unavailable

Every Part-2 wrapper should fall through to PyTorch when AITER imports fail.
Pattern:

```python
try:
    from aiter import flash_attn_varlen_func
    _AITER_OK = True
except ImportError:
    _AITER_OK = False

class AttentionMHA(nn.Module):
    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
        if _AITER_OK:
            return flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens_q,
                                          cu_seqlens_k=cu_seqlens_k,
                                          max_seqlen_q=max_seqlen_q,
                                          max_seqlen_k=max_seqlen_k,
                                          softmax_scale=self.softmax_scale,
                                          causal=self.causal)
        # Fallback: pad to dense, use PyTorch SDPA, return packed
        return _sdpa_fallback_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                     max_seqlen_q, max_seqlen_k,
                                     self.causal, self.softmax_scale)
```

The fallback path lets the same code run on NVIDIA for cross-vendor
benchmarking — required by SKILL.md Pattern A.

---

## Part 4 — `torch.compile` + CUDAGraph

### 4.1 ATOM's `@support_torch_compile` — import + usage

Verified against ATOM `d41076e5` (`atom/utils/decorators.py:375`). Canonical
reference: `atom/models/qwen3.py:46, 235`.

The decorator is **call-style with `dynamic_arg_dims`**, not a bare decorator.
`dynamic_arg_dims` maps each forward arg name to the tensor dim(s) that vary
across calls (e.g. token count = dim 0). If omitted, ATOM auto-infers from
`forward` annotations of type `torch.Tensor` / `Optional[torch.Tensor]` and
assumes dim 0; pass explicitly when in doubt.

```python
from atom.utils.decorators import support_torch_compile


@support_torch_compile(
    dynamic_arg_dims={
        "x": 0,                  # [total_tokens, hidden_size] — variable token count
        # add more here if forward takes other variable-shape tensors
    }
)
class TransformerLayerBody(nn.Module):
    """One repeating block. Inputs/outputs must have stable rank (number of
    dims) within a phase; only the dims listed in `dynamic_arg_dims` may
    vary. Variable-shape ops (sampling, dynamic KV growth) live OUTSIDE
    this module."""

    def __init__(self, hidden_size, num_heads, head_dim, intermediate_size):
        super().__init__()
        self.attn_qkv = ColumnParallelLinear(hidden_size, 3 * num_heads * head_dim)
        self.attn = AttentionMHA(num_heads, head_dim)
        self.attn_out = RowParallelLinear(num_heads * head_dim, hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.ffn_gate_up = ColumnParallelLinear(hidden_size, 2 * intermediate_size)
        self.ffn_down = RowParallelLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor, *attn_meta):
        # x: [total_tokens, hidden_size] — stable rank, variable leading dim
        # (handled by CUDAGraph bucketing — see 4.3)
        h = self.norm1(x)
        qkv = self.attn_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        h = self.attn(q, k, v, *attn_meta)
        h = self.attn_out(h)
        x = x + h
        h = self.norm2(x)
        g_u = self.ffn_gate_up(h)
        g, u = g_u.chunk(2, dim=-1)
        h = silu_and_mul(g, u)
        h = self.ffn_down(h)
        return x + h
```

### 4.2 Defining static sub-block boundaries (the only rule that matters)

A sub-block is "static enough" to CUDAGraph-capture iff **all of**:

1. Input tensor **ranks** are constant across calls.
2. Input tensor **shapes** are either constant, or only the leading dim
   varies and that dim is bucketed (see 4.3).
3. No data-dependent control flow inside (`if x.sum() > 0` style is forbidden).
4. No allocator-disturbing ops (random sampling without pre-allocated buffer).

Violate any → keep that op **outside** the decorated module. Typical
out-of-graph ops: top-k sampling, KV cache append, layernorm with dynamic
group sizes.

### 4.3 Variable-length input → bucketing pattern

For varlen MHA (per-batch token count varies), pre-define buckets and
pad up:

```python
BUCKETS = [128, 256, 512, 1024, 2048, 4096]   # power-of-two is friendliest

def pick_bucket(n_tokens: int) -> int:
    for b in BUCKETS:
        if b >= n_tokens:
            return b
    return BUCKETS[-1]   # caller must handle overflow separately

def pad_to_bucket(x: torch.Tensor, bucket: int, pad_value=0):
    n = x.shape[0]
    if n == bucket:
        return x, n
    return torch.nn.functional.pad(x, (0, 0) * (x.dim() - 1) + (0, bucket - n),
                                   value=pad_value), n
```

CUDAGraph is captured once per bucket size — memory cost is `len(BUCKETS) ×
graph_workspace`. 6 buckets is typical sweet spot.

### 4.4 Compile cache hygiene

```bash
# Clear on code change
rm -rf ~/.cache/torch_inductor ~/.cache/aiter ~/.cache/<your_app>/

# Suppress kernel log flooding
export AITER_LOG_LEVEL=WARNING

# Multiprocessing — `spawn` only (fork breaks CUDA re-init)
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
```

**Never** edit a `@support_torch_compile`-decorated module *after* the
decoration line. Dynamo's source-cache check fires. Instrument at call
sites instead.

---

## Part 5 — Validation toolkit

### 5.1 `cos_max(a, b)` — the only correctness metric you need

```python
import torch

def cos_max(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Returns the worst-case cosine similarity across rows after flattening
    everything except dim 0. Used to detect even small per-position drift
    that an aggregate `cos(flatten(a), flatten(b))` would mask.

    Thresholds (from SKILL.md Pattern F / Step 6):
      > 0.9999  -> bit-equal range, pass
      0.99 .. 0.9999  -> numerical drift, acceptable
      < 0.99    -> bug; bisect deeper
    """
    a = a.detach().float().flatten(1)
    b = b.detach().float().flatten(1)
    # Per-row cosine
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1).clamp(min=eps) * b.norm(dim=-1).clamp(min=eps)
    cos = num / den
    return cos.min().item()   # worst row, not mean
```

### 5.2 Per-layer bisect harness

When `cos_max < 0.99` at the model level, find the first broken layer:

```python
import torch
from typing import Callable

def bisect_layers(
    ref_model: torch.nn.Module,
    new_model: torch.nn.Module,
    sample_input,
    layer_filter: Callable[[str, torch.nn.Module], bool] = lambda n, m: True,
    threshold: float = 0.99,
):
    """Forward-hook both models, compare per-layer outputs, print first
    layer where new_model drops below `threshold`. Both models must share
    module names (same architecture, swapped kernels)."""

    captured_ref, captured_new = {}, {}
    def mk_hook(store, name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach()
        return hook

    handles = []
    for (n_ref, m_ref), (n_new, m_new) in zip(
        ref_model.named_modules(), new_model.named_modules()
    ):
        assert n_ref == n_new, "Model topology must match"
        if not layer_filter(n_ref, m_ref):
            continue
        handles.append(m_ref.register_forward_hook(mk_hook(captured_ref, n_ref)))
        handles.append(m_new.register_forward_hook(mk_hook(captured_new, n_new)))

    with torch.no_grad():
        ref_model(sample_input)
        new_model(sample_input)
    for h in handles:
        h.remove()

    first_bad = None
    for name in captured_ref:
        if name not in captured_new:
            continue
        c = cos_max(captured_ref[name], captured_new[name])
        marker = "FAIL" if c < threshold else "ok"
        print(f"  [{marker:>4}] cos_max={c:.6f}  {name}")
        if first_bad is None and c < threshold:
            first_bad = name
    return first_bad
```

### 5.3 Triton JIT pre-warm

First-run Triton compilation is 30s–2min per kernel; if your eval loop
runs short-lived processes, pre-warm:

```python
def prewarm_triton_cache(model, sample_inputs):
    """Run one forward to compile all Triton kernels. The JIT cache is on
    disk (TRITON_CACHE_DIR, default ~/.triton/cache/), so subsequent
    processes hit warm cache."""
    import torch
    model.eval()
    with torch.no_grad():
        for x in sample_inputs:
            _ = model(x)
    torch.cuda.synchronize()
    print("Triton cache warmed")

# For containerized eval: bake the warmed cache into the image:
#   docker cp <warmed_container>:/root/.triton /tmp/triton_cache
#   ADD /tmp/triton_cache /root/.triton
```

### 5.4 Steady-state measurement protocol (mandatory for any AITER A/B)

**Why this section exists**: in [Wan2.1 C2 experiment](../../../../overnight_tasks/wan2.1/experiments.md#c2--kernels-for-3d-replace-sdpa-with-aiter-mha-via-monkeypatch) (2026-05-21), the obvious "total wall after / total wall before" reading reported AITER `flash_attn_func` as **−51% slower** than AOTriton SDPA — when in fact the steady-state delta was **+32% faster**. The 48.4 s one-time JIT compile of `module_fmha_v3_fwd` had been folded into per-step latency, inverting the sign of the conclusion. This will silently happen to every agent that measures naively.

**Mandatory protocol** for any A/B that touches an AITER op:

1. **Run N ≥ 5 steps**, never 1. Step 1 is always cold (Triton/JIT/cache miss).
2. **Always exclude step 1** from the steady-state average. If you exclude it from one side, exclude it from both.
3. **Report a triple, not a delta**:

   ```
   (steady_avg_s_per_step, one_time_jit_cost_s, break_even_steps)
   ```

   - `steady_avg_s_per_step`: average over steps 2..N (or 3..N if step 2 also shows cache effects)
   - `one_time_jit_cost_s`: `step_1_s − steady_avg_s_per_step` (the AITER first-call compile)
   - `break_even_steps`: smallest N such that `N × baseline_per_step ≥ jit_cost + N × patched_per_step`. Below this, baseline wins; above it, AITER wins. For inference workloads with hundreds of denoise steps / batches, break-even is ~3-30 steps; for one-shot scripts it may never break even.

4. **Two paragraphs in the writeup**: one for raw timings, one for the triple. Never collapse to a single "X is N% faster".

#### Reference implementation

```python
import torch, time

def ab_steady(name, model_fn, steps=5, warmup=1, dropfirst=True):
    """Run model_fn() `steps` times after `warmup`. Return:
       per_step_s_list, steady_avg_s, jit_cost_s (= step_1 - steady_avg).
    `dropfirst=True` excludes step 1 from steady_avg even when warmup > 0,
    because JIT compile may happen during the *real* first instrumented step
    (e.g. with a fresh process / cold disk cache)."""
    # Warmup (not timed)
    for _ in range(warmup):
        out = model_fn()
        torch.cuda.synchronize()

    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        out = model_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    if dropfirst and len(times) > 1:
        steady = sum(times[1:]) / (len(times) - 1)
        jit_cost = times[0] - steady
    else:
        steady = sum(times) / len(times)
        jit_cost = 0.0
    print(f"  {name}: per-step={times}, steady={steady:.3f}s, jit={jit_cost:.3f}s")
    return times, steady, jit_cost


def break_even(jit_cost, baseline_steady, patched_steady):
    """How many steady steps until the patched path's amortized cost
    (jit + N*patched) drops below the baseline's (N*baseline)."""
    if patched_steady >= baseline_steady:
        return None  # never breaks even
    return int(jit_cost / (baseline_steady - patched_steady)) + 1


# --- Usage ---
_, b_steady, _        = ab_steady("baseline (AOTriton SDPA)", run_baseline_step, steps=5)
_, p_steady, p_jit    = ab_steady("patched (AITER MHA)",      run_patched_step,  steps=5)
n_be = break_even(p_jit, b_steady, p_steady)

print(f"steady speedup: {(b_steady - p_steady)/b_steady*100:+.1f}%")
print(f"one-time JIT cost: {p_jit:.1f}s")
print(f"break-even at step {n_be} (below this, AITER loses on wall)")
```

#### Anti-pattern (what NOT to do)

```python
# DO NOT: this is what produced the -51% mis-reading in Wan2.1 C2
t0 = time.time(); run_baseline(num_steps=5); baseline_wall = time.time() - t0
t1 = time.time(); run_patched(num_steps=5);  patched_wall  = time.time() - t1
print(f"speedup: {(baseline_wall - patched_wall)/baseline_wall*100:+.1f}%")
# ↑ this number has no meaning if either side compiled a kernel during the run
```

For multi-step diffusion / VLA decode (≥ 20 steps typical), JIT cost is always worth amortizing; for one-shot scripts (1-2 inferences), use `Part 5.3` pre-warm so JIT happens before the timed region.

---

## Part 6 — Cross-ROCm version switch

### 6.1 6.4 → 7.2 switch checklist

| Package | 6.4 install | 7.2 install / action |
|---|---|---|
| PyTorch | `--index-url .../rocm6.4` | `--index-url .../rocm7.2` |
| AITER (CK path activates) | `pip install amd-aiter` (auto-Triton; PyPI name is `amd-aiter`, **not** `aiter` — see lib-compat §FA pitfall) | `pip install amd-aiter` (auto-CK) |
| xformers | `--index-url .../rocm6.4` (works) | **No ROCm 7.x wheel as of 2026-05** — remove if possible, else stay on 6.4 |
| gsplat / amd_gsplat | 6.4 wheels only | not available — drop or stay 6.4 |
| pytorch3d | 6.4 wheels only | not available — drop or stay 6.4 |
| flash-attn (independent) | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn` | usually unneeded; AITER CK supersedes |
| triton | 6.4 wheel | bundled with PyTorch 2.8+/rocm7 |
| bitsandbytes | ≥0.45.3 | ≥0.45.3 |
| Docker base | `rocm/pytorch:rocm6.4.3_*` | `rocm/pytorch:rocm7.0.2_*` (or newer) |

**Pre-switch decision tree**:

```
Repo depends on xformers / gsplat / pytorch3d?
    ├── YES  -> stay on 6.4 (perf ceiling lower; AITER on Triton path)
    └── NO   -> migrate to 7.2 (AITER CK ~25% faster on attention)
```

### 6.2 AITER backend self-check (CK vs Triton)

```python
def aiter_backend_report():
    import aiter
    print(f"aiter version: {aiter.__version__}")
    try:
        from aiter import flash_attn_varlen_func, paged_attention_ragged
        print("CK entry points: OK")
    except ImportError:
        print("CK entry points: MISSING (Triton fallback only)")
    try:
        from aiter.ops.gemm_op_a4w4 import gemm_a4w4
        print("MXFP4 GEMM: OK (gfx950+)")
    except ImportError:
        print("MXFP4 GEMM: not available")
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8
        print("FP8/INT8 GEMM: OK")
    except ImportError:
        print("FP8/INT8 GEMM: not available")
```

Run this in CI for every env to detect silent regressions.

---

## Part 7 — When AITER doesn't cover the kernel

### 7.1 Triton linear-attn / Mamba on ROCm (current gap status, 2026-05)

| Kernel family | Upstream | ROCm status | Action |
|---|---|---|---|
| Mamba2 / Mamba2-SSD | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Partial — `causal_conv1d` works (also in AITER `ops/causal_conv1d.py`); selective_scan needs verify | Try upstream first; if broken, port to AITER `causal_conv1d` |
| GLA / RetNet | [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | Unverified on ROCm | Try as-is; file gap issue per SKILL § 6 |
| Hybrid (Jamba / SANA linear-attn block) | varies | Each variant separate | Treat each upstream kernel as its own port |

This is the **largest current gap** (SKILL § 3 table). When you hit it,
file an issue immediately — these gaps need upstream owner attention, not
local workarounds.

### 7.2 ROCm-safe Triton kernel minimal template

When you must write a Triton kernel from scratch:

```python
import triton
import triton.language as tl
import torch

# AMD wave_size = 64; NVIDIA warp_size = 32. Triton autotune handles this
# IF you give it ROCm-friendly configs.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
        for m in [16, 32, 64, 128]
        for n in [16, 32, 64, 128]
        for w in [4, 8]              # MI300X sweet spot
        for s in [2, 3]
    ],
    key=["M", "N"],
)
@triton.jit
def my_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(A_ptr + offs_m[:, None] * N + offs_n[None, :],
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    b = tl.load(B_ptr + offs_m[:, None] * N + offs_n[None, :],
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    c = a + b
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], c,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

**Avoid** on ROCm Triton (incomplete support as of 2026-05):

- `tl.dot` with `input_precision="tf32"` (no TF32 on AMD)
- Some `tl.atomic_*` reductions on FP types other than fp32
- Cooperative thread array hints (`num_ctas > 1`)
- `tl.async_load` / `tl.async_store` (SM-90/100 specific)

When in doubt, run on both AMD and NVIDIA Triton; differences = ROCm incompat.

### 7.3 GEAK (Generative Evolutionary Agentic Kernel) — entry point

For Triton kernels too complex to hand-write, AMD's GEAK can generate +
auto-verify candidates. Out of scope for this cookbook, but the trigger is:

```
mini -m claude-opus-4.6 --config geak.yaml --gpu-ids 0 --yolo \
    -t "Optimize problem_xxx.py on AMD gfx942. ..." \
    -o traj_xxx.json
```

Full GEAK recipe lives in [AMD-AIM/inference-skill `phases/07-kernel-optimize.md`](https://github.com/AMD-AIM/inference-skill).
Use when manual Triton writing exceeds 2 hours per kernel.

---

## Appendix — Quick reference back to SKILL.md

| SKILL.md section | Cookbook section |
|---|---|
| § 2 Pattern A (Wrap, don't call raw) | [Part 2](#part-2--atom-model_ops-skeletons) |
| § 2 Pattern B (CK shuffle) | [Part 3.1](#31-process_weights_after_loading-entry-point) + Part 2 `process_weights_after_loading` in each wrapper |
| § 2 Pattern C (Three TP flavors) | [Part 2.1 / 2.2 / 2.3](#21-columnparallellinear) |
| § 2 Pattern D (Piecewise compile + CUDAGraph) | [Part 4](#part-4--torchcompile--cudagraph) |
| § 2 Pattern E (Quant in model_ops) | [Part 3.2](#32-fp8-scale-name-reconciliation) + Part 2 wrappers' dtype routing |
| § 2 Pattern F (MLA / Paged / MoE) | [Part 2.4–2.6](#24-attentionmha-variable-length-the-most-common-shape) |
| § 4 Step 4 (Loader) | [Part 3](#part-3--loader-patterns) |
| § 4 Step 5 (Compile + CUDAGraph) | [Part 4](#part-4--torchcompile--cudagraph) |
| § 4 Step 6 (cos_max validate) | [Part 5](#part-5--validation-toolkit) |
| § 5 Pitfall: ROCm version mix | [Part 6](#part-6--cross-rocm-version-switch) |
| § 6 Upstream routing — linear-attn / Mamba | [Part 7.1](#71-triton-linear-attn--mamba-on-rocm-current-gap-status-2026-05) |
