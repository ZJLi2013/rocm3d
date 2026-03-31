---
name: rocm-lib-compat
version: 2.1.0
author: ZJLi2013
description: |
  ROCm library compatibility reference for porting ML repos (3D generation,
  reconstruction, world models, etc.) to AMD GPUs. Use when adapting a repo
  to ROCm, generating install scripts, or troubleshooting CUDA→ROCm build failures.
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm Library Compatibility Reference

When porting an ML repo to AMD ROCm, the key challenge is replacing
CUDA-specific libraries with ROCm-compatible equivalents. This skill
provides the canonical replacement table and install patterns.

---

## ROCm Version Strategy

**ROCm 6.4 is the primary base** — most pre-built wheels (xformers, gsplat,
pytorch3d) only have ROCm 6.4 builds. Use ROCm 7.x only when flash attention
CK performance matters and the repo does NOT depend on xformers/gsplat/pytorch3d.

| ROCm Version | PyTorch | xformers | gsplat | pytorch3d | flash-attn | aiter |
|:-------------|:-------:|:--------:|:------:|:---------:|:----------:|:-----:|
| **6.4** (default) | ✅ | ✅ wheel | ✅ wheel | ✅ wheel | ✅ FA2 Triton | ✅ Triton v3 |
| **7.0** | ✅ | ⚠️ | ✅ wheel | ❌ | — | ✅ CK + Triton |
| **7.1** | ✅ | ⚠️ experimental | ? | ❌ | — | ✅ CK + Triton |
| **7.2** | ✅ | ❌ no wheel | ? | ❌ | — | ✅ CK + Triton |

**Decision rule:**
- Repo uses xformers / gsplat / pytorch3d → **ROCm 6.4**
- Repo only uses flash-attn, want best perf → **ROCm 7.x** (AITER CK)
- Repo only uses flash-attn, want simplicity → **ROCm 6.4** (FA2 Triton or AITER Triton v3)

---

## Recommended Base Images

| ROCm | Docker Image | Python | PyTorch |
|------|-------------|--------|---------|
| **6.4** (default) | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` | 3.12 | 2.6.0 |
| **7.2** (AITER CK) | `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1` | 3.10 | 2.9.1 |

Native (non-Docker) setups: match the ROCm driver version on the host and install PyTorch via pip (Step 2).

---

## Step 1 — Scan Repo Dependencies

Read `requirements.txt`, `setup.py`, `pyproject.toml`, and `README.md`.
Identify any libraries from the table below.

## Step 2 — ROCm PyTorch Base

```bash
# ROCm 6.4 (default, most libs have pre-built wheels)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

# ROCm 7.2 (only for flash-attn CK performance path)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

## Step 3 — Replace CUDA Libraries

### Core Replacement Table

**ROCm 6.4 libraries** (pre-built wheels, no source build):

| Library | ROCm Install | Notes |
|---------|-------------|-------|
| **xformers** | `pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4` | ROCm 6.4 only; version must match torch |
| **gsplat** | `pip install gsplat --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | ROCm 6.4 / 7.0; never source build |
| **pytorch3d** | `pip install https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl` | ROCm 6.4 only; Python 3.12 |
| **triton** | `pip install triton --index-url https://download.pytorch.org/whl/rocm6.4` | Bundled with ROCm PyTorch; rarely needed |
| **torch-geometric** | `pip install torch_geometric && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<VER>+<ROCM_TAG>.html` | Match torch version |
| **diff-gaussian-rasterization** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule |
| **simple-knn** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule |
| **bitsandbytes** | Not yet ROCm-compatible | Skip `--use_int8` flags |

**Flash Attention** (tiered strategy — same importlib code auto-adapts):

| Backend | ROCm | Install | Perf |
|---------|------|---------|------|
| FA2 Triton | 6.x | `pip install flash-attn --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | baseline |
| AITER Triton v3 | 6.x | `pip install aiter` (Triton path auto-selected on 6.x) | ~same |
| **AITER CK** | **7.x** | `pip install aiter` (CK path auto-selected on 7.x) or source: `git clone https://github.com/ROCm/aiter && cd aiter && git submodule update --init 3rdparty/composable_kernel && pip install .` | **-25%** |

### Exclusion Pattern for requirements.txt

After installing ROCm replacements, strip them from `requirements.txt` before `pip install -r`:

```bash
EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers|gsplat|flash.attn|triton|torch.geometric|pyg.lib|torch.scatter|torch.sparse|torch.cluster|torch.spline.conv|pytorch3d"
grep -vEi "^(${EXCLUDE_PKGS})([<>=!~;[:space:]]|$)" requirements.txt > req_clean.txt
grep -vEi "git\+https?://[^[:space:]]*(gsplat|pytorch3d|xformers|flash.attn|flash-attn|triton)" req_clean.txt > req_final.txt
pip install -r req_final.txt
```

For `pyproject.toml`, strip pinned CUDA deps before `pip install -e .`:

```python
python3 -c "
import re, pathlib; p = pathlib.Path('pyproject.toml')
pkg = r'torch|torchvision|torchaudio|xformers|gsplat|flash\.attn|flash-attn|triton|torch\.geometric|pytorch3d'
pat1 = re.compile(r'[ \t]*\"(' + pkg + r')[^\"]*\",?[^\n]*\n', re.I)
pat2 = re.compile(r'[ \t]*(' + pkg + r')\s*=\s*\{[^\}]*git[^\}]*\}[^\n]*\n', re.I)
txt = p.read_text(); txt = pat1.sub('', txt); txt = pat2.sub('', txt); p.write_text(txt)
"
```

### Native Extension Build

For repos with C++/CUDA extensions (`ext_modules`, `build_ext`, `.gitmodules`):

```bash
export PYTORCH_ROCM_ARCH="gfx942"   # MI300X/MI308X
# export PYTORCH_ROCM_ARCH="gfx1100"  # RX 7900 XTX
```

---

## AITER Flash Attention — Integration Guide

[AITER](https://github.com/ROCm/aiter) (AI Tensor Engine for ROCm) provides
flash attention via two backends:
- **Triton**: works on ROCm 6.x and 7.x, JIT from Python
- **CK (Composable Kernel)**: ROCm 7.x only, compiles to native AMD ISA, ~25% faster

`aiter.ops.mha.flash_attn_varlen_func` is the unified entry point — AITER
automatically dispatches to CK (if available) or falls back to Triton.

### Integration Pattern

```python
try:
    import importlib as _il
    _aiter_mha = _il.import_module('aiter.ops.mha')
    _aiter_flash_attn_varlen = _aiter_mha.flash_attn_varlen_func
    AITER_AVAILABLE = True
except Exception:
    AITER_AVAILABLE = False

# Dispatch priority: FA3 (NVIDIA) → AITER (AMD, auto CK/Triton) → FA2 Triton → error
```

The `importlib` pattern bypasses AITER's eager top-level imports (which may
fail due to `_mxfp4_quant_op` etc.). Same API as `flash_attn.flash_attn_varlen_func`.

### Performance (MI300X, gfx942, Matrix-Game 3.0)

| Backend | ROCm | Steady-state iter time | vs baseline |
|---------|------|----------------------|-------------|
| FA2 Triton | 6.x | ~14s | baseline |
| AITER Triton v3 | 6.x | ~14.6s | ~same |
| **AITER CK** | **7.x** | **~11s** | **-25%** |

### Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `fmha_fwd.hpp` not found | CK submodule not checked out | `git submodule update --init 3rdparty/composable_kernel` |
| `__builtin_amdgcn_mfma_*` undeclared | ROCm 6.x hipcc too old for CK | Use ROCm 7.x (CK auto falls back to Triton on 6.x) |
| `hipDeviceAttributePciChipId` undeclared | AITER main targets ROCm 7.x HIP API | Use ROCm 7.x or pin AITER v0.1.9 |
| `_mxfp4_quant_op` import error | AITER v0.1.9 eager import broken | Use `importlib` pattern (above) |
| `ENABLE_CK=0` still needs CK headers | AITER bug: incomplete CK-free guards | Use `importlib` pattern; don't rely on ENABLE_CK=0 |

---

## Smoke Test

```bash
python -c "import torch; print(f'torch {torch.__version__} | HIP: {torch.cuda.is_available()}')"
```
