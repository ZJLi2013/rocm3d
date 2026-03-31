---
name: rocm-lib-compat
version: 2.0.0
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

## Step 1 — Scan Repo Dependencies

Read `requirements.txt`, `setup.py`, `pyproject.toml`, and `README.md`.
Identify any libraries from the table below.

## Step 2 — ROCm PyTorch Base

Always install ROCm PyTorch first, before any other dependency:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
```

For ROCm 7.x environments:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

## Step 3 — Replace CUDA Libraries

### Core Replacement Table

| Library | ROCm Install | Notes |
|---------|-------------|-------|
| **flash-attn** (ROCm 6.x) | `pip install flash-attn --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | AMD pre-built wheel; Triton backend |
| **flash-attn** (ROCm 7.x, recommended) | `pip install aiter` or source build: `git clone https://github.com/ROCm/aiter && cd aiter && git submodule update --init 3rdparty/composable_kernel && pip install .` | AITER CK backend, native ISA, ~25% faster than Triton. See "AITER Flash Attention" section below |
| **xformers** | `pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4` | Version must match torch version |
| **gsplat** | `pip install gsplat --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | Never source build |
| **triton** | `pip install triton --index-url https://download.pytorch.org/whl/rocm6.4` | Bundled with ROCm PyTorch; explicit install rarely needed |
| **pytorch3d** | `pip install https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl` | Pre-built ROCm 6.4 / Python 3.12 wheel |
| **torch-geometric** / **torch_scatter** | `pip install torch_geometric && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<VER>+<ROCM_TAG>.html` | Match torch version |
| **diff-gaussian-rasterization** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | Gaussian splatting submodule |
| **simple-knn** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | Gaussian splatting submodule |
| **bitsandbytes** | Not yet ROCm-compatible | Skip `--use_int8` flags |

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

## AITER Flash Attention (FA3 CK Backend)

[AITER](https://github.com/ROCm/aiter) (AI Tensor Engine for ROCm) provides
the best flash attention performance on AMD GPUs via Composable Kernel (CK)
backend, compiling to native AMD ISA.

### Integration Pattern

```python
try:
    import importlib as _il
    _aiter_mha = _il.import_module('aiter.ops.mha')
    _aiter_flash_attn_varlen = _aiter_mha.flash_attn_varlen_func
    AITER_AVAILABLE = True
except Exception:
    AITER_AVAILABLE = False

# Dispatch priority: FA3 (NVIDIA) → AITER CK (AMD) → FA2 Triton → error
```

The `importlib` pattern bypasses AITER's eager top-level imports which may
fail on some configurations. `aiter.ops.mha.flash_attn_varlen_func` is the
CK backend entry point with the same API as `flash_attn.flash_attn_varlen_func`.

### Key Requirements

- **ROCm 7.x required** for CK backend (hipcc needs `__builtin_amdgcn_mfma_f32_16x16x32_f16`)
- **ROCm 6.x**: Use AITER Triton path (`aiter.ops.triton.attention.mha_v3`) or FA2 Triton
- **CK submodule**: `git submodule update --init 3rdparty/composable_kernel` before `pip install .`
- **Docker recommended**: `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1`

### Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `fmha_fwd.hpp` not found | CK submodule not checked out | `git submodule update --init 3rdparty/composable_kernel` |
| `__builtin_amdgcn_mfma_*` undeclared | ROCm 6.x compiler too old | Use ROCm 7.x (Docker) |
| `hipDeviceAttributePciChipId` undeclared | AITER main targets ROCm 7.x | Use ROCm 7.x or AITER v0.1.9 |
| `_mxfp4_quant_op` import error | AITER v0.1.9 quant dep broken | Patch `aiter/__init__.py` or use importlib |

---

## Smoke Test

```bash
python -c "import torch; print(f'torch {torch.__version__} | HIP: {torch.cuda.is_available()}')"
```
