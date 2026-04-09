---
name: rocm-lib-compat
version: 2.4.0
author: ZJLi2013
description: |
  ROCm library compatibility reference for porting ML repos (3D generation,
  reconstruction, world models, VLA, video generation) to AMD GPUs.
  Provides canonical CUDA→ROCm replacement table, AITER flash-attn integration,
  Docker base images, and dependency cleaning patterns.
  Use when adapting a repo to ROCm, generating install scripts, replacing
  CUDA-specific libraries (xformers, gsplat, pytorch3d, flash-attn, triton),
  or troubleshooting CUDA→ROCm build failures.
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
| **7.2** | ✅ | ❌ no wheel | ? | ❌ | — | ✅ CK + Triton |

> **pytorch3d source build pitfall (verified 2026-03-31, MI308X):**
> `pip install "git+...pytorch3d.git" --no-build-isolation` **builds successfully**
> (~80s) but produces a **CPU-only binary** — GPU rasterization kernels are missing
> (`"Not compiled with GPU support"`). The pre-built wheel is the only reliable
> path. Requires **Python 3.12 + Docker** (`rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`).

**Decision rule:**
- Repo uses xformers / gsplat / pytorch3d → **ROCm 6.4** (+ Docker py3.12 for pytorch3d)
- Repo only uses flash-attn, want best perf → **ROCm 7.x** (AITER CK)
- Repo only uses flash-attn, want simplicity → **ROCm 6.4** (FA2 Triton or AITER Triton v3)

---

## Recommended Base Images

| ROCm | Docker Image | Python | PyTorch |
|------|-------------|--------|---------|
| **6.4** (default) | `rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0` | 3.12 | 2.6.0 |
| **7.2** (AITER CK) | `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1` | 3.10 | 2.9.1 |

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

**ROCm 6.4+ libraries** (pre-built wheels or source build on ROCm 6.4+):

| Library | ROCm Install | Notes |
|---------|-------------|-------|
| **xformers** | `pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4` | ROCm 6.4 only; version must match torch |
| **gsplat** | `pip install gsplat --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | ROCm 6.4 / 7.0; never source build |
| **pytorch3d** | `pip install https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl` | ROCm 6.4 only; Python 3.12 |
| **triton** | `pip install triton --index-url https://download.pytorch.org/whl/rocm6.4` | Bundled with ROCm PyTorch; rarely needed |
| **torch-geometric** | `pip install torch_geometric && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-<ver>+<rocm>.html` | Match torch version |
| **apex** | `git clone https://github.com/ROCm/apex && cd apex && pip install . --no-build-isolation` | [ROCm/apex](https://github.com/ROCm/apex); 已含于 ROCm PyTorch Docker; hipblasLT on gfx942 |
| **diff-gaussian-rasterization** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule; hipcc compatible ✅ |
| **simple-knn** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule; hipcc compatible ✅ |
| **bitsandbytes** | `pip install bitsandbytes` (≥v0.45.3) | ROCm 6.4+ supported since v0.45.3 ✅ |
| **custom_rasterizer** (Hunyuan3D) | `pip install -e . --no-build-isolation` | Pure PyTorch C++ ext, no raw CUDA kernel ✅ |
| **flex_gemm** | `pip install . --no-build-isolation` (hipify 自动运行) | Triton backend 全算法 ROCm ✅; [PR #18](https://github.com/JeffreyXiang/FlexGEMM/pull/18); 合并前用 fork: `pip install git+https://github.com/ZJLi2013/FlexGEMM.git@rocm` |
| **cumesh** | `GPU_ARCHS=gfx942 pip install . --no-build-isolation` (hipify 自动运行) | 全 3 扩展 ROCm ✅; `cuda::std::plus`→`cub::Sum`, `cuda::std::tuple`→`rocprim::tuple`, Vec3f 加 `__host__`, nvcc flags 分支; fork: `pip install git+https://github.com/ZJLi2013/CuMesh.git@rocm` |
| **nvdiffrast** | `GPU_ARCHS=<arch> pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation` | **ROCm 6.4 + 7.2 均验证**; RDNA3/4 (gfx1100/gfx1201) wave32 ✅ + CDNA3 (gfx942) wave64 半wavefront模拟 ✅; cudaraster 全 4 阶段 + interpolate + antialias grad + texture 全 PASS |
| **nvdiffrec** | 同 nvdiffrast | ROCm 6.4 + 7.2; RDNA4 ✅ CDNA3 ✅ |

**Flash Attention** (tiered strategy):

| Backend | ROCm | Install | Perf | 验证 |
|---------|------|---------|------|------|
| **FA2 Triton** | **6.4.3** | `pip install flash-attn --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple` | baseline | ✅ HyDRA |
| AITER Triton v3 | 6.4.3 | `pip install aiter` (Triton path auto-selected) | ~same | ✅ HyDRA |
| **AITER CK** | **7.2.1** | `pip install aiter` (CK path auto-selected) | **-25%** | ✅ Matrix-Game (AITER ≥v0.1.13) |

### Exclusion Pattern for requirements.txt

Strip CUDA libs before `pip install -r`:

```bash
EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers|gsplat|flash.attn|triton|torch.geometric|pyg.lib|torch.scatter|torch.sparse|torch.cluster|torch.spline.conv|pytorch3d|numpy|cupy|bpy"
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

For repos with C++/CUDA extensions:

```bash
export PYTORCH_ROCM_ARCH="gfx942"   # MI300X/MI308X
# export PYTORCH_ROCM_ARCH="gfx1100"  # RX 7900 XTX
```

---

## AITER Flash Attention — Integration Guide

[AITER](https://github.com/ROCm/aiter) provides flash attention via:
- **Triton**: works on ROCm 6.4.3 and 7.2.1, JIT from Python
- **CK (Composable Kernel)**: ROCm 7.2.1 only, ~25% faster

### When to use AITER vs FA2 Triton

- **大部分场景**: ROCm 6.4.3 上直接 `pip install flash-attn --index-url=pypi.amd.com` 即可，不需要 AITER
- **需要 CK 加速**: 使用 ROCm 7.2.1 镜像 + `pip install aiter`（AITER ≥v0.1.13）

### Integration Pattern (AITER)

```python
try:
    from aiter.ops.mha import flash_attn_varlen_func
    AITER_AVAILABLE = True
except Exception:
    AITER_AVAILABLE = False
```

Same API as `flash_attn.flash_attn_varlen_func`.

### Performance (MI300X, gfx942)

| Backend | ROCm | AITER 版本 | Steady-state iter time | vs baseline |
|---------|------|-----------|----------------------|-------------|
| FA2 Triton | 6.4.3 | — | ~14s | baseline |
| AITER Triton v3 | 6.4.3 | ≥v0.1.13 | ~14.6s | ~same |
| **AITER CK** | **7.2.1** | **≥v0.1.13** | **~11s** | **-25%** |

### Known Issues (AITER CK on ROCm 7.2.1)

| Issue | Cause | Fix |
|-------|-------|-----|
| `fmha_fwd.hpp` not found | CK submodule not checked out | `git submodule update --init 3rdparty/composable_kernel` |
| xformers/gsplat/pytorch3d 不可用 | 这些库无 ROCm 7.x wheel | 仅在 repo 不依赖这些库时使用 ROCm 7.2.1 |

---

## CUDA-Only Libraries (No ROCm Path)

| Library | Reason | Workaround | Verified |
|---------|--------|------------|----------|
| **cuda-python** | NVIDIA CUDA Python bindings | Remove if not in critical path | — |
| **spconv-cu\*** | CUDA-only sparse convolution (cumm/pccm 生成 CUDA kernel) | 迁移中: [ZJLi2013/spconv_rocm](https://github.com/ZJLi2013/spconv_rocm) hipBLAS 路线 | 2026-03 |
| **tinycudann** | CUDA hash grid + MLP | [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn): 编译+forward ✅, backward split_k bug 已修复 (`6f32935`), video_to_world Stage 0-1b PASS, 全 pipeline 待重跑 | 2026-04-01 MI308X |
| **cupy-cuda12x** | NVIDIA CUDA Python array | Skip or `cupy-rocm-5-0` (limited, old ROCm) | — |
| **auto_gptq** | CUDA quantization | Skip quantization or use GGUF | — |
| **nvidia-cuda-nvcc** | NVIDIA compiler | Not needed on ROCm | — |
| **transformer-engine** | NVIDIA FP8 | Not available on ROCm | — |

---

## Known Patterns & Pitfalls

### PyTorch C++ Extensions That Work on ROCm

Not all C++ extensions are CUDA-only. Extensions using `torch/extension.h` + standard
PyTorch ops (no raw `__global__` kernels) often compile with hipcc:

```bash
export PYTORCH_ROCM_ARCH=gfx942
pip install -e . --no-build-isolation
```

Verified: custom_rasterizer (Hunyuan3D ✅), o-voxel (TRELLIS.2 ✅ compile, ❌ runtime needs flex_gemm).

### numpy Pin Breaks on Python 3.12

If `requirements.txt` pins `numpy==1.24.x`, pip build isolation pulls old setuptools
→ `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`.

Fix: skip the pin, use Docker-preinstalled numpy 2.x (add `numpy` to `EXCLUDE_PKGS`).

### flash-attn: Prefer Wheel Over Source Build

Source-building flash-attn for ROCm (ROCm/flash-attention) compiles ~2199 .hip files
and takes **~40 minutes**. Always try the pre-built wheel first:

```bash
pip install flash-attn --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple
```

Only fall back to source build if the wheel version is incompatible.

### AOTriton (PyTorch Built-in Flash Attention)

PyTorch 2.6+ on ROCm uses AOTriton as the SDPA backend automatically — no need to
install flash-attn at all for repos that use `torch.nn.functional.scaled_dot_product_attention`.
Confirmed in Hunyuan3D-2.1 logs: `Using AOTriton backend for Flash Attention forward...`

---

## Smoke Test

```bash
python -c "import torch; print(f'torch {torch.__version__} | HIP: {torch.cuda.is_available()}')"
```

---

## Verified Repos

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|--------------|--------|
| PartCrafter | Part-aware 3D gen | pytorch3d | ✅ |
| ml-sharp | 3D recon | gsplat | ✅ |
| shap-e | Text/image→3D | — | ✅ |
| dust3r/fast3r | Dense stereo | croco ext | ✅ |
| Difix3D | 3D diffusion | xformers | ✅ |
| vggt | Visual grounding | — | ✅ |
| Depth-Anything-3 | Mono depth + 3DGS | xformers, gsplat | ✅ |
| Matrix-Game | Video world model | flash-attn→AITER CK | ✅ |
| **Hunyuan3D-2.1** | Image-to-3D + PBR | — (纯 PyTorch, AOTriton FA) | ✅ shape gen (60s, 344K verts, MI300X) |
| **TRELLIS.2** | Image-to-3D (O-Voxel) | flash-attn ✅, flex_gemm ✅ ([PR #18](https://github.com/JeffreyXiang/FlexGEMM/pull/18)), cumesh ✅ (`ZJLi2013/CuMesh@rocm`), nvdiffrast ✅ (`ZJLi2013/nvdiffrast@rocm`) | 🔶 集成测试进行中 |
| **video_to_world** | Video→3D recon | tinycudann→tiny-rocm-nn, gsplat, xformers | 🔶 Stage 0-1b PASS, split_k fix 待重跑 |

---

## With Other Skills

| Skill | Interaction |
|-------|-------------|
| **cursor-overnight-task-manager** | Phase 3 uses this table for ROCm lib install |
| **gpu-cluster-resource-manager** | Node selection considers ROCm version compatibility |
