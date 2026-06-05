---
name: rocm-lib-compat
version: 2.6.1
author: ZJLi2013
description: |
  ROCm library compatibility reference for porting ML repos (3D generation,
  reconstruction, world models, VLA, video generation) to AMD GPUs.
  Provides canonical CUDA→ROCm replacement table, AITER flash-attn integration,
  Docker base images, dependency cleaning patterns, and perf handoff routing
  when a profile is dominated by library-owned kernels such as gsplat, spconv,
  nvdiffrast, pytorch3d raster, torch-scatter, or flash-attn.
  Use when adapting a repo to ROCm, generating install scripts, replacing
  CUDA-specific libraries (xformers, gsplat, pytorch3d, flash-attn, triton),
  troubleshooting CUDA→ROCm build failures, or checking whether a slow kernel
  is using the intended ROCm backend before deeper optimization.
allowed-tools: [Read, Write, Glob, Grep, Shell]
---

# ROCm Library Compatibility Reference

When porting an ML repo to AMD ROCm, the key challenge is replacing
CUDA-specific libraries with ROCm-compatible equivalents. This skill
provides the canonical replacement table and install patterns.

It also owns the **library-level perf handoff**: after `rocm-perf-analysis`
identifies a top kernel, this skill answers "is the correct ROCm backend loaded,
or are we accidentally profiling a fallback?" It does not perform kernel
optimization itself; it routes the slow family to `rocm-kernels-for-3d`.

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

> **pytorch3d source build pitfall (verified 2026-03-31, MI300X):**
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
| **xformers** | `pip uninstall torchvision -y && pip install xformers torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4` | ROCm 6.4 only; 必须与 torchvision/torchaudio 一次性安装 (→ torch 2.9.1 + xformers 0.0.33.post2); 分步安装会导致 torch 版本冲突 |
| **gsplat** | `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/` | 包名 `amd_gsplat` 含预编译 `csrc.so`; import 仍为 `gsplat`; ROCm 6.4; pypi.amd.com 上的 `gsplat` 包无 compiled ext，不可用 |
| **pytorch3d** | `pip install https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl` | ROCm 6.4 only; Python 3.12 |
| **triton** | `pip install triton --index-url https://download.pytorch.org/whl/rocm6.4` | Bundled with ROCm PyTorch; rarely needed |
| **torch-geometric** | `pip install torch_geometric torch-scatter-rocm torch-sparse-rocm torch-cluster-rocm` | [pyg-rocm-build](https://github.com/ZJLi2013/pyg-rocm-build) PyPI wheels; `torch_scatter` / `torch_sparse` / `torch_cluster` 原始 import name 不变; PTv3 ROCm 7.2 MI300X 验证通过 |
| **apex** | `git clone https://github.com/ROCm/apex && cd apex && pip install . --no-build-isolation` | [ROCm/apex](https://github.com/ROCm/apex); 已含于 ROCm PyTorch Docker; hipblasLT on gfx942 |
| **diff-gaussian-rasterization** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule; hipcc compatible ✅ |
| **diff-gaussian-rasterization-w-pose** | `PYTORCH_ROCM_ARCH=gfx942 pip install git+https://github.com/ZJLi2013/diff-gaussian-rasterization-w-pose.git@rocm_support --no-build-isolation` | Gaussian Splatting SLAM pose-Jacobian fork; ROCm source fixes in [PR #4](https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose/pull/4); import name remains `diff_gaussian_rasterization` ✅ |
| **diff-triangle-rasterization** | `PYTORCH_ROCM_ARCH=gfx942 pip install git+https://github.com/ZJLi2013/diff-triangle-rasterization.git@rocm_support --no-build-isolation` | TriSplat triangle rasterizer; ROCm source fixes in [PR #3](https://github.com/trianglesplatting/diff-triangle-rasterization/pull/3); import name `diff_triangle_rasterization` ✅ |
| **simple-knn** | Source build with `PYTORCH_ROCM_ARCH=gfx942` | 3DGS submodule; hipcc compatible ✅ |
| **bitsandbytes** | `pip install bitsandbytes` (≥v0.45.3) | ROCm 6.4+ supported since v0.45.3 ✅ |
| **custom_rasterizer** (Hunyuan3D) | `pip install -e . --no-build-isolation` | Pure PyTorch C++ ext, no raw CUDA kernel ✅ |
| **flex_gemm** | `pip install . --no-build-isolation` (hipify 自动运行) | Triton backend 全算法 ROCm ✅; [PR #18](https://github.com/JeffreyXiang/FlexGEMM/pull/18); 合并前用 fork: `pip install git+https://github.com/ZJLi2013/FlexGEMM.git@rocm` |
| **cumesh** | `GPU_ARCHS=gfx942 pip install . --no-build-isolation` (hipify 自动运行) | 全 3 扩展 ROCm ✅; `cuda::std::plus`→`cub::Sum`, `cuda::std::tuple`→`rocprim::tuple`, Vec3f 加 `__host__`, nvcc flags 分支; fork: `pip install git+https://github.com/ZJLi2013/CuMesh.git@rocm` |
| **nvdiffrast** | `GPU_ARCHS=<arch> pip install git+https://github.com/ZJLi2013/nvdiffrast.git@rocm --no-build-isolation` | **ROCm 6.4 + 7.2 均验证**; RDNA3/4 (gfx1100/gfx1201) wave32 ✅ + CDNA3 (gfx942) wave64 半wavefront模拟 ✅; cudaraster 全 4 阶段 + interpolate + antialias grad + texture 全 PASS |
| **nvdiffrec** | 同 nvdiffrast | ROCm 6.4 + 7.2; RDNA4 ✅ CDNA3 ✅ |
| **spconv-cu\*** | `git clone -b rocm https://github.com/ZJLi2013/spconv_rocm.git && pip install -e .` | HIP kernel (indice pairs, Murmur3 hash) + C++ JIT ext + torch.mm GEMM; 28 tests PASS on MI300X; PTv3 standalone 40/40 类 inference ✅; ROCm 6.4+ / 7.2 |

**Flash Attention** (tiered strategy):

| Backend | ROCm | Install | Perf | 验证 |
|---------|------|---------|------|------|
| **FA2 Triton** | **6.4.3** | `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE pip install flash-attn` | baseline | ✅ TRELLIS.2 MI300X |
| AITER Triton v3 | 6.4.3 | `pip install amd-aiter` (Triton path auto-selected) | ~same | ✅ HyDRA |
| **AITER CK** | **7.2.1** | `pip install amd-aiter` (CK path auto-selected) | **-25%** | ✅ Matrix-Game (AITER ≥v0.1.13) |

FA2 Triton 安装 & 运行均需 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`，运行时 Triton JIT 编译内核。

> **⚠️ PyPI package name pitfall** — AMD AITER's PyPI distribution is **`amd-aiter`** (not `aiter`). `pip install aiter` installs the 2019 "asynchronous iterators utility" stub (`aiter-0.13.20191203`), which is unrelated and silently breaks every `from aiter.ops.mha import ...` downstream. Source: [`aiter/setup.py`](https://github.com/ROCm/aiter/blob/main/setup.py) `PACKAGE_NAME = "amd-aiter"`. Default install is JIT mode (~10 s for the wheel + ~30-60 s/kernel on first use); the "30-60 min CK pre-compile" scenario only applies when explicitly setting `PREBUILD_KERNELS=1/2/3`.

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
export PYTORCH_ROCM_ARCH="gfx942"   # MI300X/MI300X
# export PYTORCH_ROCM_ARCH="gfx1100"  # RX 7900 XTX
```

---

## Perf Handoff Routing

Use this after `rocm-perf-analysis` ranks kernels. The goal is not to tune
everything here. The goal is to verify the intended ROCm library/backend is in
use, then hand off to the right kernel-family owner.

| Profile top kernel family | Backend sanity check owned here | Handoff |
|---|---|---|
| `spconv*`, indice pair, sparse conv | Confirm `spconv_rocm` is installed/imported, not `spconv-cu*`; run the repo's smallest sparse-conv smoke and note whether time is indice generation, gather/scatter, or GEMM | `rocm-kernels-for-3d` sparse-conv TODO; upstream `spconv_rocm` issue/PR if backend kernel is wrong |
| `gsplat*`, Gaussian raster kernels | Confirm `amd_gsplat` package is installed from `pypi.amd.com` and import name `gsplat` resolves to compiled ROCm `csrc.so`, not a Python-only fallback | Owning library/upstream (`amd_gsplat` / repo-specific raster path). Do not route to generic `rocm-kernels-for-3d` |
| `nvdiffrast*`, rasterize/interpolate/antialias/texture | Confirm ROCm fork is installed with the right `GPU_ARCHS`; run import + minimal raster/interpolate smoke | ROCm nvdiffrast fork / owning repo issue. Do not route to generic `rocm-kernels-for-3d` |
| `conv2d`, `conv3d`, `miopen_convolution`, CK grouped conv, `Im3d2Col` | Confirm this is an intentional PyTorch/ROCm backend path, not CPU fallback. Record exact op names and phase (VAE encode/decode vs transformer) | `rocm-kernels-for-3d` dense/video conv + layout checklist. Do **not** default to old MIOpen tuning as the main plan |
| `copy_`, `contiguous`, transpose/permute/layout kernels | Check whether a compatibility shim or backend conversion introduced repeated layout churn | Model-side cleanup. No `rocm-kernels-for-3d` or AITER kernel replacement unless a concrete fused pattern is identified |
| `flash_attn`, `sdpa`, `attn_fwd` | Confirm which backend is actually used: PyTorch AOTriton SDPA, FA2 Triton, AITER Triton, or AITER CK. Installing `flash-attn` does not guarantee it is used by diffusers | `rocm-kernels-for-3d` attention/AITER section if attention is high-share |
| `pytorch3d` raster/ops | Confirm prebuilt ROCm wheel, not source-built CPU-only binary | pytorch3d ROCm wheel / owning repo issue; not generic kernel work |
| `torch_scatter`, `torch_sparse`, `torch_cluster` | Confirm ROCm wheels are loaded and original import names resolve correctly | If backend is correct but slow, inspect model indexing/dataflow or file against the owning library. AITER comm kernels do **not** replace tensor indexing scatter/gather |

Rule: **make it run + verify the backend + route the slow family**. If the
backend is correct but still slow, do not keep changing install recipes; move
to `rocm-kernels-for-3d` with the profile row and a minimal reproducer.

---

## AITER Flash Attention — Integration Guide

[AITER](https://github.com/ROCm/aiter) provides flash attention via:
- **Triton**: works on ROCm 6.4.3 and 7.2.1, JIT from Python
- **CK (Composable Kernel)**: ROCm 7.2.1 only, ~25% faster

### When to use AITER vs FA2 Triton

- **大部分场景**: ROCm 6.4.3 上直接 `pip install flash-attn --index-url=pypi.amd.com` 即可，不需要 AITER
- **需要 CK 加速**: 使用 ROCm 7.2.1 镜像 + `pip install amd-aiter`（AITER ≥v0.1.13；PyPI 包名是 `amd-aiter`，**不是** `aiter` — 详见上方 FA tier table 的 pitfall note）

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
| **tinycudann** | CUDA hash grid + MLP | [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn): 编译+forward ✅, backward split_k bug 已修复 (`6f32935`), video_to_world Stage 0-1b PASS, 全 pipeline 待重跑 | 2026-04-01 MI300X |
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

Verified: custom_rasterizer (Hunyuan3D ✅), o-voxel (TRELLIS.2 ✅ compile + runtime with flex_gemm).

### numpy Pin Breaks on Python 3.12

If `requirements.txt` pins `numpy==1.24.x`, pip build isolation pulls old setuptools
→ `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`.

Fix: skip the pin, use Docker-preinstalled numpy 2.x (add `numpy` to `EXCLUDE_PKGS`).

### flash-attn install pitfalls

The backend choice lives in the Flash Attention tier table above. Keep only the
install gotchas here:

- FA2 Triton requires `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` at install and
  runtime; first use JIT-compiles Triton kernels.
- `pypi.amd.com` does not provide a flash-attn wheel; install from PyPI source.
- The `rocm6.4.3` Docker image may ship a broken editable Triton install; repair
  with `pip install --force-reinstall triton --index-url https://download.pytorch.org/whl/rocm6.4`.

---

## Smoke Test

```bash
python -c "import torch; print(f'torch {torch.__version__} | HIP: {torch.cuda.is_available()}')"
```

---

## With Other Skills

| Skill | Interaction |
|-------|-------------|
| **cursor-overnight-task-manager** | Phase 3 uses this table for ROCm lib install |
| **gpu-cluster-resource-manager** | Node selection considers ROCm version compatibility |

Verified repo status lives in the root `README.md` / `README_EN.md`; keep this
skill focused on install, dependency replacement, and backend sanity checks.
