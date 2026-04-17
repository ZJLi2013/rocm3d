# rocm3d

[中文](README.md) | **English**

Cursor agent skill for porting ML repos (3D generation, reconstruction, world models, video generation, etc.) to AMD ROCm.

Provides a canonical **ROCm library replacement table** — when you encounter a CUDA-dependent library
in a repo's dependencies, the skill tells Cursor exactly how to install the ROCm equivalent.

## Usage

In Cursor, invoke the skill:

```
"Use rocm-lib-compat skill to generate ROCm install script for https://github.com/<owner>/<repo>"
```

## Supported Repos

The following repos have been verified on AMD MI300X with ROCm:

### 3D Generation & Reconstruction

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | — (pure PyTorch, AOTriton FA) | ✅ Verified |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | Part-aware 3D generation | pytorch3d | ✅ Verified |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D reconstruction | gsplat | ✅ Verified |
| [openai/shap-e](https://github.com/openai/shap-e) | Text/image to 3D | — | ✅ Verified |
| [naver/dust3r](https://github.com/naver/dust3r) | Dense stereo reconstruction | croco (ext build) | ✅ Verified |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | Fast 3D reconstruction | croco (ext build) | ✅ Verified |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D diffusion fixing | xformers | ✅ Verified |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Visual grounding | — | ✅ Verified |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Monocular depth + 3DGS | xformers, gsplat | ✅ Verified |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS (ROCm fork) | diff-gaussian-rasterization | ✅ Verified |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Map reconstruction | — | ✅ Verified |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | flash-attn, flex_gemm, cumesh, nvdiffrast | ✅ Verified ([ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)) |
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | Dense 3D reconstruction (VGGT-like, depth+pose) | — (AOTriton SDPA, FlashInfer fallback) | ✅ Verified (`--use_sdpa`, 286-frame church @ 2.5 FPS) |
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS (recurrent) | gsplat, pointops | ✅ Verified (DL3DV demo, PSNR 31.17 / SSIM 0.954 / LPIPS 0.074) |

### 3D/4D Generation (AI-generated scripts)

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | 4D Gaussians | diff-gaussian-rasterization, simple-knn | ✅ Script generated |
| [VITA-Group/Anything-3D](https://github.com/VITA-Group/Anything-3D) | Anything to 3D | — | ✅ Script generated |
| [any4d](https://github.com/) | 4D generation | — | ✅ Script generated |
| [DimensionX](https://github.com/) | Multi-dim generation | — | ✅ Script generated |
| [nv-tlabs/FLARE](https://github.com/nv-tlabs/FLARE) | Face generation | pytorch3d | ✅ Script generated |
| [Gen3C](https://github.com/) | 3D-consistent generation | — | ✅ Script generated |
| [mv-inverse](https://github.com/) | Multi-view inverse | — | ✅ Script generated |
| [jiangzhongshi/RecamMaster](https://github.com/jiangzhongshi/RecamMaster) | Camera re-rendering | — | ✅ Script generated |

### Video Generation / World Models

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | Video world model | flash-attn → **AITER CK** | ✅ Verified (PR ready) |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | Learned world model | — (device-agnostic) | ✅ [Verified](https://github.com/lucas-maes/le-wm/issues/15) (inference + 8-GPU training) |
| [H-EmbodVis/HyDRA](https://github.com/H-EmbodVis/HyDRA) | Hybrid-memory video world model | flash-attn (FA2 Triton) | ✅ Verified (4 videos, FA2 -19% vs SDPA) |
| [ABU121111/DreamWorld](https://github.com/ABU121111/DreamWorld) | Video generation (Wan2.1 + VGGT) | — | ✅ Verified (2 videos, ~39min) |
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | Monocular 4D geometry + motion | xformers, pytorch3d | 🔶 Likely (clean deps, env ready) |

### VLA / Embodied AI

| Repo | Domain | Key ROCm Libs | Status |
|------|--------|---------------|--------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model (Wan2.2 DiT) | — (pure PyTorch, deepspeed) | ✅ Verified (ActionDiT 1.02B + **LIBERO eval 5/5 success**, AOTriton SDPA, out-of-box) |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA framework (Qwen3-VL + OFT/FAST/GR00T) | — (pure PyTorch, deepspeed) | ✅ Verified (8-GPU training 20K steps + **LIBERO 3-suite eval avg 97.8%**, AOTriton SDPA, out-of-box) |
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B inference | — (pure PyTorch) | 🔶 Likely (all deps installed, no CUDA blocker) |

### Partially Working (needs extra fixes)

| Repo | Domain | Key ROCm Libs | Status | Blocker |
|------|--------|---------------|--------|---------|
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D scene generation | pytorch3d, gsplat | 🔶 85% pass | spconv lacks ROCm GPU kernel |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D scene understanding (VLA) | flash-attn | 🔶 Env ready | Needs ScanNet dataset |
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | Video → 3D reconstruction | gsplat, xformers | 🔶 In progress | tinycudann → [tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) |

## Project Structure

```
.cursor/skills/rocm-lib-compat/
  SKILL.md       # Core skill — ROCm lib replacement table + AITER FA3
```

## Core Replacement Table (highlights)

ROCm 6.4 is the default base (most libs have pre-built wheels). ROCm 7.x only for flash-attn CK acceleration.

| Library | ROCm Solution | ROCm Version |
|---------|--------------|-------------|
| flash-attn | `pip install flash-attn --index-url=https://pypi.amd.com/simple` (FA2 Triton) | 6.x |
| flash-attn | `pip install aiter` — AITER CK backend, **~25% faster** | **7.x** |
| flash-attn | `pip install aiter` — AITER Triton v3 (auto-selected on 6.x) | 6.x |
| xformers | `pip install xformers --index-url https://download.pytorch.org/whl/rocm6.4` | 6.4 only |
| gsplat | `pip install gsplat --index-url=https://pypi.amd.com/simple` | 6.4 / 7.0 |
| pytorch3d | Pre-built ROCm wheel | 6.4 only |

See [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) for the full table, AITER integration patterns, and troubleshooting guide.

## Contributing

For new ROCm library mappings: update `.cursor/skills/rocm-lib-compat/SKILL.md`
