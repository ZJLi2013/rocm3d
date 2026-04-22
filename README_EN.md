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

The following repos have been verified on AMD MI300X with ROCm.

> **License Legend**
> - 🟢 **Permissive**: Code and model weights are under permissive licenses (MIT / Apache-2.0). Free to use for ROCm migration and promotion.
> - 🟡 **Non-Commercial / Custom**: Code or model has NC / custom terms. Research use only.
> - 🔴 **Restricted Weights**: Code is permissive but **model weights are restricted** (e.g., NVIDIA License / gated). Cannot be redistributed or promoted alongside ROCm migration.
> - ❓ **Unlicensed**: No explicit license found in repo. Migration verification is for internal reference only.
>
> **This project only verifies ROCm technical compatibility. It does not modify or relicense the original repos. Verify license compliance before use.**

### 3D Generation & Reconstruction

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | 🟡 Tencent Community (custom; EU/UK restrictions; >1M MAU requires approval) | — (pure PyTorch, AOTriton FA) | ✅ Verified |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | Part-aware 3D generation | 🟢 MIT | pytorch3d | ✅ Verified |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D reconstruction | 🟡 Apple Sample Code | gsplat | ✅ Verified |
| [openai/shap-e](https://github.com/openai/shap-e) | Text/image to 3D | 🟢 MIT | — | ✅ Verified |
| [naver/dust3r](https://github.com/naver/dust3r) | Dense stereo reconstruction | 🟡 CC BY-NC-SA 4.0 | croco (ext build) | ✅ Verified |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | Fast 3D reconstruction | 🟡 FAIR Non-Commercial | croco (ext build) | ✅ Verified |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D diffusion fixing | 🟡 NVIDIA + Stability AI (non-commercial) | xformers | ✅ Verified |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Visual grounding | 🟡 Meta VGGT License (custom) | — | ✅ Verified |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Monocular depth + 3DGS | 🟢 Apache-2.0 | xformers, gsplat | ✅ Verified |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS (ROCm fork) | 🟡 Inria/MPII non-commercial | diff-gaussian-rasterization | ✅ Verified |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Map reconstruction | 🟢 Apache-2.0 | — | ✅ Verified |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | 🟢 MIT | flash-attn, flex_gemm, cumesh, nvdiffrast | ✅ Verified ([ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)) |
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | Dense 3D reconstruction | 🟢 Apache-2.0 | — (AOTriton SDPA) | ✅ Verified (church 286 frames @ 2.5 FPS) |
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS | 🟢 MIT | gsplat, pointops | ✅ Verified (PSNR 31.17 / SSIM 0.954) |
| [Nelipot-Lee/SegviGen](https://github.com/Nelipot-Lee/SegviGen) | 3D part segmentation | 🟢 MIT | flash-attn, flex_gemm, cumesh | ✅ Verified (66K verts, ~107s) |
| [nv-tlabs/TokenGS](https://github.com/nv-tlabs/TokenGS) | Feed-forward 3DGS prediction | 🟢 Apache-2.0 | **amd_gsplat**, fused-ssim | ✅ Verified (1.25s/scene, MI300X) |
| [kaichen-z/PAGE4D](https://github.com/kaichen-z/PAGE4D) | 4D perception (VGGT) | 🟢 Apache-2.0 | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (poses+depth+points, ~70s) |

### 3D/4D Generation (AI-generated scripts)

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | 4D Gaussians | 🟢 MIT | diff-gaussian-rasterization, simple-knn | ✅ Script generated |
| [VITA-Group/Anything-3D](https://github.com/VITA-Group/Anything-3D) | Anything to 3D | 🟢 MIT | — | ✅ Script generated |
| [any4d](https://github.com/) | 4D generation | ❓ | — | ✅ Script generated |
| [DimensionX](https://github.com/) | Multi-dim generation | ❓ | — | ✅ Script generated |
| [nv-tlabs/FLARE](https://github.com/nv-tlabs/FLARE) | Face generation | ❓ (repo 404) | pytorch3d | ✅ Script generated |
| [Gen3C](https://github.com/) | 3D-consistent generation | ❓ | — | ✅ Script generated |
| [mv-inverse](https://github.com/) | Multi-view inverse | ❓ | — | ✅ Script generated |
| [jiangzhongshi/RecamMaster](https://github.com/jiangzhongshi/RecamMaster) | Camera re-rendering | ❓ (repo 404) | — | ✅ Script generated |

### Video Generation / World Models

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | Video world model | 🟢 MIT | flash-attn → **AITER CK** | ✅ Verified (PR ready) |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | Learned world model | 🟢 MIT | — (device-agnostic) | ✅ [Verified](https://github.com/lucas-maes/le-wm/issues/15) (inference + 8-GPU training) |
| [H-EmbodVis/HyDRA](https://github.com/H-EmbodVis/HyDRA) | Hybrid-memory video world model | ❓ No LICENSE file; depends on Wan2.1 weights | flash-attn (FA2 Triton) | ✅ Verified (4 videos) |
| [ABU121111/DreamWorld](https://github.com/ABU121111/DreamWorld) | Video generation (Wan2.1 + VGGT) | ❓ No LICENSE file | — | ✅ Verified (2 videos, ~39min) |
| [nv-tlabs/Lyra-2](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2) | Image→3D world (Wan2.1 + DA3 + GS) | 🔴 Code: Apache-2.0; **Weights: NVIDIA License (non-commercial, gated)** | flash-attn, **TE→SDPA**, megatron stub | ✅ Verified (zoom-in/out video, 14B, ~2h, MI300X) |
| [Sim2Reason/Sim2Reason](https://github.com/Sim2Reason/Sim2Reason) | LLM physics reasoning (VERL + Qwen2.5) | ❓ No LICENSE file; verl_v4 subtree Apache-2.0 | vLLM→HF generate, liger-kernel | ✅ Verified (JEEBench 123q, ~100min, MI300X) |
| [OpenImagingLab/AnyRecon](https://github.com/OpenImagingLab/AnyRecon) | Arbitrary-view 3D reconstruction (Wan2.1 14B + DiffSynth) | ❓ No LICENSE file | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (chair video 5.0MB, ~7.4min, MI300X) |
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | Monocular 4D geometry + motion | 🟡 Academic Only (custom; no commercial; EU restricted) | xformers, pytorch3d | 🔶 Likely |

### VLA / Embodied AI

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model (Wan2.2 DiT) | 🟢 MIT | — (pure PyTorch, deepspeed) | ✅ Verified (LIBERO 5/5 success) |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA framework (Qwen3-VL) | 🟢 MIT | — (pure PyTorch, deepspeed) | ✅ Verified (LIBERO avg 97.8%) |
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B inference | 🟢 Apache-2.0 | — (pure PyTorch) | 🔶 Likely |

### Partially Working (needs extra fixes)

| Repo | Domain | License | Status | Blocker |
|------|--------|---------|--------|---------|
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | Video → 3D reconstruction | 🟢 MIT | 🔶 Stage 0-1b PASS | tinycudann split_k fix |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D scene generation | 🟡 CC BY-NC-SA 4.0 | 🔶 85% pass | spconv lacks ROCm GPU kernel |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D scene understanding (VLA) | 🟢 Apache-2.0 | 🔶 Env ready | Needs ScanNet dataset |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | Animate-ready 3D assets (TRELLIS) | ❓ No LICENSE file | 🔶 Model loading OK | spconv GPU kernel CUDA-only; pytorch3d/nvdiffrast/flash-attn all OK |

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
| gsplat | `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/` | 6.4 (pkg name `amd_gsplat`, import as `gsplat`) |
| pytorch3d | Pre-built ROCm wheel | 6.4 only |

See [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) for the full table, AITER integration patterns, and troubleshooting guide.

## Contributing

For new ROCm library mappings: update `.cursor/skills/rocm-lib-compat/SKILL.md`
