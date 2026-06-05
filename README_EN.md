# rocm3d

[中文](README.md) | **English**

A set of Cursor agent skills targeting open-source repos in 3D / video / world models / VLA / embodied AI, for **porting, performance analysis, and kernel optimization** on AMD ROCm.

## Core Value (four skills, four responsibilities)

```
[ rocm-lib-compat ]   →   [ rocm-perf-analysis ]   →   [ rocm-kernels-for-3d ]
 (make the repo run)      (measure + classify kernels)  (wire existing fast kernels)

                                   ↘ [ rocm-new-kernels ]
                                      (design missing kernels)
```

| Skill | Problem it solves | When to invoke |
|---|---|---|
| [`rocm-lib-compat`](.cursor/skills/rocm-lib-compat/SKILL.md) | Generate ROCm install scripts and verify whether gsplat / spconv / nvdiffrast / flash-attn use the intended ROCm backend | When the repo doesn't run yet, or when a profile top kernel is library-owned |
| [`rocm-perf-analysis`](.cursor/skills/rocm-perf-analysis/SKILL.md) | TraceLens + phase-split roofline + GPU peak TFLOPS; produces full kernel ranking + forced kernel_type classification + handoff | After the repo runs, when deciding what to optimize |
| [`rocm-kernels-for-3d`](.cursor/skills/rocm-kernels-for-3d/SKILL.md) | Wire existing AITER/ATOM/ROCm-library fast paths into 3D/VLA/WM models: attention / GEMM / norm / quant / MoE / compile / loader / validation | When the top kernel has a concrete existing owner |
| [`rocm-new-kernels`](.cursor/skills/rocm-new-kernels/SKILL.md) | Design missing forward-only HIP/Triton/CK kernels with K/R/W, standalone harnesses, ledgers, MI300/MI350 hardware knowhow, and GEAK-style iteration | For missing ops such as GroundingDINO MsDeformAttn |

## Usage

Invoke the appropriate skill in Cursor based on your scenario:

```
"Use rocm-lib-compat skill to generate ROCm install script for https://github.com/<owner>/<repo>"

"Use rocm-perf-analysis skill to run a phase-split roofline report for <model>"

"Use rocm-kernels-for-3d skill to optimize <model>'s top kernels by kernel_type from perf-analysis"

"Use rocm-new-kernels skill to design a ROCm forward kernel for <model>'s <missing-op>"
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

### Visual Detection / Segmentation

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [ZJLi2013/GroundingDINO](https://github.com/ZJLi2013/GroundingDINO/tree/rocm_supported) | Open-vocabulary detection | 🟢 Apache-2.0 | HIP `MsDeformAttn` forward, torch.library custom op | ✅ Verified (ROCm inference; GroundingDINO→SAM e2e PASS) |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Image/video segmentation | 🟢 Apache-2.0 | — (SAM2 image predictor) | ✅ Verified (`sam2.1_hiera_base_plus.pt`, official `truck.jpg` image example) |
| [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Grounded detection + SAM2 segmentation/tracking | 🟢 Apache-2.0 + upstream component licenses | SAM2, HF GroundingDINO | ✅ Verified (HF GroundingDINO tiny + SAM2.1 base-plus, 4 annotations; recommended base) |
| [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | GroundingDINO + SAM legacy pipeline | 🟢 Apache-2.0 | GroundingDINO ROCm fork, SAM vit_b | ✅ Verified (legacy image e2e outputs `truck` mask; compatibility fallback) |
| [DCDmllm/InstructSAM](https://github.com/DCDmllm/InstructSAM) | Instruction-driven instance segmentation (Qwen3-VL + SAM3) | ❓ No LICENSE file; `facebook/sam3` gated weights | flash-attn (FA2 Triton), SAM3 | ✅ Verified (InstructSAM-2B + `fused_attention`, `truck.jpg` outputs 10 masks, peak 6016MB) |

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
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | Dense 3D reconstruction (VGGT-like) | 🟢 Apache-2.0 | — (AOTriton SDPA) | ✅ Verified (church 286 frames @ 2.5 FPS) |
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS | 🟢 MIT | gsplat, pointops | ✅ Verified (PSNR 31.17 / SSIM 0.954) |
| [cvg/ZipSplat](https://github.com/cvg/ZipSplat) | Feed-forward 3D Gaussian compression / reconstruction | 🟢 Apache-2.0 | **amd_gsplat**, AOTriton SDPA | ✅ Verified (office 5 images, 51,840 Gaussians, render PASS; demo-candidate) |
| [Nelipot-Lee/SegviGen](https://github.com/Nelipot-Lee/SegviGen) | 3D part segmentation | 🟢 MIT | flash-attn, flex_gemm, cumesh | ✅ Verified (66K verts, ~107s) |
| [nv-tlabs/TokenGS](https://github.com/nv-tlabs/TokenGS) | Feed-forward 3DGS prediction | 🟢 Apache-2.0 | **amd_gsplat**, fused-ssim | ✅ Verified (1.25s/scene, MI300X) |
| [kaichen-z/PAGE4D](https://github.com/kaichen-z/PAGE4D) | 4D perception (VGGT) | 🟢 Apache-2.0 | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (poses+depth+points, ~70s) |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | Point cloud backbone (CVPR'24 Oral) | 🟢 MIT | spconv_rocm, flash-attn (FA2 Triton), torch_scatter shim | ✅ Verified (ModelNet40 40/40 classes PASS, MI300X) |
| [ChenYutongTHU/GGPT](https://github.com/ChenYutongTHU/GGPT) | Multiview 3D reconstruction (CVPR'26, PTv3 + VGGT + SfM) | 🟢 MIT | spconv_rocm, flash-attn (ROCm fork), torch_scatter shim | ✅ Verified (68MB PLY point cloud, ~11min, MI300X) |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D scene generation | 🟡 CC BY-NC-SA 4.0 | spconv_rocm, pytorch3d, flash-attn | ✅ Verified (sparse encoder GPU PASS, MI300X) |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | Animation-ready 3D assets (TRELLIS) | ❓ No LICENSE file | spconv_rocm, pytorch3d, nvdiffrast, flash-attn | ✅ Verified (sparse encoder GPU PASS, MI300X) |
| [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | Stereo depth estimation (Transformer) | 🔴 **NVIDIA License (non-commercial)** — study only | xformers | ✅ Verified (540x960 inference, 374.5M params, MI300XHF) |
| [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | Pixel-Aligned Image-to-3D (TRELLIS.2) | ❓ No LICENSE file | flash-attn, flex_gemm, cumesh, nvdiffrast, natten→SDPA shim | ✅ Verified (8.7MB GLB, ~198s, MI300X) |
| [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | Single-image 3D Gaussian generation | 🟢 MIT | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (20-step / 262144 Gaussian `.ply/.splat`, visual check passed, MI300X) |

| [Fictionarry/AmbiSuR](https://github.com/Fictionarry/AmbiSuR) | 3DGS surface reconstruction (ICML'26, DA3 depth) | ❓ No LICENSE file | diff-plane-rasterization (hipcc), simple-knn, pytorch3d | ✅ Verified (ROCmBuildExtension 8 fixes, MI300X) |
| [AaronNZH/LeGS](https://github.com/AaronNZH/LeGS) | 3DGS RL density control (ICML'26) | ❓ No LICENSE file | diff-gaussian-rasterization_fastgs (hipcc), simple-knn, fused-ssim | ✅ Verified (100 iter @ 105 it/s, MI300X) |

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
| [CIntellifusion/GeometryForcing](https://github.com/CIntellifusion/GeometryForcing) | Video diffusion + 3D geometry (ICLR 2026) | 🟢 MIT | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (3 demo GIFs, 16f×256², ~69s, MI300X) |
| [Eyeline-Labs/Vista4D](https://github.com/Eyeline-Labs/Vista4D) | 4D video generation (Wan2.1 14B + LoRA) | ❓ No LICENSE file | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (384p 49 frames, ~44min, MI300X) |
| [mlzxy/rla-wm](https://github.com/mlzxy/rla-wm) | Visual world model + robot learning (arXiv'26) | ❓ No LICENSE file | amd_gsplat, nvdiffrast (ROCm fork), torch-scatter | ✅ Verified (521M model load, 2.08GB, MI300X) |
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | Monocular 4D geometry + motion | 🟡 Academic Only (custom; no commercial; EU restricted) | xformers, pytorch3d | 🔶 Likely |

### VLA / Embodied AI

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model (Wan2.2 DiT) | 🟢 MIT | — (pure PyTorch, deepspeed) | ✅ Verified (LIBERO 5/5 success) |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA framework (Qwen3-VL) | 🟢 MIT | — (pure PyTorch, deepspeed) | ✅ Verified (LIBERO avg 97.8%) |
| [JIAjindou/A2A_Flow_Matching](https://github.com/JIAjindou/A2A_Flow_Matching) | Action-to-Action Flow Matching (RSS'26) | ❓ No LICENSE file | — (pure PyTorch, torchcfm) | ✅ Verified (GPU smoke test PASS, MI300X) |
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B inference | 🟢 Apache-2.0 | — (pure PyTorch) | 🔶 Likely |

### Grasping

> **⚠️ License Warning: The two NVlabs repos (GraspGen, contact_graspnet) use a custom NVIDIA License (non-commercial). Model weights are equally restricted.**
> **For academic research / technical verification only (study purpose only). Commercial use and redistribution are strictly prohibited.**

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) | 6-DoF grasp detection (GraspNet-1Billion) | ❓ No explicit license | pointnet2 (HIPified), knn shim (pure PyTorch) | ✅ Verified (325/119 grasps, 5.54s, MI300XHF) |
| [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) | 6-DoF diffusion grasp generation | 🔴 **NVIDIA License (non-commercial)** — study only | pointnet2_ops (HIPified), torch-cluster | ✅ Verified (3 objects demo, 0.4-1.9s, MI300X) |
| [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) → [PyTorch port](https://github.com/elchun/contact_graspnet_pytorch) | 6-DoF scene-level grasping | 🔴 **NVIDIA License (non-commercial)** — study only | — (pure PyTorch PointNet2, zero migration) | ✅ Verified (3 scenes, 308-382 grasps, 6-10s, MI300X) |

### Partially Working (needs extra fixes)

| Repo | Domain | License | Status | Blocker |
|------|--------|---------|--------|---------|
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | Video → 3D reconstruction | 🟢 MIT | 🔶 Stage 0-1b PASS | tinycudann split_k fix |
| [AIGeeksGroup/Lite3R](https://github.com/AIGeeksGroup/Lite3R) | Lightweight 3D reconstruction compression (FP8 QAT) | ❓ No LICENSE file | 🔶 SDPA/FP8/_scaled_mm OK | torchao >=0.17 requires PyTorch >=2.11 |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D scene understanding (VLA) | 🟢 Apache-2.0 | 🔶 Env ready | Needs ScanNet dataset |

### ❌ NVIDIA-only (cannot migrate)

| Repo | Domain | License | Blocker |
|------|--------|---------|---------|
| [NVlabs/sage](https://github.com/NVlabs/sage) | Scene-level 3D manipulation | 🟢 Apache-2.0 (code) | Isaac Sim, cuRobo, warp-lang — deeply tied to NVIDIA ecosystem |
| [Simulation-Intelligence/PAT3D](https://github.com/Simulation-Intelligence/PAT3D) | Physics-aware 3D generation | ❓ No LICENSE file | pyuipc (CUDA 13 private wheel) + CUDA 13 nightly torch — physics engine deeply tied to NVIDIA |

## Project Structure

```
rocm3d/
├── README.md                              # Chinese entry
├── README_EN.md                           # English entry (this file)
└── .cursor/skills/
    ├── rocm-lib-compat/                   # Skill 1: compatibility
    │   └── SKILL.md                       #   ROCm lib replacement table + AITER FA3
    ├── rocm-perf-analysis/                # Skill 2: performance analysis
    │   ├── SKILL.md                       #   TraceLens phase-split roofline workflow
    │   └── gpu-specs.md                   #   MI300X/MI350X/MI355X/H100/H200/B200 peak TFLOPS table
    ├── rocm-kernels-for-3d/               # Skill 3: wire existing fast kernels
    │   ├── SKILL.md                       #   AITER/ATOM wrapper routing + recipes
    │   ├── cookbook.md                    #   Executable code skeletons: model_ops wrappers / loader / compile / validate / version switch
    │   └── aiter-api.md                   #   AITER kernel reference (dtype matrix + SGLang/vLLM integration map, self-contained)
    └── rocm-new-kernels/                  # Skill 4: design missing kernels
        └── SKILL.md                       #   K/R/W + harness + MI300/MI350 knowhow + GEAK-style iteration
```

Design principles:

- **SKILL.md docs as the primary surface**; no standalone .py scripts maintained. Every executable snippet is inlined as `python -c "..."` / shell blocks that the agent can copy-paste directly.
- **One responsibility per skill**, no cross-contamination. Each skill can be loaded independently, so a large skill is never force-injected into every context.
- **External tools first**: prefer [TraceLens](https://github.com/AMD-AGI/TraceLens-internal) / [AITER](https://github.com/ROCm/aiter) / [ATOM](https://github.com/ROCm/ATOM) / [inference-skill](https://github.com/AMD-AIM/inference-skill) over reinvention.
- **Aligned with AMD's official toolchain**: `rocm-perf-analysis` is a lightweight, 3D-neighborhood port of AMD-AIM's `inferencex-optimize`, reusing the same TraceLens + GPU peak table + phase-split methodology.

## Contributing

| What you want to add/change… | Update here |
|---|---|
| ROCm library mapping (new repo support) | [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) |
| Performance analysis methodology (phase split / roofline workflow) | [`.cursor/skills/rocm-perf-analysis/SKILL.md`](.cursor/skills/rocm-perf-analysis/SKILL.md) |
| New GPU SKU peak TFLOPS data | [`.cursor/skills/rocm-perf-analysis/gpu-specs.md`](.cursor/skills/rocm-perf-analysis/gpu-specs.md) |
| Kernel-family optimization routing / recipes | [`.cursor/skills/rocm-kernels-for-3d/SKILL.md`](.cursor/skills/rocm-kernels-for-3d/SKILL.md) |
| Executable code skeletons (model_ops / loader / compile / validate) | [`.cursor/skills/rocm-kernels-for-3d/cookbook.md`](.cursor/skills/rocm-kernels-for-3d/cookbook.md) |
| AITER kernel reference (dtype matrix + integration map) | [`.cursor/skills/rocm-kernels-for-3d/aiter-api.md`](.cursor/skills/rocm-kernels-for-3d/aiter-api.md) |
| Missing-op HIP/Triton/CK kernel design | [`.cursor/skills/rocm-new-kernels/SKILL.md`](.cursor/skills/rocm-new-kernels/SKILL.md) |
| Supported repo list (tables above) | This README_EN.md |
