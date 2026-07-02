# rocm3d

[中文](README.md) | **English**

`rocm3d` is a Cursor agent skill collection for porting 3D / video / world-model / VLA / embodied-AI repos to AMD ROCm. The public release focuses on one stable entry point:

| Skill | Problem it solves | When to invoke |
|---|---|---|
| [`rocm-lib-compat`](.cursor/skills/rocm-lib-compat/SKILL.md) | Generates ROCm install / migration guidance and checks whether xformers / gsplat / pytorch3d / nvdiffrast / spconv / flash-attn have usable ROCm paths | When you need to run a 3D / VLA / world-model repo on AMD GPUs |

## Usage

Invoke the appropriate skill in Cursor based on your scenario:

```
"Use rocm-lib-compat skill to port and verify https://github.com/<owner>/<repo> on MI300 + ROCm"
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

### Visual Detection / Segmentation / Tracking

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [ZJLi2013/GroundingDINO](https://github.com/ZJLi2013/GroundingDINO/tree/rocm_supported) | Open-vocabulary detection | 🟢 Apache-2.0 | HIP `MsDeformAttn` forward, torch.library custom op | ✅ Verified (ROCm inference; GroundingDINO→SAM e2e PASS) |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Image/video segmentation | 🟢 Apache-2.0 | — (SAM2 image predictor) | ✅ Verified (`sam2.1_hiera_base_plus.pt`, official `truck.jpg` image example) |
| [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Grounded detection + SAM2 segmentation/tracking | 🟢 Apache-2.0 + upstream component licenses | SAM2, HF GroundingDINO | ✅ Verified (HF GroundingDINO tiny + SAM2.1 base-plus, 4 annotations; recommended base) |
| [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | GroundingDINO + SAM legacy pipeline | 🟢 Apache-2.0 | GroundingDINO ROCm fork, SAM vit_b | ✅ Verified (legacy image e2e outputs `truck` mask; compatibility fallback) |
| [DCDmllm/InstructSAM](https://github.com/DCDmllm/InstructSAM) | Instruction-driven instance segmentation (Qwen3-VL + SAM3) | ❓ No LICENSE file; `facebook/sam3` gated weights | flash-attn (FA2 Triton), SAM3 | ✅ Verified (InstructSAM-2B + `fused_attention`, `truck.jpg` outputs 10 masks, peak 6016MB) |
| [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) (torch port [ibaiGorordo/Tapir-Pytorch-Inference](https://github.com/ibaiGorordo/Tapir-Pytorch-Inference)) | Point trajectory tracking (TAPIR; human-video pipeline component) | 🟢 Apache-2.0 | — (pure PyTorch, public weights) | ✅ Verified (supported-smoke, torch2.9.1+rocm7.2.1, zero patch, MI300X) |

### Generative 3D Assets (image/text → mesh / voxel / gaussian)

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | 🟡 Tencent Community (custom; EU/UK restrictions; >1M MAU requires approval) | — (pure PyTorch, AOTriton FA) | ✅ Verified |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | Part-aware 3D generation | 🟢 MIT | pytorch3d | ✅ Verified |
| [openai/shap-e](https://github.com/openai/shap-e) | Text/image to 3D | 🟢 MIT | — | ✅ Verified |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | 🟢 MIT | flash-attn, flex_gemm, cumesh, nvdiffrast | ✅ Verified ([ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)) |
| [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | Pixel-Aligned Image-to-3D (TRELLIS.2) | ❓ No LICENSE file | flash-attn, flex_gemm, cumesh, nvdiffrast, natten→SDPA shim | ✅ Verified (8.7MB GLB, ~198s, MI300X) |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | Animation-ready 3D assets (TRELLIS) | ❓ No LICENSE file | spconv_rocm, pytorch3d, nvdiffrast, flash-attn | ✅ Verified (sparse encoder GPU PASS, MI300X) |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D scene generation | 🟡 CC BY-NC-SA 4.0 | spconv_rocm, pytorch3d, flash-attn | ✅ Verified (sparse encoder GPU PASS, MI300X) |
| [Roblox/cube/tree/main/cubepart](https://github.com/Roblox/cube/tree/main/cubepart) | Part-aware 3D mesh generation | ❓ TBD | diffusers, transformers, fpsample, warp fallback | ✅ Verified (jellyfish_car 8 parts + combined GLB, MI300X) |
| [Nelipot-Lee/SegviGen](https://github.com/Nelipot-Lee/SegviGen) | 3D part segmentation | 🟢 MIT | flash-attn, flex_gemm, cumesh | ✅ Verified (66K verts, ~107s) |
| [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC) | 3D mesh tokenizer / Faithful Contouring (CVPR'26 Oral) | 🟢 Apache-2.0 | atom3d JIT, FaithC `_C` (hipify kernels.cu), torch_scatter shim | ✅ Verified (demo.py encode→decode, GLB output, 0.32s, MI300X) |

### Dense Reconstruction / Multi-view Geometry / SfM (VGGT · DUSt3R family)

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [naver/dust3r](https://github.com/naver/dust3r) | Dense stereo reconstruction | 🟡 CC BY-NC-SA 4.0 | croco (ext build) | ✅ Verified |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | Fast 3D reconstruction | 🟡 FAIR Non-Commercial | croco (ext build) | ✅ Verified |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Visual grounding | 🟡 Meta VGGT License (custom) | — | ✅ Verified |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | Map reconstruction | 🟢 Apache-2.0 | — | ✅ Verified |
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | Dense 3D reconstruction (VGGT-like) | 🟢 Apache-2.0 | — (AOTriton SDPA) | ✅ Verified (church 286 frames @ 2.5 FPS) |
| [ChenYutongTHU/GGPT](https://github.com/ChenYutongTHU/GGPT) | Multiview 3D reconstruction (CVPR'26, PTv3 + VGGT + SfM) | 🟢 MIT | spconv_rocm, flash-attn (ROCm fork), torch_scatter shim | ✅ Verified (68MB PLY point cloud, ~11min, MI300X) |
| [colmap/gluemap](https://github.com/colmap/gluemap) | SfM / global mapping (VGGT backend) | ❓ TBD | pygluemap (Ceres/pybind), VGGT, CPU pycolmap | ✅ Verified (32 tests PASS; VGGT coarse demo 5 images, 22.69s) |
| [kaichen-z/PAGE4D](https://github.com/kaichen-z/PAGE4D) | 4D perception (VGGT) | 🟢 Apache-2.0 | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (poses+depth+points, ~70s) |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | Point cloud backbone (CVPR'24 Oral) | 🟢 MIT | spconv_rocm, flash-attn (FA2 Triton), torch_scatter shim | ✅ Verified (ModelNet40 40/40 classes PASS, MI300X) |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D reconstruction | 🟡 Apple Sample Code | gsplat | ✅ Verified |
| [LiteReality/LiteReality](https://github.com/LiteReality/LiteReality) | Graphics-ready 3D scene reconstruction (RGB-D, NeurIPS'25) | ❓ TBD | GroundingDINO ROCm fork (HIP MSDA), SAM ViT-H, Qwen3-VL-8B | ✅ Verified (perception core supported-demo: GroundingDINO HIP+SAM 119 detections; Qwen3-VL-8B VLM supported-smoke, MI300X; Blender Cycles CPU verified, GPU(HIP) limited on CDNA/gfx942 & known kernel-load bug on RDNA4; ~200GB material DB defer-heavy) |

### Depth Estimation / Diffusion-based Geometry

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Monocular depth + 3DGS | 🟢 Apache-2.0 | xformers, gsplat | ✅ Verified |
| [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | Stereo depth estimation (Transformer) | 🔴 **NVIDIA License (non-commercial)** — study only | xformers | ✅ Verified (540x960 inference, 374.5M params, MI300X) |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D diffusion fixing | 🟡 NVIDIA + Stability AI (non-commercial) | xformers | ✅ Verified |
| [Duisterhof/modality-forcing](https://github.com/Duisterhof/modality-forcing) | Single-DiT joint image-depth generation (FLUX.2; joint/i2d/d2i) | 🟡 Code Apache-2.0; weights CC BY-NC 4.0 | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (3 modes @ 512², text→RGB+depth+point cloud, zero patch, MI300X) |
| [haoz19/world-tracing](https://github.com/haoz19/world-tracing) | Image→multilayer geometry point cloud (scene/object/dynamic; flow-matching diffusion, Wan2.1 init) | 🟡 Code+weights CC BY-NC-ND 4.0 (weights gated, manual approval) | — (pure PyTorch, AOTriton SDPA; DINOv2/MoGe encoder; flash-attn optional) | ✅ Verified (scene `r69l` 840×840, 4.23M-point colored cloud + 360° turntable, ~79s, MI300X; object/dynamic weights gated-pending) |

### 3DGS / Splatting (feed-forward · optimization · differentiable rasterizers)

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS | 🟢 MIT | gsplat, pointops | ✅ Verified (PSNR 31.17 / SSIM 0.954) |
| [cvg/ZipSplat](https://github.com/cvg/ZipSplat) | Feed-forward 3D Gaussian compression / reconstruction | 🟢 Apache-2.0 | **amd_gsplat**, AOTriton SDPA | ✅ Verified (office 5 images, 51,840 Gaussians, render PASS; demo-candidate) |
| [nv-tlabs/TokenGS](https://github.com/nv-tlabs/TokenGS) | Feed-forward 3DGS prediction | 🟢 Apache-2.0 | **amd_gsplat**, fused-ssim | ✅ Verified (1.25s/scene, MI300X) |
| [ziplab/TriSplat](https://github.com/ziplab/TriSplat) | Feed-forward mesh / triangle splatting | 🟢 MIT | diff-gaussian-rasterization-w-pose, diff-triangle-rasterization, simple-knn, CroCo curope | ✅ Verified (ROCm native extensions; LLFF room mesh export) |
| [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | Single-image 3D Gaussian generation | 🟢 MIT | — (pure PyTorch, AOTriton SDPA) | ✅ Verified (20-step / 262144 Gaussian `.ply/.splat`, visual check passed, MI300X) |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS (ROCm fork) | 🟡 Inria/MPII non-commercial | diff-gaussian-rasterization | ✅ Verified |
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
| [microsoft/VITRA](https://github.com/microsoft/VITRA) | Egocentric-hand VLA (PaliGemma2-3B + DiT action head; human-video→action) | 🟢 Code MIT; `VITRA-VLA-3B` free MIT, `paligemma2-3b` base gated (accepted) | — (pure PyTorch/transformers==4.47.1, AOTriton SDPA) | ✅ Verified (supported-smoke, full inference action `[1,16,192]`, zero source patch, MI300X) |

### Dexterous Manipulation / Human-to-Robot

> A new embodied direction: turning **internet / egocentric / generated human video** into robot-executable
> dexterous manipulation. Video says "what to do"; physics simulation (sampling-MPC on MuJoCo Warp) recovers
> the contact/tactile dynamics video can't see. ROCm runs that physics engine on AMD.

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do) | Human-video→dexterous-hand retargeting (sampling-MPC physics optimization, arXiv'26) | 🟢 Code MIT | warp / mujoco_warp on ROCm (see `rocm-lib-compat`) + repo-side `mjwp.py` graph-capture→eager fallback | ✅ Verified (retargeting stages 1–5 full-config end-to-end, MPC converges object tracking pos=0.024m/quat~6.8°, MI300X; [ROCm fork `rocm-support`](https://github.com/ZJLi2013/do-as-i-do/tree/rocm-support), PR pending; reconstruction perception chain gated/blocked-assets) |

### Grasping

> **⚠️ License Warning: NVlabs grasping repos / model weights include NVIDIA custom or Open Model License terms; some are non-commercial or distribution-restricted.**
> **For academic research / technical verification only (study purpose only). Confirm code and weight licenses repo by repo before use.**

| Repo | Domain | License | Key ROCm Libs | Status |
|------|--------|---------|---------------|--------|
| [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) | 6-DoF grasp detection (GraspNet-1Billion) | ❓ No explicit license | pointnet2 (HIPified), knn shim (pure PyTorch) | ✅ Verified (325/119 grasps, 5.54s, MI300X) |
| [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) | 6-DoF diffusion grasp generation | 🔴 **NVIDIA License (non-commercial)** — study only | pointnet2_ops (HIPified), torch-cluster | ✅ Verified (3 objects demo, 0.4-1.9s, MI300X) |
| [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) | Cross-embodiment 6-DoF grasp generation | 🔴 Code Apache-2.0; **model weights NVIDIA Open Model License** | — (pure PyTorch; end2end cuRobo/Newton optional) | ✅ Verified (ROCm inference) |
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
| [NVlabs/sage](https://github.com/NVlabs/sage) | Scene-level 3D manipulation | 🟢 Apache-2.0 (code) | Isaac Sim + cuRobo deeply tied to NVIDIA, no ROCm path → still NVIDIA-only |
| [Simulation-Intelligence/PAT3D](https://github.com/Simulation-Intelligence/PAT3D) | Physics-aware 3D generation | ❓ No LICENSE file | pyuipc (CUDA 13 private wheel) + CUDA 13 nightly torch — physics engine deeply tied to NVIDIA |

## Project Structure

```
rocm3d/
├── README.md                              # Chinese entry
├── README_EN.md                           # English entry (this file)
└── .cursor/skills/
    └── rocm-lib-compat/
        └── SKILL.md                       # ROCm library replacement table + install / compatibility patterns
```

## Contributing

| What you want to add/change… | Update here |
|---|---|
| ROCm library mapping (new repo support) | [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) |
| Supported repo list (tables above) | This README_EN.md |
