# rocm3d

**中文** | [English](README_EN.md)

`rocm3d` 是面向 3D / 视频 / 世界模型 / VLA / 具身智能开源仓库的 ROCm 兼容性 skill 集合。Public release 版本聚焦一个稳定入口：

| Skill | 解决什么问题 | 何时调用 |
|---|---|---|
| [`rocm-lib-compat`](.cursor/skills/rocm-lib-compat/SKILL.md) | 给 CUDA-only repo 生成 ROCm 安装/迁移方案，并确认 xformers / gsplat / pytorch3d / nvdiffrast / spconv / flash-attn 等依赖是否有可用 ROCm 路径 | 需要把 3D / VLA / 世界模型 repo 跑在 AMD GPU 上时 |

## 使用方式

在 Cursor 中按场景调用对应 skill：

```
"使用 rocm-lib-compat skill，在 MI300 + ROCm 上迁移并验证 https://github.com/<owner>/<repo>"
```



## 已支持的 Repo

以下 repo 已在 AMD MI300X + ROCm 上验证通过。

> **License 说明**
> - 🟢 **Permissive**: 代码与模型均为宽松许可 (MIT / Apache-2.0)，可自由用于 ROCm 迁移与推广
> - 🟡 **Non-Commercial / Custom**: 代码或模型含 NC / 自定义条款，仅限研究用途
> - 🔴 **Restricted Weights**: 代码许可宽松但**模型权重受限**（如 NVIDIA License / gated），不可随 ROCm 迁移一起分发或推广
> - ❓ **Unlicensed**: 未在 repo 中发现明确许可证，迁移验证仅供内部参考
>
> **本项目仅验证 ROCm 技术兼容性，不对原始 repo 的许可证做任何修改或再授权。使用前请自行确认许可证合规性。**

### 视觉检测 / 分割 / 跟踪

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [ZJLi2013/GroundingDINO](https://github.com/ZJLi2013/GroundingDINO/tree/rocm_supported) | Open-vocabulary detection | 🟢 Apache-2.0 | HIP `MsDeformAttn` forward, torch.library custom op | ✅ 已验证（ROCm inference；GroundingDINO→SAM e2e PASS） |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Image/video segmentation | 🟢 Apache-2.0 | — (SAM2 image predictor) | ✅ 已验证（`sam2.1_hiera_base_plus.pt`, official `truck.jpg` image example） |
| [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | Grounded detection + SAM2 segmentation/tracking | 🟢 Apache-2.0 + upstream component licenses | SAM2, HF GroundingDINO | ✅ 已验证（HF GroundingDINO tiny + SAM2.1 base-plus，4 annotations；推荐后续 base） |
| [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) | GroundingDINO + SAM legacy pipeline | 🟢 Apache-2.0 | GroundingDINO ROCm fork, SAM vit_b | ✅ 已验证（legacy image e2e 输出 `truck` mask；作为兼容后备） |
| [DCDmllm/InstructSAM](https://github.com/DCDmllm/InstructSAM) | Instruction-driven instance segmentation (Qwen3-VL + SAM3) | ❓ 无 LICENSE 文件；`facebook/sam3` gated weights | flash-attn (FA2 Triton), SAM3 | ✅ 已验证（InstructSAM-2B + `fused_attention`，`truck.jpg` 输出 10 masks，peak 6016MB） |
| [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet) (torch port [ibaiGorordo/Tapir-Pytorch-Inference](https://github.com/ibaiGorordo/Tapir-Pytorch-Inference)) | 点轨迹跟踪 (TAPIR; 人类视频 pipeline 组件) | 🟢 Apache-2.0 | — (纯 PyTorch, 公开权重) | ✅ 已验证（supported-smoke, torch2.9.1+rocm7.2.1, 零 patch, MI300X） |
| [Robbyant/lingbot-vision](https://github.com/Robbyant/lingbot-vision) | 自监督 ViT backbone / 稠密空间感知 (DINOv2/v3 lineage; boundary-centric 掩码建模, ViT-S..1.1B ViT-g) | 🟢 Apache-2.0 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（supported-demo, 零改动；ViT-L + 1.1B ViT-g PCA 稠密特征, MI300X） |
| [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | 单目视频世界坐标全身人体运动恢复 (SMPLX; SIGGRAPH Asia'24) | 🟡 Other (自定义; SMPL/SMPLX 注册墙) | pytorch3d (ROCm wheel); 权重走 HF mirror (camenduru/GVHMR + SMPLer-X); DPVO 可选(跳过) | ✅ 已验证（demo.py -s 端到端 312 帧：YOLOv8x→ViTPose-h→HMR2.0a→GVHMR GV transformer→SMPLX incam+global→pytorch3d 渲染, 零源码 patch, MI300X） |

### 生成式 3D 资产 (image/text → mesh / voxel / gaussian)

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | 🟡 Tencent Community (自定义; EU/UK 限制; >1M MAU 需审批) | — (纯 PyTorch, AOTriton FA) | ✅ 已验证 |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | 部件感知 3D 生成 | 🟢 MIT | pytorch3d | ✅ 已验证 |
| [openai/shap-e](https://github.com/openai/shap-e) | 文本/图像转 3D | 🟢 MIT | — | ✅ 已验证 |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | 🟢 MIT | flash-attn, flex_gemm, cumesh, nvdiffrast | ✅ 已验证（[ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)） |
| [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | Pixel-Aligned Image-to-3D (TRELLIS.2) | ❓ 无 LICENSE 文件 | flash-attn, flex_gemm, cumesh, nvdiffrast, natten→SDPA shim | ✅ 已验证（8.7MB GLB, ~198s, MI300X） |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | 动画就绪 3D 资产 (TRELLIS) | ❓ 无 LICENSE 文件 | spconv_rocm, pytorch3d, nvdiffrast, flash-attn | ✅ 已验证（sparse encoder GPU PASS, MI300X） |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D 场景生成 | 🟡 CC BY-NC-SA 4.0 | spconv_rocm, pytorch3d, flash-attn | ✅ 已验证（sparse encoder GPU PASS, MI300X） |
| [Roblox/cube/tree/main/cubepart](https://github.com/Roblox/cube/tree/main/cubepart) | Part-aware 3D mesh generation | ❓ 待确认 | diffusers, transformers, fpsample, warp fallback | ✅ 已验证（jellyfish_car 8 parts + combined GLB, MI300X） |
| [Nelipot-Lee/SegviGen](https://github.com/Nelipot-Lee/SegviGen) | 3D 部件分割 | 🟢 MIT | flash-attn, flex_gemm, cumesh | ✅ 已验证（66K verts, ~107s） |
| [Luo-Yihao/FaithC](https://github.com/Luo-Yihao/FaithC) | 3D mesh tokenizer / Faithful Contouring (CVPR'26 Oral) | 🟢 Apache-2.0 | atom3d JIT, FaithC `_C` (hipify kernels.cu), torch_scatter shim | ✅ 已验证（demo.py encode→decode, GLB 输出, 0.32s, MI300X） |
| [kasothaphie/GenRecon](https://github.com/kasothaphie/GenRecon) | 多视图 3D 场景重建 (TRELLIS.2 族; O-Voxel 表示) | 🟢 MIT（权重公开：TUM kaldir + TRELLIS.2 base） | o_voxel `_C` (auto-hipify, GPU_ARCHS=gfx942), flash-attn / flex_gemm / cumesh (TRELLIS.2 recipe); nvdiffrast/nvdiffrec 仅 CUDA | ✅ 已验证（O-Voxel core supported-demo：mesh2ovox→1.6M-voxel→ovox2mesh 彩色 ply + GPU turntable, 零 C++/CUDA 改动, MI300X；全场景 e2e defer-heavy） |

### 稠密重建 / 多视图几何 / SfM (VGGT · DUSt3R 族)

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [naver/dust3r](https://github.com/naver/dust3r) | 稠密立体重建 | 🟡 CC BY-NC-SA 4.0 | croco (ext build) | ✅ 已验证 |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | 快速 3D 重建 | 🟡 FAIR Non-Commercial | croco (ext build) | ✅ 已验证 |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | 视觉定位 | 🟡 Meta VGGT License (自定义) | — | ✅ 已验证 |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 地图重建 | 🟢 Apache-2.0 | — | ✅ 已验证 |
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | 稠密 3D 重建 (VGGT-like) | 🟢 Apache-2.0 | — (AOTriton SDPA) | ✅ 已验证（church 286 帧 2.5 FPS） |
| [ChenYutongTHU/GGPT](https://github.com/ChenYutongTHU/GGPT) | Multiview 3D 重建 (CVPR'26, PTv3 + VGGT + SfM) | 🟢 MIT | spconv_rocm, flash-attn (ROCm fork), torch_scatter shim | ✅ 已验证（68MB PLY 点云, ~11min, MI300X） |
| [colmap/gluemap](https://github.com/colmap/gluemap) | SfM / global mapping (VGGT backend) | ❓ 待确认 | pygluemap (Ceres/pybind), VGGT, CPU pycolmap | ✅ 已验证（32 tests PASS；VGGT coarse demo 5 images, 22.69s） |
| [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | 单目/双目/RGB-D 视觉 SLAM (NeurIPS'21) | 🟢 BSD-3 | droid_backends (hipify kernels, altcorr 原子头可移植化), torch-scatter-rocm | ✅ 已验证（[上游 PR #171](https://github.com/princeton-vl/DROID-SLAM/pull/171) 单文件；build + 220 帧 ETH3D sfm_bench demo, ~15 it/s, MI300X） |
| [princeton-vl/lietorch](https://github.com/princeton-vl/lietorch) | SE3/Sim3 李群 PyTorch 后端 (SLAM/BA 基础库) | 🟢 BSD-3 | lietorch_gpu.cu (hipify, `.template` 消歧) | ✅ 已验证（DROID-SLAM 的李群后端；[上游 PR #53](https://github.com/princeton-vl/lietorch/pull/53) 单文件, v0.3, MI300X） |
| [kaichen-z/PAGE4D](https://github.com/kaichen-z/PAGE4D) | 4D 感知 (VGGT) | 🟢 Apache-2.0 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（poses+depth+points, ~70s） |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | 点云 Backbone (CVPR'24 Oral) | 🟢 MIT | spconv_rocm, flash-attn (FA2 Triton), torch_scatter shim | ✅ 已验证（ModelNet40 40/40 类 PASS, MI300X） |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D 重建 | 🟡 Apple Sample Code | gsplat | ✅ 已验证 |
| [LiteReality/LiteReality](https://github.com/LiteReality/LiteReality) | Graphics-ready 3D 场景重建 (RGB-D, NeurIPS'25) | ❓ 待确认 | GroundingDINO ROCm fork (HIP MSDA), SAM ViT-H, Qwen3-VL-8B | ✅ 已验证（感知核心 supported-demo：GroundingDINO HIP+SAM 119 检测；Qwen3-VL-8B VLM supported-smoke, MI300X；Blender Cycles 渲染 CPU 已验证，GPU(HIP) 在 CDNA/gfx942 受限、RDNA4 为已知 kernel-load bug；~200GB 素材库 defer-heavy） |

### 深度估计 / 扩散式几何

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 单目深度 + 3DGS | 🟢 Apache-2.0 | xformers, gsplat | ✅ 已验证 |
| [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 立体深度估计 (Transformer) | 🔴 **NVIDIA License (非商用)** — study only | xformers | ✅ 已验证（540x960 推理, 374.5M params, MI300X） |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D 扩散修复 | 🟡 NVIDIA License + Stability AI (非商用) | xformers | ✅ 已验证 |
| [Duisterhof/modality-forcing](https://github.com/Duisterhof/modality-forcing) | 单 DiT 联合 image-depth 生成 (FLUX.2; joint/i2d/d2i 三模态) | 🟡 代码 Apache-2.0; 模型权重 CC BY-NC 4.0 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（三模态 512², text→RGB+depth+点云, 零改动, MI300X） |
| [haoz19/world-tracing](https://github.com/haoz19/world-tracing) | Image→多层几何点云 (scene/object/dynamic; flow-matching diffusion, Wan2.1 init) | 🟡 代码+权重 CC BY-NC-ND 4.0（权重 gated 手动审批） | — (纯 PyTorch, AOTriton SDPA; DINOv2/MoGe encoder; flash-attn 可选) | ✅ 已验证（scene `r69l` 840×840, 4.23M-point 彩色点云 + 360° turntable, ~79s, MI300X；object/dynamic 权重 gated 待批） |

### 3DGS / Splatting (前馈预测 · 优化 · 可微渲染器)

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS | 🟢 MIT | gsplat, pointops | ✅ 已验证（PSNR 31.17 / SSIM 0.954） |
| [cvg/ZipSplat](https://github.com/cvg/ZipSplat) | Feed-forward 3D Gaussian compression / reconstruction | 🟢 Apache-2.0 | **amd_gsplat**, AOTriton SDPA | ✅ 已验证（office 5 images, 51,840 Gaussians, render PASS；demo-candidate） |
| [nv-tlabs/TokenGS](https://github.com/nv-tlabs/TokenGS) | 前馈式 3DGS 预测 | 🟢 Apache-2.0 | **amd_gsplat**, fused-ssim | ✅ 已验证（1.25s/scene, MI300X） |
| [ziplab/TriSplat](https://github.com/ziplab/TriSplat) | Feed-forward mesh / triangle splatting | 🟢 MIT | diff-gaussian-rasterization-w-pose, diff-triangle-rasterization, simple-knn, CroCo curope | ✅ 已验证（ROCm native extensions；LLFF room mesh export） |
| [VAST-AI-Research/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) | Single-image 3D Gaussian generation | 🟢 MIT | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（20-step / 262144 Gaussian `.ply/.splat`，视觉检查通过，MI300X） |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS（ROCm 分支） | 🟡 Inria/MPII 非商用 | diff-gaussian-rasterization | ✅ 已验证 |
| [Fictionarry/AmbiSuR](https://github.com/Fictionarry/AmbiSuR) | 3DGS 表面重建 (ICML'26, DA3 depth) | ❓ 无 LICENSE 文件 | diff-plane-rasterization (hipcc), simple-knn, pytorch3d | ✅ 已验证（ROCmBuildExtension 8 fixes, MI300X） |
| [AaronNZH/LeGS](https://github.com/AaronNZH/LeGS) | 3DGS RL 密度控制 (ICML'26) | ❓ 无 LICENSE 文件 | diff-gaussian-rasterization_fastgs (hipcc), simple-knn, fused-ssim | ✅ 已验证（100 iter 105 it/s, MI300X） |

### 3D/4D 生成

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | 4D 高斯 | 🟢 MIT | diff-gaussian-rasterization, simple-knn | ✅ 脚本生成 |
| [VITA-Group/Anything-3D](https://github.com/VITA-Group/Anything-3D) | 万物转 3D | 🟢 MIT | — | ✅ 脚本生成 |
| [any4d](https://github.com/) | 4D 生成 | ❓ | — | ✅ 脚本生成 |
| [DimensionX](https://github.com/) | 多维生成 | ❓ | — | ✅ 脚本生成 |
| [nv-tlabs/FLARE](https://github.com/nv-tlabs/FLARE) | 人脸生成 | ❓ (repo 404) | pytorch3d | ✅ 脚本生成 |
| [Gen3C](https://github.com/) | 3D 一致性生成 | ❓ | — | ✅ 脚本生成 |
| [mv-inverse](https://github.com/) | 多视角逆向 | ❓ | — | ✅ 脚本生成 |
| [jiangzhongshi/RecamMaster](https://github.com/jiangzhongshi/RecamMaster) | 相机重渲染 | ❓ (repo 404) | — | ✅ 脚本生成 |

### 视频生成 / 世界模型

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | 视频世界模型 | 🟢 MIT | flash-attn → **AITER CK** | ✅ 已验证（PR ready） |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | 学习型世界模型 | 🟢 MIT | — (device-agnostic) | ✅ [已验证](https://github.com/lucas-maes/le-wm/issues/15)（推理 + 8-GPU 训练） |
| [H-EmbodVis/HyDRA](https://github.com/H-EmbodVis/HyDRA) | 混合记忆视频世界模型 | ❓ 无 LICENSE 文件; 模型依赖 Wan2.1 | flash-attn (FA2 Triton) | ✅ 已验证（4 videos） |
| [ABU121111/DreamWorld](https://github.com/ABU121111/DreamWorld) | 视频生成 (Wan2.1 + VGGT) | ❓ 无 LICENSE 文件 | — | ✅ 已验证（2 videos, ~39min） |
| [nv-tlabs/Lyra-2](https://github.com/nv-tlabs/lyra/tree/main/Lyra-2) | 图像→3D 世界 (Wan2.1 + DA3 + GS) | 🔴 代码 Apache-2.0; **模型权重 NVIDIA License (非商用, gated)** | flash-attn, **TE→SDPA**, megatron stub | ✅ 已验证（zoom-in/out 视频, 14B, ~2h, MI300X） |
| [Sim2Reason/Sim2Reason](https://github.com/Sim2Reason/Sim2Reason) | LLM 物理推理 (VERL + Qwen2.5) | ❓ 无 LICENSE 文件; verl_v4 子目录 Apache-2.0 | vLLM→HF generate, liger-kernel | ✅ 已验证（JEEBench 123q, ~100min, MI300X） |
| [OpenImagingLab/AnyRecon](https://github.com/OpenImagingLab/AnyRecon) | 任意视角 3D 重建 (Wan2.1 14B + DiffSynth) | ❓ 无 LICENSE 文件 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（chair 视频 5.0MB, ~7.4min, MI300X） |
| [CIntellifusion/GeometryForcing](https://github.com/CIntellifusion/GeometryForcing) | 视频扩散 + 3D 几何 (ICLR 2026) | 🟢 MIT | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（3 demo GIFs, 16f×256², ~69s, MI300X） |
| [Eyeline-Labs/Vista4D](https://github.com/Eyeline-Labs/Vista4D) | 4D 视频生成 (Wan2.1 14B + LoRA) | ❓ 无 LICENSE 文件 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（384p 49 frames, ~44min, MI300X） |
| [mlzxy/rla-wm](https://github.com/mlzxy/rla-wm) | 视觉世界模型 + 机器人学习 (arXiv'26) | ❓ 无 LICENSE 文件 | amd_gsplat, nvdiffrast (ROCm fork), torch-scatter | ✅ 已验证（521M model load, 2.08GB, MI300X） |
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | 单目 4D 几何+运动重建 | 🟡 Academic Only (自定义; 禁止商用; EU 限制) | xformers, pytorch3d | 🔶 大概率 |

### VLA / 具身智能

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model (Wan2.2 DiT) | 🟢 MIT | — (纯 PyTorch, deepspeed) | ✅ 已验证（LIBERO 5/5 success） |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA 框架 (Qwen3-VL) | 🟢 MIT | — (纯 PyTorch, deepspeed) | ✅ 已验证（LIBERO avg 97.8%） |
| [JIAjindou/A2A_Flow_Matching](https://github.com/JIAjindou/A2A_Flow_Matching) | Action-to-Action Flow Matching (RSS'26) | ❓ 无 LICENSE 文件 | — (纯 PyTorch, torchcfm) | ✅ 已验证（GPU smoke test PASS, MI300X） |
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B 推理 | 🟢 Apache-2.0 | — (纯 PyTorch) | 🔶 大概率 |
| [microsoft/VITRA](https://github.com/microsoft/VITRA) | Egocentric-hand VLA (PaliGemma2-3B + DiT 动作头; 人类视频→动作) | 🟢 代码 MIT；`VITRA-VLA-3B` free MIT，`paligemma2-3b` base gated（已接受） | — (纯 PyTorch/transformers==4.47.1, AOTriton SDPA) | ✅ 已验证（supported-smoke，完整推理 action `[1,16,192]`，零源码 patch, MI300X） |

### 灵巧操作 / 人类示范重定向 (Dexterous Manipulation / Human-to-Humanoid)

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [ThunderVVV/HaWoR](https://github.com/ThunderVVV/HaWoR) | Egocentric 世界坐标 3D 手部运动重建 (masked DROID-SLAM; CVPR'25)；do-as-i-do / Wh0 的手部感知/标注**共同原语** | 🟡 代码 CC BY-NC-ND 4.0；MANO 权重注册墙 | masked DROID-SLAM (`droid_backends` hipify + altcorr 原子头), lietorch (`.template` 消歧), Metric3D v2-L, torch-scatter-rocm, WiLoR YOLO | ✅ 已验证（整条 pipeline e2e：WiLoR 检测→HaWoR 运动估计→masked DROID-SLAM→Metric3D→infiller，世界坐标连续 3D 手轨产出, MI300X；[ROCm fork](https://github.com/ZJLi2013/HaWoR/tree/amd_support)） |
| [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do) | 人类视频→灵巧手 重定向（采样式 MPC 物理优化, arXiv'26） | 🟢 代码 MIT | warp / mujoco_warp on ROCm（见 `rocm-lib-compat`）+ repo 侧 `mjwp.py` graph-capture→eager 回退 | ✅ 已验证（retargeting 阶段 1–5 端到端，MPC 收敛 object tracking pos=0.024m / quat≈6.8°, MI300X；[ROCm fork](https://github.com/ZJLi2013/do-as-i-do/tree/rocm-support)） |

### 具身柔体仿真 (Embodied Soft-Body / Cloth Simulation)

> 为具身 / 可变形世界提供物理仿真数据的两个子方向：**柔体 / 布料 FEM 求解**（BS-Cloth）与
> **physics-aligned 数据放大器**（SIM1）。前者是 CPU-only 求解器（无 GPU 代码，ROCm 不适用，但可在 AMD host 原生跑）；
> 后者的 GPU 加速走 NVIDIA Warp，核心层已在 gfx942 用官方 ROCm fork 实测跑通。

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [Simulation-Intelligence/BS-Cloth](https://github.com/Simulation-Intelligence/BS-Cloth) | B-Spline 有限元布料仿真 (ACM TOG / SIGGRAPH'26) | ❓ 无 LICENSE 文件 | — (无 GPU 代码; CPU-only 求解器) | ✅ 已验证（supported-demo-cpu：31 步布料仿真 35.8s，32 帧 OBJ 序列，AMD MI300X host CPU 原生跑；ROCm 不适用） |
| [InternRobotics/SIM1](https://github.com/InternRobotics/SIM1) | Physics-aligned 仿真器 / 可变形世界零样本数据放大 (arXiv:2604.08544) | ❓ 无 LICENSE 文件 | warp / mujoco_warp on ROCm（`ROCm/warp@amd-integration` 1.13.0+rocm.0 + `ROCm/mujoco_warp@amd-integration` 3.8.1） | 🔶 partial-rocm（warp/mujoco_warp 核心层 gfx942 实测 L1+L2 PASS；bundled newton + 布料 solver(VBD/Style3D) 未验证） |

### 抓取 (Grasping)

> **⚠️ License 警告：NVlabs 抓取相关 repo / 模型权重包含 NVIDIA 自定义或 Open Model License 条款，部分仅限非商用或受限分发。**
> **仅供学术研究 / 技术验证使用 (study purpose only)，使用前请逐项确认代码与权重许可证。**

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) | 6-DoF 抓取检测 (GraspNet-1Billion) | ❓ 无明确许可 | pointnet2 (HIPified), knn shim (纯 PyTorch) | ✅ 已验证（325/119 grasps, 5.54s, MI300X） |
| [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) | 6-DoF 扩散抓取生成 | 🔴 **NVIDIA License (非商用)** — study only | pointnet2_ops (HIPified), torch-cluster | ✅ 已验证（3 物体 demo, 0.4-1.9s, MI300X） |
| [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) | 跨 embodiment 6-DoF 抓取生成 | 🔴 代码 Apache-2.0; **模型权重 NVIDIA Open Model License** | — (纯 PyTorch; end2end cuRobo/Newton 非必需) | ✅ 已验证（ROCm inference） |
| [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) → [PyTorch port](https://github.com/elchun/contact_graspnet_pytorch) | 6-DoF 场景级抓取 | 🔴 **NVIDIA License (非商用)** — study only | — (纯 PyTorch PointNet2, 零迁移) | ✅ 已验证（3 场景 308-382 grasps, 6-10s, MI300X） |

### 部分通过 (需额外修复)

| Repo | 领域 | 许可 | 状态 | Blocker |
|------|------|------|------|---------|
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | 视频→3D 重建 | 🟢 MIT | 🔶 Stage 0-1b PASS | tinycudann split_k fix |
| [AIGeeksGroup/Lite3R](https://github.com/AIGeeksGroup/Lite3R) | 轻量 3D 重建压缩 (FP8 QAT) | ❓ 无 LICENSE 文件 | 🔶 SDPA/FP8/_scaled_mm OK | torchao >=0.17 需 PyTorch >=2.11 |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D 场景理解 (VLA) | 🟢 Apache-2.0 | 🔶 环境就绪 | 需 ScanNet 数据集 |
| [mschneider456/worldmesh](https://github.com/mschneider456/worldmesh) | Mesh 条件扩散可导航多房间场景 (ECCV'26) | 🟡 Other (自定义) | 🔶 ROCm 切片已验证（procedural layout + Apple Depth Pro 深度, MI300X） | tiny-cuda-nn / kaolin / nvdiffrast (NV-only) + 人在环 Gradio 掩码 + gated FLUX.2/sam3 |

### ❌ NVIDIA-only (不可迁移)

| Repo | 领域 | 许可 | Blocker |
|------|------|------|---------|
| [NVlabs/sage](https://github.com/NVlabs/sage) | 场景级 3D 操控 | 🟢 Apache-2.0 (代码) | Isaac Sim + cuRobo 深度绑定 NVIDIA，无 ROCm 路径 → 整体仍 NVIDIA-only |
| [Simulation-Intelligence/PAT3D](https://github.com/Simulation-Intelligence/PAT3D) | 物理感知 3D 生成 | ❓ 无 LICENSE 文件 | pyuipc (CUDA 13 私有 wheel) + CUDA 13 nightly torch — 物理引擎深度绑定 NVIDIA |
| [ShirleyMaxx/REST3D](https://github.com/ShirleyMaxx/REST3D) | 单图→物理稳定交互 3D 场景 (CMU) | 🟡 CC BY-NC 4.0 | 核心 stage3 稳定+replay 全建于 NVIDIA Isaac Gym (闭源 CUDA/PhysX, 无 AMD 后端, 已弃用)；感知 stage 依赖 gated sam3 / sam-3d-objects + Gemini API |

## 项目结构

```txt
rocm3d/
├── README.md                              # 项目入口（本文件）
├── README_EN.md                           # English version
└── .cursor/skills/
    └── rocm-lib-compat/
        └── SKILL.md                       # ROCm 库替换表 + install / compatibility patterns
``` 

## 贡献

| 你想新增/修改… | 更新这里 |
|---|---|
| ROCm 库映射（新增 repo 支持） | [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) |
| 已支持的 repo 列表（上方表格） | 本 README.md |


