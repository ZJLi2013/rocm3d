# rocm3d

**中文** | [English](README_EN.md)

一组 Cursor agent skill，专门面向 3D / 视频 / 世界模型 / VLA / 具身智能开源仓库在 AMD ROCm 平台上的**移植、性能分析、与 kernel 优化**。

## 核心价值（三个 skill, 三个职责）

```
[ rocm-lib-compat ]   →   [ rocm-perf-analysis ]   →   [ rocm-kernels-for-3d ]
 (让 repo 跑起来)         (量化分析 + kernel 分类)      (按 kernel family 优化)
```

| Skill | 解决什么问题 | 何时调用 |
|---|---|---|
| [`rocm-lib-compat`](.cursor/skills/rocm-lib-compat/SKILL.md) | 给 CUDA-only repo 出 ROCm 安装脚本，并确认 gsplat / spconv / nvdiffrast / flash-attn 等是否走正确 ROCm backend | repo 还跑不起来，或 profile top 落在库级 kernel 时 |
| [`rocm-perf-analysis`](.cursor/skills/rocm-perf-analysis/SKILL.md) | TraceLens + phase-split roofline + GPU peak TFLOPS，输出全 kernel ranking + 强制 kernel_type 分类 + handoff | repo 跑通后，决定优化方向时 |
| [`rocm-kernels-for-3d`](.cursor/skills/rocm-kernels-for-3d/SKILL.md) | 按实际 profile 优化有明确 kernel owner 的 family：attention / GEMM / conv / sparse conv / collective comm / fused elementwise 等；`copy/layout`、tensor indexing、`gsplat` / `nvdiffrast` 等走 handoff routing | 拿到 perf-analysis 的分类优化清单后 |

## 使用方式

在 Cursor 中按场景调用对应 skill：

```
"使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"

"使用 rocm-perf-analysis skill，给 <model> 跑 phase-split roofline 报告"

"使用 rocm-kernels-for-3d skill，根据 perf-analysis 的 kernel_type 优化 <model> 的 top kernels"
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

### 3D 生成与重建

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | 🟡 Tencent Community (自定义; EU/UK 限制; >1M MAU 需审批) | — (纯 PyTorch, AOTriton FA) | ✅ 已验证 |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | 部件感知 3D 生成 | 🟢 MIT | pytorch3d | ✅ 已验证 |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D 重建 | 🟡 Apple Sample Code | gsplat | ✅ 已验证 |
| [openai/shap-e](https://github.com/openai/shap-e) | 文本/图像转 3D | 🟢 MIT | — | ✅ 已验证 |
| [naver/dust3r](https://github.com/naver/dust3r) | 稠密立体重建 | 🟡 CC BY-NC-SA 4.0 | croco (ext build) | ✅ 已验证 |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | 快速 3D 重建 | 🟡 FAIR Non-Commercial | croco (ext build) | ✅ 已验证 |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D 扩散修复 | 🟡 NVIDIA License + Stability AI (非商用) | xformers | ✅ 已验证 |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | 视觉定位 | 🟡 Meta VGGT License (自定义) | — | ✅ 已验证 |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 单目深度 + 3DGS | 🟢 Apache-2.0 | xformers, gsplat | ✅ 已验证 |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS（ROCm 分支） | 🟡 Inria/MPII 非商用 | diff-gaussian-rasterization | ✅ 已验证 |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 地图重建 | 🟢 Apache-2.0 | — | ✅ 已验证 |
| [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Image-to-3D (O-Voxel, 4B) | 🟢 MIT | flash-attn, flex_gemm, cumesh, nvdiffrast | ✅ 已验证（[ROCm fork](https://github.com/ZJLi2013/TRELLIS.2/tree/rocm)） |
| [robbyant/lingbot-map](https://github.com/robbyant/lingbot-map) | 稠密 3D 重建 (VGGT-like) | 🟢 Apache-2.0 | — (AOTriton SDPA) | ✅ 已验证（church 286 帧 2.5 FPS） |
| [cvg/resplat](https://github.com/cvg/resplat) | Feed-forward 3DGS | 🟢 MIT | gsplat, pointops | ✅ 已验证（PSNR 31.17 / SSIM 0.954） |
| [Nelipot-Lee/SegviGen](https://github.com/Nelipot-Lee/SegviGen) | 3D 部件分割 | 🟢 MIT | flash-attn, flex_gemm, cumesh | ✅ 已验证（66K verts, ~107s） |
| [nv-tlabs/TokenGS](https://github.com/nv-tlabs/TokenGS) | 前馈式 3DGS 预测 | 🟢 Apache-2.0 | **amd_gsplat**, fused-ssim | ✅ 已验证（1.25s/scene, MI300X） |
| [kaichen-z/PAGE4D](https://github.com/kaichen-z/PAGE4D) | 4D 感知 (VGGT) | 🟢 Apache-2.0 | — (纯 PyTorch, AOTriton SDPA) | ✅ 已验证（poses+depth+points, ~70s） |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | 点云 Backbone (CVPR'24 Oral) | 🟢 MIT | spconv_rocm, flash-attn (FA2 Triton), torch_scatter shim | ✅ 已验证（ModelNet40 40/40 类 PASS, MI300X） |
| [ChenYutongTHU/GGPT](https://github.com/ChenYutongTHU/GGPT) | Multiview 3D 重建 (CVPR'26, PTv3 + VGGT + SfM) | 🟢 MIT | spconv_rocm, flash-attn (ROCm fork), torch_scatter shim | ✅ 已验证（68MB PLY 点云, ~11min, MI300X） |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D 场景生成 | 🟡 CC BY-NC-SA 4.0 | spconv_rocm, pytorch3d, flash-attn | ✅ 已验证（sparse encoder GPU PASS, MI300X） |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | 动画就绪 3D 资产 (TRELLIS) | ❓ 无 LICENSE 文件 | spconv_rocm, pytorch3d, nvdiffrast, flash-attn | ✅ 已验证（sparse encoder GPU PASS, MI300X） |
| [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 立体深度估计 (Transformer) | 🔴 **NVIDIA License (非商用)** — study only | xformers | ✅ 已验证（540x960 推理, 374.5M params, MI300XHF） |
| [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | Pixel-Aligned Image-to-3D (TRELLIS.2) | ❓ 无 LICENSE 文件 | flash-attn, flex_gemm, cumesh, nvdiffrast, natten→SDPA shim | ✅ 已验证（8.7MB GLB, ~198s, MI300X） |

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

### 抓取 (Grasping)

> **⚠️ License 警告：NVlabs 的两个 repo (GraspGen, contact_graspnet) 为 NVIDIA 自定义许可（非商用），模型权重同样受限。**
> **仅供学术研究 / 技术验证使用 (study purpose only)，严禁商业用途或再分发。**

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) | 6-DoF 抓取检测 (GraspNet-1Billion) | ❓ 无明确许可 | pointnet2 (HIPified), knn shim (纯 PyTorch) | ✅ 已验证（325/119 grasps, 5.54s, MI300XHF） |
| [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen) | 6-DoF 扩散抓取生成 | 🔴 **NVIDIA License (非商用)** — study only | pointnet2_ops (HIPified), torch-cluster | ✅ 已验证（3 物体 demo, 0.4-1.9s, MI300X） |
| [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) → [PyTorch port](https://github.com/elchun/contact_graspnet_pytorch) | 6-DoF 场景级抓取 | 🔴 **NVIDIA License (非商用)** — study only | — (纯 PyTorch PointNet2, 零迁移) | ✅ 已验证（3 场景 308-382 grasps, 6-10s, MI300X） |

### 部分通过 (需额外修复)

| Repo | 领域 | 许可 | 状态 | Blocker |
|------|------|------|------|---------|
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | 视频→3D 重建 | 🟢 MIT | 🔶 Stage 0-1b PASS | tinycudann split_k fix |
| [AIGeeksGroup/Lite3R](https://github.com/AIGeeksGroup/Lite3R) | 轻量 3D 重建压缩 (FP8 QAT) | ❓ 无 LICENSE 文件 | 🔶 SDPA/FP8/_scaled_mm OK | torchao >=0.17 需 PyTorch >=2.11 |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D 场景理解 (VLA) | 🟢 Apache-2.0 | 🔶 环境就绪 | 需 ScanNet 数据集 |

### ❌ NVIDIA-only (不可迁移)

| Repo | 领域 | 许可 | Blocker |
|------|------|------|---------|
| [NVlabs/sage](https://github.com/NVlabs/sage) | 场景级 3D 操控 | 🟢 Apache-2.0 (代码) | Isaac Sim, cuRobo, warp-lang — 深度绑定 NVIDIA 生态 |
| [Simulation-Intelligence/PAT3D](https://github.com/Simulation-Intelligence/PAT3D) | 物理感知 3D 生成 | ❓ 无 LICENSE 文件 | pyuipc (CUDA 13 私有 wheel) + CUDA 13 nightly torch — 物理引擎深度绑定 NVIDIA |

## 项目结构

```
rocm3d/
├── README.md                              # 项目入口（本文件）
├── README_EN.md                           # English version
└── .cursor/skills/
    ├── rocm-lib-compat/                   # Skill 1：兼容性
    │   └── SKILL.md                       #   ROCm 库替换表 + AITER FA3
    ├── rocm-perf-analysis/                # Skill 2：性能分析
    │   ├── SKILL.md                       #   TraceLens phase-split roofline 工作流
    │   └── gpu-specs.md                   #   MI300X/MI350X/MI355X/H100/H200/B200 peak TFLOPS 表
    └── rocm-kernels-for-3d/               # Skill 3：按 kernel family 优化
        ├── SKILL.md                       #   attention/GEMM/conv/sparse/comm/elementwise 等优化 routing + recipe
        ├── cookbook.md                    #   可执行代码骨架：model_ops wrappers / loader / compile / validate / version switch
        └── aiter-api.md                   #   AITER kernel 全景速查（dtype 矩阵 + SGLang/vLLM 集成清单，自包含）
```

设计原则：

- **以 SKILL.md 文档为主**，不维护独立 .py 脚本。所有可执行片段以 `python -c "..."` / shell block 形式直接嵌入 markdown，agent 可以直接 copy-paste。
- **三个 skill 单一职责**，互不污染；每个都可以独立加载，避免大 skill 被强制塞到所有 context 里。
- **外部工具优先**：能用 [TraceLens](https://github.com/AMD-AGI/TraceLens-internal) / [AITER](https://github.com/ROCm/aiter) / [ATOM](https://github.com/ROCm/ATOM) / [inference-skill](https://github.com/AMD-AIM/inference-skill) 的就不自己写。
- **与 AMD 官方工具链呼应**：`rocm-perf-analysis` 是 AMD-AIM 的 `inferencex-optimize` 在 3D 邻域的轻量版本，复用同样的 TraceLens + GPU peak 表 + phase-split 方法论。

## 贡献

| 你想新增/修改… | 更新这里 |
|---|---|
| ROCm 库映射（新增 repo 支持） | [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md) |
| 性能分析方法（phase split / roofline workflow） | [`.cursor/skills/rocm-perf-analysis/SKILL.md`](.cursor/skills/rocm-perf-analysis/SKILL.md) |
| 新 GPU SKU peak TFLOPS 数据 | [`.cursor/skills/rocm-perf-analysis/gpu-specs.md`](.cursor/skills/rocm-perf-analysis/gpu-specs.md) |
| Kernel family 优化 routing / recipe | [`.cursor/skills/rocm-kernels-for-3d/SKILL.md`](.cursor/skills/rocm-kernels-for-3d/SKILL.md) |
| 可执行代码骨架（model_ops / loader / compile / validate） | [`.cursor/skills/rocm-kernels-for-3d/cookbook.md`](.cursor/skills/rocm-kernels-for-3d/cookbook.md) |
| AITER kernel 全景速查（dtype 矩阵 + 集成清单） | [`.cursor/skills/rocm-kernels-for-3d/aiter-api.md`](.cursor/skills/rocm-kernels-for-3d/aiter-api.md) |
| 已支持的 repo 列表（上方表格） | 本 README.md |
