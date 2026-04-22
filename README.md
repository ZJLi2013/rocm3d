# rocm3d

**中文** | [English](README_EN.md)

Cursor agent skill，用于将 ML 开源仓库（3D 生成、重建、世界模型、视频生成等）移植到 AMD ROCm 平台。

核心价值：提供一张 **ROCm 库替换表** — 当 repo 依赖中出现 CUDA 专用库时，告诉 Cursor 如何安装对应的 ROCm 兼容版本。

## 使用方式

在 Cursor 中调用 skill：

```
"使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"
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
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | 单目 4D 几何+运动重建 | 🟡 Academic Only (自定义; 禁止商用; EU 限制) | xformers, pytorch3d | 🔶 大概率 |

### VLA / 具身智能

| Repo | 领域 | 许可 | 关键 ROCm 库 | 状态 |
|------|------|------|-------------|------|
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | World Action Model (Wan2.2 DiT) | 🟢 MIT | — (纯 PyTorch, deepspeed) | ✅ 已验证（LIBERO 5/5 success） |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | VLA 框架 (Qwen3-VL) | 🟢 MIT | — (纯 PyTorch, deepspeed) | ✅ 已验证（LIBERO avg 97.8%） |
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B 推理 | 🟢 Apache-2.0 | — (纯 PyTorch) | 🔶 大概率 |

### 部分通过 (需额外修复)

| Repo | 领域 | 许可 | 状态 | Blocker |
|------|------|------|------|---------|
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | 视频→3D 重建 | 🟢 MIT | 🔶 Stage 0-1b PASS | tinycudann split_k fix |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D 场景生成 | 🟡 CC BY-NC-SA 4.0 | 🔶 85% 通过 | spconv 无 ROCm GPU kernel |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D 场景理解 (VLA) | 🟢 Apache-2.0 | 🔶 环境就绪 | 需 ScanNet 数据集 |
| [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | 动画就绪 3D 资产 (TRELLIS) | ❓ 无 LICENSE 文件 | 🔶 模型加载 OK | spconv GPU kernel CUDA-only; pytorch3d/nvdiffrast/flash-attn 均 OK |

### ❌ NVIDIA-only (不可迁移)

| Repo | 领域 | 许可 | Blocker |
|------|------|------|---------|
| [NVlabs/sage](https://github.com/NVlabs/sage) | 场景级 3D 操控 | 🟢 Apache-2.0 (代码) | Isaac Sim, cuRobo, warp-lang — 深度绑定 NVIDIA 生态 |

## 项目结构

```
.cursor/skills/rocm-lib-compat/
  SKILL.md       # 核心 skill — ROCm 库替换表 + AITER FA3
```

## 核心替换表（摘要）

ROCm 6.4 为默认基础环境（大部分库有 pre-built wheels），ROCm 7.x 仅用于 flash-attn CK 加速。

| 库 | ROCm 方案 | ROCm 版本 |
|----|----------|----------|
| flash-attn | `pip install flash-attn --index-url=https://pypi.amd.com/simple`（FA2 Triton） | 6.x |
| flash-attn | `pip install aiter` — AITER CK 后端，**快 ~25%** | **7.x** |
| flash-attn | `pip install aiter` — AITER Triton v3（6.x 自动选择） | 6.x |
| xformers | `pip install xformers --index-url https://download.pytorch.org/whl/rocm6.4` | 6.4 only |
| gsplat | `pip install amd_gsplat --extra-index-url=https://pypi.amd.com/rocm-6.4.3/simple/` | 6.4 (包名 `amd_gsplat`, import 仍为 `gsplat`) |
| pytorch3d | 预编译 ROCm wheel | 6.4 only |
| bitsandbytes | `pip install bitsandbytes` (≥v0.45.3) | 6.4+ |
| flex_gemm | `pip install . --no-build-isolation` (Triton backend) | 6.4+ |
| cumesh | `GPU_ARCHS=gfx942 pip install . --no-build-isolation` ([fork](https://github.com/ZJLi2013/CuMesh)) | 6.4+ |

完整替换表、AITER 集成模式和问题排查见 [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md)。

## 贡献

新增 ROCm 库映射请更新 `.cursor/skills/rocm-lib-compat/SKILL.md`
