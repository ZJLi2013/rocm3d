# rocm3d-autorun

**中文** | [English](README_EN.md)

Cursor agent skill，用于将 ML 开源仓库（3D 生成、重建、世界模型、视频生成等）移植到 AMD ROCm 平台。

核心价值：提供一张 **ROCm 库替换表** — 当 repo 依赖中出现 CUDA 专用库时，告诉 Cursor 如何安装对应的 ROCm 兼容版本。

## 使用方式

在 Cursor 中调用 skill：

```
"使用 rocm-lib-compat skill，给 https://github.com/<owner>/<repo> 生成 ROCm install 脚本"
```

## 已支持的 Repo

以下 repo 已在 AMD MI300X + ROCm 上验证通过：

### 3D 生成与重建

| Repo | 领域 | 关键 ROCm 库 | 状态 |
|------|------|-------------|------|
| [Tencent/Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) | Image-to-3D + PBR | — (纯 PyTorch, AOTriton FA) | ✅ 已验证 |
| [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | 部件感知 3D 生成 | pytorch3d | ✅ 已验证 |
| [apple/ml-sharp](https://github.com/apple/ml-sharp) | 3D 重建 | gsplat | ✅ 已验证 |
| [openai/shap-e](https://github.com/openai/shap-e) | 文本/图像转 3D | — | ✅ 已验证 |
| [naver/dust3r](https://github.com/naver/dust3r) | 稠密立体重建 | croco (ext build) | ✅ 已验证 |
| [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | 快速 3D 重建 | croco (ext build) | ✅ 已验证 |
| [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | 3D 扩散修复 | xformers | ✅ 已验证 |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | 视觉定位 | — | ✅ 已验证 |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 单目深度 + 3DGS | xformers, gsplat | ✅ 已验证 |
| [expenses/gaussian-splatting](https://github.com/expenses/gaussian-splatting) | 3DGS（ROCm 分支） | diff-gaussian-rasterization | ✅ 已验证 |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | 地图重建 | — | ✅ 已验证 |

### 3D/4D 生成（AI 生成脚本）

| Repo | 领域 | 关键 ROCm 库 | 状态 |
|------|------|-------------|------|
| [fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting) | 4D 高斯 | diff-gaussian-rasterization, simple-knn | ✅ 脚本生成 |
| [VITA-Group/Anything-3D](https://github.com/VITA-Group/Anything-3D) | 万物转 3D | — | ✅ 脚本生成 |
| [any4d](https://github.com/) | 4D 生成 | — | ✅ 脚本生成 |
| [DimensionX](https://github.com/) | 多维生成 | — | ✅ 脚本生成 |
| [nv-tlabs/FLARE](https://github.com/nv-tlabs/FLARE) | 人脸生成 | pytorch3d | ✅ 脚本生成 |
| [Gen3C](https://github.com/) | 3D 一致性生成 | — | ✅ 脚本生成 |
| [mv-inverse](https://github.com/) | 多视角逆向 | — | ✅ 脚本生成 |
| [jiangzhongshi/RecamMaster](https://github.com/jiangzhongshi/RecamMaster) | 相机重渲染 | — | ✅ 脚本生成 |

### 视频生成 / 世界模型

| Repo | 领域 | 关键 ROCm 库 | 状态 |
|------|------|-------------|------|
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | 视频世界模型 | flash-attn → **AITER CK** | ✅ 已验证（PR ready） |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | 学习型世界模型 | — (device-agnostic) | ✅ [已验证](https://github.com/lucas-maes/le-wm/issues/15)（推理 + 8-GPU 训练） |
| [H-EmbodVis/HyDRA](https://github.com/H-EmbodVis/HyDRA) | 混合记忆视频世界模型 | flash-attn (FA2 Triton) | ✅ 已验证（4 videos, FA2 -19% vs SDPA） |
| [ABU121111/DreamWorld](https://github.com/ABU121111/DreamWorld) | 视频生成 (Wan2.1 + VGGT) | — | ✅ 已验证（2 videos, ~39min） |
| [TencentARC/MotionCrafter](https://github.com/TencentARC/MotionCrafter) | 单目 4D 几何+运动重建 | xformers, pytorch3d | 🔶 大概率（deps 干净，环境已就绪） |

### VLA / 具身智能

| Repo | 领域 | 关键 ROCm 库 | 状态 |
|------|------|-------------|------|
| [open-gigaai/giga-brain-0](https://github.com/open-gigaai/giga-brain-0) | VLA 3.5B 推理 | — (纯 PyTorch) | 🔶 大概率（deps 全部安装成功，无 CUDA blocker） |

### 部分通过 (需额外修复)

| Repo | 领域 | 关键 ROCm 库 | 状态 | Blocker |
|------|------|-------------|------|---------|
| [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) (v2) | Image-to-3D (O-Voxel) | flash-attn, flex_gemm, cumesh | 🔶 依赖逐步修复中 | flex_gemm ✅ ([PR #18](https://github.com/JeffreyXiang/FlexGEMM/pull/18)), cumesh ✅ ([fork](https://github.com/ZJLi2013/CuMesh)), 集成测试待验证 |
| [lukasHoel/video_to_world](https://github.com/lukasHoel/video_to_world) | 视频→3D 重建 | gsplat, xformers, tinycudann | 🔶 Stage 0-1b PASS | tinycudann→[tiny-rocm-nn](https://github.com/ZJLi2013/tiny-rocm-nn) split_k fix 待重跑全 pipeline |
| [liuwei283/RealWonder](https://github.com/liuwei283/RealWonder) | 3D 场景生成 | pytorch3d, gsplat | 🔶 85% 通过 | spconv 无 ROCm GPU kernel ([迁移中](https://github.com/ZJLi2013/spconv_rocm)) |
| [H-EmbodVis/VEGA-3D](https://github.com/H-EmbodVis/VEGA-3D) | 3D 场景理解 (VLA) | flash-attn | 🔶 环境就绪 | 需 ScanNet 数据集 |

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
| gsplat | `pip install gsplat --index-url=https://pypi.amd.com/simple` | 6.4 / 7.0 |
| pytorch3d | 预编译 ROCm wheel | 6.4 only |
| bitsandbytes | `pip install bitsandbytes` (≥v0.45.3) | 6.4+ |
| flex_gemm | `pip install . --no-build-isolation` (Triton backend) | 6.4+ |
| cumesh | `GPU_ARCHS=gfx942 pip install . --no-build-isolation` ([fork](https://github.com/ZJLi2013/CuMesh)) | 6.4+ |

完整替换表、AITER 集成模式和问题排查见 [`.cursor/skills/rocm-lib-compat/SKILL.md`](.cursor/skills/rocm-lib-compat/SKILL.md)。

## 贡献

新增 ROCm 库映射请更新 `.cursor/skills/rocm-lib-compat/SKILL.md`
