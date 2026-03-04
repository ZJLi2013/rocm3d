# 手写 Docker 运行脚本（3D 生成/重建 · ROCm）

基于 [知乎：AMD 显卡玩转 3D 生成与重建](https://zhihu.com) 各 repo 的 install/run 步骤整理，供 **code-autorun** 的 `docker_agent --install-script` / `--run-script` 使用。

## 脚本列表

| 脚本 | 目标 Repo | 说明 |
|------|-----------|------|
| `partcrafter.sh` | [wgsxm/PartCrafter](https://github.com/wgsxm/PartCrafter) | conda + 卸 base torch + ROCm PyTorch + settings/setup.sh |
| `ml_sharp.sh` | [apple/ml-sharp](https://github.com/apple/ml-sharp) | conda + ROCm PyTorch + gsplat + requirements_rocm.txt |
| `shap_e.sh` | [openai/shap-e](https://github.com/openai/shap-e) | conda + 卸 torch + pip install -e . + sample_image_to_3d |
| `dust3r.sh` | [naver/dust3r](https://github.com/naver/dust3r) | 安装+运行一体（遗留）；建议用 `dust3r_install.sh` + `dust3r_run.sh` 两阶段 |
| `dust3r_install.sh` | [naver/dust3r](https://github.com/naver/dust3r) | 仅安装（conda + croco build_ext），供 commit 后单独跑 sample |
| `dust3r_run.sh` | [naver/dust3r](https://github.com/naver/dust3r) | 仅 sample 运行（checkpoints + quick_start），在 saved image 中执行 |
| `difix3d.sh` | [nv-tlabs/Difix3D](https://github.com/nv-tlabs/Difix3D) | conda + xformers + 过滤 requirements + inference_difix |
| `fast3r.sh` | [facebookresearch/fast3r](https://github.com/facebookresearch/fast3r) | conda + 过滤 requirements + pip install -e . + croco + quick_start |
| `map_anything.sh` | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | conda + 卸 torch + pip install -e . + quick_start |
| `vggt.sh` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | conda + 过滤 requirements + quick_start |
| `depth_anything_3.sh` | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | conda + xformers + gsplat + 过滤 requirements + basic_gs |
| `gaussian_splatting_rocm.sh` | [expenses/gaussian-splatting (rocm)](https://github.com/expenses/gaussian-splatting) | 需 clone `-b rocm --recursive`，安装 submodules 后 train |

所有脚本假定容器内 **工作目录 `/workspace` 即为 clone 后的 repo 根目录**（与 `docker_agent` 挂载方式一致）。

## 使用方式

- **单脚本**：`--install-script` 指定安装脚本，执行后 commit 镜像；不加 `--run-script` 时仅安装并保存镜像。
- **两阶段（安装 → commit → sample）**：`--install-script` 为安装脚本，`--run-script` 为 sample 运行脚本。先执行安装并 commit，再在保存的镜像中执行 sample 脚本（便于后续插入「从 README 由 LLM 生成 sample」等步骤）。
- 脚本路径可为相对 code-autorun 根目录或绝对路径，会拷贝进 clone 出的 repo（安装脚本）或挂载进 sample 容器（运行脚本）后执行。

## 远程运行命令示例

在**已安装 Git、Python、Docker、docker-py 的远程节点**上，进入 **code-autorun** 仓库根目录后执行。

**基础镜像**（ROCm 6.4，按需替换 tag）：

```text
rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

**示例（vggt）**

```bash
cd /path/to/code-autorun
export PYTHONPATH=./src
python -m docker_agent \
  --repo_url https://github.com/facebookresearch/vggt \
  --base-image rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  --install-script samples/manually_scripts/vggt.sh \
  --run-timeout 3600
```

**示例（dust3r，两阶段：安装 → commit → sample）**

```bash
python -m docker_agent \
  --repo_url https://github.com/naver/dust3r.git \
  --base-image rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  --install-script samples/manually_scripts/dust3r_install.sh \
  --run-script samples/manually_scripts/dust3r_run.sh \
  --run-timeout 3600
```

仅安装并保存镜像（不跑 sample）：省略 `--run-script` 即可。


## 依赖

- 远程节点：Docker、Git、Python 3、`pip install -r requirements.txt`（code-autorun 根目录）
- AMD GPU 节点需已安装 ROCm 驱动，Docker 能访问 `--device=/dev/kfd`、`/dev/dri`（若 docker_agent 未传这些参数，需在 Docker 默认配置或 run 时另行加上）。
