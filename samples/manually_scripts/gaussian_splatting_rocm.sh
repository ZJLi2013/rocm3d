#!/usr/bin/env bash
# 3d-gs (Gaussian Splatting) ROCm 分支
# 需 clone: git clone -b rocm --recursive https://github.com/expenses/gaussian-splatting.git
# 运行前提：/workspace 为 clone 后的 gaussian-splatting 根目录（含 submodules）
set -e
cd /workspace

pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization

# 可选：指定数据路径；无数据时可改为仅跑 eval 或跳过 train
python3 train.py --source_path /dataset/db/playroom/ --iterations 8000 --eval 2>/dev/null || echo "Skip train (no dataset); run: python3 train.py --source_path /path/to/data --iterations 8000 --eval"
