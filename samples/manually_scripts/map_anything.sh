#!/usr/bin/env bash
# MapAnything - https://github.com/facebookresearch/map-anything
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n mapanything python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

pip install -e .
# python3 quick_start.py --images /workspace/dataset/colors/*.jpg 2>/dev/null || python3 quick_start.py
