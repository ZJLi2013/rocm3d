#!/usr/bin/env bash
# Depth-Anything-3 - https://github.com/ByteDance-Seed/Depth-Anything-3
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n da3 python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate da3

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4
pip install gsplat --index-url=https://pypi.amd.com/simple --extra-index-url https://pypi.org/simple

EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers|gsplat"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

# python3 basic_gs.py
