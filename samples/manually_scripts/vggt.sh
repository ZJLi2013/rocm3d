#!/usr/bin/env bash
# vggt - https://github.com/facebookresearch/vggt
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n vggt python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate vggt

pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

#python3 quick_start.py
