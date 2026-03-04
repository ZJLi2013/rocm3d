#!/usr/bin/env bash
# dust3r - https://github.com/naver/dust3r
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n dust3r python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate dust3r

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_optional.txt

export PYTORCH_ROCM_ARCH="gfx942"
# avoid "dubious ownership" when /workspace is bind-mounted (container user != host owner)
git config --global --add safe.directory /workspace
git config --global --add safe.directory /workspace/croco
# apply this PR first https://github.com/naver/croco/pull/48/changes
cd croco
git fetch origin pull/48/head:pr48-rocm
git merge pr48-rocm --no-edit
cd ..
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# --- sample run（建议拆为两阶段：用 dust3r_install.sh 安装并 commit，再用 dust3r_run.sh 在 saved image 中跑 sample）---
# mkdir -p checkpoints/
# wget -q https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/ || true
# python3 quick_start.py
