#!/usr/bin/env bash
# fast3r - https://github.com/facebookresearch/fast3r
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n fast3r python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate fast3r

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt
pip install -e .
python3 -m pip install --force-reinstall --no-cache-dir "setuptools==80.9.0"

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

#python3 quick_start.py --images /workspace/dataset/colors/sub/*.jpg 2>/dev/null || python3 quick_start.py
