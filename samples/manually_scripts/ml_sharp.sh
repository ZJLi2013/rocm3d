#!/usr/bin/env bash
# ml-sharp - https://github.com/apple/ml-sharp
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n sharp python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate sharp

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install gsplat --index-url=https://pypi.amd.com/simple
pip install -r requirements_rocm.txt

sharp predict -i data/teaser.jpg -o data/output/ --render
