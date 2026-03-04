#!/usr/bin/env bash
# shap-e - https://github.com/openai/shap-e
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n shap-e python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate shap-e

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

cd shap-e
pip install -e .
pip install pyyaml
#python -m shap_e.examples.sample_image_to_3d
