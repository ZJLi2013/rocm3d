#!/usr/bin/env bash
# PartCrafter - https://github.com/wgsxm/PartCrafter
# 运行前提：/workspace 为 clone 后的 repo 根目录（docker_agent 挂载）
set -e
cd /workspace

conda create -n partcrafter python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate partcrafter

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4

bash settings/setup.sh

#python3 -m scripts.inference_partcrafter_scene --image_path assets/images_scene/np6_0192a842-531c-419a-923e-28db4add8656_DiningRoom-31158.png
