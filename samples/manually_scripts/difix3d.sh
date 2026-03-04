#!/usr/bin/env bash
# Difix3D - https://github.com/nv-tlabs/Difix3D
# 运行前提：/workspace 为 clone 后的 repo 根目录
set -e
cd /workspace

conda create -n difix3d python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate difix3d

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4

EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

# python src/inference_difix.py \
#   --model_name "nvidia/difix_ref" \
#   --input_image "assets/example_input.png" \
#   --ref_image "assets/example_ref.png" \
#   --prompt "remove degradation" \
#   --output_dir "outputs/difix"
