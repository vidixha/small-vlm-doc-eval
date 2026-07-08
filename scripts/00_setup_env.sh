#!/bin/bash
# 00_setup_env.sh — Miniconda + vlmeval env setup for Colab (T4).
# Rerun this script from scratch after any Colab disconnect:
#   bash /content/drive/MyDrive/vlm_eval/scripts/00_setup_env.sh
set -euo pipefail

CONDA_DIR=/content/miniconda3
ENV_NAME=vlmeval

# 1. Miniconda (local disk — fast; ephemeral, hence this script)
if [ ! -d "$CONDA_DIR" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
unset PYTHONPATH  # Colab sets this; it leaks system site-packages into the env

# Accept Anaconda channel ToS (required non-interactively since mid-2025)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 2. Dedicated env (python 3.10 — VLMEvalKit-supported)
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# 3. VLMEvalKit + deps
if [ ! -d /content/VLMEvalKit ]; then
  git clone https://github.com/open-compass/VLMEvalKit.git /content/VLMEvalKit
fi
cd /content/VLMEvalKit
pip install -e . -q
pip install timm einops rouge_score -q
# torch with CUDA comes in via VLMEvalKit deps; verify:
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== setup complete: conda env '$ENV_NAME' ready ==="
echo "Activate with: source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"
