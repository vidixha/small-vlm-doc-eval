#!/bin/bash
# 05_setup_vllm_env.sh — separate conda env for vLLM serving (vLLM pins its own
# torch; keeping it out of the `vlmeval` env preserves the HF fallback path).
#   bash /content/drive/MyDrive/vlm_eval/scripts/05_setup_vllm_env.sh
set -euo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

if ! conda env list | grep -q "^vllm "; then
  conda create -y -n vllm python=3.12
fi
conda activate vllm
pip install -q vllm
python -c "import vllm; print('vllm', vllm.__version__)"
echo "=== vllm env ready ==="
