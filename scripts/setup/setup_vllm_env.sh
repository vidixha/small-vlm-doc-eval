#!/bin/bash
# setup/setup_vllm_env.sh: separate conda env for vLLM serving (vLLM pins its own
# torch; keeping it separate avoids clobbering the vlmeval env).
#   bash /content/drive/MyDrive/vlm_eval/scripts/setup/setup_vllm_env.sh
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
