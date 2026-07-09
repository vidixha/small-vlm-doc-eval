#!/bin/bash
# inference/smoke_test.sh: mandatory 5-sample smoke test per model on DocVQA_VAL
# (TASK.md Cell 5). Run AFTER setup/setup_env.sh and setup/register_models.py.
#   bash /content/drive/MyDrive/vlm_eval/scripts/inference/smoke_test.sh [ModelName ...]
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH
conda activate vlmeval
cd /content/VLMEvalKit

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData  # datasets persist on Drive
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation on 15GB T4
WORK_DIR=/content/drive/MyDrive/eval_work
MODELS=("${@:-Qwen3.5-0.8B InternVL3-1B SmolVLM-500M}")
[ $# -eq 0 ] && MODELS=(Qwen3.5-0.8B InternVL3-1B SmolVLM-500M)

# Current VLMEvalKit has no --limit; a 5-row custom TSV (built by
# setup/make_subsets.py) + --data-config serves as the smoke set.
DC='{"DocVQA_VAL_SMOKE": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL_SMOKE"}}'

for M in "${MODELS[@]}"; do
  echo "=== SMOKE TEST: $M ==="
  python run.py --model "$M" --data DocVQA_VAL_SMOKE --data-config "$DC" \
    --work-dir "$WORK_DIR" 2>&1 | tail -30
  echo "=== exit code: $? for $M ==="
done
