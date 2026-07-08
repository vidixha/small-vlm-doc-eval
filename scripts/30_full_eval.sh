#!/bin/bash
# 30_full_eval.sh — full runs: 3 models x {DocVQA_VAL, InfoVQA_VAL}.
# Results land in Drive work-dir, so a Colab disconnect loses nothing;
# rerunning skips completed (model, dataset) pairs via VLMEvalKit --reuse.
#   bash /content/drive/MyDrive/vlm_eval/scripts/30_full_eval.sh
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH
conda activate vlmeval
cd /content/VLMEvalKit

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData  # datasets persist on Drive
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation on 15GB T4
WORK_DIR=/content/drive/MyDrive/eval_work

# Fixed-seed 300-sample subsets (15_make_subsets.py, seed 42) — identical
# subset for every model per TASK.md §4. Full VAL sets are too slow for 6
# T4 runs in a day.
DC='{"DocVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL_SUB300"},
     "InfoVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "InfoVQA_VAL_SUB300"}}'

for M in Qwen3.5-0.8B InternVL3-1B SmolVLM-500M; do
  echo "=== FULL EVAL: $M ==="
  python run.py --model "$M" --data DocVQA_VAL_SUB300 InfoVQA_VAL_SUB300 \
    --data-config "$DC" --work-dir "$WORK_DIR" --reuse 2>&1 | tail -40
  echo "=== exit code: $? for $M ==="
done
