#!/bin/bash
# 72_lora_eval.sh — evaluate the LoRA-merged Qwen3.5-0.8B on the SAME 300-sample
# subsets with the SAME serve/eval flags as the baseline: InfoVQA (target) +
# DocVQA (regression check). Model name Qwen3.5-0.8B-LoRA keeps results separate.
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData
WORK_DIR=/content/drive/MyDrive/eval_work
LOG_DIR=/content/drive/MyDrive/vlm_eval/logs
MERGED=/content/drive/MyDrive/vlm_eval/lora/merged
NAME=Qwen3.5-0.8B-LoRA
PORT=8000
DC='{"DocVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL_SUB300"},
     "InfoVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "InfoVQA_VAL_SUB300"}}'

conda activate vllm
vllm serve "$MERGED" --served-model-name "$NAME" --port $PORT \
  --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --mm-processor-cache-gb 0 \
  --mm-processor-kwargs '{"min_pixels":200704,"max_pixels":1003520}' \
  > "$LOG_DIR/vllm_${NAME}.log" 2>&1 &
SERVER_PID=$!
ok=0
for _ in $(seq 1 540); do
  curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200 && { ok=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || break
  sleep 5
done
[ $ok -ne 1 ] && { echo "!!! LoRA server failed to start"; exit 1; }

conda activate vlmeval
cd /content/VLMEvalKit
python run.py --model "$NAME" --data InfoVQA_VAL_SUB300 DocVQA_VAL_SUB300 \
  --data-config "$DC" --work-dir "$WORK_DIR" --reuse \
  --base-url "http://127.0.0.1:$PORT/v1" --key EMPTY \
  --temperature 0 --max-tokens 512 --api-nproc 4 --retry 3
RC=$?
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
echo "=== [LoRA eval done] exit $RC ==="
