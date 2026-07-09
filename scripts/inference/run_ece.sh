#!/bin/bash
# inference/run_ece.sh: serve each model (warm caches) and run the ECE+latency pass
# (inference/ece_latency.py) over both 300-sample subsets. Same serve flags as the
# main eval for comparability. Resumable: inference/ece_latency.py appends JSONL and
# skips already-done indices.
#   bash /content/drive/MyDrive/vlm_eval/scripts/inference/run_ece.sh [Model ...]
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData
LOG_DIR=/content/drive/MyDrive/vlm_eval/logs
SCRIPTS=/content/drive/MyDrive/vlm_eval/scripts
PORT=8000

declare -A REPO=(
  [Qwen3.5-0.8B]="Qwen/Qwen3.5-0.8B"
  [InternVL3-1B]="OpenGVLab/InternVL3-1B"
  [SmolVLM-500M]="HuggingFaceTB/SmolVLM-500M-Instruct"
)
declare -A EXTRA=(
  [Qwen3.5-0.8B]="--mm-processor-kwargs {\"min_pixels\":200704,\"max_pixels\":1003520}"
  [InternVL3-1B]="--trust-remote-code"
  [SmolVLM-500M]=""
)

MODELS=("$@")
[ $# -eq 0 ] && MODELS=(Qwen3.5-0.8B InternVL3-1B SmolVLM-500M)

for NAME in "${MODELS[@]}"; do
  echo "=== [ECE] serving $NAME ==="
  conda activate vllm
  # shellcheck disable=SC2086
  vllm serve "${REPO[$NAME]}" --served-model-name "$NAME" --port $PORT \
    --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --mm-processor-cache-gb 0 \
    ${EXTRA[$NAME]} > "$LOG_DIR/vllm_ece_${NAME}.log" 2>&1 &
  SERVER_PID=$!
  ok=0
  for _ in $(seq 1 540); do
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200 && { ok=1; break; }
    kill -0 $SERVER_PID 2>/dev/null || break
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! [ECE] server for $NAME failed to start"
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    continue
  fi

  conda activate vlmeval
  python "$SCRIPTS/inference/ece_latency.py" --model "$NAME" \
    --data DocVQA_VAL InfoVQA_VAL \
    --concurrency 4 --latency-samples 30
  RC=$?
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  sleep 10
  echo "=== [ECE done] $NAME (exit $RC) ==="
done
