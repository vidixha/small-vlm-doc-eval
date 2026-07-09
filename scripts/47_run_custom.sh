#!/bin/bash
# 47_run_custom.sh — run the CoT-vs-direct prompting eval on the hand-annotated
# custom document set (CustomDocVQA), built by 95_build_custom_tsv.py.
#
# Same serving flags and driver as 45_run_prompting.sh, just pointed at the custom
# TSV. Build the TSV first:  python scripts/95_build_custom_tsv.py
#   bash /content/drive/MyDrive/vlm_eval/scripts/47_run_custom.sh [Model ...]
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData
SCRIPTS=/content/drive/MyDrive/vlm_eval/scripts
LOG_DIR=/content/drive/MyDrive/vlm_eval/logs
PORT=8000
DATA="CustomDocVQA"

if [ ! -f "$LMUData/CustomDocVQA.tsv" ]; then
  echo "!!! $LMUData/CustomDocVQA.tsv not found — run: python $SCRIPTS/95_build_custom_tsv.py"
  exit 1
fi

declare -A REPO=(
  [Qwen3.5-0.8B]="Qwen/Qwen3.5-0.8B"
  [InternVL3-1B]="OpenGVLab/InternVL3-1B"
  [SmolVLM-500M]="HuggingFaceTB/SmolVLM-500M-Instruct"
)
declare -A EXTRA=(
  [Qwen3.5-0.8B]='--mm-processor-kwargs {"min_pixels":200704,"max_pixels":1003520}'
  [InternVL3-1B]="--trust-remote-code"
  [SmolVLM-500M]=""
)

MODELS=("$@")
[ $# -eq 0 ] && MODELS=(Qwen3.5-0.8B InternVL3-1B SmolVLM-500M)
mkdir -p "$LOG_DIR"

serve_and_eval() {
  local NAME=$1
  echo "=== [vLLM] serving $NAME (${REPO[$NAME]}) ==="
  conda activate vllm
  # shellcheck disable=SC2086
  vllm serve "${REPO[$NAME]}" --served-model-name "$NAME" --port $PORT \
    --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --mm-processor-cache-gb 0 \
    ${EXTRA[$NAME]} > "$LOG_DIR/vllm_custom_${NAME}.log" 2>&1 &
  local SPID=$!

  local ok=0
  for _ in $(seq 1 540); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200; then
      ok=1; break
    fi
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! vLLM server for $NAME failed to start — see $LOG_DIR/vllm_custom_${NAME}.log"
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    return 1
  fi

  echo "=== [eval] $NAME on CustomDocVQA : direct + cot ==="
  conda activate vlmeval
  python "$SCRIPTS/40_prompt_eval.py" --model "$NAME" --data $DATA \
    --modes direct cot --port $PORT --concurrency 4 --latency-samples 0

  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 10
  echo "=== [done] $NAME ==="
}

for M in "${MODELS[@]}"; do
  serve_and_eval "$M" || echo "!!! $M failed under vLLM"
done
echo "=== custom eval complete — aggregate with: python $SCRIPTS/66_analyze_custom.py ==="
