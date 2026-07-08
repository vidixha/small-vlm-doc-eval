#!/bin/bash
# 45_run_prompting.sh — CoT vs direct (non-CoT) prompting eval for the sub-1B VLMs.
#
# Per model: serve with vLLM using the SAME flags as 35_full_eval_vllm.sh (so the
# `direct` mode reproduces the headline eval), wait for /v1/models, then run
# 40_prompt_eval.py in BOTH --modes direct cot on the 300-sample subsets, then
# kill the server. Greedy decoding enforced client-side.
#   bash /content/drive/MyDrive/vlm_eval/scripts/45_run_prompting.sh [Model ...]
#   SMOKE=1 bash .../45_run_prompting.sh Qwen3.5-0.8B    # 5-sample DocVQA smoke
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData
SCRIPTS=/content/drive/MyDrive/vlm_eval/scripts
LOG_DIR=/content/drive/MyDrive/vlm_eval/logs
PORT=8000

if [ "${SMOKE:-0}" = "1" ]; then
  DATA="DocVQA_VAL_SMOKE"
else
  DATA="DocVQA_VAL_SUB300 InfoVQA_VAL_SUB300"
fi

declare -A REPO=(
  [Qwen3.5-0.8B]="Qwen/Qwen3.5-0.8B"
  [InternVL3-1B]="OpenGVLab/InternVL3-1B"
  [SmolVLM-500M]="HuggingFaceTB/SmolVLM-500M-Instruct"
)
declare -A EXTRA=(
  [Qwen3.5-0.8B]="--mm-processor-kwargs {\\\"min_pixels\\\":200704,\\\"max_pixels\\\":1003520}"
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
  # --mm-processor-cache-gb 0: same 12GB-RAM OOM guard as the main eval
  vllm serve "${REPO[$NAME]}" --served-model-name "$NAME" --port $PORT \
    --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --mm-processor-cache-gb 0 \
    ${EXTRA[$NAME]} > "$LOG_DIR/vllm_prompt_${NAME}.log" 2>&1 &
  local SPID=$!

  local ok=0
  for _ in $(seq 1 540); do   # up to 45 min (Qwen3.5 Triton autotune is slow, one-time)
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200; then
      ok=1; break
    fi
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! vLLM server for $NAME failed to start — see $LOG_DIR/vllm_prompt_${NAME}.log"
    kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
    return 1
  fi

  echo "=== [eval] $NAME : direct + cot ==="
  conda activate vlmeval
  # shellcheck disable=SC2086
  python "$SCRIPTS/40_prompt_eval.py" --model "$NAME" --data $DATA \
    --modes direct cot --port $PORT --concurrency 4
  local RC=$?

  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
  sleep 10   # let GPU memory drain before the next model
  echo "=== [done] $NAME (eval exit $RC) ==="
  return $RC
}

for M in "${MODELS[@]}"; do
  serve_and_eval "$M" || echo "!!! $M failed under vLLM"
done
echo "=== prompting eval complete — aggregate with: python $SCRIPTS/65_analyze_prompting.py ==="
