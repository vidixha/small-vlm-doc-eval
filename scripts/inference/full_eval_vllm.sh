#!/bin/bash
# inference/full_eval_vllm.sh: full eval with vLLM-hosted models (continuous batching
# + parallel client requests; much faster than sequential HF generate on T4).
#
# Per model: serve with vLLM (env `vllm`) -> wait for /v1/models -> run
# VLMEvalKit in API mode (env `vlmeval`, --base-url) -> kill server.
# Greedy decoding enforced client-side (--temperature 0). Qwen pixel caps match
# the HF-path config (min 256*28*28, max 1280*28*28) for comparability.
#   bash /content/drive/MyDrive/vlm_eval/scripts/inference/full_eval_vllm.sh [Model ...]
set -uo pipefail
source /content/miniconda3/etc/profile.d/conda.sh
unset PYTHONPATH

export LMUData=/content/drive/MyDrive/vlm_eval/LMUData
WORK_DIR=/content/drive/MyDrive/eval_work
LOG_DIR=/content/drive/MyDrive/vlm_eval/logs
PORT=8000
# SMOKE=1 -> 5-sample DocVQA smoke set instead of the full 300-sample subsets
if [ "${SMOKE:-0}" = "1" ]; then
  DATA="DocVQA_VAL_SMOKE"
  DC='{"DocVQA_VAL_SMOKE": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL_SMOKE"}}'
else
  DATA="DocVQA_VAL_SUB300 InfoVQA_VAL_SUB300"
  DC='{"DocVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "DocVQA_VAL_SUB300"},
       "InfoVQA_VAL_SUB300": {"class": "ImageVQADataset", "dataset": "InfoVQA_VAL_SUB300"}}'
fi

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

serve_and_eval() {
  local NAME=$1
  echo "=== [vLLM] serving $NAME (${REPO[$NAME]}) ==="
  conda activate vllm
  # shellcheck disable=SC2086
  # --mm-processor-cache-gb 0: the 4GB default MM cache + concurrent big
  # infographic payloads got the engine OOM-killed (12GB system RAM, no swap)
  vllm serve "${REPO[$NAME]}" --served-model-name "$NAME" --port $PORT \
    --dtype half --gpu-memory-utilization 0.85 --max-model-len 8192 \
    --mm-processor-cache-gb 0 \
    ${EXTRA[$NAME]} > "$LOG_DIR/vllm_${NAME}.log" 2>&1 &
  local SERVER_PID=$!

  # wait up to 45 min for readiness: Qwen3.5's linear-attention Triton
  # autotune alone can take 20+ min on Colab's 2-core CPU (one-time, cached)
  local ok=0
  for _ in $(seq 1 540); do
    if curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" | grep -q 200; then
      ok=1; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
    sleep 5
  done
  if [ $ok -ne 1 ]; then
    echo "!!! vLLM server for $NAME failed to start: see $LOG_DIR/vllm_${NAME}.log"
    kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    return 1
  fi

  echo "=== [eval] $NAME via API mode ==="
  conda activate vlmeval
  cd /content/VLMEvalKit
  # shellcheck disable=SC2086
  python run.py --model "$NAME" --data $DATA \
    --data-config "$DC" --work-dir "$WORK_DIR" --reuse \
    --base-url "http://127.0.0.1:$PORT/v1" --key EMPTY \
    --temperature 0 --max-tokens 512 --api-nproc 4 --retry 3  # nproc 8 -> 4: RAM headroom
  local RC=$?

  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
  sleep 10  # let GPU memory drain before next model
  echo "=== [done] $NAME (eval exit $RC) ==="
  return $RC
}

mkdir -p "$LOG_DIR"
for M in "${MODELS[@]}"; do
  serve_and_eval "$M" || echo "!!! $M failed under vLLM: fall back to inference/full_eval_hf.sh $M"
done
