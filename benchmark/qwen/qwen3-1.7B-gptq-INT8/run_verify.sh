#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_PATH="$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/stage1-fused-results}"
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR%/}/backend-correctness-${RUN_ID}.json}"
REPORT_MD="${REPORT_MD:-${OUTPUT_JSON%.*}.md}"

DEFAULT_PYTHON="python3"
if [[ -x "$SCRIPT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$SCRIPT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

REFERENCE_BACKEND="${REFERENCE_BACKEND:-reference}"
CANDIDATE_BACKEND="${CANDIDATE_BACKEND:-fused}"
PROMPT="${PROMPT:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-1}"
DECODE_STEPS="${DECODE_STEPS:-4}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_CACHED_BLOCKS="${MAX_CACHED_BLOCKS:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ATOL="${ATOL:-0.1}"
RTOL="${RTOL:-0.02}"
MIN_ALLCLOSE_RATE="${MIN_ALLCLOSE_RATE:-0.999}"
MIN_TOP1_RATE="${MIN_TOP1_RATE:-0.99}"
MAX_MEAN_KL="${MAX_MEAN_KL:-1e-3}"
MIN_PROJECTION_ALLCLOSE_RATE="${MIN_PROJECTION_ALLCLOSE_RATE:-}"
KEEP_LOGITS="${KEEP_LOGITS:-0}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON" "$SCRIPT_DIR/verify_gptq_backends.py" --help
fi

for flag in CHAT_TEMPLATE KEEP_LOGITS; do
  value="${!flag}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    printf '%s must be 0 or 1, got: %s\n' "$flag" "$value" >&2
    exit 2
  fi
done
VERIFY_ARGS=(
  --model-path "$MODEL_PATH"
  --reference-backend "$REFERENCE_BACKEND"
  --candidate-backend "$CANDIDATE_BACKEND"
  --decode-steps "$DECODE_STEPS"
  --max-model-length "$MAX_MODEL_LENGTH"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-cached-blocks "$MAX_CACHED_BLOCKS"
  --block-size "$BLOCK_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --atol "$ATOL"
  --rtol "$RTOL"
  --min-allclose-rate "$MIN_ALLCLOSE_RATE"
  --min-top1-rate "$MIN_TOP1_RATE"
  --max-mean-kl "$MAX_MEAN_KL"
)
EXTRA_OUTPUT_ARG=0
EXTRA_REPORT_ARG=0
EXTRA_PROMPT_ARGS=0
for arg in "$@"; do
  case "$arg" in
    --prompt|--prompt=*|--prompt-file|--prompt-file=*) EXTRA_PROMPT_ARGS=1 ;;
    --output|--output=*) EXTRA_OUTPUT_ARG=1 ;;
    --report|--report=*) EXTRA_REPORT_ARG=1 ;;
  esac
done
if [[ "$EXTRA_OUTPUT_ARG" == "0" ]]; then
  VERIFY_ARGS+=(--output "$OUTPUT_JSON")
fi
if [[ "$EXTRA_REPORT_ARG" == "0" && "$EXTRA_OUTPUT_ARG" == "0" ]]; then
  VERIFY_ARGS+=(--report "$REPORT_MD")
fi
if [[ "$EXTRA_PROMPT_ARGS" == "0" && -n "$PROMPT_FILE" ]]; then
  VERIFY_ARGS+=(--prompt-file "$PROMPT_FILE")
elif [[ "$EXTRA_PROMPT_ARGS" == "0" && -n "$PROMPT" ]]; then
  VERIFY_ARGS+=(--prompt "$PROMPT")
fi
if [[ "$CHAT_TEMPLATE" == "1" ]]; then
  VERIFY_ARGS+=(--chat-template)
fi
if [[ -n "$MIN_PROJECTION_ALLCLOSE_RATE" ]]; then
  VERIFY_ARGS+=(--min-projection-allclose-rate "$MIN_PROJECTION_ALLCLOSE_RATE")
fi
if [[ "$KEEP_LOGITS" == "1" ]]; then
  VERIFY_ARGS+=(--keep-logits)
fi

printf 'Running GPTQ correctness comparison: %s vs %s ...\n' \
  "$REFERENCE_BACKEND" "$CANDIDATE_BACKEND"
"$PYTHON" "$SCRIPT_DIR/verify_gptq_backends.py" "${VERIFY_ARGS[@]}" "$@"
