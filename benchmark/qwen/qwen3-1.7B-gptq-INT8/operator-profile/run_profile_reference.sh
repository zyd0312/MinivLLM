#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/operator-profile/reference/run-${RUN_ID}}"
DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

ARGS=(
  --model-path "$MODEL_PATH"
  --gptq-backend reference
  --profile-projections
  --iterations "${ITERATIONS:-5}"
  --warmup "${WARMUP:-2}"
  --batch-size "${BATCH_SIZE:-1}"
  --max-tokens "${MAX_TOKENS:-32}"
  --max-model-length "${MAX_MODEL_LENGTH:-512}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}"
  --max-cached-blocks "${MAX_CACHED_BLOCKS:-1024}"
  --block-size "${BLOCK_SIZE:-256}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --chat-template
  --output "${OUTPUT_DIR%/}/projection-profile.json"
  --report "${OUTPUT_DIR%/}/projection-profile.md"
)
if [[ -n "${PROMPT_FILE:-}" ]]; then
  ARGS+=(--prompt-file "$PROMPT_FILE")
fi
mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$ROOT_DIR/performance/benchmark.py" "${ARGS[@]}" "$@"
