#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results.json}"
ITERATIONS="${ITERATIONS:-5}"
WARMUP="${WARMUP:-2}"
PYTHON="${PYTHON:-python3}"
PROFILE_PROJECTIONS="${PROFILE_PROJECTIONS:-0}"

ARGS=(
  --model-path "$MODEL_PATH"
  --output "$OUTPUT"
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --chat-template
)
if [[ "$PROFILE_PROJECTIONS" == "1" ]]; then
  ARGS+=(--profile-projections)
fi
if [[ -n "$REFERENCE_MODEL_PATH" ]]; then
  ARGS+=(--reference-model-path "$REFERENCE_MODEL_PATH" --correctness)
fi

exec "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${ARGS[@]}" "$@"
