#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-}"
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
ITERATIONS="${ITERATIONS:-5}"
WARMUP="${WARMUP:-2}"
if [[ -z "${PYTHON:-}" && -x "$ROOT_DIR/../../../.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/../../../.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
PROFILE_PROJECTIONS="${PROFILE_PROJECTIONS:-0}"
GPTQ_BACKEND="${GPTQ_BACKEND:-fused}"
if [[ "$PROFILE_PROJECTIONS" == "1" ]]; then
  GPTQ_BACKEND="reference"
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/operator-profile/reference/run-${RUN_ID}}"
  OUTPUT="${OUTPUT:-${OUTPUT_DIR}/projection-profile.json}"
  REPORT="${REPORT:-${OUTPUT%.*}.md}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/performance/${GPTQ_BACKEND}/run-${RUN_ID}}"
  OUTPUT="${OUTPUT:-${OUTPUT_DIR}/results.json}"
  REPORT="${REPORT:-${OUTPUT%.*}.md}"
fi

ARGS=(
  --model-path "$MODEL_PATH"
  --gptq-backend "$GPTQ_BACKEND"
  --output "$OUTPUT"
  --report "$REPORT"
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --chat-template
)
mkdir -p "$(dirname -- "$OUTPUT")"
if [[ "$PROFILE_PROJECTIONS" == "1" ]]; then
  ARGS+=(--profile-projections)
fi
if [[ -n "$REFERENCE_MODEL_PATH" ]]; then
  ARGS+=(--reference-model-path "$REFERENCE_MODEL_PATH" --correctness)
fi

exec "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${ARGS[@]}" "$@"
