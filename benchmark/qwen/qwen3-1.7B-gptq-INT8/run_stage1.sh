#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/stage1-results}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-}"
ITERATIONS="${ITERATIONS:-5}"
WARMUP="${WARMUP:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_CACHED_BLOCKS="${MAX_CACHED_BLOCKS:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PROMPT_FILE="${PROMPT_FILE:-}"
DEFAULT_PYTHON="python3"
if [[ -x "$SCRIPT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$SCRIPT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_PATH" >&2
  exit 2
fi

# These options describe one fixed workload. Use --prompt/--prompt-file after
# the script command to change the workload while keeping both runs identical.
COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --iterations "$ITERATIONS"
  --warmup "$WARMUP"
  --batch-size "$BATCH_SIZE"
  --max-tokens "$MAX_TOKENS"
  --max-model-length "$MAX_MODEL_LENGTH"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-cached-blocks "$MAX_CACHED_BLOCKS"
  --block-size "$BLOCK_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --chat-template
)
if [[ -n "$PROMPT_FILE" ]]; then
  COMMON_ARGS+=(--prompt-file "$PROMPT_FILE")
fi

mkdir -p "$OUTPUT_DIR"
BASELINE_JSON="${OUTPUT_DIR%/}/baseline.json"
BASELINE_MD="${OUTPUT_DIR%/}/baseline.md"
PROFILE_JSON="${OUTPUT_DIR%/}/projection-profile.json"
PROFILE_MD="${OUTPUT_DIR%/}/projection-profile.md"

printf 'Running formal baseline (profiler disabled) ...\n'
BASELINE_ARGS=("${COMMON_ARGS[@]}" --output "$BASELINE_JSON" --report "$BASELINE_MD")
if [[ -n "$REFERENCE_MODEL_PATH" ]]; then
  BASELINE_ARGS+=(--reference-model-path "$REFERENCE_MODEL_PATH" --correctness)
fi
PROFILE_PROJECTIONS=0 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${BASELINE_ARGS[@]}" "$@"

printf '\nRunning projection profile (separate process, profiler enabled) ...\n'
PROFILE_ARGS=("${COMMON_ARGS[@]}" --output "$PROFILE_JSON" --report "$PROFILE_MD" --profile-projections)
PROFILE_PROJECTIONS=1 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${PROFILE_ARGS[@]}" "$@"

printf '\nStage 1 reports:\n  baseline JSON: %s\n  baseline Markdown: %s\n  profile JSON: %s\n  profile Markdown: %s\n' \
  "$(realpath "$BASELINE_JSON")" \
  "$(realpath "$BASELINE_MD")" \
  "$(realpath "$PROFILE_JSON")" \
  "$(realpath "$PROFILE_MD")"
