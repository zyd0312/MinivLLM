#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=("$@")
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/stage1-fused-results}"
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
GPTQ_BACKEND="${GPTQ_BACKEND:-fused}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-0}"
CORRECTNESS_BACKEND="${CORRECTNESS_BACKEND:-$GPTQ_BACKEND}"
CORRECTNESS_DECODE_STEPS="${CORRECTNESS_DECODE_STEPS:-4}"
DEFAULT_PYTHON="python3"
if [[ -x "$SCRIPT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$SCRIPT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -d "$MODEL_PATH" ]]; then
  printf 'Model directory does not exist: %s\n' "$MODEL_PATH" >&2
  exit 2
fi
if [[ "$RUN_BASELINE" != "0" && "$RUN_BASELINE" != "1" ]]; then
  printf 'RUN_BASELINE must be 0 or 1, got: %s\n' "$RUN_BASELINE" >&2
  exit 2
fi
if [[ "$RUN_CORRECTNESS" != "0" && "$RUN_CORRECTNESS" != "1" ]]; then
  printf 'RUN_CORRECTNESS must be 0 or 1, got: %s\n' "$RUN_CORRECTNESS" >&2
  exit 2
fi
if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  case "$CORRECTNESS_BACKEND" in
    fused|fused_decode|fused_prefill) ;;
    *) printf 'Unsupported CORRECTNESS_BACKEND: %s\n' "$CORRECTNESS_BACKEND" >&2; exit 2 ;;
  esac
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

if [[ "$RUN_BASELINE" == "1" ]]; then
  printf 'Running formal baseline (profiler disabled) ...\n'
  BASELINE_ARGS=("${COMMON_ARGS[@]}" --gptq-backend "$GPTQ_BACKEND" --output "$BASELINE_JSON" --report "$BASELINE_MD")
  if [[ -n "$REFERENCE_MODEL_PATH" ]]; then
    BASELINE_ARGS+=(--reference-model-path "$REFERENCE_MODEL_PATH" --correctness)
  fi
  PROFILE_PROJECTIONS=0 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${BASELINE_ARGS[@]}" "$@"
else
  printf 'Skipping formal baseline (RUN_BASELINE=0) ...\n'
fi

printf '\nRunning projection profile (separate process, profiler enabled) ...\n'
PROFILE_ARGS=("${COMMON_ARGS[@]}" --gptq-backend reference --output "$PROFILE_JSON" --report "$PROFILE_MD" --profile-projections)
PROFILE_PROJECTIONS=1 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${PROFILE_ARGS[@]}" "$@"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  printf '\nRunning reference/candidate correctness comparison ...\n'
  CORRECTNESS_ARGS=(
    --model-path "$MODEL_PATH"
    --reference-backend reference
    --candidate-backend "$CORRECTNESS_BACKEND"
    --decode-steps "$CORRECTNESS_DECODE_STEPS"
    --max-model-length "$MAX_MODEL_LENGTH"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-cached-blocks "$MAX_CACHED_BLOCKS"
    --block-size "$BLOCK_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --chat-template
    --output "${OUTPUT_DIR%/}/backend-correctness.json"
    --report "${OUTPUT_DIR%/}/backend-correctness.md"
  )
  if [[ -n "$PROMPT_FILE" ]]; then
    CORRECTNESS_ARGS+=(--prompt-file "$PROMPT_FILE")
  fi
  CORRECTNESS_PROMPT_SEEN=0
  for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
      --prompt)
        if [[ "$CORRECTNESS_PROMPT_SEEN" == "0" && $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
          CORRECTNESS_ARGS+=(--prompt "${EXTRA_ARGS[$((i + 1))]}")
          CORRECTNESS_PROMPT_SEEN=1
        fi
        i=$((i + 1))
        ;;
      --prompt=*)
        if [[ "$CORRECTNESS_PROMPT_SEEN" == "0" ]]; then
          CORRECTNESS_ARGS+=("${EXTRA_ARGS[$i]}")
          CORRECTNESS_PROMPT_SEEN=1
        fi
        ;;
      --prompt-file)
        if [[ $((i + 1)) -lt ${#EXTRA_ARGS[@]} ]]; then
          CORRECTNESS_ARGS+=(--prompt-file "${EXTRA_ARGS[$((i + 1))]}")
        fi
        i=$((i + 1))
        ;;
      --prompt-file=*)
        CORRECTNESS_ARGS+=("${EXTRA_ARGS[$i]}")
        ;;
    esac
  done
  "$PYTHON" "$SCRIPT_DIR/verify_gptq_backends.py" "${CORRECTNESS_ARGS[@]}"
fi

printf '\nStage 1 reports:\n  baseline JSON: %s\n  baseline Markdown: %s\n  profile JSON: %s\n  profile Markdown: %s\n' \
  "$(realpath "$BASELINE_JSON")" \
  "$(realpath "$BASELINE_MD")" \
  "$(realpath "$PROFILE_JSON")" \
  "$(realpath "$PROFILE_MD")"
if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  printf '  correctness JSON: %s\n  correctness Markdown: %s\n' \
    "$(realpath "${OUTPUT_DIR%/}/backend-correctness.json")" \
    "$(realpath "${OUTPUT_DIR%/}/backend-correctness.md")"
fi
