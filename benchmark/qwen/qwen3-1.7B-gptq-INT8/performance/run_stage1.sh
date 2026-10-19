#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXTRA_ARGS=("$@")
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
RESULTS_ROOT="${RESULTS_ROOT:-${OUTPUT_DIR:-${ROOT_DIR}/results}}"
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
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${RESULTS_ROOT}/performance/${GPTQ_BACKEND}/run-${RUN_ID}}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${RESULTS_ROOT}/operator-profile/reference/run-${RUN_ID}}"
CORRECTNESS_OUTPUT_DIR="${CORRECTNESS_OUTPUT_DIR:-${RESULTS_ROOT}/correctness/${CORRECTNESS_BACKEND}/run-${RUN_ID}}"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/../../../.venv/bin/python"
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

BASELINE_JSON="${BASELINE_OUTPUT_DIR%/}/baseline.json"
BASELINE_MD="${BASELINE_OUTPUT_DIR%/}/baseline.md"
PROFILE_JSON="${PROFILE_OUTPUT_DIR%/}/projection-profile.json"
PROFILE_MD="${PROFILE_OUTPUT_DIR%/}/projection-profile.md"
CORRECTNESS_JSON="${CORRECTNESS_OUTPUT_DIR%/}/backend-correctness.json"
CORRECTNESS_MD="${CORRECTNESS_OUTPUT_DIR%/}/backend-correctness.md"

if [[ "$RUN_BASELINE" == "1" ]]; then
  mkdir -p "$BASELINE_OUTPUT_DIR"
  printf 'Running formal %s baseline (profiler disabled) ...\n' "$GPTQ_BACKEND"
  BASELINE_ARGS=("${COMMON_ARGS[@]}" --gptq-backend "$GPTQ_BACKEND" --output "$BASELINE_JSON" --report "$BASELINE_MD")
  if [[ -n "$REFERENCE_MODEL_PATH" ]]; then
    BASELINE_ARGS+=(--reference-model-path "$REFERENCE_MODEL_PATH" --correctness)
  fi
  PROFILE_PROJECTIONS=0 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${BASELINE_ARGS[@]}" "$@"
else
  printf 'Skipping formal baseline (RUN_BASELINE=0) ...\n'
fi

mkdir -p "$PROFILE_OUTPUT_DIR"
printf '\nRunning reference projection profile (separate process, profiler enabled) ...\n'
PROFILE_ARGS=("${COMMON_ARGS[@]}" --gptq-backend reference --output "$PROFILE_JSON" --report "$PROFILE_MD" --profile-projections)
PROFILE_PROJECTIONS=1 "$PYTHON" "$SCRIPT_DIR/benchmark.py" "${PROFILE_ARGS[@]}" "$@"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  mkdir -p "$CORRECTNESS_OUTPUT_DIR"
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
    --output "$CORRECTNESS_JSON"
    --report "$CORRECTNESS_MD"
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
  "$PYTHON" "$ROOT_DIR/correctness/verify_gptq_backends.py" "${CORRECTNESS_ARGS[@]}"
fi

printf '\nStage 1 reports:\n'
if [[ "$RUN_BASELINE" == "1" ]]; then
  printf '  baseline JSON: %s\n  baseline Markdown: %s\n' \
    "$(realpath "$BASELINE_JSON")" \
    "$(realpath "$BASELINE_MD")"
fi
printf '  reference operator profile JSON: %s\n  reference operator profile Markdown: %s\n' \
  "$(realpath "$PROFILE_JSON")" \
  "$(realpath "$PROFILE_MD")"
if [[ "$RUN_CORRECTNESS" == "1" ]]; then
  printf '  correctness JSON: %s\n  correctness Markdown: %s\n' \
    "$(realpath "$CORRECTNESS_JSON")" \
    "$(realpath "$CORRECTNESS_MD")"
fi
