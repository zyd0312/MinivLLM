#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
BACKEND="${BACKEND:-fused}"
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/system-profile/${BACKEND}/run-${RUN_ID}}"

DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ -z "${NSYS:-}" ]]; then
  NSYS="$(command -v nsys || true)"
  if [[ -n "$NSYS" ]] && ! "$NSYS" --version >/dev/null 2>&1; then
    NSYS=""
  fi
  if [[ -z "$NSYS" ]]; then
    for candidate in /opt/nvidia/nsight-systems/*/target-linux-x64/nsys /usr/local/cuda-*/nsight-systems-*/target-linux-x64/nsys; do
      if [[ -x "$candidate" ]]; then
        NSYS="$candidate"
        break
      fi
    done
  fi
fi
if [[ -z "$NSYS" || ! -x "$NSYS" ]]; then
  printf 'Nsight Systems CLI was not found. Set NSYS=/path/to/nsys.\n' >&2
  exit 2
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON" "$SCRIPT_DIR/profile_fused.py" --help
fi

mkdir -p "$OUTPUT_DIR"
exec "$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="${OUTPUT_DIR%/}/fused" \
  "$PYTHON" "$SCRIPT_DIR/profile_fused.py" \
  --model-path "$MODEL_PATH" \
  --backend "$BACKEND" \
  --output-dir "$OUTPUT_DIR" \
  --no-torch-profiler \
  "$@"
