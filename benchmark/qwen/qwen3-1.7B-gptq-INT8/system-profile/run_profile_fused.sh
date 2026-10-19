#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUN_ID="$(date +'%Y%m%d-%H%M%S')-$$"
MODEL_PATH="${MODEL_PATH:-$HOME/project/LLM/models/qwen/Qwen3-1.7B-GPTQ-Int8}"
BACKEND="${BACKEND:-fused}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/results/system-profile/${BACKEND}/run-${RUN_ID}}"
DEFAULT_PYTHON="python3"
if [[ -x "$ROOT_DIR/../../../.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$ROOT_DIR/../../../.venv/bin/python"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PYTHON" "$SCRIPT_DIR/profile_fused.py" --help
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON" "$SCRIPT_DIR/profile_fused.py" \
  --model-path "$MODEL_PATH" \
  --backend "$BACKEND" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
