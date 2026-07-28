#!/usr/bin/env bash
# Serve the policy VLM behind an OpenAI-compatible endpoint.
#
#   DatasetConstruct/start_qwen25vl.sh --model models/Qwen2.5-VL-7B-Instruct --port 8000
#
# Uses .venv-qwen25vl/bin/python when that virtualenv exists (GPU serving usually
# wants its own environment), otherwise falls back to $PYTHON or python3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT/.venv-qwen25vl/bin/python"

if [[ -x "$VENV_PY" ]]; then
  PY="$VENV_PY"
else
  PY="${PYTHON:-python3}"
fi

exec "$PY" "$ROOT/DatasetConstruct/serve_qwen25vl.py" "$@"
