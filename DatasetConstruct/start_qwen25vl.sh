#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv-qwen25vl/bin/python" \
  "$ROOT/DatasetConstruct/serve_qwen25vl.py" "$@"
