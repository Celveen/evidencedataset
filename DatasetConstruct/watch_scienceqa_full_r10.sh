#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/media/wenke/DCF3AE9C25B8124A/datasetForAiii/evidencetree_runs/scienceqa_latest}"
RUN_ROOT="$(readlink -f "$RUN_ROOT")"
DS_DIR="$(find "$RUN_ROOT" -maxdepth 1 -type d -name 'full*' | head -n 1)"
DS_DIR="${DS_DIR:+$DS_DIR/scienceqa}"

echo "run_root: $RUN_ROOT"
[[ -f "$RUN_ROOT/status.txt" ]] && echo "status: $(cat "$RUN_ROOT/status.txt")"
[[ -f "$RUN_ROOT/run.log" ]] && echo "last_log: $(tail -n 1 "$RUN_ROOT/run.log")"

if [[ -n "$DS_DIR" && -d "$DS_DIR" ]]; then
  for file in trajectories.jsonl scored.jsonl rationales.jsonl; do
    path="$DS_DIR/$file"
    [[ -f "$path" ]] && printf '%s: %s rows\n' "$file" "$(wc -l < "$path")"
  done
  if [[ -f "$DS_DIR/trajectories.jsonl" ]]; then
    /home/wenke/SQJ/code/EvidenceTree/.venv-qwen25vl/bin/python - "$DS_DIR/trajectories.jsonl" <<'PY'
import json
import sys
from pathlib import Path

qids = set()
for line in Path(sys.argv[1]).open(encoding="utf-8"):
    if line.strip():
        qids.add(json.loads(line)["query_id"])
print(f"completed_queries: {len(qids)}/10332 ({len(qids)/10332:.2%})")
PY
  fi
fi

echo "tmux:"
tmux list-sessions 2>/dev/null | grep scienceqa_full_r10 || echo "not running"
