#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

RUN_STAMP="${RUN_STAMP:-20260705_222632}"
RUN_ROOT="${RUN_ROOT:-/media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/infoseek_20k_r10_${RUN_STAMP}}"
FULL_ROOT="$RUN_ROOT/full20k"
FULL_DS="$FULL_ROOT/infoseek"
LOG="$RUN_ROOT/run.log"

mkdir -p "$FULL_DS"
exec > >(tee -a "$LOG") 2>&1

echo "===== resume InfoSeek 20k r10 after Step1: $(date -Is) ====="
echo "[run_root] $RUN_ROOT"
echo "[full_ds] $FULL_DS"

export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HTTP_PROXY="http://127.0.0.1:7898"
export HTTPS_PROXY="http://127.0.0.1:7898"
export http_proxy="http://127.0.0.1:7898"
export https_proxy="http://127.0.0.1:7898"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

pipeline_base_args=(
  --config DatasetConstruct/config.qwen25vl.yaml
  --set benchmark=infoseek
  --set data.data_dir=data/corpus/infoseek
  --set mcts.rollouts=10
  --set output.trajectories="$FULL_DS/trajectories.jsonl"
  --set output.scored="$FULL_DS/scored.jsonl"
  --set output.rationales="$FULL_DS/rationales.jsonl"
  --set output.dataset_dir="$FULL_DS/dataset"
  --set step2.support_concurrency="${STEP2_SUPPORT_CONCURRENCY:-8}"
  --set step2.support_max_pending="${STEP2_SUPPORT_MAX_PENDING:-32}"
  --set rationale.concurrency="${RATIONALE_CONCURRENCY:-4}"
  --set rationale.max_pending="${RATIONALE_MAX_PENDING:-16}"
  --set rationale.flush_every="${RATIONALE_FLUSH_EVERY:-100}"
  --set rationale.progress_every="${RATIONALE_PROGRESS_EVERY:-1000}"
  --set rationale.retry_connection_errors="${RATIONALE_RETRY_CONNECTION_ERRORS:-true}"
  --n 20000
)

count_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

count_queries() {
  .venv-qwen25vl/bin/python - "$FULL_DS/trajectories.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
qids = set()
if path.exists():
    for line in path.open(encoding="utf-8"):
        if line.strip():
            qids.add(json.loads(line)["query_id"])
print(len(qids))
PY
}

summarize_root() {
  PYTHONPATH=src .venv-qwen25vl/bin/python DatasetConstruct/evaluate_pipeline_outputs.py \
    --root "$FULL_ROOT" \
    --datasets infoseek \
    --soft-threshold 0.75 \
    --examples 5 \
    --output "$FULL_ROOT/summary.md" || true
}

queries_done="$(count_queries)"
traj_rows="$(count_file "$FULL_DS/trajectories.jsonl")"
echo "[resume] existing trajectories: $traj_rows rows, $queries_done queries"
if [[ "$queries_done" -lt 20000 ]]; then
  echo "[resume] ERROR: Step1 is incomplete ($queries_done/20000 queries)."
  exit 2
fi

echo "===== [resume] Step 2 ====="
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  "${pipeline_base_args[@]}" \
  --steps 2

echo "===== [resume] Step 3 ====="
for attempt in $(seq 1 30); do
  echo "[resume] Step3 attempt $attempt/30"
  if .venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
    "${pipeline_base_args[@]}" \
    --steps 3; then
    echo "[resume] Step3 completed"
    break
  fi
  echo "[resume] Step3 failed; completed rationales: $(count_file "$FULL_DS/rationales.jsonl"). Retrying after 60s."
  sleep 60
  if [[ "$attempt" -eq 30 ]]; then
    echo "[resume] ERROR: Step3 did not complete after 30 attempts."
    exit 3
  fi
done

echo "===== [resume] Step 4 ====="
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  "${pipeline_base_args[@]}" \
  --steps 4 --force

summarize_root

echo "===== resume InfoSeek 20k r10 finished: $(date -Is) ====="
echo "[summary] $FULL_ROOT/summary.md"
