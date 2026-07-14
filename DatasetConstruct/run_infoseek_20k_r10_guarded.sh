#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

RUN_STAMP="${RUN_STAMP:-20260705_222632}"
RUN_ROOT="${RUN_ROOT:-/media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/infoseek_20k_r10_${RUN_STAMP}}"
SMOKE_ROOT="$RUN_ROOT/smoke5"
SMOKE_DS="$SMOKE_ROOT/infoseek"
FULL_ROOT="$RUN_ROOT/full20k"
FULL_DS="$FULL_ROOT/infoseek"
LOG="$RUN_ROOT/run.log"

mkdir -p "$SMOKE_DS" "$FULL_DS"
exec > >(tee -a "$LOG") 2>&1

echo "===== guarded InfoSeek 20k r10 started: $(date -Is) ====="
echo "[run_root] $RUN_ROOT"
echo "[smoke_root] $SMOKE_ROOT"
echo "[full_root] $FULL_ROOT"

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

wait_for_qwen() {
  echo "[qwen] waiting for http://127.0.0.1:8000/health"
  for i in $(seq 1 180); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health >/tmp/evidencetree_qwen_health.json 2>/dev/null; then
      echo "[qwen] ready: $(cat /tmp/evidencetree_qwen_health.json)"
      return 0
    fi
    if (( i % 6 == 0 )); then
      echo "[qwen] still waiting... $((i * 10))s"
    fi
    sleep 10
  done
  echo "[qwen] ERROR: Qwen server did not become ready in time"
  return 2
}

pipeline_base_args=(
  --config DatasetConstruct/config.qwen25vl.yaml
  --set benchmark=infoseek
  --set data.data_dir=data/corpus/infoseek
  --set mcts.rollouts=10
)

run_steps() {
  local n="$1"
  local ds="$2"
  local steps="$3"
  local force_flag="${4:-}"
  local -a force_args=()
  if [[ "$force_flag" == "force" ]]; then
    force_args=(--force)
  fi
  .venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
    "${pipeline_base_args[@]}" \
    --n "$n" \
    --steps "$steps" \
    "${force_args[@]}" \
    --set output.trajectories="$ds/trajectories.jsonl" \
    --set output.scored="$ds/scored.jsonl" \
    --set output.rationales="$ds/rationales.jsonl" \
    --set output.dataset_dir="$ds/dataset"
}

count_queries() {
  local ds="$1"
  .venv-qwen25vl/bin/python - "$ds/trajectories.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit
qids = set()
for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        qids.add(json.loads(line)["query_id"])
print(len(qids))
PY
}

check_full_pipeline() {
  local ds="$1"
  local min_queries="$2"
  .venv-qwen25vl/bin/python - "$ds" "$min_queries" <<'PY'
import json
import sys
from pathlib import Path

ds = Path(sys.argv[1])
min_queries = int(sys.argv[2])
required = [
    ds / "trajectories.jsonl",
    ds / "scored.jsonl",
    ds / "rationales.jsonl",
    ds / "dataset" / "stats.json",
]
missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
if missing:
    raise SystemExit("missing or empty outputs: " + ", ".join(missing))

qids = set()
for line in (ds / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        qids.add(json.loads(line)["query_id"])
if len(qids) < min_queries:
    raise SystemExit(f"only {len(qids)} queries completed, expected >= {min_queries}")

stats = json.loads((ds / "dataset" / "stats.json").read_text(encoding="utf-8"))
if int(stats.get("kept", 0)) <= 0:
    raise SystemExit("quality filter kept 0 samples")
print(json.dumps({"queries": len(qids), "stats": stats}, ensure_ascii=False))
PY
}

run_step3_with_retries() {
  local n="$1"
  local ds="$2"
  local max_attempts="${3:-10}"
  for attempt in $(seq 1 "$max_attempts"); do
    echo "[step3] attempt $attempt/$max_attempts for $ds"
    if run_steps "$n" "$ds" "3"; then
      echo "[step3] completed for $ds"
      return 0
    fi
    echo "[step3] failed, will resume after 60s"
    sleep 60
  done
  echo "[step3] ERROR: failed after $max_attempts attempts for $ds"
  return 3
}

summarize_root() {
  local root="$1"
  PYTHONPATH=src .venv-qwen25vl/bin/python DatasetConstruct/evaluate_pipeline_outputs.py \
    --root "$root" \
    --datasets infoseek \
    --soft-threshold 0.75 \
    --examples 5 \
    --output "$root/summary.md" || true
}

wait_for_qwen

echo "===== [smoke] running 5-query full pipeline first ====="
run_steps 5 "$SMOKE_DS" "1,2" force
run_step3_with_retries 5 "$SMOKE_DS" 5
run_steps 5 "$SMOKE_DS" "4" force
check_full_pipeline "$SMOKE_DS" 5
summarize_root "$SMOKE_ROOT"
echo "===== [smoke] passed; continuing to full 20k ====="

echo "===== [full] Step 1: generating 20k trajectories, rollout=10 ====="
run_steps 20000 "$FULL_DS" "1" force
for retry in $(seq 1 5); do
  completed="$(count_queries "$FULL_DS")"
  echo "[full] Step1 completed query count after pass $retry: $completed/20000"
  if [[ "$completed" -ge 20000 ]]; then
    break
  fi
  echo "[full] retrying missing Step1 queries after 60s"
  sleep 60
  run_steps 20000 "$FULL_DS" "1"
done

completed="$(count_queries "$FULL_DS")"
if [[ "$completed" -lt 20000 ]]; then
  echo "[full] WARNING: Step1 has $completed/20000 queries after retries; continuing with completed queries."
fi

echo "===== [full] Step 2 ====="
run_steps 20000 "$FULL_DS" "2" force

echo "===== [full] Step 3 ====="
run_step3_with_retries 20000 "$FULL_DS" 20

echo "===== [full] Step 4 ====="
run_steps 20000 "$FULL_DS" "4" force
check_full_pipeline "$FULL_DS" 1
summarize_root "$FULL_ROOT"

echo "===== guarded InfoSeek 20k r10 finished: $(date -Is) ====="
echo "[run_root] $RUN_ROOT"
echo "[log] $LOG"
