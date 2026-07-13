#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

N_QUERIES="${N_QUERIES:-10332}"
ROLLOUTS="${ROLLOUTS:-10}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/media/wenke/DCF3AE9C25B8124A/datasetForAiii/evidencetree_runs/scienceqa_${N_QUERIES}_r${ROLLOUTS}_${RUN_STAMP}}"
SMOKE_ROOT="$RUN_ROOT/smoke5"
SMOKE_DS="$SMOKE_ROOT/scienceqa"
FULL_ROOT="$RUN_ROOT/full${N_QUERIES}"
FULL_DS="$FULL_ROOT/scienceqa"
LOG="$RUN_ROOT/run.log"
STATUS="$RUN_ROOT/status.txt"

mkdir -p "$SMOKE_DS" "$FULL_DS"
ln -sfn "$RUN_ROOT" "$(dirname "$RUN_ROOT")/scienceqa_latest"
ln -sfn "$RUN_ROOT" \
  /media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/scienceqa_latest
exec > >(tee -a "$LOG") 2>&1

write_status() {
  printf '%s\t%s\n' "$(date -Is)" "$1" > "$STATUS"
}

on_exit() {
  local code=$?
  if [[ "$code" -eq 0 ]]; then
    write_status "complete"
  else
    write_status "failed(exit=$code); rerun with RUN_ROOT=$RUN_ROOT to resume"
  fi
}
trap on_exit EXIT

echo "===== guarded ScienceQA full r${ROLLOUTS} started: $(date -Is) ====="
echo "[run_root] $RUN_ROOT"
echo "[queries] $N_QUERIES"
echo "[rollouts] $ROLLOUTS"
write_status "initializing"

export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7898}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7898}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"

wait_for_qwen() {
  echo "[qwen] waiting for http://127.0.0.1:8000/health"
  for i in $(seq 1 180); do
    if curl -fsS --max-time 5 http://127.0.0.1:8000/health \
      >/tmp/evidencetree_qwen_health.json 2>/dev/null; then
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
  --set benchmark=scienceqa
  --set data.data_dir=data/corpus/scienceqa
  --set mcts.rollouts="$ROLLOUTS"
  --set step2.support_concurrency="${STEP2_SUPPORT_CONCURRENCY:-8}"
  --set step2.support_max_pending="${STEP2_SUPPORT_MAX_PENDING:-32}"
  --set step2.flush_every="${STEP2_FLUSH_EVERY:-200}"
  --set step2.progress_every="${STEP2_PROGRESS_EVERY:-200}"
  --set rationale.concurrency="${RATIONALE_CONCURRENCY:-4}"
  --set rationale.max_pending="${RATIONALE_MAX_PENDING:-16}"
  --set rationale.flush_every="${RATIONALE_FLUSH_EVERY:-100}"
  --set rationale.progress_every="${RATIONALE_PROGRESS_EVERY:-1000}"
  --set rationale.retry_connection_errors=true
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

count_lines() {
  [[ -f "$1" ]] && wc -l < "$1" || echo 0
}

count_queries() {
  .venv-qwen25vl/bin/python - "$1/trajectories.jsonl" <<'PY'
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

check_full_pipeline() {
  .venv-qwen25vl/bin/python - "$1" "$2" <<'PY'
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
qids = {
    json.loads(line)["query_id"]
    for line in (ds / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
}
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
  local max_attempts="${3:-30}"
  for attempt in $(seq 1 "$max_attempts"); do
    echo "[step3] attempt $attempt/$max_attempts"
    if run_steps "$n" "$ds" "3"; then
      echo "[step3] completed"
      return 0
    fi
    echo "[step3] incomplete; $(count_lines "$ds/rationales.jsonl") rows saved. Retrying in 60s."
    sleep 60
  done
  return 3
}

summarize_root() {
  PYTHONPATH=src .venv-qwen25vl/bin/python \
    DatasetConstruct/evaluate_pipeline_outputs.py \
    --root "$1" --datasets scienceqa --soft-threshold 0.75 --examples 5 \
    --output "$1/summary.md" || true
}

wait_for_qwen

write_status "smoke: Step1-4 on 5 queries"
echo "===== [smoke] 5-query real full pipeline ====="
run_steps 5 "$SMOKE_DS" "1,2" force
run_step3_with_retries 5 "$SMOKE_DS" 5
run_steps 5 "$SMOKE_DS" "4" force
check_full_pipeline "$SMOKE_DS" 5
summarize_root "$SMOKE_ROOT"
echo "===== [smoke] passed ====="

write_status "full: Step1 trajectories"
echo "===== [full] Step1: $N_QUERIES queries, rollout=$ROLLOUTS ====="
run_steps "$N_QUERIES" "$FULL_DS" "1"
for retry in $(seq 1 10); do
  completed="$(count_queries "$FULL_DS")"
  echo "[full] Step1 query count after pass $retry: $completed/$N_QUERIES"
  [[ "$completed" -ge "$N_QUERIES" ]] && break
  echo "[full] retrying missing queries in 60s"
  sleep 60
  run_steps "$N_QUERIES" "$FULL_DS" "1"
done

completed="$(count_queries "$FULL_DS")"
if [[ "$completed" -lt "$N_QUERIES" ]]; then
  echo "[full] WARNING: Step1 has $completed/$N_QUERIES queries after retries."
fi

write_status "full: Step2 scoring"
echo "===== [full] Step2 ====="
run_steps "$N_QUERIES" "$FULL_DS" "2"

write_status "full: Step3 rationales"
echo "===== [full] Step3 ====="
run_step3_with_retries "$N_QUERIES" "$FULL_DS" 30

write_status "full: Step4 quality filter"
echo "===== [full] Step4 ====="
run_steps "$N_QUERIES" "$FULL_DS" "4" force
check_full_pipeline "$FULL_DS" 1
summarize_root "$FULL_ROOT"

echo "===== ScienceQA construction complete: $(date -Is) ====="
echo "[run_root] $RUN_ROOT"
echo "[summary] $FULL_ROOT/summary.md"
