#!/usr/bin/env bash
set -uo pipefail

ROOT="${1:?Usage: $0 OUTPUT_ROOT}"
cd /home/wenke/SQJ/code/EvidenceTree || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH=src

mkdir -p "$ROOT/infoseek"

{
  echo "[start] $(date '+%F %T')"
  echo "[root] $ROOT"

  .venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
    --config DatasetConstruct/config.qwen25vl.yaml \
    --n 20000 \
    --force \
    --set benchmark=infoseek \
    --set data.data_dir=data/corpus/infoseek \
    --set mcts.rollouts=10 \
    --set output.trajectories="$ROOT/infoseek/trajectories.jsonl" \
    --set output.scored="$ROOT/infoseek/scored.jsonl" \
    --set output.rationales="$ROOT/infoseek/rationales.jsonl" \
    --set output.dataset_dir="$ROOT/infoseek/dataset"
  STATUS=$?
  echo "[pipeline_status] $STATUS"

  if [ "$STATUS" -eq 0 ]; then
    echo "[summary] $(date '+%F %T')"
    .venv-qwen25vl/bin/python DatasetConstruct/evaluate_pipeline_outputs.py \
      --root "$ROOT" \
      --soft-threshold 0.75 \
      --examples 5 \
      --output "$ROOT/summary.md"

    echo "[validate] $(date '+%F %T')"
    .venv-qwen25vl/bin/python DatasetConstruct/validate_dataset.py \
      --dataset-dir "$ROOT/infoseek/dataset" \
      --require-rationale \
      --max-errors 20 | tee "$ROOT/validate.log"
  fi

  echo "[end] $(date '+%F %T')"
  exit "$STATUS"
} 2>&1 | tee "$ROOT/run.log"
