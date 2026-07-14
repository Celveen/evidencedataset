#!/usr/bin/env bash
set -euo pipefail

cd /home/wenke/SQJ/code/EvidenceTree || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7898}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7898}"
export http_proxy="${http_proxy:-$HTTP_PROXY}"
export https_proxy="${https_proxy:-$HTTPS_PROXY}"
unset ALL_PROXY all_proxy

SOURCE_ROOT="/media/wenke/BBC23084DC1B0A00/datasetForAiii/infoseek"
LOG_DIR="/media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/infoseek_corpus_images_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SOURCE_ROOT" "$LOG_DIR"
ln -sfn "$LOG_DIR" /media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/infoseek_corpus_images_latest

{
  echo "[start] $(date '+%F %T')"
  echo "[source_root] $SOURCE_ROOT"
  echo "[log_dir] $LOG_DIR"

  scripts/download_wiki6m_by_ranges.sh \
    "https://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz" \
    "$SOURCE_ROOT/Wiki6M_ver_1_0.jsonl.gz" \
    7408031599 \
    64

  .venv-qwen25vl/bin/python DatasetConstruct/prepare_infoseek_corpus_images.py \
    --data-dir data/corpus/infoseek \
    --source-root "$SOURCE_ROOT" \
    --workers 4 \
    --thumbnail-width 512 \
    --min-sleep 0.3 \
    --max-sleep 0.8 \
    --max-attempts 3 \
    --shuffle-pending \
    --device cpu \
    --batch-size 32
  STATUS=$?
  echo "[status] $STATUS"
  echo "[end] $(date '+%F %T')"
  exit "$STATUS"
} 2>&1 | tee "$LOG_DIR/run.log"
