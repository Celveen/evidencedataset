#!/usr/bin/env bash
set -euo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

RAW="/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA/raw"
LOG="$RAW/evqa_corpus_images_tmux_job.log"

exec > >(tee -a "$LOG") 2>&1

echo "===== EVQA corpus image tmux job started: $(date -Is) ====="

export HTTP_PROXY=http://127.0.0.1:7898
export HTTPS_PROXY=http://127.0.0.1:7898
export http_proxy=http://127.0.0.1:7898
export https_proxy=http://127.0.0.1:7898

.venv-qwen25vl/bin/python DatasetConstruct/download_evqa_corpus_images.py \
  --n 30000 \
  --split train \
  --proxy http://127.0.0.1:7898 \
  --thumbnail-width 512 \
  --min-sleep 0.1 \
  --max-sleep 0.3 \
  --max-attempts 1 \
  --workers 4 \
  --max-urls-per-entity 4 \
  --shuffle-pending

rm -rf data/corpus/evqa/clip_index
.venv-qwen25vl/bin/python scripts/build_index.py \
  --corpus data/corpus/evqa/corpus.jsonl \
  --out data/corpus/evqa/clip_index \
  --clip \
  --device cuda

.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml \
  --n 1 --force \
  --set benchmark=evqa \
  --set data.data_dir=data/corpus/evqa \
  --set mcts.rollouts=1 \
  --set policy.backend=mock \
  --set retriever.image.device=cuda \
  --set verifier.backend=lexical \
  --set rationale.backend=mock \
  --set output.trajectories=data/evqa_30k_corpus_images_smoke/trajectories.jsonl \
  --set output.scored=data/evqa_30k_corpus_images_smoke/scored.jsonl \
  --set output.rationales=data/evqa_30k_corpus_images_smoke/rationales.jsonl \
  --set output.dataset_dir=data/evqa_30k_corpus_images_smoke/dataset

echo "===== EVQA corpus image tmux job finished: $(date -Is) ====="
