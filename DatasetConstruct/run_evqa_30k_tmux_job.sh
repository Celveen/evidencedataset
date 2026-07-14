#!/usr/bin/env bash
set -euo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

RAW="/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA/raw"
LOG="$RAW/evqa_30k_tmux_job.log"

exec > >(tee -a "$LOG") 2>&1

echo "===== EVQA 30k tmux job started: $(date -Is) ====="

.venv-qwen25vl/bin/python DatasetConstruct/download_evqa_train_mini_resumable.py \
  --workers 4 \
  --parts 16 \
  --report-seconds 120

.venv-qwen25vl/bin/python DatasetConstruct/extract_evqa_query_images.py \
  --source-root /media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA \
  --data-dir data/corpus/evqa

.venv-qwen25vl/bin/python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

stats_path = Path("data/corpus/evqa/stats.json")
queries_path = Path("data/corpus/evqa/queries.jsonl")

stats = json.loads(stats_path.read_text(encoding="utf-8"))
existing = 0
missing = 0
unique = set()
for line in queries_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    image_path = row.get("image_path")
    if image_path and Path(image_path).exists():
        existing += 1
        unique.add(image_path)
    else:
        missing += 1

stats.pop("query_images_pending_extraction", None)
stats["query_images_existing_rows"] = existing
stats["query_images_extracted_unique"] = len(unique)
stats["missing_query_images"] = missing
stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
PY

rm -f "$RAW/train_mini.tar.gz"
rm -rf "$RAW/train_mini_parts"

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
  --set output.trajectories=data/evqa_30k_smoke/trajectories.jsonl \
  --set output.scored=data/evqa_30k_smoke/scored.jsonl \
  --set output.rationales=data/evqa_30k_smoke/rationales.jsonl \
  --set output.dataset_dir=data/evqa_30k_smoke/dataset

echo "===== EVQA 30k tmux job finished: $(date -Is) ====="
