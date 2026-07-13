#!/usr/bin/env bash
set -euo pipefail

cd /home/wenke/SQJ/code/EvidenceTree

MAIN_SESSION="infoseek_imgs"
WATCH_LOG="/media/wenke/BBC23084DC1B0A00/datasetForAiii/evidencetree_runs/infoseek_corpus_images_progress_watch.log"
PART_DIR="/media/wenke/BBC23084DC1B0A00/datasetForAiii/infoseek/Wiki6M_ver_1_0.jsonl.gz.parts"
FINAL="/media/wenke/BBC23084DC1B0A00/datasetForAiii/infoseek/Wiki6M_ver_1_0.jsonl.gz"
EXPECTED_SIZE=7408031599

while true; do
  ts="$(date '+%F %T')"
  parts=0
  bytes=0
  if [ -d "$PART_DIR" ]; then
    parts="$(find "$PART_DIR" -name '*.part' -type f | wc -l)"
    bytes="$(du -sb "$PART_DIR" | awk '{print $1}')"
  fi
  final_size=0
  if [ -f "$FINAL" ]; then
    final_size="$(stat -c%s "$FINAL")"
  fi
  pct="$(python3 - <<PY
expected=$EXPECTED_SIZE
done=max($bytes, $final_size)
print(f"{done/expected*100:.2f}")
PY
)"

  if tmux has-session -t "$MAIN_SESSION" 2>/dev/null; then
    status="running"
  else
    status="stopped"
  fi

  echo "[$ts] status=$status parts=$parts/111 bytes=$bytes final_size=$final_size pct=$pct%" | tee -a "$WATCH_LOG"

  if [ "$final_size" -eq "$EXPECTED_SIZE" ]; then
    echo "[$ts] final Wiki6M file complete; watchdog stops." | tee -a "$WATCH_LOG"
    exit 0
  fi

  if [ "$status" = "stopped" ]; then
    echo "[$ts] main session stopped before completion; restarting $MAIN_SESSION." | tee -a "$WATCH_LOG"
    tmux new-session -d -s "$MAIN_SESSION" -c /home/wenke/SQJ/code/EvidenceTree \
      "scripts/run_infoseek_corpus_images_tmux.sh"
  fi

  sleep 300
done
