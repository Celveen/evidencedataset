#!/usr/bin/env bash
set -euo pipefail

URL="${1:-https://storage.googleapis.com/gresearch/open-vision-language/Wiki6M_ver_1_0.jsonl.gz}"
OUT="${2:-/media/wenke/BBC23084DC1B0A00/datasetForAiii/infoseek/Wiki6M_ver_1_0.jsonl.gz}"
SIZE="${3:-7408031599}"
CHUNK_MB="${4:-64}"

CHUNK_SIZE=$((CHUNK_MB * 1024 * 1024))
PART_DIR="${OUT}.parts"
CURL_BIN="${CURL_BIN:-/home/wenke/anaconda3/bin/curl}"
mkdir -p "$PART_DIR" "$(dirname "$OUT")"

if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -eq "$SIZE" ]; then
  echo "[wiki6m] already complete: $OUT"
  exit 0
fi

find "$PART_DIR" -maxdepth 1 -name '*.tmp' -type f -delete

echo "[wiki6m] url=$URL"
echo "[wiki6m] out=$OUT"
echo "[wiki6m] size=$SIZE chunk_mb=$CHUNK_MB part_dir=$PART_DIR"
echo "[wiki6m] curl=$CURL_BIN"

start=0
idx=0
while [ "$start" -lt "$SIZE" ]; do
  end=$((start + CHUNK_SIZE - 1))
  if [ "$end" -ge "$SIZE" ]; then
    end=$((SIZE - 1))
  fi
  expected=$((end - start + 1))
  part="$PART_DIR/$(printf '%012d-%012d.part' "$start" "$end")"
  tmp="${part}.tmp"
  idx=$((idx + 1))

  if [ -f "$part" ] && [ "$(stat -c%s "$part")" -eq "$expected" ]; then
    if [ $((idx % 10)) -eq 0 ]; then
      echo "[wiki6m] chunk $idx exists"
    fi
    start=$((end + 1))
    continue
  fi

  attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "[wiki6m] downloading chunk $idx bytes=$start-$end attempt=$attempt"
    rm -f "$tmp"
    if "$CURL_BIN" --http1.1 -L --connect-timeout 30 --retry 10 --retry-delay 5 \
      -r "${start}-${end}" -o "$tmp" "$URL"; then
      got="$(stat -c%s "$tmp")"
      if [ "$got" -eq "$expected" ]; then
        mv "$tmp" "$part"
        break
      fi
      echo "[wiki6m] bad chunk $idx: expected=$expected got=$got" >&2
    else
      echo "[wiki6m] curl failed for chunk $idx attempt=$attempt" >&2
    fi
    sleep_time=$((attempt < 12 ? attempt * 5 : 60))
    echo "[wiki6m] retry chunk $idx after ${sleep_time}s" >&2
    sleep "$sleep_time"
  done
  start=$((end + 1))
done

echo "[wiki6m] assembling $OUT"
tmp_out="${OUT}.tmp"
rm -f "$tmp_out"
for part in "$PART_DIR"/*.part; do
  cat "$part" >> "$tmp_out"
done
got="$(stat -c%s "$tmp_out")"
if [ "$got" -ne "$SIZE" ]; then
  echo "[wiki6m] bad assembled file: expected=$SIZE got=$got" >&2
  exit 1
fi
mv "$tmp_out" "$OUT"
echo "[wiki6m] complete: $OUT"
