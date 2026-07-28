#!/usr/bin/env bash
set -euo pipefail

ROOT="${GQA_ROOT:-data/raw/GQA}"
RAW="$ROOT/raw"
LOG="$ROOT/gqa_download.log"

mkdir -p "$RAW"
exec > >(tee -a "$LOG") 2>&1

echo "===== GQA download started: $(date -Is) ====="
echo "ROOT=$ROOT"

download() {
  local url="$1"
  local name="$2"
  local out="$RAW/$name"
  echo
  echo "===== downloading $name ====="
  if [[ -s "$out" ]]; then
    echo "Existing file found, resuming/checking: $out"
  fi
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy \
    wget -c --progress=dot:giga -O "$out" "$url"
  echo "Downloaded $name: $(du -h "$out" | cut -f1)"
}

download "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip" "questions1.2.zip"
download "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip" "sceneGraphs.zip"
download "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" "images.zip"

echo
echo "===== zip listings ====="
unzip -l "$RAW/questions1.2.zip" | head -40
unzip -l "$RAW/sceneGraphs.zip" | head -40
unzip -l "$RAW/images.zip" | head -40

echo "===== GQA download finished: $(date -Is) ====="
