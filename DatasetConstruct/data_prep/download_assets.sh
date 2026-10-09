#!/usr/bin/env bash
# Download everything the construction server can fetch directly (ModelScope),
# bypassing any proxy. InfoSeek data and GQA scene graphs are NOT reachable from
# the server: build them with DatasetConstruct/fetch_kit/fetch_infoseek.py on a
# machine with internet access and copy them over.
#
# Usage (from the repository root):
#   bash DatasetConstruct/data_prep/download_assets.sh            # everything
#   bash DatasetConstruct/data_prep/download_assets.sh models     # one group
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/../site.env" ] && . "$HERE/../site.env"   # WEIGHTS_DIR (see site.env.example)
WEIGHTS_DIR="${WEIGHTS_DIR:-models}"           # datasets always go to data/raw
NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy)
WHAT="${1:-all}"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "Installing the ModelScope CLI (Aliyun PyPI mirror)..."
  "${NOPROXY[@]}" pip install -q modelscope
fi

ms() { "${NOPROXY[@]}" modelscope download "$@"; }

if [[ "$WHAT" == all || "$WHAT" == models ]]; then
  ms --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir "$WEIGHTS_DIR/Qwen2.5-VL-7B-Instruct"   # ~16 GB
  ms --model sentence-transformers/clip-ViT-B-32 --local_dir "$WEIGHTS_DIR/clip-ViT-B-32"     # ~0.6 GB
fi

if [[ "$WHAT" == all || "$WHAT" == gqa ]]; then
  for part in train_balanced_instructions train_balanced_images \
              testdev_balanced_instructions testdev_balanced_images; do
    ms --dataset lmms-lab/GQA --include "${part}/*" --local_dir data/raw/GQA         # ~10.5 GB total
  done
fi

if [[ "$WHAT" == all || "$WHAT" == scienceqa ]]; then
  ms --dataset AI-ModelScope/ScienceQA --local_dir data/raw/ScienceQA                 # ~0.6 GB
fi

echo "Done. Next: data_prep/prepare_scienceqa.py, data_prep/prepare_gqa.py (needs sceneGraphs.zip),"
echo "and data_prep/import_infoseek_kit.py (needs the InfoSeek fetch kit)."
