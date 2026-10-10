#!/usr/bin/env bash
# Serve the training-free policy VLM (step 1) with vLLM in containers — one
# OpenAI-compatible server per GPU on ports 8100, 8101, ... (localhost only).
# `policy.base_url` in configs/base.yaml lists those URLs; step 1 rotates
# requests over them.
#
# Usage (from the repository root):
#   bash DatasetConstruct/serve_policy_vllm.sh            # one server per POLICY_GPUS entry
#   POLICY_GPUS=0,1 bash DatasetConstruct/serve_policy_vllm.sh
# Check: python DatasetConstruct/preflight.py    Logs: docker logs -f etbench_policy_gpu<N>
# Stop:  docker rm -f $(docker ps -aq --filter label=purpose=etbench-policy)
#
# Machine-specific settings (weights dir, image, GPUs) come from
# DatasetConstruct/site.env — see site.env.example. The policy and the
# rationale server may share GPUs only one at a time (stop one, start the other).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# site.env supplies defaults; a variable set on the command line wins.
if [ -f "$HERE/site.env" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    if [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then export "$key=$value"; fi
  done < "$HERE/site.env"
fi

WEIGHTS_DIR="${WEIGHTS_DIR:-$HERE/../models}"
MODEL_DIR="${MODEL_DIR:-$WEIGHTS_DIR/Qwen2.5-VL-7B-Instruct}"
NAME="${NAME:-Qwen2.5-VL-7B-Instruct}"       # must equal policy.model in the config
GPUS="${POLICY_GPUS:-0,1,2,3}"
PORT0="${PORT0:-8100}"
MAX_LEN="${MAX_LEN:-16384}"
MEM="${GPU_MEM:-0.85}"
DEFAULT_LIMIT_MM='{"image": 5}'              # query image + up to 4 retrieved images
LIMIT_MM="${LIMIT_MM:-$DEFAULT_LIMIT_MM}"
DEFAULT_MM_KWARGS='{"max_pixels": 1003520}'  # <= 1280 visual tokens per image; caps the
MM_KWARGS="${MM_KWARGS:-$DEFAULT_MM_KWARGS}"    # profiling peak on 24 GB cards
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}"

for g in ${RESERVED_GPUS//,/ }; do
  case ",$GPUS," in *",$g,"*) echo "GPU $g is reserved (RESERVED_GPUS)" >&2; exit 1;; esac
done
MODEL_DIR="$(realpath "$MODEL_DIR")"

port=$PORT0
IFS=',' read -ra ids <<< "$GPUS"
for gpu in "${ids[@]}"; do
  docker run -d --name "etbench_policy_gpu${gpu}" \
    --label project=evidencedataset --label purpose=etbench-policy \
    --gpus "\"device=${gpu}\"" --ipc=host \
    -p "127.0.0.1:${port}:8000" \
    -v "${MODEL_DIR}:/model:ro" \
    -e NAME="$NAME" -e MAX_LEN="$MAX_LEN" -e MEM="$MEM" -e LIMIT_MM="$LIMIT_MM" -e MM_KWARGS="$MM_KWARGS" \
    -e API_KEY="${LOCAL_QWEN_API_KEY:-local}" \
    --entrypoint bash "$IMAGE" -c '
      f=$(python3 -c "import transformers.tokenization_utils_base as t; print(t.__file__)")
      sed -i "s/_config\.model_type/_config.get(\"model_type\")/g" "$f"
      exec python3 -m vllm.entrypoints.openai.api_server --model /model \
        --served-model-name "$NAME" --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$MEM" --limit-mm-per-prompt "$LIMIT_MM" \
        --mm-processor-kwargs "$MM_KWARGS" \
        --host 0.0.0.0 --port 8000 --api-key "$API_KEY"' >/dev/null
  echo "GPU $gpu -> http://127.0.0.1:${port}/v1 (etbench_policy_gpu${gpu})"
  port=$((port + 1))
done
echo "Started ${#ids[@]} policy server(s); loading takes a few minutes."
