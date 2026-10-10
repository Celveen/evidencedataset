#!/usr/bin/env bash
# Serve the local rationale / support-judge model with vLLM in a container,
# tensor-parallel over RATIONALE_GPUS, bound to localhost only. Steps 4-5 then
# use `--provider local_qwen3_coder` (configs/providers/).
#
# Usage (from the repository root):
#   bash DatasetConstruct/serve_rationale_vllm.sh                 # Qwen3-Coder-30B-A3B
#   MODEL_DIR=/path/to/Qwen3-30B-A3B-Instruct-2507 NAME=qwen3-30b-a3b-instruct \
#       bash DatasetConstruct/serve_rationale_vllm.sh
# Check: curl -s http://127.0.0.1:8020/v1/models     Logs: docker logs -f etbench_rationale
# Stop:  docker stop etbench_rationale && docker rm etbench_rationale
#
# Machine-specific settings (weights dir, image, GPUs) come from
# DatasetConstruct/site.env — see site.env.example.
#
# transformers 4.57.2 crashes loading a local tokenizer whose config.json
# records transformers_version <= 4.57.2 (it reads config.json as a dict, then
# asks for `.model_type`). The container patches that one expression at
# start-up (a no-op on other versions); image and model files stay untouched.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# site.env supplies defaults; a variable set on the command line wins.
if [ -f "$HERE/site.env" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    if [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && [ -z "${!key+x}" ]; then export "$key=$value"; fi
  done < "$HERE/site.env"
fi

WEIGHTS_DIR="${WEIGHTS_DIR:-$HERE/../models}"
MODEL_DIR="${MODEL_DIR:-$WEIGHTS_DIR/Qwen3-Coder-30B-A3B-Instruct}"
NAME="${NAME:-qwen3-coder-30b-a3b}"          # must equal `model` in the provider preset
GPUS="${RATIONALE_GPUS:-0,1,2,3}"
PORT="${PORT:-8020}"
MAX_LEN="${MAX_LEN:-16384}"
MEM="${GPU_MEM:-0.80}"                       # 0.88 OOMs at warm-up next to a 0.8 GB process
MAX_SEQS="${MAX_SEQS:-64}"                   # concurrent sequences (step 5 sends 16)
CONTAINER="${CONTAINER:-etbench_rationale}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.11.2}"
TP=$(awk -F, '{print NF}' <<< "$GPUS")

for g in ${RESERVED_GPUS//,/ }; do
  case ",$GPUS," in *",$g,"*) echo "GPU $g is reserved (RESERVED_GPUS)" >&2; exit 1;; esac
done
MODEL_DIR="$(realpath "$MODEL_DIR")"

docker run -d --name "$CONTAINER" \
  --label project=evidencedataset --label purpose=etbench-rationale-judge \
  --gpus "\"device=${GPUS}\"" --ipc=host \
  -p "127.0.0.1:${PORT}:8000" \
  -v "${MODEL_DIR}:/model:ro" \
  -e NAME="$NAME" -e TP="$TP" -e MAX_LEN="$MAX_LEN" -e MEM="$MEM" -e MAX_SEQS="$MAX_SEQS" \
  -e API_KEY="${LOCAL_QWEN_API_KEY:-local}" \
  --entrypoint bash "$IMAGE" -c '
    f=$(python3 -c "import transformers.tokenization_utils_base as t; print(t.__file__)")
    sed -i "s/_config\.model_type/_config.get(\"model_type\")/g" "$f"
    exec python3 -m vllm.entrypoints.openai.api_server --model /model \
      --served-model-name "$NAME" --tensor-parallel-size "$TP" \
      --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM" --max-num-seqs "$MAX_SEQS" \
      --host 0.0.0.0 --port 8000 --api-key "$API_KEY"'
echo "Started $CONTAINER ($NAME) on GPUs $GPUS -> http://127.0.0.1:${PORT}/v1 (loading takes a few minutes)"
