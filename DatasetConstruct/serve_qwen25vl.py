"""OpenAI-compatible local server for Qwen2.5-VL-7B-Instruct."""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"

app = FastAPI(title="Local Qwen2.5-VL")
model = None
processor = None
model_name = "Qwen2.5-VL-7B-Instruct"
generation_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            converted = []
            for item in content:
                item_type = item.get("type")
                if item_type == "image_url":
                    value = item.get("image_url", {})
                    url = value.get("url") if isinstance(value, dict) else value
                    converted.append({"type": "image", "image": url})
                elif item_type == "image":
                    converted.append(item)
                elif item_type == "text":
                    converted.append({"type": "text", "text": item.get("text", "")})
            content = converted
        normalized.append({"role": message.get("role", "user"), "content": content})
    return normalized


def run_generation(request: ChatRequest) -> str:
    messages = normalize_messages(request.messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    sampling = request.temperature > 0
    kwargs = {
        "max_new_tokens": request.max_tokens,
        "do_sample": sampling,
    }
    if sampling:
        kwargs.update(temperature=request.temperature, top_p=request.top_p)
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


@app.get("/health")
def health():
    return {"status": "ok", "model": model_name, "cuda": torch.cuda.is_available()}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        async with generation_lock:
            output = await asyncio.to_thread(run_generation, request)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    prompt_tokens = 0
    completion_tokens = len(output.split())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main() -> int:
    global model, processor, model_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    args = parser.parse_args()

    model_name = args.model.name
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=256 * 28 * 28,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
