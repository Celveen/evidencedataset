"""Run one local Qwen2.5-VL-7B inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"


def load_model(model_path: Path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        local_files_only=True,
    )
    return model, processor


def generate(model, processor, prompt: str, image: Path | None, max_tokens: int) -> str:
    content = []
    if image is not None:
        content.append({"type": "image", "image": str(image.resolve())})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

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

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens)
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model, processor = load_model(args.model)
    print(generate(model, processor, args.prompt, args.image, args.max_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
