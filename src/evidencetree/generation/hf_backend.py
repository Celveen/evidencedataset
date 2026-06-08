"""HuggingFace transformers generation backend (local model).

Real backend for the GPU server (or slow local CPU/MPS dev). Loads a causal LM
and generates a short answer from a RAG prompt. Requires ``requirements/models.txt``.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Generator, GenerationConfig


class HFGenerator(Generator):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        if not config.model:
            raise ValueError("HFGenerator requires config.model (a HF model id).")
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - exercised only on server
            raise ImportError(
                "HF backend needs torch + transformers. "
                "Install with: pip install -r requirements/models.txt"
            ) from e

        self._torch = __import__("torch")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def generate(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        prompt = self._format_chat(question, context_docs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.temperature > 0,
                temperature=max(self.config.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _format_chat(self, question: str, context_docs: Sequence[str]) -> str:
        user = self._build_prompt(question, context_docs)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{self.config.system_prompt}\n\n{user}"
