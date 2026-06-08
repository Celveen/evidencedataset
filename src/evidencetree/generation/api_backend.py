"""External-API generation backend (Claude / GPT-4o).

Reads the API key from the environment (``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY``). Per the implementation report, the only sanctioned external
API call is offline rationale generation; reusing this backend for vanilla-RAG
generation during the pilot is fine. NEVER use it for the retrieval step.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from .base import Generator, GenerationConfig

_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",
}


class APIGenerator(Generator):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self.provider = config.provider.lower()
        self.model = config.model or _DEFAULT_MODELS.get(self.provider)
        if self.model is None:
            raise ValueError(f"No default model for provider {self.provider!r}.")
        self._client = self._make_client()

    def _make_client(self):
        if self.provider == "anthropic":
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise EnvironmentError("ANTHROPIC_API_KEY not set.")
            import anthropic

            return anthropic.Anthropic()
        if self.provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise EnvironmentError("OPENAI_API_KEY not set.")
            import openai

            return openai.OpenAI()
        raise ValueError(f"Unknown provider {self.provider!r}.")

    def generate(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        user = self._build_prompt(question, context_docs)
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text.strip()
        # openai
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()
