"""Configurable generation backend.

A ``Generator`` turns (question, retrieved context) into an answer string. This
is used by Stage 0.1's vanilla RAG and (later) by rationale generation.

Backends are selected via config (``backend: mock | hf | api``) and imported
lazily so that the mock smoke test never imports torch / transformers / API SDKs.

The optional ``reference`` keyword on :meth:`Generator.generate` is an
EVALUATION-ONLY hint consumed solely by the mock backend to control simulated
accuracy. Real backends ignore it; passing it during real inference would be
cheating.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class GenerationConfig:
    """Backend-agnostic generation settings."""

    backend: str = "mock"           # "mock" | "hf" | "api"
    model: str | None = None        # HF model id or API model name
    provider: str = "anthropic"     # for api backend: "anthropic" | "openai"
    max_new_tokens: int = 64
    temperature: float = 0.0
    system_prompt: str = (
        "You are a question-answering assistant. Using ONLY the provided context, "
        "answer the question with a short, specific phrase. If the context is "
        "insufficient, answer with your best guess in a few words."
    )
    # mock-only knobs
    mock_accuracy: float = 0.6
    mock_seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GenerationConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        cfg = cls(**kwargs)
        cfg.extra.update(extra)
        return cfg


class Generator(ABC):
    """Abstract generation backend."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        """Return an answer string given a question and retrieved context."""

    def _build_prompt(self, question: str, context_docs: Sequence[str]) -> str:
        """Assemble a standard RAG prompt from context + question."""
        context = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(context_docs))
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )


def build_generator(config: GenerationConfig | dict | None) -> Generator:
    """Factory: instantiate the configured backend, importing it lazily."""
    if not isinstance(config, GenerationConfig):
        config = GenerationConfig.from_dict(config)

    backend = config.backend.lower()
    if backend == "mock":
        from .mock_backend import MockGenerator

        return MockGenerator(config)
    if backend == "hf":
        from .hf_backend import HFGenerator

        return HFGenerator(config)
    if backend == "api":
        from .api_backend import APIGenerator

        return APIGenerator(config)
    raise ValueError(
        f"Unknown generation backend {backend!r}. Expected 'mock', 'hf', or 'api'."
    )
