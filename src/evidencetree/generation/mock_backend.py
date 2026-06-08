"""Mock generation backend — no model, no network.

Simulates a weak RAG generator with a tunable accuracy so the Stage 0.1 pipeline
(and its correlation analysis) can be exercised locally with *signal*.

Behaviour:
* If an evaluation ``reference`` (gold answer) is provided, return it with
  probability ``config.mock_accuracy``; otherwise return a plausible wrong
  answer drawn from a distractor pool.
* If no reference is provided, fall back to a naive heuristic: echo the most
  salient capitalized token found in the retrieved context.

The ``reference`` hint is a smoke-test device only; real backends never see it.
"""

from __future__ import annotations

import random
import re
from typing import Any, Sequence

from .base import Generator, GenerationConfig

_WRONG_POOL = [
    "London", "Tokyo", "Berlin", "Cairo", "Madrid",
    "Canada", "Brazil", "Germany", "France", "Spain",
]
_CAP_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z]+)\b")


class MockGenerator(Generator):
    def __init__(self, config: GenerationConfig) -> None:
        super().__init__(config)
        self._rng = random.Random(config.mock_seed)

    def generate(
        self, question: str, context_docs: Sequence[str], **kwargs: Any
    ) -> str:
        reference = kwargs.get("reference")
        if reference is not None:
            gold = reference[0] if isinstance(reference, (list, tuple)) else reference
            if self._rng.random() < self.config.mock_accuracy:
                return str(gold)
            wrong = [w for w in _WRONG_POOL if w.lower() != str(gold).lower()]
            return self._rng.choice(wrong)

        # No hint: naive extraction from context.
        for doc in context_docs:
            caps = _CAP_TOKEN_RE.findall(doc)
            if caps:
                return caps[-1]
        return "unknown"
