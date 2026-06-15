"""Modality-specific grounding verifiers: g(action, state) -> float in [0, 1].

All grounding scores are GRADED (continuous 0-1), never binary (report §3.2).

Backends:
    lexical — deterministic token-overlap alignment. Offline, free; used by
              mock smoke runs and as a sanity baseline.
    api     — LLM judge through the generation API backend, returns a graded
              score. Used by DatasetConstruct when no local verifier is up.

Per-action-type semantics (report §Stage 2):
    text_search  -> alignment of the search query with the original question
                    (lexical now; local cross-encoder lands with full Stage 2)
    image_search -> needs CLIP / VL matching; the lexical backend returns a
                    neutral 0.5 (documented limitation — CLIP verifier is
                    server-side Stage 2 work). The api backend judges normally.
    answer       -> None: not locally scored, outcome decides.
"""

from __future__ import annotations

import re
from typing import Sequence

from evidencetree.generation.base import Generator

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "are", "was", "were",
    "which", "what", "who", "whom", "whose", "where", "when", "how", "name",
    "can", "be", "found", "located", "does", "do", "did",
}

_JUDGE_PROMPT = """You are grading one retrieval action inside a multi-step \
retrieval trajectory.

Question: {question}
Evidence already collected before this action:
{evidence}

Proposed action: {action_type}({action_input})

Rate how well-grounded this action is: does it plausibly move toward answering
the question given what is already known? 0.0 = clearly unjustified or
redundant, 1.0 = clearly the right move. Reply with ONLY a number in [0, 1]."""

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    }


class GroundingVerifier:
    """g(action, state) with a pluggable backend."""

    def __init__(self, backend: str = "lexical", generator: Generator | None = None):
        backend = backend.lower()
        if backend not in {"lexical", "api"}:
            raise ValueError(f"Unknown verifier backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api verifier backend needs a generator.")
        self.backend = backend
        self.generator = generator

    # ------------------------------------------------------------------ #
    def score(
        self,
        *,
        question: str,
        action_type: str,
        action_input: str,
        state_evidence_texts: Sequence[str] = (),
    ) -> float | None:
        """Graded grounding score, or None for actions outcome decides."""
        if action_type == "answer":
            return None
        if self.backend == "lexical":
            return self._lexical(question, action_type, action_input)
        return self._api(question, action_type, action_input, state_evidence_texts)

    # ------------------------------------------------------------------ #
    def _lexical(self, question: str, action_type: str, action_input: str) -> float:
        if action_type == "image_search":
            # Lexical features cannot judge an image query; neutral score.
            # The CLIP-based verifier (Stage 2, GPU server) replaces this.
            return 0.5
        q_tokens = _content_tokens(question)
        a_tokens = _content_tokens(action_input)
        if not a_tokens or not q_tokens:
            return 0.0
        # Precision of the query w.r.t. the question: drifting queries score low.
        return len(q_tokens & a_tokens) / len(a_tokens)

    def _api(
        self,
        question: str,
        action_type: str,
        action_input: str,
        state_evidence_texts: Sequence[str],
    ) -> float:
        evidence = "\n".join(state_evidence_texts) or "(none)"
        prompt = _JUDGE_PROMPT.format(
            question=question,
            evidence=evidence,
            action_type=action_type,
            action_input=action_input,
        )
        try:
            raw = self.generator.generate(prompt, [])
            match = _NUMBER_RE.search(raw)
            if match is None:
                return 0.5
            return min(1.0, max(0.0, float(match.group(1))))
        except Exception:
            return 0.5  # judge failure -> neutral, never crash the pipeline


def grounding_score(action, state, verifier: GroundingVerifier | None = None) -> float | None:
    """Convenience wrapper over typed Action/SearchState objects."""
    verifier = verifier or GroundingVerifier()
    action_input = getattr(action, "query", None) or getattr(action, "text", "") or ""
    return verifier.score(
        question=state.question,
        action_type=action.action_type,
        action_input=str(action_input),
        state_evidence_texts=state.evidence_texts(),
    )
