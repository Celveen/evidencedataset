"""Rationale generation with a strong LLM (Stage 3, offline one-time).

The ONLY sanctioned external API in the project; retrieval stays offline.
Each (state, action, score) sample gets a natural-language rationale that
explains the score — this is the generative-PRM training signal.

Quality filter (report §3.5): a rationale must
    (a) cite >= 1 evidence_id visible in the sample (state or observation),
    (b) explicitly mention the action type,
    (c) be 50-150 tokens long (whitespace tokens as the proxy).
Failures are regenerated up to ``max_attempts``; still-failing samples are
dropped by DatasetConstruct step 4.

Backends: ``api`` (Claude / GPT-4o via the generation backend) and ``mock``
(deterministic template that passes QC — pipeline smoke tests only).
"""

from __future__ import annotations

import re
from typing import Any

from evidencetree.generation.base import Generator

MIN_TOKENS, MAX_TOKENS = 50, 150

_PROMPT = """You are writing a training rationale for a process reward model \
that judges retrieval actions.

Question: {question}
Evidence collected BEFORE this action:
{evidence_before}
Action taken: {action_type}({action_input})
Evidence retrieved BY this action:
{observation}
Assigned quality score: {score:.2f}  (0 = poor action, 1 = excellent action)

Write a rationale of 50-150 tokens explaining WHY this score is appropriate.
Requirements:
- explicitly mention the action type "{action_type}";
- cite at least one evidence id in square brackets, e.g. [e0];
- be concrete about how the evidence supports (or fails to support) progress
  toward answering the question.
Output ONLY the rationale text."""

_EVIDENCE_ID_RE = re.compile(r"\be\d+\b")


def visible_evidence_ids(sample: dict[str, Any]) -> set[str]:
    ids = {e["evidence_id"] for e in sample["state"]["evidence_before"]}
    ids |= {e["evidence_id"] for e in sample.get("observation_evidence", [])}
    return ids


def check_rationale(text: str, sample: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the report's three QC rules; return (ok, failure reasons)."""
    reasons = []
    n_tokens = len(text.split())
    if not (MIN_TOKENS <= n_tokens <= MAX_TOKENS):
        reasons.append(f"length {n_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]")
    if sample["action"]["type"] not in text:
        reasons.append("does not mention the action type")
    cited = set(_EVIDENCE_ID_RE.findall(text))
    if not (cited & visible_evidence_ids(sample)):
        reasons.append("cites no evidence_id visible in the sample")
    return (not reasons, reasons)


class RationaleGenerator:
    """Generate + QC-retry rationales for step-level samples."""

    def __init__(
        self,
        backend: str = "mock",
        generator: Generator | None = None,
        max_attempts: int = 3,
    ) -> None:
        backend = backend.lower()
        if backend not in {"mock", "api"}:
            raise ValueError(f"Unknown rationale backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api rationale backend needs a generator.")
        self.backend = backend
        self.generator = generator
        self.max_attempts = max(1, int(max_attempts))

    # ------------------------------------------------------------------ #
    def generate_for(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return rationale fields to merge into the sample."""
        text, ok, reasons, attempts = "", False, ["not attempted"], 0
        for attempts in range(1, self.max_attempts + 1):
            text = (
                self._mock(sample) if self.backend == "mock" else self._api(sample)
            )
            ok, reasons = check_rationale(text, sample)
            if ok:
                break
        return {
            "rationale": text,
            "rationale_backend": self.backend,
            "rationale_attempts": attempts,
            "rationale_qc_pass": ok,
            "rationale_qc_reasons": reasons if not ok else [],
        }

    # ------------------------------------------------------------------ #
    def _api(self, sample: dict[str, Any]) -> str:
        state = sample["state"]
        before = "\n".join(
            f"[{e['evidence_id']}] {e['title']}: {e['text']}"
            for e in state["evidence_before"]
        ) or "(none)"
        observation = "\n".join(
            f"[{e['evidence_id']}] {e['title']}: {e['text']}"
            for e in sample.get("observation_evidence", [])
        ) or "(none)"
        prompt = _PROMPT.format(
            question=sample["question"],
            evidence_before=before,
            action_type=sample["action"]["type"],
            action_input=sample["action"]["input"],
            observation=observation,
            score=float(sample["score"]),
        )
        return self.generator.generate(prompt, []).strip()

    def _mock(self, sample: dict[str, Any]) -> str:
        """Deterministic template that satisfies the QC rules (smoke only)."""
        ids = sorted(visible_evidence_ids(sample))
        cite = f"[{ids[0]}]" if ids else "[none]"
        action = sample["action"]
        return (
            f"This {action['type']} step receives a score of "
            f"{float(sample['score']):.2f} under the dual-source reward. "
            f"The action input {action['input']!r} stays aligned with the "
            f"original question, and the retrieved passage {cite} contains "
            f"the entity tokens that the question asks about, so the local "
            f"grounding component is consistent with the assigned value. "
            f"Among all rollout trajectories passing through this node, the "
            f"observed downstream success rate matches the outcome component "
            f"of the label, so the fused score is a faithful estimate of this "
            f"{action['type']} action's true contribution to answering."
        )
