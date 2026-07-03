"""Rationale generation with a strong LLM (Stage 3, offline one-time).

The ONLY sanctioned external API in the project; retrieval stays offline.
Each (state, action, score) sample gets a natural-language rationale that
explains the score — this is the generative-PRM training signal.

Quality filter (report §3.5, extended): a rationale must
    (a) cite >= 1 evidence_id visible in the sample (state or observation),
    (b) explicitly mention the action type,
    (c) be 30-150 tokens long (whitespace tokens as the proxy),
    (d) NOT leak the gold answer: no evaluation phrasing ("ground truth",
        "correct answer", ...) and no gold-answer string that is absent from
        the sample's visible evidence (quoting the answer FROM cited evidence
        is legitimate; knowing it without evidence is leakage),
    (e) end with a ``VERDICT: good|mixed|poor`` line that does not hard-
        contradict the assigned score (a "good" verdict on a <=0.4 score or a
        "poor" verdict on a >=0.6 score means rationale and label disagree —
        training on such pairs teaches the PRM to say one thing and score
        another). The verdict line is stripped from the stored rationale and
        kept in ``rationale_verdict``.
Failures are regenerated up to ``max_attempts``; still-failing samples are
dropped by DatasetConstruct step 4.

Backends: ``api`` (Claude / GPT-4o via the generation backend) and ``mock``
(deterministic template that passes QC — pipeline smoke tests only).
"""

from __future__ import annotations

import re
from typing import Any

from evidencetree.generation.base import Generator

MIN_TOKENS, MAX_TOKENS = 30, 150

_PROMPT = """You are writing a training rationale for a process reward model \
that judges retrieval actions.

Question: {question}
Evidence collected BEFORE this action:
{evidence_before}
Action taken: {action_type}({action_input})
Evidence retrieved BY this action:
{observation}
Assigned quality score: {score:.2f}  (0 = poor action, 1 = excellent action)

Write a rationale of 30-150 tokens explaining WHY this score is appropriate.
Requirements:
- explicitly mention the action type "{action_type}";
- cite at least one evidence id in square brackets, e.g. [e0];
- be concrete about how the evidence supports (or fails to support) progress
  toward answering the question;
- judge the step ONLY from the question, the evidence, and the action — as if
  the final outcome were still unknown. NEVER mention or hint at a gold /
  ground-truth answer and never use evaluation language such as "the answer is
  correct" or "matches the ground truth";
- if the evidence does not actually support the step's usefulness, say so
  plainly.
After the rationale, end with ONE final line of exactly this form (it must be
consistent with the assigned score):
VERDICT: good     (clearly useful step)   or
VERDICT: mixed    (partially useful)      or
VERDICT: poor     (useless or harmful)"""

_EVIDENCE_ID_RE = re.compile(r"\be\d+\b")
_VERDICT_RE = re.compile(r"VERDICT:\s*(good|mixed|poor)\b", re.IGNORECASE)
# Evaluation phrasing that must never appear in a training rationale.
_LEAK_PHRASES = (
    "gold answer", "ground truth", "ground-truth", "correct answer",
    "标准答案", "参考答案", "正确答案",
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def split_verdict(text: str) -> tuple[str, str | None]:
    """Split raw generation into (rationale body, verdict|None)."""
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    body = (text[: last.start()] + text[last.end():]).strip()
    return body, last.group(1).lower()


def _gold_strings(sample: dict[str, Any]) -> list[str]:
    """Normalize gold answers to comparable strings (>=3 chars, non-boolean)."""
    out: list[str] = []
    for g in sample.get("gold_answers", []) or []:
        if isinstance(g, dict):
            g = g.get("wikidata", "")
        g = str(g).strip().lower()
        if len(g) >= 3 and g not in {"yes", "no", "none", "true", "false"}:
            out.append(g)
    return out


def visible_evidence_ids(sample: dict[str, Any]) -> set[str]:
    ids = {e["evidence_id"] for e in sample["state"]["evidence_before"]}
    ids |= {e["evidence_id"] for e in sample.get("observation_evidence", [])}
    return ids


def check_rationale(text: str, sample: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the QC rules to a raw generation; return (ok, failure reasons).

    ``text`` is the raw model output (verdict line included); length / mention /
    citation rules run on the body with the verdict line stripped.
    """
    reasons = []
    body, verdict = split_verdict(text)
    n_tokens = len(body.split())
    if not (MIN_TOKENS <= n_tokens <= MAX_TOKENS):
        reasons.append(f"length {n_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]")
    if sample["action"]["type"] not in body:
        reasons.append("does not mention the action type")
    cited = set(_EVIDENCE_ID_RE.findall(body))
    if not (cited & visible_evidence_ids(sample)):
        reasons.append("cites no evidence_id visible in the sample")

    # (d) gold-answer leakage — the rationale must argue ex-ante.
    body_lower = body.lower()
    for phrase in _LEAK_PHRASES:
        if phrase in body_lower:
            reasons.append(f"evaluation phrasing leaks the outcome: {phrase!r}")
            break
    visible_text = " ".join(
        e.get("text", "") + " " + e.get("title", "")
        for e in sample["state"].get("evidence_before", [])
    )
    visible_text += " " + " ".join(
        e.get("text", "") + " " + e.get("title", "")
        for e in sample.get("observation_evidence", [])
    )
    # The action input is part of the sample the PRM sees (for an answer step it
    # IS the answer under evaluation) — quoting it is not leakage.
    visible_text = (
        visible_text
        + " "
        + sample.get("question", "")
        + " "
        + str(sample["action"].get("input", ""))
    ).lower()
    body_words = set(_WORD_RE.findall(body_lower))
    for gold in _gold_strings(sample):
        in_body = gold in body_lower if " " in gold else gold in body_words
        if in_body and gold not in visible_text:
            reasons.append("mentions a gold answer absent from visible evidence")
            break

    # (e) verdict present and not contradicting the score.
    if verdict is None:
        reasons.append("missing VERDICT line")
    else:
        score = float(sample["score"])
        if score >= 0.6 and verdict == "poor":
            reasons.append(f"verdict 'poor' contradicts score {score:.2f}")
        elif score <= 0.4 and verdict == "good":
            reasons.append(f"verdict 'good' contradicts score {score:.2f}")
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
        body, verdict = split_verdict(text)
        return {
            "rationale": body,
            "rationale_verdict": verdict,
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
        score = float(sample["score"])
        verdict = "good" if score >= 0.6 else ("poor" if score <= 0.4 else "mixed")
        return (
            f"This {action['type']} step receives a score of "
            f"{score:.2f} under the dual-source reward. "
            f"The action input {action['input']!r} stays aligned with the "
            f"original question, and the retrieved passage {cite} contains "
            f"the entity tokens that the question asks about, so the local "
            f"grounding component is consistent with the assigned value. "
            f"Among all rollout trajectories passing through this node, the "
            f"observed downstream success rate matches the outcome component "
            f"of the label, so the fused score is a faithful estimate of this "
            f"{action['type']} action's true contribution to answering.\n"
            f"VERDICT: {verdict}"
        )
