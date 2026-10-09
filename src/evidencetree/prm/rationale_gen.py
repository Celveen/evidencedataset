"""Rationale generation with a strong LLM (Stage 3, offline one-time).

The ONLY sanctioned external API in the project; retrieval stays offline.
Each (state, action, score) sample gets a natural-language rationale that
explains the score — this is the generative-PRM training signal.

Quality filter (report §3.5, extended): a rationale must
    (a) cite >= 1 evidence_id visible in the sample (state or observation),
    (b) explicitly mention the action type,
    (c) be 30-150 tokens long (whitespace tokens as the proxy),
    (d) not leak the gold answer or use evaluation phrasing,
    (e) end with ``VERDICT: good|mixed|poor`` and keep that verdict consistent
        with the assigned score.
Failures are regenerated up to ``max_attempts``; still-failing samples are
dropped by DatasetConstruct step 4.

Backends: ``api`` (any provider via the generation backend) and ``mock``
(deterministic template that passes QC — pipeline smoke tests only).

Cost: the fixed instructions come first and the per-sample content last, so a
provider's automatic prefix cache can reuse the instruction block (and, when
samples are sent grouped by query, the question and earlier evidence as well).
Evidence gathered before a SEARCH step is shown truncated — the rationale is
about what this step retrieved — while answer steps, whose support lives in
that earlier evidence, see it in full.

Group mode (:meth:`RationaleGenerator.generate_group`): siblings — candidate
actions taken from the SAME state — share one request; the question and the
shared evidence are sent once and the model writes one labeled rationale per
candidate. Every rationale must stand alone (the PRM later sees one candidate
at a time), so QC additionally rejects a rationale that names evidence only
another candidate retrieved; a candidate whose block is missing or fails QC is
regenerated alone. Groups never span states, so no rationale is written with
hindsight of later steps.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Sequence

from evidencetree.generation.base import Generator

from .data_gen import format_evidence_line

MIN_TOKENS, MAX_TOKENS = 30, 150

_PROMPT = """You are writing a training rationale for a process reward model \
that judges retrieval actions.

Write a rationale of 30-150 tokens explaining WHY the assigned score below is
appropriate for the action. Requirements:
- explicitly mention the action type (text_search, image_search or answer)
  exactly as written in the sample;
- cite at least one evidence id in square brackets, e.g. [e0];
- be concrete about how the evidence supports (or fails to support) progress
  toward answering the question;
- judge the step ONLY from the question, the evidence, and the action. Never
  mention or hint at a gold / ground-truth answer, and never use evaluation
  language such as "the answer is correct" or "matches the ground truth";
- if the evidence does not actually support the step's usefulness, say so
  plainly;
- write only the rationale in plain prose: do not repeat the question, the
  evidence, or any section header, and quote at most a few words of evidence.
How to judge retrieval steps: a search that identifies the entity shown in the
question image (e.g. an image_search returning that entity's page) is real
progress even before the asked attribute is found; a search that only repeats
information already in the earlier evidence adds little; an answer is well
supported only when the earlier evidence states it.
After the rationale, end with ONE final line of exactly this form:
VERDICT: good
or
VERDICT: mixed
or
VERDICT: poor
The VERDICT must be consistent with the assigned score (0 = poor action,
1 = excellent action). Output ONLY the rationale text plus the final VERDICT
line.

--- SAMPLE ---
Question: {question}
Evidence collected BEFORE this action:
{evidence_before}
Action taken: {action_type}({action_input})
Evidence retrieved BY this action:
{observation}
Assigned quality score: {score:.2f}"""

_GROUP_PROMPT = """You are writing training rationales for a process reward model \
that judges retrieval actions.

The {k} candidate actions below were each taken from the SAME state: the same
question and the same evidence collected so far. For EACH candidate, write a
rationale of 30-150 tokens explaining WHY its assigned score is appropriate.
Every rationale must stand alone: it will be shown together with ONLY its own
action and evidence, so never mention, compare with, or rely on the other
candidates or the evidence they retrieved.

Requirements for each rationale:
- explicitly mention its action type (text_search, image_search or answer)
  exactly as written;
- cite at least one evidence id in square brackets, e.g. [e0], from the shared
  evidence or from ITS OWN retrieved evidence;
- be concrete about how the evidence supports (or fails to support) progress
  toward answering the question;
- judge the step ONLY from the question, the evidence, and the action. Never
  mention or hint at a gold / ground-truth answer, and never use evaluation
  language such as "the answer is correct" or "matches the ground truth";
- if the evidence does not actually support the step's usefulness, say so
  plainly;
- write only the rationale in plain prose: do not repeat the question, the
  evidence, or any section header, and quote at most a few words of evidence.
How to judge retrieval steps: a search that identifies the entity shown in the
question image (e.g. an image_search returning that entity's page) is real
progress even before the asked attribute is found; a search that only repeats
information already in the earlier evidence adds little; an answer is well
supported only when the earlier evidence states it.
Format: for every candidate, in order, output its label alone on a line
([[C1]], [[C2]], ...), then its rationale, then ONE line of exactly the form
VERDICT: good, VERDICT: mixed or VERDICT: poor, consistent with ITS score
(0 = poor action, 1 = excellent action). Output nothing else.

--- SHARED STATE ---
Question: {question}
Evidence collected BEFORE these actions:
{evidence_before}

--- CANDIDATES ---
{candidates}"""

_CANDIDATE = """[[C{index}]]
Action taken: {action_type}({action_input})
Evidence retrieved BY this action:
{observation}
Assigned quality score: {score:.2f}"""

_LABEL_RE = re.compile(r"\[\[C(\d+)\]\]")

_EVIDENCE_ID_RE = re.compile(r"\be\d+\b")
# Prompt scaffolding a rationale must not echo back.
_SCAFFOLD = (
    "evidence retrieved by this action", "evidence collected before",
    "assigned quality score", "action taken:", "--- sample", "--- shared state",
    "--- candidates", "[[c",
)
_RATIONALE_LABEL_RE = re.compile(r"(?im)^\s*\**rationale\**\s*:\s*")
_COPY_WORDS = 12  # this many consecutive evidence words = verbatim copying
_VERDICT_RE = re.compile(r"VERDICT:\s*(good|mixed|poor)\b", re.IGNORECASE)
# Evaluation phrasing that leaks the outcome. Chinese variants are included
# because the rationale model occasionally answers in Chinese.
_LEAK_PHRASES = (
    "gold answer", "ground truth", "ground-truth", "correct answer",
    "标准答案", "参考答案", "正确答案",
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def salvage_rationale(text: str) -> str:
    """Drop a leading ``Rationale:`` label, and when the model echoed the prompt
    (question / evidence / headers) before such a label, keep only what follows
    the last label — the rationale it actually wrote."""
    labels = list(_RATIONALE_LABEL_RE.finditer(text))
    if not labels:
        return text.strip()
    head = text[: labels[-1].start()].lower()
    if labels[-1].start() == 0 or any(m in head for m in _SCAFFOLD) or "question:" in head:
        return text[labels[-1].end():].strip()
    return text.strip()


def copied_span(body: str, sample: dict[str, Any], n: int = _COPY_WORDS) -> str | None:
    """First run of ``n`` consecutive words copied verbatim from the evidence."""
    words = _WORD_RE.findall(body.lower())
    grams = {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    if not grams:
        return None
    for e in list(sample["state"].get("evidence_before", [])) + list(
        sample.get("observation_evidence", [])
    ):
        ew = _WORD_RE.findall(str(e.get("text", "")).lower())
        for i in range(len(ew) - n + 1):
            if tuple(ew[i:i + n]) in grams:
                return " ".join(ew[i:i + n])
    return None


def split_verdict(text: str) -> tuple[str, str | None]:
    """Split raw generation into (rationale body, verdict)."""
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        return text.strip(), None
    last = matches[-1]
    body = (text[: last.start()] + text[last.end():]).strip()
    return body, last.group(1).lower()


def _gold_strings(sample: dict[str, Any]) -> list[str]:
    """Normalize gold answers to comparable strings."""
    out: list[str] = []
    for gold in sample.get("gold_answers", []) or []:
        if isinstance(gold, dict):
            gold = gold.get("wikidata", "")
        gold = str(gold).strip().lower()
        if len(gold) >= 3 and gold not in {"yes", "no", "none", "true", "false"}:
            out.append(gold)
    return out


def visible_evidence_ids(sample: dict[str, Any]) -> set[str]:
    ids = {e["evidence_id"] for e in sample["state"]["evidence_before"]}
    ids |= {e["evidence_id"] for e in sample.get("observation_evidence", [])}
    return ids


def _visible_text(sample: dict[str, Any]) -> str:
    """Everything a rationale may draw on, lower-cased: question, action input,
    and the titles/texts of the evidence before and retrieved by the action."""
    parts = [sample.get("question", ""), str(sample["action"].get("input", ""))]
    for e in list(sample["state"].get("evidence_before", [])) + list(
        sample.get("observation_evidence", [])
    ):
        parts += [e.get("text", ""), e.get("title", "")]
    return " ".join(parts).lower()


def sibling_leak(body: str, sample: dict[str, Any], siblings: Sequence[dict[str, Any]]) -> str | None:
    """Title of evidence that only ANOTHER candidate retrieved and that ``body``
    names, else None (group mode: a rationale must not rely on its siblings)."""
    body_lower = body.lower()
    own = _visible_text(sample)
    for sib in siblings:
        if sib is sample:
            continue
        for e in sib.get("observation_evidence", []):
            title = (e.get("title") or "").strip()
            if len(title) >= 4 and title.lower() in body_lower and title.lower() not in own:
                return title
    return None


def check_rationale(
    text: str, sample: dict[str, Any], siblings: Sequence[dict[str, Any]] = ()
) -> tuple[bool, list[str]]:
    """Apply rationale QC rules; return (ok, failure reasons).

    ``text`` is the raw model output. Length, action mention and citation rules
    run on the body with the verdict line stripped. ``siblings`` (group mode)
    adds the stand-alone rule: no evidence that only another candidate saw.
    """
    reasons = []
    body, verdict = split_verdict(salvage_rationale(text))
    n_tokens = len(body.split())
    if not (MIN_TOKENS <= n_tokens <= MAX_TOKENS):
        reasons.append(f"length {n_tokens} outside [{MIN_TOKENS}, {MAX_TOKENS}]")
    if sample["action"]["type"] not in body:
        reasons.append("does not mention the action type")
    cited = set(_EVIDENCE_ID_RE.findall(body))
    if not (cited & visible_evidence_ids(sample)):
        reasons.append("cites no evidence_id visible in the sample")

    body_lower = body.lower()
    for phrase in _LEAK_PHRASES:
        if phrase in body_lower:
            reasons.append(f"evaluation phrasing leaks the outcome: {phrase!r}")
            break

    visible_text = _visible_text(sample)
    body_words = set(_WORD_RE.findall(body_lower))
    for gold in _gold_strings(sample):
        in_body = gold in body_lower if " " in gold else gold in body_words
        if in_body and gold not in visible_text:
            reasons.append("mentions a gold answer absent from visible evidence")
            break

    if any(m in body_lower for m in _SCAFFOLD):
        reasons.append("echoes the prompt instead of writing a rationale")
    copied = copied_span(body, sample)
    if copied:
        reasons.append(f"copies evidence verbatim: '{copied[:60]}...'")

    if siblings:
        leaked = sibling_leak(body, sample, siblings)
        if leaked:
            reasons.append(f"relies on evidence another candidate retrieved: {leaked!r}")

    if verdict is None:
        reasons.append("missing VERDICT line")
    else:
        score = float(sample["score"])
        if score >= 0.6 and verdict == "poor":
            reasons.append(f"verdict 'poor' contradicts score {score:.2f}")
        elif score <= 0.4 and verdict == "good":
            reasons.append(f"verdict 'good' contradicts score {score:.2f}")
    return (not reasons, reasons)


def parse_group_output(text: str, k: int) -> dict[int, str]:
    """Split a group reply into {candidate index: raw block (rationale + VERDICT)}."""
    marks = list(_LABEL_RE.finditer(text))
    out: dict[int, str] = {}
    for i, m in enumerate(marks):
        index = int(m.group(1))
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        if 1 <= index <= k and index not in out:
            out[index] = text[m.end():end].strip()
    return out


def group_samples(
    samples: Sequence[dict[str, Any]], max_group: int = 6
) -> list[list[dict[str, Any]]]:
    """Siblings (same query and parent state) in chunks of at most ``max_group``;
    first-appearance order is kept."""
    groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for s in samples:
        key = (s["query_id"], s.get("parent_key") or s["sample_id"])
        groups.setdefault(key, []).append(s)
    size = max(1, int(max_group))
    return [g[i:i + size] for g in groups.values() for i in range(0, len(g), size)]


class RationaleGenerator:
    """Generate + QC-retry rationales for step-level samples."""

    def __init__(
        self,
        backend: str = "mock",
        generator: Generator | None = None,
        max_attempts: int = 3,
        before_chars: int | None = 300,
        before_chars_answer: int | None = 1500,
        observation_chars: int | None = 1500,
    ) -> None:
        backend = backend.lower()
        if backend not in {"mock", "api"}:
            raise ValueError(f"Unknown rationale backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api rationale backend needs a generator.")
        self.backend = backend
        self.generator = generator
        self.max_attempts = max(1, int(max_attempts))
        self.before_chars = before_chars
        self.before_chars_answer = before_chars_answer
        self.observation_chars = observation_chars

    # ------------------------------------------------------------------ #
    def generate_for(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return rationale fields to merge into the sample."""
        text, ok, reasons, attempts = "", False, ["not attempted"], 0
        for attempts in range(1, self.max_attempts + 1):
            text = (
                self._mock(sample)
                if self.backend == "mock"
                else self._api(sample, reasons if attempts > 1 else None)
            )
            ok, reasons = check_rationale(text, sample)
            if ok:
                break
        body, verdict = split_verdict(salvage_rationale(text))
        return {
            "rationale": body,
            "rationale_verdict": verdict,
            "rationale_backend": self.backend,
            "rationale_attempts": attempts,
            "rationale_qc_pass": ok,
            "rationale_qc_reasons": reasons if not ok else [],
            "rationale_mode": "single",
        }

    def generate_group(self, samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rationale fields for siblings from ONE request, in input order.

        Each block is QC-checked against its own sample (plus the stand-alone
        rule); a missing or failing block is regenerated alone with
        :meth:`generate_for`, its attempt count including the group request.
        Transport errors propagate (the caller retries the whole group).
        """
        samples = list(samples)
        if len(samples) == 1:
            return [self.generate_for(samples[0])]
        text = (
            self._mock_group(samples)
            if self.backend == "mock"
            else self.generator.generate(self.build_group_prompt(samples), []).strip()
        )
        blocks = parse_group_output(text, len(samples))
        results: list[dict[str, Any]] = []
        for index, sample in enumerate(samples, 1):
            block = blocks.get(index)
            if block is not None:
                ok, _ = check_rationale(block, sample, siblings=samples)
                if ok:
                    body, verdict = split_verdict(salvage_rationale(block))
                    results.append({
                        "rationale": body,
                        "rationale_verdict": verdict,
                        "rationale_backend": self.backend,
                        "rationale_attempts": 1,
                        "rationale_qc_pass": True,
                        "rationale_qc_reasons": [],
                        "rationale_mode": "group",
                    })
                    continue
            fields = self.generate_for(sample)
            fields["rationale_attempts"] += 1
            fields["rationale_mode"] = "group_fallback"
            results.append(fields)
        return results

    # ------------------------------------------------------------------ #
    def build_prompt(
        self, sample: dict[str, Any], previous_reasons: list[str] | None = None
    ) -> str:
        state = sample["state"]
        is_answer = sample["action"]["type"] == "answer"
        limit = self.before_chars_answer if is_answer else self.before_chars
        before = "\n".join(
            format_evidence_line(e, limit) for e in state["evidence_before"]
        ) or "(none)"
        observation = "\n".join(
            format_evidence_line(e, self.observation_chars)
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
        if previous_reasons:
            prompt += (
                "\n\nThe previous attempt failed validation because: "
                + "; ".join(previous_reasons)
                + ". Rewrite it and correct every listed issue. Aim for about "
                "85 whitespace-separated words."
            )
        return prompt

    def build_group_prompt(self, samples: Sequence[dict[str, Any]]) -> str:
        first = samples[0]
        has_answer = any(s["action"]["type"] == "answer" for s in samples)
        limit = self.before_chars_answer if has_answer else self.before_chars
        before = "\n".join(
            format_evidence_line(e, limit) for e in first["state"]["evidence_before"]
        ) or "(none)"
        candidates = "\n\n".join(
            _CANDIDATE.format(
                index=i,
                action_type=s["action"]["type"],
                action_input=s["action"]["input"],
                observation="\n".join(
                    format_evidence_line(e, self.observation_chars)
                    for e in s.get("observation_evidence", [])
                ) or "(none)",
                score=float(s["score"]),
            )
            for i, s in enumerate(samples, 1)
        )
        return _GROUP_PROMPT.format(
            k=len(samples), question=first["question"], evidence_before=before,
            candidates=candidates,
        )

    def _api(
        self, sample: dict[str, Any], previous_reasons: list[str] | None = None
    ) -> str:
        return self.generator.generate(self.build_prompt(sample, previous_reasons), []).strip()

    def _mock_group(self, samples: Sequence[dict[str, Any]]) -> str:
        return "\n".join(f"[[C{i}]]\n{self._mock(s)}" for i, s in enumerate(samples, 1))

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
