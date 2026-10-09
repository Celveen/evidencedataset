"""Answer-correctness metrics (SQuAD-style EM / F1, benchmark outcome rules).

Used to compute the *outcome correctness* of a trajectory's final answer
against the gold answer(s) — during dataset construction (step 1) and at
evaluation time. Every rule is deterministic; no LLM judge is involved.

:func:`outcome_match` dispatches to each benchmark's official rule:

* InfoSeek  — relaxed accuracy: numeric range for VALUE questions, normalized
  exact match over the gold aliases for STRING questions.
* ScienceQA — multiple choice: the prediction must resolve to the gold option,
  whether it names the option letter, the option text, or both.
* GQA (and anything else) — normalized exact match.
"""

from __future__ import annotations

import ast
import re
import string
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def normalize_answer(text: str) -> str:
    """Lower-case, remove punctuation, articles, and extra whitespace (SQuAD)."""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, ground_truths: str | Sequence[str]) -> float:
    """1.0 if normalized prediction equals any normalized gold answer, else 0.0."""
    golds = _as_list(ground_truths)
    if _numeric_range_match(prediction, golds):
        return 1.0
    pred_norm = normalize_answer(prediction)
    return float(any(pred_norm == normalize_answer(g) for g in golds))


def f1_score(prediction: str, ground_truths: str | Sequence[str]) -> float:
    """Max token-level F1 of the prediction against any gold answer."""
    golds = _as_list(ground_truths)
    if _numeric_range_match(prediction, golds):
        return 1.0
    return max((_f1_single(prediction, g) for g in golds), default=0.0)


def _numeric_range_match(prediction: str, golds: Sequence[str]) -> bool:
    """Return True when a predicted number falls in an InfoSeek range label."""

    pred_numbers = _extract_numbers(prediction)
    if not pred_numbers:
        return False
    for gold in golds:
        label = _parse_mapping(gold)
        if not isinstance(label, dict):
            continue
        answer_range = label.get("range")
        if (
            not isinstance(answer_range, Sequence)
            or isinstance(answer_range, str)
            or len(answer_range) != 2
        ):
            continue
        try:
            lo, hi = sorted(float(value) for value in answer_range)
        except (TypeError, ValueError):
            continue
        if any(lo <= value <= hi for value in pred_numbers):
            return True
    return False


def _parse_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None


def _f1_single(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        # If either is empty, F1 is 1.0 only when both are empty.
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _as_list(ground_truths: str | Sequence[str]) -> list[str]:
    if isinstance(ground_truths, str):
        return [ground_truths]
    return list(ground_truths)


# --------------------------------------------------------------------------- #
# InfoSeek relaxed accuracy (rule-based, NO LLM judge)
# --------------------------------------------------------------------------- #
# InfoSeek answers are of two kinds:
#   * STRING questions  -> gold is a list of acceptable strings (answer_eval).
#   * VALUE questions   -> gold is a numeric range {"wikidata": v, "range": [lo, hi]}.
# Bare exact_match silently scores EVERY value question 0, because it string-
# compares the prediction against the dict's repr. This is the official, fully
# deterministic relaxed metric: a predicted number inside [lo, hi] counts.
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_numbers(text: str) -> list[float]:
    nums = []
    for m in _NUM_RE.findall(text):
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return nums


def _numeric_range_gold(golds: Sequence) -> tuple[float, float] | None:
    """Return (lo, hi) if any gold encodes an InfoSeek numeric range (dict or its
    string repr), else None."""
    for g in golds:
        obj = g
        if isinstance(g, str) and "range" in g and "{" in g:
            try:
                obj = ast.literal_eval(g)
            except (ValueError, SyntaxError):
                continue
        if isinstance(obj, dict) and isinstance(obj.get("range"), (list, tuple)) \
                and len(obj["range"]) == 2:
            try:
                lo, hi = float(obj["range"][0]), float(obj["range"][1])
            except (TypeError, ValueError):
                continue
            return (min(lo, hi), max(lo, hi))
    return None


def infoseek_accuracy(prediction: str, ground_truths: str | Sequence) -> float:
    """InfoSeek-style relaxed accuracy. Numeric (VALUE) questions: any number in
    the prediction within the gold ``range`` -> 1.0. String questions: relaxed
    exact match over the gold list. Deterministic and transparent (no model)."""
    golds = _as_list(ground_truths)
    rng = _numeric_range_gold(golds)
    if rng is not None:
        lo, hi = rng
        return float(any(lo <= n <= hi for n in _extract_numbers(prediction)))
    return exact_match(prediction, golds)


# --------------------------------------------------------------------------- #
# Multiple choice (ScienceQA)
# --------------------------------------------------------------------------- #
_LEADING_LETTER_RE = re.compile(r"^\s*(?:option\s+)?\(?([A-Za-z])\)?(?:[.):,]|\s|$)")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*",
    re.IGNORECASE,
)


def parse_choice(prediction: str, choices: Sequence[str]) -> int | None:
    """Resolve a free-form prediction to an option index, or None.

    Accepted forms: the option text ("mammal"), the option letter ("B",
    "(B)", "B."), both ("B. mammal"), or a sentence naming exactly one option
    ("the answer is mammal"). A letter that contradicts the option text that
    follows it, or a sentence naming several options, is ambiguous -> None.
    """
    if not choices:
        return None
    text = _ANSWER_PREFIX_RE.sub("", str(prediction).strip())
    norm_choices = [normalize_answer(str(c)) for c in choices]
    norm_pred = normalize_answer(text)
    if norm_pred:
        exact = [i for i, c in enumerate(norm_choices) if c == norm_pred]
        if len(exact) == 1:
            return exact[0]

    m = _LEADING_LETTER_RE.match(text)
    if m:
        index = ord(m.group(1).upper()) - ord("A")
        if 0 <= index < len(choices):
            rest = normalize_answer(text[m.end():])
            if not rest or rest == norm_choices[index]:
                return index
            named = [i for i, c in enumerate(norm_choices) if c and c == rest]
            if not named:
                return index  # trailing explanation that names no other option
            return None  # "B. <text of another option>" is contradictory

    mentioned = [
        i for i, c in enumerate(norm_choices)
        if c and re.search(rf"(?<!\w){re.escape(c)}(?!\w)", norm_pred)
    ]
    # Drop options that are substrings of a longer mentioned option.
    mentioned = [
        i for i in mentioned
        if not any(j != i and norm_choices[i] in norm_choices[j] for j in mentioned)
    ]
    return mentioned[0] if len(mentioned) == 1 else None


def multiple_choice_match(
    prediction: str, choices: Sequence[str], answer_index: int
) -> float:
    """1.0 if the prediction resolves to the gold option, else 0.0."""
    return float(parse_choice(prediction, choices) == int(answer_index))


# --------------------------------------------------------------------------- #
# Benchmark dispatch
# --------------------------------------------------------------------------- #
def _benchmark_key(benchmark: str | None) -> str:
    return str(benchmark or "").lower().replace("-", "").replace("_", "")


def outcome_metric_name(benchmark: str | None) -> str:
    """Name of the rule :func:`outcome_match` applies (recorded in provenance)."""
    key = _benchmark_key(benchmark)
    if key == "infoseek":
        return "infoseek_relaxed"
    if key == "scienceqa":
        return "scienceqa_multiple_choice"
    return "normalized_exact_match"


def outcome_match(
    prediction: str,
    gold_answers: str | Sequence[Any],
    benchmark: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> float:
    """Outcome correctness under the benchmark's official rule."""
    key = _benchmark_key(benchmark)
    meta = metadata or {}
    if key == "scienceqa" and meta.get("choices") and meta.get("answer_index") is not None:
        return multiple_choice_match(prediction, meta["choices"], int(meta["answer_index"]))
    if key == "infoseek":
        return infoseek_accuracy(prediction, gold_answers)
    return exact_match(prediction, gold_answers)


def answer_type(
    gold_answers: str | Sequence[Any], metadata: Mapping[str, Any] | None = None
) -> str:
    """``choice`` (multiple choice), ``value`` (numeric range) or ``string``."""
    if (metadata or {}).get("choices"):
        return "choice"
    if _numeric_range_gold(_as_list(gold_answers)) is not None:
        return "value"
    return "string"


def aggregate(
    predictions: Iterable[str], references: Iterable[str | Sequence[str]]
) -> dict[str, float]:
    """Return mean EM and F1 over a dataset."""
    ems, f1s = [], []
    for pred, ref in zip(predictions, references):
        ems.append(exact_match(pred, ref))
        f1s.append(f1_score(pred, ref))
    n = len(ems)
    return {
        "exact_match": sum(ems) / n if n else 0.0,
        "f1": sum(f1s) / n if n else 0.0,
        "n": float(n),
    }
