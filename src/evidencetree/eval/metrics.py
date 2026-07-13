"""Answer-correctness metrics (SQuAD-style EM / F1).

Used in Stage 0.1 to compute *outcome correctness* of a trajectory's final
answer against the gold answer(s), and reused by Stage 7 evaluation.
"""

from __future__ import annotations

import ast
import re
import string
from collections import Counter
from typing import Any, Iterable, Sequence


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


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


def _extract_numbers(text: str) -> list[float]:
    return [float(match.replace(",", "")) for match in _NUMBER_RE.findall(text)]


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
