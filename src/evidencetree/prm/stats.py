"""Distribution report for an ETBench-Open build (``stats.json``, SCHEMA §4)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def histogram(values: Iterable[float | None], bins: int = 10) -> list[int]:
    """Counts over ``bins`` equal-width bins on [0, 1] (1.0 lands in the last bin)."""
    counts = [0] * bins
    for v in values:
        if v is None:
            continue
        counts[min(bins - 1, max(0, int(float(v) * bins)))] += 1
    return counts


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def split_stats(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-split statistics of released samples."""
    answers = [s for s in samples if s["action"]["type"] == "answer"]
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for s in answers:
        by_type.setdefault(str(s.get("answer_type") or "unknown"), []).append(s)
    trajs = {s["traj_id"]: s.get("traj_len") for s in samples}
    with_image = sum(
        1 for s in samples
        if any(
            e.get("image_path") and not e.get("text")
            for e in list(s["state"]["evidence_before"]) + list(s["observation_evidence"])
        )
    )
    gains = [s["marginal_gain"] for s in samples if s.get("marginal_gain") is not None]
    return {
        "samples": len(samples),
        "queries": len({s["query_id"] for s in samples}),
        "action_types": dict(Counter(s["action"]["type"] for s in samples)),
        "score_sources": dict(Counter(s.get("score_source") for s in samples)),
        "mean_trajectory_length": _mean([float(v) for v in trajs.values() if v is not None]),
        "hist_score": histogram(s["score"] for s in samples),
        "hist_local_grounding": histogram(s["local_grounding"] for s in samples),
        "hist_outcome_credit": histogram(s["outcome_credit"] for s in samples),
        "mean_score": _mean([float(s["score"]) for s in samples]),
        "mean_marginal_gain": _mean(gains),
        "samples_with_image_evidence": with_image,
        "answer_outcome_positive_rate": {
            t: _rate(sum(1 for s in rows if float(s["outcome_credit"]) >= 0.5), len(rows))
            for t, rows in sorted(by_type.items())
        },
        "answer_support_labels": dict(
            Counter(str(s.get("answer_support_label")) for s in answers)
        ),
        "unsupported_correct": sum(1 for s in answers if s.get("unsupported_correct")),
    }


def trajectory_stats(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Construction-level statistics over all step-1 rollouts."""
    by_type: dict[str, list[float]] = {}
    for t in trajectories:
        by_type.setdefault(str(t.get("answer_type") or "unknown"), []).append(float(t["outcome_em"]))
    n_q = len({t["query_id"] for t in trajectories})
    return {
        "trajectories": len(trajectories),
        "queries": n_q,
        "trajectories_per_query": round(len(trajectories) / n_q, 4) if n_q else None,
        "mean_steps": _mean([float(len(t["steps"])) for t in trajectories]),
        "outcome_positive_rate": {
            t: _rate(int(sum(v)), len(v)) for t, v in sorted(by_type.items())
        },
        "step_action_types": dict(
            Counter(s["action_type"] for t in trajectories for s in t["steps"])
        ),
    }


def rationale_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """QC statistics over every generated rationale (passed or not)."""
    reasons = Counter(
        r.split(":")[0].split(" ")[0] if r else ""
        for row in rows if not row.get("rationale_qc_pass")
        for r in (row.get("rationale_qc_reasons") or [])
    )
    return {
        "generated": len(rows),
        "qc_pass_rate": _rate(sum(1 for r in rows if r.get("rationale_qc_pass")), len(rows)),
        "mean_attempts": _mean([float(r.get("rationale_attempts", 1)) for r in rows]),
        "verdicts": dict(Counter(str(r.get("rationale_verdict")) for r in rows)),
        "failure_reasons": dict(reasons),
    }
