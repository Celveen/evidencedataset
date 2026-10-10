"""Step labelling for ETBench-Open (used by DatasetConstruct steps 2 and 4).

For each step of each trajectory:
    * local grounding label — grounding verifier, graded 0-1 (None for answer)
    * outcome label         — TREE-LEVEL credit: the mean outcome of ALL
                              trajectories passing through this node (Monte
                              Carlo estimate, paper Eq. 4). A node is the
                              action-sequence prefix within one query. This is
                              NOT trajectory-uniform final-reward averaging.
    * score = alpha * local + (1 - alpha) * outcome  (non-answer steps)
    * answer steps additionally get an answer-support label. Their score is
      outcome * (floor + (1 - floor) * support), so correct-but-unsupported
      guesses land near the floor instead of receiving a perfect label.
      support=None falls back to outcome-only (verifier off / cannot judge /
      judge ruled NOT_REQUIRED — the question is answerable from the question
      + image alone). The raw judge label is recorded in
      ``answer_support_label`` for diagnostics.

Construction labels in two phases so the paid judge only sees what is kept:
:func:`iter_label_steps` with ``defer_support=True`` computes the free labels
for every step and marks answer steps ``pending``; after node-level sampling
:func:`judge_answer_samples` calls the support judge on the sampled answer
steps only, and skips wrong answers entirely (their score is 0 whatever the
support is).

Each sample also records:
    * ``node_key`` / ``parent_key`` — hashes of (query, action prefix) with and
      without this action. Samples sharing ``parent_key`` are siblings: the same
      state, different actions (the unit of node-level sampling and DPO pairs).
    * ``grounding_detail`` — modality and raw cosine of the best result.
    * ``evidence_redundancy`` / ``marginal_gain`` — similarity of the best
      result to the evidence already held, and the paper's naive utility
      g * (1 - redundancy). Recorded for analysis; not a training label.

Trajectory dicts follow the DatasetConstruct step-1 JSONL schema (see
DatasetConstruct/SCHEMA.md).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Iterable, Iterator, Sequence

from .verifiers import AnswerSupportVerifier, GroundingVerifier

PrefixKey = tuple[str, tuple[tuple[str, str], ...]]  # (query_id, action prefix)

PENDING = "pending"                # answer step awaiting the support judge
SKIPPED_INCORRECT = "skipped_incorrect"  # wrong answer: judging cannot change its 0 score


def tree_level_credit(
    trajectories: Iterable[dict[str, Any]],
) -> dict[PrefixKey, tuple[float, int]]:
    """Map each node (query_id, action-prefix) to (mean outcome, #trajectories).

    Every trajectory contributes its ``outcome_em`` to every prefix along its
    own path; a node shared by several trajectories therefore gets the success
    rate over all of them.
    """
    acc: dict[PrefixKey, list[float]] = {}
    for traj in trajectories:
        outcome = float(traj["outcome_em"])
        prefix: tuple[tuple[str, str], ...] = ()
        for step in traj["steps"]:
            prefix = prefix + ((step["action_type"], step["action_input"]),)
            acc.setdefault((traj["query_id"], prefix), []).append(outcome)
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


def node_key(query_id: str, prefix: Sequence[tuple[str, str]]) -> str:
    """Stable id of the tree node reached by ``prefix`` in query ``query_id``."""
    payload = json.dumps([query_id, [list(p) for p in prefix]], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fuse_score(local: float | None, outcome: float, alpha: float = 0.5) -> float:
    """Dual-source reward fusion; local=None steps are outcome-only."""
    if local is None:
        return outcome
    return alpha * float(local) + (1.0 - alpha) * float(outcome)


def fuse_answer_score(
    outcome: float, support: float | None, floor: float = 0.3
) -> float:
    """Answer-step score gated by evidence support.

    Wrong answers stay at 0 via the outcome multiplier. Correct but unsupported
    answers land near ``floor``: lower than supported answers, but distinct from
    genuinely wrong answers and less brittle to support-verifier misses.
    """
    if support is None:
        return float(outcome)
    return float(outcome) * (floor + (1.0 - floor) * float(support))


def finalize_answer(
    sample: dict[str, Any],
    support_label: str | None,
    support: float | None,
    floor: float = 0.3,
    unsupported_threshold: float = 0.3,
    reason: str | None = None,
) -> dict[str, Any]:
    """Write the answer-support fields and the final score into ``sample``."""
    outcome = float(sample["outcome_credit"])
    sample["answer_support"] = support
    sample["answer_support_label"] = support_label
    sample["answer_support_reason"] = reason or None
    sample["support_floor"] = floor
    sample["score"] = fuse_answer_score(outcome, support, floor)
    sample["unsupported_correct"] = bool(
        support is not None and support <= unsupported_threshold and outcome >= 0.5
    )
    sample["score_source"] = "answer_support" if support is not None else "outcome_only"
    return sample


def format_evidence_line(e: dict[str, Any], max_chars: int | None = None) -> str:
    """``[e3] Title: text`` for text units, ``[e3] [image] Title`` for images."""
    eid = e.get("evidence_id", "")
    title = e.get("title") or ""
    text = e.get("text") or ""
    if not text and e.get("image_path"):
        return f"[{eid}] [image] {title or '(untitled picture)'}".strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + " …"
    body = f"{title}: {text}" if title else text
    return f"[{eid}] {body}".strip()


def support_evidence_lines(evidence: Sequence[dict[str, Any]]) -> list[str]:
    """Evidence bundle as the support judge sees it (titles and images kept)."""
    return [
        format_evidence_line(e)
        for e in evidence
        if e.get("text") or e.get("image_path")
    ]


def label_steps(
    trajectories: list[dict[str, Any]],
    verifier: GroundingVerifier | None = None,
    alpha: float = 0.5,
    support_verifier: AnswerSupportVerifier | None = None,
    support_floor: float = 0.3,
    unsupported_threshold: float = 0.3,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Produce one step-level training sample per (trajectory, step)."""
    return list(
        iter_label_steps(
            trajectories,
            verifier=verifier,
            alpha=alpha,
            support_verifier=support_verifier,
            support_floor=support_floor,
            unsupported_threshold=unsupported_threshold,
            **kwargs,
        )
    )


def _evidence_record(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": e["evidence_id"],
        "doc_id": e.get("doc_id"),
        "title": e.get("title", ""),
        "text": e.get("text", ""),
        "image_path": e.get("image_path"),
        "score": e.get("score"),
        "result_modality": e.get("result_modality"),
    }


def iter_label_steps(
    trajectories: list[dict[str, Any]],
    verifier: GroundingVerifier | None = None,
    alpha: float = 0.5,
    support_verifier: AnswerSupportVerifier | None = None,
    support_floor: float = 0.3,
    unsupported_threshold: float = 0.3,
    skip_sample_ids: set[str] | None = None,
    support_concurrency: int = 1,
    support_max_pending: int | None = None,
    defer_support: bool = False,
    benchmark: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Produce one step-level training sample per (trajectory, step).

    Each sample snapshots the state BEFORE the action (actions + evidence so
    far), the action itself, the evidence it observed, and the labels.

    ``defer_support=True`` leaves answer steps ``pending`` (score = outcome,
    provisionally) for :func:`judge_answer_samples`; otherwise answer steps are
    judged inline with ``support_verifier`` (None = outcome-only).

    ``skip_sample_ids`` lets DatasetConstruct resume from a partially written
    output without recomputing labels.

    ``support_concurrency`` only parallelizes answer-support judging, which is
    the slow API-bound part of step labelling. Local grounding remains
    sequential to avoid sharing CLIP model state across worker threads.
    """
    verifier = verifier or GroundingVerifier()
    credit = tree_level_credit(trajectories)
    skip_sample_ids = skip_sample_ids or set()
    support_concurrency = max(1, int(support_concurrency))
    if support_max_pending is None:
        support_max_pending = max(1, support_concurrency * 4)
    support_max_pending = max(1, int(support_max_pending))

    executor: ThreadPoolExecutor | None = None
    pending: dict[Future[tuple[str, float | None]], dict[str, Any]] = {}
    if support_verifier is not None and support_concurrency > 1 and not defer_support:
        executor = ThreadPoolExecutor(max_workers=support_concurrency)

    def _drain_pending(block: bool) -> list[dict[str, Any]]:
        if not pending:
            return []
        if block:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending if future.done()}
        samples: list[dict[str, Any]] = []
        for future in done:
            sample = pending.pop(future)
            try:
                support_label, support = future.result()
            except Exception:  # noqa: BLE001 - keep long jobs resumable
                support_label, support = ("unparseable", None)
            samples.append(
                finalize_answer(sample, support_label, support, support_floor, unsupported_threshold)
            )
        return samples

    try:
        for traj in trajectories:
            actions_before: list[str] = []
            evidence_before: list[dict[str, Any]] = []
            prefix: tuple[tuple[str, str], ...] = ()
            query_id = traj["query_id"]
            for step in traj["steps"]:
                sample_id = f"{traj['traj_id']}#s{step['step_index']}"
                parent = node_key(query_id, prefix)
                prefix = prefix + ((step["action_type"], step["action_input"]),)
                obs = step.get("evidence", [])
                if sample_id in skip_sample_ids:
                    actions_before.append(f"{step['action_type']}({step['action_input']})")
                    evidence_before.extend(obs)
                    continue
                is_answer = step["action_type"] == "answer"
                detail = verifier.score_results(
                    question=traj["question"],
                    action_type=step["action_type"],
                    results=obs,
                )
                local = detail.score
                redundancy = marginal = None
                n_repeat = None
                if not is_answer:
                    seen_docs = {e.get("doc_id") for e in evidence_before if e.get("doc_id")}
                    n_repeat = sum(1 for e in obs if e.get("doc_id") in seen_docs)
                    if detail.best_index is not None:
                        redundancy = verifier.redundancy(obs[detail.best_index], evidence_before)
                    elif obs:
                        redundancy = 0.0
                    marginal = (
                        float(local) * (1.0 - redundancy)
                        if local is not None and redundancy is not None
                        else 0.0
                    )
                outcome, n_through = credit[(query_id, prefix)]
                sample = {
                    "sample_id": sample_id,
                    "traj_id": traj["traj_id"],
                    "query_id": query_id,
                    "benchmark": traj.get("benchmark", benchmark),
                    "question": traj["question"],
                    "image_path": traj.get("image_path"),
                    "node_key": node_key(query_id, prefix),
                    "parent_key": parent,
                    "depth": step["step_index"],
                    "traj_len": len(traj["steps"]),
                    "state": {
                        "actions_before": list(actions_before),
                        "evidence_before": [_evidence_record(e) for e in evidence_before],
                    },
                    "action": {
                        "type": step["action_type"],
                        "input": step["action_input"],
                    },
                    "observation_evidence": [_evidence_record(e) for e in obs],
                    "local_grounding": local,
                    "grounding_detail": (
                        None if is_answer
                        else {"modality": detail.modality, "raw_cos": detail.raw_cos}
                    ),
                    "evidence_redundancy": redundancy,
                    "marginal_gain": marginal,
                    "n_repeat_docs": n_repeat,
                    "outcome_credit": outcome,
                    "n_traj_through": n_through,
                    "alpha": alpha,
                    "answer_support": None,
                    "answer_support_label": None,
                    "answer_support_reason": None,
                    "support_floor": support_floor,
                    "unsupported_correct": False,
                    "gold_answers": traj.get("gold_answers", []),
                    "answer_type": traj.get("answer_type"),
                    "score": 0.0,
                    "score_source": "",
                }
                if is_answer:
                    if defer_support:
                        sample["answer_support_label"] = PENDING
                        sample["score"] = float(outcome)  # provisional until judged
                        sample["score_source"] = "pending_support"
                        yield sample
                    elif support_verifier is None:
                        yield finalize_answer(sample, None, None, support_floor, unsupported_threshold)
                    elif executor is None:
                        support_label, support = support_verifier.classify(
                            question=traj["question"],
                            answer=step["action_input"],
                            evidence_texts=support_evidence_lines(evidence_before),
                        )
                        yield finalize_answer(
                            sample, support_label, support, support_floor, unsupported_threshold
                        )
                    else:
                        while len(pending) >= support_max_pending:
                            yield from _drain_pending(block=True)
                        future = executor.submit(
                            support_verifier.classify,
                            question=traj["question"],
                            answer=step["action_input"],
                            evidence_texts=support_evidence_lines(evidence_before),
                        )
                        pending[future] = sample
                        yield from _drain_pending(block=False)
                else:
                    sample["score"] = fuse_score(local, outcome, alpha)
                    sample["score_source"] = (
                        "outcome_only" if local is None else "local_plus_outcome"
                    )
                    yield sample
                actions_before.append(f"{step['action_type']}({step['action_input']})")
                evidence_before.extend(obs)
        while pending:
            yield from _drain_pending(block=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


def judge_answer_samples(
    samples: Iterable[dict[str, Any]],
    support_verifier: AnswerSupportVerifier | None,
    floor: float = 0.3,
    unsupported_threshold: float = 0.3,
    concurrency: int = 1,
    max_pending: int | None = None,
    skip_incorrect: bool = True,
) -> Iterator[dict[str, Any]]:
    """Finalize ``pending`` answer samples; every other sample passes through.

    Wrong answers (``outcome_credit == 0``) are not sent to the judge when
    ``skip_incorrect``: Eq. 6 multiplies support by the outcome, so their score
    is 0 either way. They are labelled ``skipped_incorrect``. Output order may
    differ from input order when ``concurrency > 1``.
    """
    concurrency = max(1, int(concurrency))
    max_pending = max(1, int(max_pending or concurrency * 4))
    executor = ThreadPoolExecutor(max_workers=concurrency) if concurrency > 1 else None
    pending: dict[Future[tuple[str, float | None, str]], dict[str, Any]] = {}

    def _judge(sample: dict[str, Any]) -> tuple[str, float | None, str]:
        return support_verifier.judge(
            question=sample["question"],
            answer=sample["action"]["input"],
            evidence_texts=support_evidence_lines(sample["state"]["evidence_before"]),
        )

    def _drain(block: bool) -> list[dict[str, Any]]:
        if not pending:
            return []
        done = wait(pending, return_when=FIRST_COMPLETED)[0] if block else {
            f for f in pending if f.done()
        }
        out = []
        for future in done:
            sample = pending.pop(future)
            try:
                label, support, reason = future.result()
            except Exception:  # noqa: BLE001 - keep long jobs resumable
                label, support, reason = ("unparseable", None, "")
            out.append(finalize_answer(sample, label, support, floor, unsupported_threshold, reason))
        return out

    try:
        for sample in samples:
            if sample["action"]["type"] != "answer" or sample.get("answer_support_label") != PENDING:
                yield sample
                continue
            if support_verifier is None:
                yield finalize_answer(sample, None, None, floor, unsupported_threshold)
                continue
            if skip_incorrect and float(sample["outcome_credit"]) <= 0.0:
                yield finalize_answer(sample, SKIPPED_INCORRECT, None, floor, unsupported_threshold)
                continue
            if executor is None:
                label, support, reason = _judge(sample)
                yield finalize_answer(sample, label, support, floor, unsupported_threshold, reason)
                continue
            while len(pending) >= max_pending:
                yield from _drain(block=True)
            pending[executor.submit(_judge, sample)] = sample
            yield from _drain(block=False)
        while pending:
            yield from _drain(block=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
