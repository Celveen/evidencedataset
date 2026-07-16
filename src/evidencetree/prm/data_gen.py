"""Step labelling for ETBench-Open (Stage 3, used by DatasetConstruct step 2).

For each step of each trajectory:
    * local grounding label — Stage 2 verifier, graded 0-1 (None for answer)
    * outcome label         — TREE-LEVEL credit: the mean outcome of ALL
                              trajectories passing through this node (Monte
                              Carlo estimate, report §3.2). A node is the
                              action-sequence prefix within one query. This is
                              NOT trajectory-uniform final-reward averaging —
                              that is the classic bug (report §4.1.1).
    * score = alpha * local + (1 - alpha) * outcome  (non-answer steps)
    * answer steps additionally get an answer-support label. Their score is
      outcome * (floor + (1 - floor) * support), so correct-but-unsupported
      guesses land near the floor instead of receiving a perfect label.
      support=None falls back to outcome-only (verifier off / cannot judge /
      judge ruled NOT_REQUIRED — the question is answerable from the question
      + image alone). The raw judge label is recorded in
      ``answer_support_label`` for diagnostics.

Trajectory dicts follow the DatasetConstruct step-1 JSONL schema (see
DatasetConstruct/README.md).
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Iterable, Iterator

from .verifiers import AnswerSupportVerifier, GroundingVerifier

PrefixKey = tuple[str, tuple[tuple[str, str], ...]]  # (query_id, action prefix)


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


def label_steps(
    trajectories: list[dict[str, Any]],
    verifier: GroundingVerifier | None = None,
    alpha: float = 0.5,
    support_verifier: AnswerSupportVerifier | None = None,
    support_floor: float = 0.3,
    unsupported_threshold: float = 0.3,
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
        )
    )


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
) -> Iterator[dict[str, Any]]:
    """Produce one step-level training sample per (trajectory, step).

    Each sample snapshots the state BEFORE the action (actions + evidence so
    far), the action itself, the evidence it observed, and the three labels.

    ``skip_sample_ids`` lets DatasetConstruct step 2 resume from a partially
    written scored JSONL without re-calling expensive support judges.

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
    if support_verifier is not None and support_concurrency > 1:
        executor = ThreadPoolExecutor(max_workers=support_concurrency)

    def _finalize_answer(
        sample: dict[str, Any], support_label: str | None, support: float | None
    ) -> dict[str, Any]:
        outcome = float(sample["outcome_credit"])
        sample["answer_support"] = support
        sample["answer_support_label"] = support_label
        sample["score"] = fuse_answer_score(outcome, support, support_floor)
        sample["unsupported_correct"] = bool(
            support is not None
            and support <= unsupported_threshold
            and outcome >= 0.5
        )
        sample["score_source"] = (
            "answer_support" if support is not None else "outcome_only"
        )
        return sample

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
            samples.append(_finalize_answer(sample, support_label, support))
        return samples

    try:
        for traj in trajectories:
            actions_before: list[str] = []
            evidence_before: list[dict[str, Any]] = []
            prefix: tuple[tuple[str, str], ...] = ()
            for step in traj["steps"]:
                sample_id = f"{traj['traj_id']}#s{step['step_index']}"
                prefix = prefix + ((step["action_type"], step["action_input"]),)
                if sample_id in skip_sample_ids:
                    actions_before.append(f"{step['action_type']}({step['action_input']})")
                    evidence_before.extend(step.get("evidence", []))
                    continue
                obs = step.get("evidence", [])
                local = verifier.score(
                    question=traj["question"],
                    action_type=step["action_type"],
                    result_texts=[e["text"] for e in obs if e.get("text")],
                    result_image_paths=[
                        e["image_path"] for e in obs if e.get("image_path")
                    ],
                )
                outcome, n_through = credit[(traj["query_id"], prefix)]
                is_answer = step["action_type"] == "answer"
                sample = {
                "sample_id": sample_id,
                "traj_id": traj["traj_id"],
                "query_id": traj["query_id"],
                "question": traj["question"],
                "image_path": traj.get("image_path"),
                "state": {
                    "actions_before": list(actions_before),
                    "evidence_before": [
                        {"evidence_id": e["evidence_id"],
                         "title": e.get("title", ""),
                         "text": e["text"],
                         "image_path": e.get("image_path"),
                         "score": e.get("score"),
                         "result_modality": e.get("result_modality")}
                        for e in evidence_before
                    ],
                },
                "action": {
                    "type": step["action_type"],
                    "input": step["action_input"],
                },
                "observation_evidence": [
                    {"evidence_id": e["evidence_id"],
                     "title": e.get("title", ""),
                     "text": e["text"],
                     "image_path": e.get("image_path"),
                     "score": e.get("score"),
                     "result_modality": e.get("result_modality")}
                    for e in step.get("evidence", [])
                ],
                "local_grounding": local,
                "outcome_credit": outcome,
                "n_traj_through": n_through,
                "alpha": alpha,
                "answer_support": None,
                "answer_support_label": None,
                "support_floor": support_floor,
                "unsupported_correct": False,
                "gold_answers": traj.get("gold_answers", []),
                "score": 0.0,
                "score_source": "",
            }
                if is_answer:
                    if support_verifier is None:
                        yield _finalize_answer(sample, None, None)
                    elif executor is None:
                        support_label, support = support_verifier.classify(
                            question=traj["question"],
                            answer=step["action_input"],
                            evidence_texts=[
                                e["text"] for e in evidence_before if e.get("text")
                            ],
                        )
                        yield _finalize_answer(sample, support_label, support)
                    else:
                        while len(pending) >= support_max_pending:
                            yield from _drain_pending(block=True)
                        future = executor.submit(
                            support_verifier.classify,
                            question=traj["question"],
                            answer=step["action_input"],
                            evidence_texts=[
                                e["text"] for e in evidence_before if e.get("text")
                            ],
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
                evidence_before.extend(step.get("evidence", []))
        while pending:
            yield from _drain_pending(block=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
