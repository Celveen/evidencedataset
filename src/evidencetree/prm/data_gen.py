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
    * answer steps additionally get an ANSWER-SUPPORT label — is the answer
      backed by the accumulated evidence bundle? Their score is
      outcome * (floor + (1 - floor) * support), which keeps the ordering
      wrong (0) < correct-but-unsupported (~floor) < correct-and-supported (~1)
      instead of rewarding parametric lucky guesses with a perfect label.
      support=None (verifier off / cannot judge) falls back to outcome-only.

Trajectory dicts follow the DatasetConstruct step-1 JSONL schema (see
DatasetConstruct/README.md).
"""

from __future__ import annotations

from typing import Any, Iterable

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
    """Dual-source reward fusion; answer steps (local=None) are outcome-only."""
    if local is None:
        return outcome
    return alpha * float(local) + (1.0 - alpha) * float(outcome)


def fuse_answer_score(
    outcome: float, support: float | None, floor: float = 0.3
) -> float:
    """Answer-step score: outcome gated by evidence support.

    Multiplicative in outcome (a wrong answer stays low no matter how
    "supported" a spurious passage looks), but floor-lifted in support so a
    correct-but-unsupported answer lands near ``floor`` — clearly below a
    supported one, clearly above a wrong one — rather than being crushed to 0
    and becoming indistinguishable from a wrong answer (which would also make
    the label brittle to support-verifier noise).
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
    """Produce one step-level training sample per (trajectory, step).

    Each sample snapshots the state BEFORE the action (actions + evidence so
    far), the action itself, the evidence it observed, and the three labels.
    """
    verifier = verifier or GroundingVerifier()
    credit = tree_level_credit(trajectories)

    samples: list[dict[str, Any]] = []
    for traj in trajectories:
        actions_before: list[str] = []
        evidence_before: list[dict[str, Any]] = []
        prefix: tuple[tuple[str, str], ...] = ()
        for step in traj["steps"]:
            prefix = prefix + ((step["action_type"], step["action_input"]),)
            # Grounding scores the RETRIEVED RESULT of this step vs the question.
            obs = step.get("evidence", [])
            local = verifier.score(
                question=traj["question"],
                action_type=step["action_type"],
                result_texts=[e["text"] for e in obs if e.get("text")],
                result_image_paths=[e["image_path"] for e in obs if e.get("image_path")],
            )
            outcome, n_through = credit[(traj["query_id"], prefix)]
            is_answer = step["action_type"] == "answer"
            support: float | None = None
            if is_answer and support_verifier is not None:
                support = support_verifier.score(
                    question=traj["question"],
                    answer=step["action_input"],
                    evidence_texts=[e["text"] for e in evidence_before if e.get("text")],
                )
            score = (
                fuse_answer_score(outcome, support, support_floor)
                if is_answer
                else fuse_score(local, outcome, alpha)
            )
            # "Correct but unsupported" answers are the parametric-guess failure
            # mode: flagged (not dropped) — they are ready-made negatives for the
            # Stage 4.2 same-state DPO pairs.
            unsupported_correct = bool(
                is_answer
                and support is not None
                and support <= unsupported_threshold
                and outcome >= 0.5
            )
            samples.append(
                {
                    "sample_id": f"{traj['traj_id']}#s{step['step_index']}",
                    "traj_id": traj["traj_id"],
                    "query_id": traj["query_id"],
                    "question": traj["question"],
                    "image_path": traj.get("image_path"),
                    "state": {
                        "actions_before": list(actions_before),
                        "evidence_before": [
                            {"evidence_id": e["evidence_id"],
                             "title": e.get("title", ""),
                             "text": e["text"]}
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
                         "text": e["text"]}
                        for e in step.get("evidence", [])
                    ],
                    "local_grounding": local,
                    "outcome_credit": outcome,
                    "n_traj_through": n_through,
                    "alpha": alpha,
                    "answer_support": support,
                    "support_floor": support_floor,
                    "unsupported_correct": unsupported_correct,
                    "gold_answers": traj.get("gold_answers", []),
                    "score": score,
                }
            )
            actions_before.append(f"{step['action_type']}({step['action_input']})")
            evidence_before.extend(step.get("evidence", []))
    return samples
