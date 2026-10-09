"""Same-state preference pairs for ET-PRM's action-typed DPO stage.

Pairs are mined inside sibling groups of the final dataset — nodes sharing a
``parent_key``, i.e. the same state with different actions — so the contrast
is exactly the decision the searcher faces. Each pair references its two
samples by ``sample_id`` (the trainer joins them with the split file).

Pair types (paper §3.4 and Appendix H):

* ``action_type``      — text_search vs image_search from the same state;
* ``query_choice``     — two searches of the same type with different inputs;
* ``search_vs_answer`` — a search and an answer from the same state;
* ``answer_support``   — a supported answer vs a correct-but-unsupported one
  (the latter retained precisely to serve as this rejected side);
* ``answer_outcome``   — two answers that differ in correctness.

All of the above require a score margin of at least ``margin``. Optionally,
``answer_now_vs_search_more`` pairs a supported, successful answer (chosen)
with a successful search sibling (rejected) whose score is within the margin:
a label-free signal for when to stop searching. They are tagged so a trainer
can include or exclude them.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterable

from .sampling import sibling_groups


def _pair_type(a: dict[str, Any], b: dict[str, Any]) -> str:
    ta, tb = a["action"]["type"], b["action"]["type"]
    if ta == "answer" and tb == "answer":
        if a.get("unsupported_correct") or b.get("unsupported_correct"):
            return "answer_support"
        return "answer_outcome"
    if ta == "answer" or tb == "answer":
        return "search_vs_answer"
    return "action_type" if ta != tb else "query_choice"


def _record(chosen: dict[str, Any], rejected: dict[str, Any], pair_type: str) -> dict[str, Any]:
    return {
        "pair_id": f"{chosen['sample_id']}>{rejected['sample_id']}",
        "query_id": chosen["query_id"],
        "parent_key": chosen["parent_key"],
        "pair_type": pair_type,
        "chosen_sample_id": chosen["sample_id"],
        "rejected_sample_id": rejected["sample_id"],
        "chosen_action": chosen["action"]["type"],
        "rejected_action": rejected["action"]["type"],
        "chosen_score": chosen["score"],
        "rejected_score": rejected["score"],
        "margin": round(float(chosen["score"]) - float(rejected["score"]), 6),
    }


def mine_pairs(
    samples: Iterable[dict[str, Any]],
    margin: float = 0.3,
    stop_pairs: bool = True,
) -> list[dict[str, Any]]:
    """Mine same-state preference pairs from final (judged) samples."""
    pairs: list[dict[str, Any]] = []
    for members in sibling_groups(samples).values():
        for a, b in combinations(members, 2):
            diff = float(a["score"]) - float(b["score"])
            if abs(diff) >= margin:
                chosen, rejected = (a, b) if diff > 0 else (b, a)
                pairs.append(_record(chosen, rejected, _pair_type(a, b)))
            elif stop_pairs:
                for ans, srch in ((a, b), (b, a)):
                    if (
                        ans["action"]["type"] == "answer"
                        and srch["action"]["type"] != "answer"
                        and float(ans["outcome_credit"]) >= 0.5
                        and ans.get("answer_support") == 1.0
                        and float(srch["outcome_credit"]) >= 0.5
                    ):
                        pairs.append(_record(ans, srch, "answer_now_vs_search_more"))
    return pairs


def pair_stats(pairs: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(p["pair_type"] for p in pairs))
