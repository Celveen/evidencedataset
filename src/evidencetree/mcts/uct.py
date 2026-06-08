"""UCT selection with modality novelty bonus.

Placeholder — implemented in **Stage 5**.

    UCT(a|s) = Q(a|s) + c * sqrt(ln N(s) / N(s,a)) + lambda * nu(a, pi)

    Q(a|s)    : PRM score (exploitation)
    2nd term  : classic exploration
    nu(a, pi) : modality novelty bonus (new modality +w_mod, new granularity +w_gran)

IMPORTANT: the modality bonus lives in SELECTION, never in the reward function.
"""

from __future__ import annotations

_STAGE = "Stage 5 — MCTS 搜索（固定 λ 版本）"


def uct_score(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"uct_score is not implemented yet (planned for {_STAGE})."
    )
