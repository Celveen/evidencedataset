"""Main MCTS search loop (selection -> expansion -> simulation -> backup).

Placeholder — implemented in **Stage 5** (fixed lambda), extended in **Stage 6**
(bandit-adaptive lambda).

Defaults: P=10 rollouts, max_depth=3. Expansion uses the policy LLM to propose
candidate actions, the PRM to score them, and keeps the top-k as children.
Early stop when the best path's PRM Q exceeds a threshold.
"""

from __future__ import annotations

_STAGE = "Stage 5 — MCTS 搜索（固定 λ 版本）"


class MCTS:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"MCTS search is not implemented yet (planned for {_STAGE})."
        )
