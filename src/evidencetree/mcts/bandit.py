"""Thompson-Sampling bandit for self-adjusting lambda.

Placeholder — implemented in **Stage 6** (Algorithm 1).

    arms: lambda in {0.1, 0.3, 0.5, 0.7, 1.0}, each Beta(1, 1) prior
    warm-up: first 3 round-robin over {0.1, 0.5, 1.0}
    then: Thompson-sample lambda each rollout
    update: reward > running median -> arm alpha+1 else beta+1

IMPORTANT: bandit state is PER-QUERY (fresh each query; never shared across
queries — sharing would break the frozen-model evaluation paradigm).
"""

from __future__ import annotations

_STAGE = "Stage 6 — Self-Adjusting Bandit"


class ThompsonBandit:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"ThompsonBandit is not implemented yet (planned for {_STAGE})."
        )
