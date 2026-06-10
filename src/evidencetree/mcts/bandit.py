"""Thompson-Sampling bandit for self-adjusting lambda (Algorithm 1, v1.3 §3.3.1).

    arms     : lambda in {0.1, 0.3, 0.5, 0.7, 1.0}, each with a Beta(1,1) prior
    warm-up  : first 3 selections round-robin over {0.1, 0.5, 1.0}
    then     : sample theta_k ~ Beta(alpha_k, beta_k), pick argmax
    update   : z = 1[reward > median(previous rewards)]; alpha_k* += z,
               beta_k* += 1 - z (only the arm just played; first reward only
               seeds the history — no posterior update, per Algorithm 1)

IMPORTANT: bandit state is PER-QUERY. Create a fresh instance for every query;
never share one across queries (it would break the frozen-model evaluation
paradigm — report §4.1.3).
"""

from __future__ import annotations

import random
from statistics import median


class ThompsonBandit:
    """Per-query Thompson Sampling over discrete lambda arms."""

    DEFAULT_ARMS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)
    DEFAULT_WARMUP_ARMS: tuple[float, ...] = (0.1, 0.5, 1.0)

    def __init__(
        self,
        arms: tuple[float, ...] = DEFAULT_ARMS,
        warmup: int = 3,
        warmup_arms: tuple[float, ...] | None = None,
        seed: int = 0,
    ) -> None:
        if not arms:
            raise ValueError("Bandit needs at least one arm.")
        self.arms = tuple(float(a) for a in arms)
        self.warmup = int(warmup)
        self.warmup_arms = tuple(
            float(a) for a in (warmup_arms or self.DEFAULT_WARMUP_ARMS)
            if float(a) in self.arms
        ) or self.arms
        self._rng = random.Random(seed)
        k = len(self.arms)
        self._alpha = [1.0] * k
        self._beta = [1.0] * k
        self._history: list[float] = []
        self._t = 0                       # selections made so far
        self._pending: int | None = None  # arm awaiting its update()

    # ------------------------------------------------------------------ #
    def select(self) -> float:
        """Pick the lambda for the next rollout."""
        if self._pending is not None:
            raise RuntimeError("select() called twice without update().")
        if self._t < self.warmup:
            arm_value = self.warmup_arms[self._t % len(self.warmup_arms)]
            idx = self.arms.index(arm_value)
        else:
            samples = [
                self._rng.betavariate(self._alpha[k], self._beta[k])
                for k in range(len(self.arms))
            ]
            idx = max(range(len(self.arms)), key=samples.__getitem__)
        self._pending = idx
        self._t += 1
        return self.arms[idx]

    def update(self, reward: float) -> None:
        """Feed back the rollout reward for the lambda just selected."""
        if self._pending is None:
            raise RuntimeError("update() called before select().")
        if self._history:  # first reward only seeds the median history
            z = 1.0 if reward > median(self._history) else 0.0
            self._alpha[self._pending] += z
            self._beta[self._pending] += 1.0 - z
        self._history.append(float(reward))
        self._pending = None

    # ------------------------------------------------------------------ #
    # Introspection (ablation A6.2/A6.3 read these)
    # ------------------------------------------------------------------ #
    @property
    def history(self) -> list[float]:
        return list(self._history)

    @property
    def posterior_means(self) -> list[float]:
        return [
            a / (a + b) for a, b in zip(self._alpha, self._beta)
        ]

    def converged_lambda(self) -> float:
        """lambda-hat_q: the arm with the highest posterior mean (A6.2)."""
        means = self.posterior_means
        return self.arms[max(range(len(self.arms)), key=means.__getitem__)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        stats = ", ".join(
            f"{arm}: {m:.2f}" for arm, m in zip(self.arms, self.posterior_means)
        )
        return f"ThompsonBandit(t={self._t}, posterior_means={{{stats}}})"
