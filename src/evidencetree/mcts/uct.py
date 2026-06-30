"""UCT selection for retrieval-action MCTS.

    UCT(a|s) = Q(a|s) + c * sqrt(ln N(s) / N(s,a))

    Q(a|s)   : backed-up mean reward (PRM prior when the edge is unvisited)
    2nd term : classic UCB1 exploration (Hoeffding lineage)

Search is guided purely by the PRM's value estimate Q plus standard UCB1
exploration. There is deliberately NO modality-coverage / lambda novelty term:
the PRM's Q is the single quality signal, and bolting an extra exploration prior
on top only diluted it and added hyperparameters (lambda, w_mod, w_gran) to tune.

Pure function so the math is unit-testable in isolation; the searcher composes
it with node statistics.
"""

from __future__ import annotations

import math

# Unvisited children sort first (classic UCT treats N=0 as infinite urgency);
# adding q on top breaks ties by PRM prior.
UNVISITED_BASE = 1e6


def uct_score(
    q_value: float,
    visits: int,
    parent_visits: int,
    c: float = 1.0,
) -> float:
    """UCB1 score for one child edge: Q + c * sqrt(ln N_parent / N_child)."""
    if visits == 0:
        return UNVISITED_BASE + q_value
    exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / visits)
    return q_value + exploration
