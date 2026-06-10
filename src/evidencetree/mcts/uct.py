"""UCT selection with modality novelty bonus (Modality-Coverage UCB).

    UCT(a|s) = Q(a|s) + c * sqrt(ln N(s) / N(s,a)) + lambda * nu(a, pi)

    Q(a|s)    : backed-up mean reward (PRM prior when unvisited)
    2nd term  : classic UCB1 exploration (Hoeffding lineage)
    nu(a, pi) : modality novelty — +w_mod for an action type unseen on the
                path pi, +w_gran for an unseen visual granularity

IMPORTANT: the modality bonus lives in SELECTION, never in the reward function
(reward judges facts; selection judges exploration policy — report §4.1.5).

Pure functions so the math is unit-testable in isolation; the searcher
composes them with node statistics.
"""

from __future__ import annotations

import math
from typing import Sequence

from evidencetree.actions.action_space import Action

# Unvisited children sort first (classic UCT treats N=0 as infinite urgency);
# adding q + lambda*nu on top breaks ties by PRM prior and novelty.
UNVISITED_BASE = 1e6


def modality_novelty(
    action: Action,
    path_actions: Sequence[Action],
    w_mod: float = 1.0,
    w_gran: float = 0.5,
) -> float:
    """nu(a, pi): bonus for introducing a new action type / granularity.

    ``path_actions`` is the path pi the candidate would extend (root -> parent).
    Answer actions get no novelty — they terminate, they don't explore.
    """
    if action.modality == "answer":
        return 0.0
    nu = 0.0
    if action.action_type not in {a.action_type for a in path_actions}:
        nu += w_mod
    seen_gran = {a.granularity for a in path_actions if a.granularity != "n/a"}
    if action.granularity != "n/a" and action.granularity not in seen_gran:
        nu += w_gran
    return nu


def uct_score(
    q_value: float,
    visits: int,
    parent_visits: int,
    novelty: float = 0.0,
    c: float = 1.0,
    lam: float = 0.3,
) -> float:
    """Modality-Coverage UCB score for one child edge."""
    if visits == 0:
        return UNVISITED_BASE + q_value + lam * novelty
    exploration = c * math.sqrt(math.log(max(parent_visits, 1)) / visits)
    return q_value + exploration + lam * novelty
