"""MCTS tree node.

Domain-agnostic (mirrors MCTS-RAG's backbone/domain split): a node carries an
immutable :class:`~evidencetree.actions.action_space.SearchState`, the action
edge that produced it, visit statistics, and a PRM prior assigned at expansion
time. All selection math lives in :mod:`evidencetree.mcts.uct`; all domain
logic (what actions exist, how they execute) lives in ``evidencetree.actions``.
"""

from __future__ import annotations

from evidencetree.actions.action_space import Action, SearchState


class MCTSNode:
    """One node of the search tree."""

    __slots__ = ("state", "parent", "action", "prior", "children", "visits",
                 "total_value")

    def __init__(
        self,
        state: SearchState,
        parent: "MCTSNode | None" = None,
        action: Action | None = None,
        prior: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action          # edge from parent (None at root)
        self.prior = float(prior)     # PRM score at proposal time
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.total_value = 0.0

    # ------------------------------------------------------------------ #
    @property
    def q_value(self) -> float:
        """Mean backed-up reward; falls back to the PRM prior when unvisited."""
        return self.total_value / self.visits if self.visits > 0 else self.prior

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def is_expanded(self) -> bool:
        return bool(self.children)

    @property
    def depth(self) -> int:
        return self.state.depth

    # ------------------------------------------------------------------ #
    def path_actions(self) -> list[Action]:
        """Actions on the path from the root to (and including) this node."""
        return list(self.state.actions_taken)

    def add_child(self, child: "MCTSNode") -> None:
        self.children.append(child)

    def update(self, reward: float) -> None:
        """Backup one rollout reward into this node."""
        self.visits += 1
        self.total_value += float(reward)

    def best_child_by_visits(self) -> "MCTSNode | None":
        return max(self.children, key=lambda c: c.visits) if self.children else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        act = self.action.describe() if self.action else "<root>"
        return (
            f"MCTSNode({act}, N={self.visits}, Q={self.q_value:.3f}, "
            f"children={len(self.children)}, terminal={self.is_terminal})"
        )
