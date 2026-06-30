"""Main MCTS loop: selection -> expansion -> simulation -> backup.

Retrieval-action MCTS guided by a pluggable ``TrajectoryScorer`` (today: mock /
frozen VisualPRM; later: EvidenceTree-PRM in score-only mode), so the framework
runs before any PRM is trained. Selection is standard UCB1 over the PRM's Q —
there is no bandit and no modality-coverage lambda term (both removed; the PRM's
Q is the single quality signal).

Defaults follow the implementation report: P=10 rollouts, max_depth=3, expansion
keeps the top-k PRM-ranked candidates, early stop when a rollout's reward exceeds
``early_stop_q`` (tau_stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evidencetree.actions.action_space import SearchState
from evidencetree.actions.executor import ActionExecutor
from evidencetree.mcts.node import MCTSNode
from evidencetree.mcts.proposer import ActionProposer
from evidencetree.mcts.uct import uct_score
from evidencetree.prm.model import Trajectory, TrajectoryStep


class TrajectoryScorer(Protocol):
    """PRM interface: anything with ``score(Trajectory) -> float`` in [0, 1]."""

    def score(self, trajectory: Trajectory, **kwargs: Any) -> float: ...


# --------------------------------------------------------------------------- #
# Config & result containers
# --------------------------------------------------------------------------- #
@dataclass
class SearchConfig:
    rollouts: int = 10            # P
    max_depth: int = 3
    top_k_children: int = 3
    c_uct: float = 1.0
    early_stop_q: float = 0.9     # tau_stop
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SearchConfig":
        """Build from a configs/mcts.yaml-style nested dict."""
        d = dict(d or {})
        search = dict(d.get("search", {}))
        kwargs: dict[str, Any] = {}
        for key in ("rollouts", "max_depth", "top_k_children", "c_uct",
                    "early_stop_q", "seed"):
            if key in search:
                kwargs[key] = search[key]
        return cls(**kwargs)


@dataclass
class RolloutRecord:
    t: int
    reward: float
    answer: str | None
    depth: int
    state: SearchState | None = None  # terminal state (full trajectory; used
                                      # by dataset construction & Stage 0.2)


@dataclass
class SearchResult:
    answer: str
    best_reward: float
    best_state: SearchState
    rollouts_run: int
    rollout_log: list[RolloutRecord] = field(default_factory=list)
    root: MCTSNode | None = None


# --------------------------------------------------------------------------- #
# Trajectory adapter (search state -> PRM input)
# --------------------------------------------------------------------------- #
def state_to_trajectory(state: SearchState) -> Trajectory:
    """Flatten a (possibly partial) state into the PRM's Trajectory format."""
    steps = []
    for i, action in enumerate(state.actions_taken):
        observation = " | ".join(
            f"{e.title}: {e.text}" if e.title else e.text
            for e in state.evidence
            if e.step_index == i
        )
        action_input = getattr(action, "query", None) or getattr(action, "text", "")
        steps.append(
            TrajectoryStep(
                action_type=action.action_type,
                action_input=str(action_input) if action_input else action.describe(),
                observation=observation,
            )
        )
    return Trajectory(
        question=state.question,
        steps=steps,
        final_answer=state.final_answer or "",
        image_path=state.image_path,
    )


# --------------------------------------------------------------------------- #
# Searcher
# --------------------------------------------------------------------------- #
class MCTSSearcher:
    """Retrieval-action MCTS guided by a pluggable PRM scorer."""

    def __init__(
        self,
        executor: ActionExecutor,
        proposer: ActionProposer,
        scorer: TrajectoryScorer,
        config: SearchConfig | None = None,
        tracer=None,
    ) -> None:
        self.executor = executor
        self.proposer = proposer
        self.scorer = scorer
        self.cfg = config or SearchConfig()
        # Optional callable(dict) -> None for step-by-step inspection of the
        # search (selection UCT scores, expansion priors, simulation, backup).
        # Default None = zero overhead, so normal runs/tests are unaffected.
        self.tracer = tracer

    def _emit(self, **event) -> None:
        if self.tracer is not None:
            self.tracer(event)

    # ------------------------------------------------------------------ #
    def search(self, question: str, image_path: str | None = None) -> SearchResult:
        """Run P rollouts for one query; return the best-scored answer."""
        cfg = self.cfg
        root = MCTSNode(SearchState(question=question, image_path=image_path))

        best_state: SearchState | None = None
        best_reward = float("-inf")
        log: list[RolloutRecord] = []

        for t in range(cfg.rollouts):
            self._emit(event="rollout_start", t=t)
            terminal_state, reward = self._rollout(root)
            log.append(
                RolloutRecord(
                    t=t, reward=reward,
                    answer=terminal_state.final_answer,
                    depth=terminal_state.depth,
                    state=terminal_state,
                )
            )
            if reward > best_reward:
                best_reward, best_state = reward, terminal_state
            if reward >= cfg.early_stop_q:  # tau_stop early stop
                break

        assert best_state is not None  # rollouts >= 1
        return SearchResult(
            answer=best_state.final_answer or "",
            best_reward=best_reward,
            best_state=best_state,
            rollouts_run=len(log),
            rollout_log=log,
            root=root,
        )

    # ------------------------------------------------------------------ #
    # One rollout = selection -> expansion -> simulation -> backup
    # ------------------------------------------------------------------ #
    def _rollout(self, root: MCTSNode) -> tuple[SearchState, float]:
        node = self._select(root)
        if not node.is_terminal and node.depth < self.cfg.max_depth:
            self._expand(node)
            if node.children:
                node = self._best_child(node)
        terminal_state, reward = self._simulate(node.state)
        self._backup(node, reward)
        return terminal_state, reward

    def _select(self, root: MCTSNode) -> MCTSNode:
        node = root
        while node.is_expanded and not node.is_terminal:
            node = self._best_child(node)
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        scored = []
        for ch in node.children:
            u = uct_score(
                q_value=ch.q_value, visits=ch.visits,
                parent_visits=node.visits, c=self.cfg.c_uct,
            )
            scored.append((u, ch))
        best = max(scored, key=lambda x: x[0])[1]
        self._emit(
            event="select",
            candidates=[
                {"action": ch.action.describe(), "uct": u, "q": ch.q_value,
                 "visits": ch.visits, "prior": ch.prior}
                for u, ch in scored
            ],
            chosen=best.action.describe(),
        )
        return best

    def _expand(self, node: MCTSNode) -> None:
        """Propose candidates, execute them, PRM-rank, keep top-k children."""
        # Ask for a couple extra so top-k still has choice after dedup.
        candidates = self.proposer.propose(node.state, k=self.cfg.top_k_children + 2)
        seen = {ch.action for ch in node.children}
        scored: list[MCTSNode] = []
        for action in candidates:
            if action in seen:
                continue
            seen.add(action)
            child_state = self.executor.execute(node.state, action)
            prior = float(self.scorer.score(state_to_trajectory(child_state)))
            scored.append(MCTSNode(child_state, parent=node, action=action, prior=prior))
        scored.sort(key=lambda ch: ch.prior, reverse=True)
        kept = scored[: self.cfg.top_k_children]
        for child in kept:
            node.add_child(child)
        self._emit(
            event="expand",
            proposed=[
                {"action": ch.action.describe(), "prior": ch.prior,
                 "n_evidence": len(ch.state.evidence)}
                for ch in scored
            ],
            kept=[ch.action.describe() for ch in kept],
        )

    def _simulate(self, state: SearchState) -> tuple[SearchState, float]:
        """Greedy continuation to a terminal state, then PRM-score the
        trajectory. Forces an answer at max_depth so every rollout is scorable."""
        rolled = []
        while not state.is_terminal and state.depth < self.cfg.max_depth:
            candidates = self.proposer.propose(state, k=1)
            if not candidates:
                break
            state = self.executor.execute(state, candidates[0])
            rolled.append(candidates[0].describe())
        if not state.is_terminal:
            answer = self.proposer.propose_answer(state)
            state = self.executor.execute(state, answer)
            rolled.append(answer.describe())
        reward = float(self.scorer.score(state_to_trajectory(state)))
        self._emit(event="simulate", rollout_actions=rolled,
                   answer=state.final_answer, reward=reward)
        return state, reward

    def _backup(self, node: MCTSNode, reward: float) -> None:
        path = []
        current: MCTSNode | None = node
        while current is not None:
            current.update(reward)
            path.append(
                (current.action.describe() if current.action else "<root>",
                 current.visits, current.q_value)
            )
            current = current.parent
        self._emit(event="backup", reward=reward, path=path)
