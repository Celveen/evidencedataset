"""Main MCTS loop: selection -> expansion -> simulation -> backup.

Stage 5 (fixed lambda) and Stage 6 (bandit-adaptive lambda) in one searcher —
``SearchConfig.adaptive_lambda`` flips between them. The PRM is a pluggable
``TrajectoryScorer`` (today: mock / frozen VisualPRM; later: EvidenceTree-PRM
in score-only mode), so the framework runs before any PRM is trained.

Defaults follow the implementation report: P=10 rollouts, max_depth=3,
expansion keeps the top-k PRM-ranked candidates, early stop when a rollout's
reward exceeds ``early_stop_q`` (tau_stop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evidencetree.actions.action_space import SearchState
from evidencetree.actions.executor import ActionExecutor
from evidencetree.mcts.bandit import ThompsonBandit
from evidencetree.mcts.node import MCTSNode
from evidencetree.mcts.proposer import ActionProposer
from evidencetree.mcts.uct import modality_novelty, uct_score
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
    lam: float = 0.3              # fixed lambda (ignored when adaptive_lambda)
    w_mod: float = 1.0
    w_gran: float = 0.5
    early_stop_q: float = 0.9     # tau_stop
    adaptive_lambda: bool = False  # Stage 6 bandit on/off
    bandit_arms: tuple[float, ...] = ThompsonBandit.DEFAULT_ARMS
    bandit_warmup: int = 3
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "SearchConfig":
        """Build from a configs/mcts.yaml-style nested dict."""
        d = dict(d or {})
        search = dict(d.get("search", {}))
        uct = dict(d.get("uct", {}))
        bandit = dict(d.get("bandit", {}))
        kwargs: dict[str, Any] = {}
        for key in ("rollouts", "max_depth", "top_k_children", "c_uct", "lam",
                    "early_stop_q", "seed"):
            if key in search:
                kwargs[key] = search[key]
        for key in ("w_mod", "w_gran"):
            if key in uct:
                kwargs[key] = uct[key]
        if "enabled" in bandit:
            kwargs["adaptive_lambda"] = bool(bandit["enabled"])
        if "arms" in bandit:
            kwargs["bandit_arms"] = tuple(float(a) for a in bandit["arms"])
        if "warmup" in bandit:
            kwargs["bandit_warmup"] = int(bandit["warmup"])
        return cls(**kwargs)


@dataclass
class RolloutRecord:
    t: int
    lam: float
    reward: float
    answer: str | None
    depth: int


@dataclass
class SearchResult:
    answer: str
    best_reward: float
    best_state: SearchState
    rollouts_run: int
    rollout_log: list[RolloutRecord] = field(default_factory=list)
    root: MCTSNode | None = None
    bandit: ThompsonBandit | None = None


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
    ) -> None:
        self.executor = executor
        self.proposer = proposer
        self.scorer = scorer
        self.cfg = config or SearchConfig()

    # ------------------------------------------------------------------ #
    def search(self, question: str, image_path: str | None = None) -> SearchResult:
        """Run P rollouts for one query; return the best-scored answer.

        A fresh bandit is created per call — per-query independence is
        structural, not a convention to remember.
        """
        cfg = self.cfg
        root = MCTSNode(SearchState(question=question, image_path=image_path))
        bandit = (
            ThompsonBandit(
                arms=cfg.bandit_arms, warmup=cfg.bandit_warmup, seed=cfg.seed
            )
            if cfg.adaptive_lambda
            else None
        )

        best_state: SearchState | None = None
        best_reward = float("-inf")
        log: list[RolloutRecord] = []

        for t in range(cfg.rollouts):
            lam = bandit.select() if bandit else cfg.lam
            terminal_state, reward = self._rollout(root, lam)
            if bandit:
                bandit.update(reward)
            log.append(
                RolloutRecord(
                    t=t, lam=lam, reward=reward,
                    answer=terminal_state.final_answer,
                    depth=terminal_state.depth,
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
            bandit=bandit,
        )

    # ------------------------------------------------------------------ #
    # One rollout = selection -> expansion -> simulation -> backup
    # ------------------------------------------------------------------ #
    def _rollout(self, root: MCTSNode, lam: float) -> tuple[SearchState, float]:
        node = self._select(root, lam)
        if not node.is_terminal and node.depth < self.cfg.max_depth:
            self._expand(node)
            if node.children:
                node = self._best_child(node, lam)
        terminal_state, reward = self._simulate(node.state)
        self._backup(node, reward)
        return terminal_state, reward

    def _select(self, root: MCTSNode, lam: float) -> MCTSNode:
        node = root
        while node.is_expanded and not node.is_terminal:
            node = self._best_child(node, lam)
        return node

    def _best_child(self, node: MCTSNode, lam: float) -> MCTSNode:
        path = node.path_actions()
        return max(
            node.children,
            key=lambda ch: uct_score(
                q_value=ch.q_value,
                visits=ch.visits,
                parent_visits=node.visits,
                novelty=modality_novelty(
                    ch.action, path, w_mod=self.cfg.w_mod, w_gran=self.cfg.w_gran
                ),
                c=self.cfg.c_uct,
                lam=lam,
            ),
        )

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
        for child in scored[: self.cfg.top_k_children]:
            node.add_child(child)

    def _simulate(self, state: SearchState) -> tuple[SearchState, float]:
        """Greedy continuation to a terminal state, then PRM-score the
        trajectory. Forces an answer at max_depth so every rollout is scorable."""
        while not state.is_terminal and state.depth < self.cfg.max_depth:
            candidates = self.proposer.propose(state, k=1)
            if not candidates:
                break
            state = self.executor.execute(state, candidates[0])
        if not state.is_terminal:
            state = self.executor.execute(state, self.proposer.propose_answer(state))
        reward = float(self.scorer.score(state_to_trajectory(state)))
        return state, reward

    @staticmethod
    def _backup(node: MCTSNode, reward: float) -> None:
        current: MCTSNode | None = node
        while current is not None:
            current.update(reward)
            current = current.parent
