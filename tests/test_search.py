"""End-to-end tests for the MCTS searcher on mock data (no GPU, no downloads)."""

import pytest

from evidencetree.actions import ActionExecutor, BM25Retriever
from evidencetree.eval.benchmarks import load_infoseek
from evidencetree.generation import build_generator
from evidencetree.mcts import (
    HeuristicProposer,
    MCTSSearcher,
    SearchConfig,
    state_to_trajectory,
)
from evidencetree.prm.model import HeuristicOverlapScorer


def make_searcher(config: SearchConfig | None = None, scorer=None, accuracy=1.0):
    queries, corpus = load_infoseek(n=10, mock=True)
    retriever = BM25Retriever().build(corpus)
    executor = ActionExecutor(text_retriever=retriever, top_k=3)
    generator = build_generator(
        {"backend": "mock", "mock_accuracy": accuracy, "mock_seed": 0}
    )
    gold_by_question = {q.question: q.gold_answers for q in queries}
    proposer = HeuristicProposer(
        generator,
        answer_kwargs_fn=lambda s: {"reference": gold_by_question.get(s.question)},
    )
    searcher = MCTSSearcher(
        executor=executor,
        proposer=proposer,
        scorer=scorer or HeuristicOverlapScorer(),
        config=config or SearchConfig(),
    )
    return queries, searcher


def test_search_returns_evidence_backed_answer():
    queries, searcher = make_searcher()
    q = queries[0]
    result = searcher.search(q.question)
    assert result.answer  # produces an answer
    assert result.best_state.is_terminal
    assert result.best_state.evidence  # answer is backed by retrieved evidence
    assert result.answer in q.gold_answers  # gold-hinted generator + overlap scorer
    assert result.best_reward > 0.5


def test_tree_statistics_consistent():
    cfg = SearchConfig(rollouts=6, early_stop_q=2.0)  # disable early stop
    queries, searcher = make_searcher(cfg)
    result = searcher.search(queries[1].question)
    assert result.rollouts_run == 6
    assert result.root.visits == 6  # one backup per rollout reaches the root
    assert len(result.root.children) <= cfg.top_k_children
    assert all(ch.visits <= result.root.visits for ch in result.root.children)


def test_early_stop_triggers():
    class PerfectScorer:
        def score(self, trajectory, **kwargs):
            return 1.0

    cfg = SearchConfig(rollouts=10, early_stop_q=0.9)
    queries, searcher = make_searcher(cfg, scorer=PerfectScorer())
    result = searcher.search(queries[0].question)
    assert result.rollouts_run == 1  # tau_stop hit on the first rollout


def test_state_to_trajectory_roundtrip():
    queries, searcher = make_searcher()
    result = searcher.search(queries[2].question)
    traj = state_to_trajectory(result.best_state)
    assert traj.question == queries[2].question
    assert traj.steps[-1].action_type == "answer"
    assert traj.final_answer == result.answer
    search_steps = [s for s in traj.steps if s.action_type == "text_search"]
    assert search_steps and all(s.observation for s in search_steps)


def test_bandit_integration():
    cfg = SearchConfig(rollouts=8, early_stop_q=2.0, adaptive_lambda=True, seed=0)
    queries, searcher = make_searcher(cfg)
    result = searcher.search(queries[3].question)
    assert result.bandit is not None
    lams = [r.lam for r in result.rollout_log]
    assert lams[:3] == [0.1, 0.5, 1.0]  # warm-up round-robin
    assert all(l in cfg.bandit_arms for l in lams)
    assert len(result.bandit.history) == result.rollouts_run
    assert result.bandit.converged_lambda() in cfg.bandit_arms


def test_bandit_state_fresh_per_query():
    cfg = SearchConfig(rollouts=4, early_stop_q=2.0, adaptive_lambda=True, seed=0)
    queries, searcher = make_searcher(cfg)
    r1 = searcher.search(queries[0].question)
    r2 = searcher.search(queries[1].question)
    assert r1.bandit is not r2.bandit
    assert len(r2.bandit.history) == r2.rollouts_run  # not carried over


def test_fixed_lambda_runs_without_bandit():
    cfg = SearchConfig(rollouts=4, adaptive_lambda=False, lam=0.3, early_stop_q=2.0)
    queries, searcher = make_searcher(cfg)
    result = searcher.search(queries[4].question)
    assert result.bandit is None
    assert all(r.lam == pytest.approx(0.3) for r in result.rollout_log)


def test_search_config_from_dict():
    cfg = SearchConfig.from_dict(
        {
            "search": {"rollouts": 5, "max_depth": 2, "lam": 0.7, "seed": 3},
            "uct": {"w_mod": 2.0, "w_gran": 0.25},
            "bandit": {"enabled": True, "arms": [0.1, 0.5], "warmup": 2},
        }
    )
    assert cfg.rollouts == 5 and cfg.max_depth == 2
    assert cfg.lam == pytest.approx(0.7)
    assert cfg.w_mod == pytest.approx(2.0) and cfg.w_gran == pytest.approx(0.25)
    assert cfg.adaptive_lambda is True
    assert cfg.bandit_arms == (0.1, 0.5) and cfg.bandit_warmup == 2
