"""End-to-end tests for the MCTS searcher on mock data (no GPU, no downloads)."""

import pytest

from evidencetree.actions import (
    ActionExecutor,
    AnswerAction,
    BM25Retriever,
    Evidence,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
)
from evidencetree.eval.benchmarks import load_infoseek
from evidencetree.generation import build_generator
from evidencetree.mcts import (
    HeuristicProposer,
    LLMProposer,
    MCTSSearcher,
    SearchConfig,
    state_to_trajectory,
)
from evidencetree.mcts.proposer import GatedProposer
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





def test_search_config_from_dict():
    cfg = SearchConfig.from_dict(
        {
            "search": {"rollouts": 5, "max_depth": 2, "seed": 3, "c_uct": 2.0},
        }
    )
    assert cfg.rollouts == 5 and cfg.max_depth == 2 and cfg.seed == 3
    assert cfg.c_uct == pytest.approx(2.0)


def test_llm_proposer_rejects_answer_before_retrieval():
    class DirectAnswerGenerator:
        def generate(self, *args, **kwargs):
            return '{"type": "answer", "text": "150 grams"}'

    proposer = LLMProposer(DirectAnswerGenerator())
    actions = proposer.propose(SearchState(question="How heavy is the bird?"), k=3)

    assert actions
    assert any(isinstance(action, TextSearchAction) for action in actions)
    assert not any(isinstance(action, AnswerAction) for action in actions)


def test_llm_proposer_respects_disabled_image_search():
    class ImageSearchGenerator:
        def generate(self, *args, **kwargs):
            return "\n".join(
                [
                    '{"type": "image_search"}',
                    '{"type": "text_search", "query": "bird weight"}',
                ]
            )

    proposer = LLMProposer(ImageSearchGenerator(), enable_image_search=False)
    actions = proposer.propose(
        SearchState(question="How heavy is the bird?", image_path="/tmp/bird.jpg"),
        k=3,
    )

    assert actions == [TextSearchAction(query="bird weight")]


def test_llm_proposer_uses_github_query_guard_without_hard_freezing():
    class RecordingGenerator:
        def __init__(self):
            self.prompt = ""

        def generate(self, prompt, *args, **kwargs):
            self.prompt = prompt
            return '{"type": "text_search", "query": "bird average weight"}'

    generator = RecordingGenerator()
    proposer = LLMProposer(generator, freeze_action_queries=False)
    actions = proposer.propose(
        SearchState(question="How heavy is this bird?", image_path="/tmp/bird.jpg"),
        k=3,
    )

    assert actions[0] == TextSearchAction(query="bird average weight")
    assert "Do NOT invent or assume a specific entity identity" in generator.prompt
    assert "Keep every search query faithful" in generator.prompt


def test_llm_proposer_can_freeze_text_queries_to_original_question():
    class RewritingGenerator:
        def generate(self, *args, **kwargs):
            return '{"type": "text_search", "query": "mourning dove weight"}'

    state = SearchState(question="How heavy is this bird?", image_path="/tmp/bird.jpg")
    proposer = LLMProposer(RewritingGenerator(), freeze_action_queries=True)

    actions = proposer.propose(state, k=3)

    assert actions[0] == TextSearchAction(query="How heavy is this bird?")


def test_llm_proposer_allows_exact_existing_evidence_query_when_frozen():
    class EvidenceCopyGenerator:
        def generate(self, *args, **kwargs):
            return '{"type": "text_search", "query": "White-winged dove"}'

    state = SearchState(question="How heavy is this bird?")
    state = state.advanced(
        TextSearchAction(query=state.question),
        [
            Evidence(
                evidence_id="e0",
                step_index=0,
                source_action="text_search",
                doc_id="doc",
                title="White-winged dove",
                text="White-winged doves weigh 150 g.",
                image_path=None,
                score=1.0,
            )
        ],
    )
    proposer = LLMProposer(EvidenceCopyGenerator(), freeze_action_queries=True)

    actions = proposer.propose(state, k=3)

    assert actions[0] == TextSearchAction(query="White-winged dove")


def test_llm_proposer_adds_visual_branch_and_passes_retrieved_images():
    class RecordingGenerator:
        def __init__(self):
            self.kwargs = []

        def generate(self, *args, **kwargs):
            self.kwargs.append(kwargs)
            return '{"type": "text_search", "query": "bird weight"}'

    generator = RecordingGenerator()
    proposer = LLMProposer(generator, enable_image_search=True)
    state = SearchState(question="How heavy is the bird?", image_path="/tmp/query.jpg")

    actions = proposer.propose(state, k=3)

    assert actions == [
        TextSearchAction(query="bird weight"),
        ImageSearchAction(),
    ]
    assert generator.kwargs[0]["image_paths"] == ["/tmp/query.jpg"]


def test_gated_proposer_filters_actions_by_runtime_availability():
    class FullSpaceProposer:
        def propose(self, state, k=4):
            return [
                TextSearchAction(query="text"),
                ImageSearchAction(),
                AnswerAction(text="answer"),
            ]

        def propose_answer(self, state):
            return AnswerAction(text="answer")

    proposer = GatedProposer(
        FullSpaceProposer(),
        benchmark="infoseek",
        mode="dataset",
        available_actions={"text_search", "answer"},
    )
    actions = proposer.propose(SearchState(question="what is this?", image_path="q.jpg"), k=5)

    assert [action.action_type for action in actions] == [
        "text_search",
        "answer",
    ]
