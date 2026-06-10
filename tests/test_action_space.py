"""Tests for the typed action space and the executor (Stage 1)."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from evidencetree.actions import (
    ACTION_REGISTRY,
    Action,
    ActionExecutor,
    AnswerAction,
    BM25Retriever,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
    register_action,
)
from evidencetree.eval.benchmarks import Document


@pytest.fixture
def executor():
    corpus = [
        Document(doc_id="d0", title="Eiffel Tower",
                 text="The Eiffel Tower is a lattice tower in Paris, France."),
        Document(doc_id="d1", title="Mount Fuji",
                 text="Mount Fuji is the tallest mountain in Japan."),
        Document(doc_id="d2", title="Noise",
                 text="Photosynthesis converts light into chemical energy."),
    ]
    return ActionExecutor(text_retriever=BM25Retriever().build(corpus), top_k=2)


def test_text_search_accumulates_evidence(executor):
    s0 = SearchState(question="In which city is the Eiffel Tower located?")
    s1 = executor.execute(s0, TextSearchAction(query="Eiffel Tower"))
    s2 = executor.execute(s1, TextSearchAction(query="Mount Fuji"))

    assert len(s1.evidence) == 2 and len(s2.evidence) == 4  # bundle grows
    assert s0.evidence == ()                                # immutability
    assert [e.step_index for e in s2.evidence] == [0, 0, 1, 1]
    assert [e.evidence_id for e in s2.evidence] == ["e0", "e1", "e2", "e3"]
    assert all(e.source_action == "text_search" for e in s2.evidence)
    assert s2.evidence[0].doc_id == "d0"
    assert s2.depth == 2 and not s2.is_terminal


def test_answer_action_is_terminal(executor):
    s0 = SearchState(question="q")
    s1 = executor.execute(s0, AnswerAction(text="Paris"))
    assert s1.is_terminal and s1.final_answer == "Paris"
    with pytest.raises(ValueError):
        executor.execute(s1, TextSearchAction(query="more"))


def test_image_search_without_retriever_raises(executor):
    state = SearchState(question="q", image_path="img.jpg")
    with pytest.raises(RuntimeError):
        executor.execute(state, ImageSearchAction())


def test_image_search_without_any_image_raises():
    class FakeImageRetriever:
        def search_image(self, image_path, region=None, top_k=5):
            return []

    ex = ActionExecutor(image_retriever=FakeImageRetriever())
    with pytest.raises(ValueError):
        ex.execute(SearchState(question="q"), ImageSearchAction())


def test_action_granularity_and_registry():
    assert TextSearchAction(query="x").granularity == "n/a"
    assert ImageSearchAction().granularity == "whole"
    assert ImageSearchAction(region=(0.1, 0.1, 0.5, 0.5)).granularity == "region"
    assert set(ACTION_REGISTRY) >= {"text_search", "image_search", "answer"}


def test_new_action_type_plugs_in_without_rewrite(executor):
    """Acceptance: adding crop/zoom later must not require refactoring."""

    @register_action
    @dataclass(frozen=True)
    class CropAction(Action):
        region: tuple = (0.0, 0.0, 1.0, 1.0)
        action_type: ClassVar[str] = "crop"
        modality: ClassVar[str] = "image"

        @property
        def granularity(self) -> str:
            return "region"

    def handle_crop(state, action):
        return state.advanced(action, [])

    executor.register_handler(CropAction, handle_crop)
    state = executor.execute(SearchState(question="q"), CropAction())
    assert state.depth == 1 and state.actions_taken[0].action_type == "crop"
    assert "crop" in ACTION_REGISTRY


def test_unknown_action_type_raises(executor):
    @dataclass(frozen=True)
    class MysteryAction(Action):
        action_type: ClassVar[str] = "mystery"

    with pytest.raises(TypeError):
        executor.execute(SearchState(question="q"), MysteryAction())
