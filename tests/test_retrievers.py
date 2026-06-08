"""Tests for BM25Retriever."""

import pytest

from evidencetree.actions.retrievers import BM25Retriever
from evidencetree.eval.benchmarks import Document, load_infoseek


def _corpus():
    return [
        Document(doc_id="d0", title="Eiffel Tower",
                 text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
        Document(doc_id="d1", title="Mount Fuji",
                 text="Mount Fuji is the tallest mountain in Japan."),
        Document(doc_id="d2", title="Photosynthesis",
                 text="Photosynthesis converts light energy into chemical energy in plants."),
    ]


def test_build_then_search_returns_relevant_doc():
    retriever = BM25Retriever().build(_corpus())
    hits = retriever.search("Where is the Eiffel Tower located?", top_k=2)
    assert len(hits) == 2
    assert hits[0].doc_id == "d0"          # most relevant first
    assert hits[0].score >= hits[1].score   # descending score


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("anything")


def test_build_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25Retriever().build([])


def test_bm25_retrieves_supporting_doc_for_mock_queries():
    """On the mock dataset, BM25 should surface each query's supporting doc."""
    queries, corpus = load_infoseek(mock=True)
    retriever = BM25Retriever().build(corpus)
    hit_rate = 0
    for q in queries:
        hits = retriever.search(q.question, top_k=3)
        support = q.metadata["supporting_doc"]
        if any(h.doc_id == support for h in hits):
            hit_rate += 1
    # The supporting doc shares salient tokens (entity name) -> high recall@3.
    assert hit_rate / len(queries) >= 0.8
