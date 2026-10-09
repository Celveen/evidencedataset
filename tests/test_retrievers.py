"""Tests for BM25Retriever (the lexical retrieval backend)."""

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
    assert hits[0].doc_id == "d0"            # most relevant first
    assert hits[0].score >= hits[1].score    # descending score
    assert all(h.result_modality == "text" for h in hits)


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


def _bigger_corpus():
    queries, corpus = load_infoseek(mock=True)
    return queries, corpus + [
        Document(doc_id="img_only", title="Eiffel Tower", text="", image_path="x.png"),
    ]


def test_sparse_bm25_matches_rank_bm25():
    """The sparse implementation reproduces rank_bm25 scores and ranking."""
    from rank_bm25 import BM25Okapi

    from evidencetree.actions.retrievers import _tokenize

    queries, corpus = _bigger_corpus()
    retriever = BM25Retriever().build(corpus)
    docs = [d for d in corpus if d.text.strip()]
    reference = BM25Okapi([_tokenize(f"{d.title} {d.text}") for d in docs])
    for q in queries[:25] + [Document("x", "the the Eiffel Eiffel")]:
        text = q.question if hasattr(q, "question") else q.text
        ref_scores = reference.get_scores(_tokenize(text))
        ref_order = sorted(range(len(docs)), key=lambda i: ref_scores[i], reverse=True)[:5]
        hits = retriever.search(text, top_k=5)
        assert [h.doc_id for h in hits] == [docs[i].doc_id for i in ref_order]
        for h, i in zip(hits, ref_order):
            assert h.score == pytest.approx(ref_scores[i], rel=1e-4, abs=1e-5)


def test_bm25_skips_image_only_docs_and_dedupes_entities():
    docs = [
        Document(doc_id="e1_p0", title="Eiffel Tower", text="Eiffel Tower is in Paris.", entity_id="E1"),
        Document(doc_id="e1_p1", title="Eiffel Tower", text="The Eiffel Tower opened in 1889 in Paris.", entity_id="E1"),
        Document(doc_id="e1_img", title="Eiffel Tower", text="", image_path="e.jpg", entity_id="E1"),
        Document(doc_id="e2_p0", title="Louvre", text="The Louvre is a museum in Paris.", entity_id="E2"),
    ] + [  # unrelated pages so idf is positive (BM25 floors it on tiny corpora)
        Document(doc_id=f"x{i}", title=f"Topic {i}", text=f"Unrelated filler text number {i}.")
        for i in range(8)
    ]
    plain = BM25Retriever().build(docs)
    assert len(plain) == 11
    assert "e1_img" not in {h.doc_id for h in plain.search("Eiffel Tower Paris", top_k=5)}
    deduped = BM25Retriever(dedupe_key="entity").build(docs)
    hits = deduped.search("Eiffel Tower Paris", top_k=3)
    assert [h.doc_id for h in hits][:1] in (["e1_p0"], ["e1_p1"])
    assert len({h.doc_id.split("_")[0] for h in hits}) == len(hits)


def test_bm25_save_load_roundtrip(tmp_path):
    _, corpus = _bigger_corpus()
    built = BM25Retriever().build(corpus)
    built.save(tmp_path / "bm25", "fp1")
    loaded = BM25Retriever.load(tmp_path / "bm25", corpus, "fp1")
    assert loaded is not None
    q = "In which city is the Eiffel Tower located?"
    assert [h.doc_id for h in loaded.search(q)] == [h.doc_id for h in built.search(q)]
    assert BM25Retriever.load(tmp_path / "bm25", corpus, "other") is None
