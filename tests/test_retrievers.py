"""Tests for BM25Retriever and DenseRetriever (fake encoder — no downloads)."""

import re
import zlib

import pytest

from evidencetree.actions.retrievers import (
    BM25Retriever,
    ClipImageRetriever,
    CragImageRetriever,
    CragWebRetriever,
    DenseRetriever,
)
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


# --------------------------------------------------------------------------- #
# DenseRetriever (deterministic fake encoder; faiss only, no model download)
# --------------------------------------------------------------------------- #
def _hash_encoder(items):
    """Deterministic bag-of-hashed-tokens embedding (64-dim)."""
    import numpy as np

    vecs = np.zeros((len(items), 64), dtype="float32")
    for i, text in enumerate(items):
        for tok in re.findall(r"[a-z0-9]+", str(text).lower()):
            vecs[i, zlib.crc32(tok.encode()) % 64] += 1.0
    return vecs


def test_dense_retriever_finds_relevant_doc():
    retriever = DenseRetriever(encoder=_hash_encoder).build(_corpus())
    hits = retriever.search("Eiffel Tower Paris", top_k=2)
    assert hits[0].doc_id == "d0"
    assert hits[0].score >= hits[1].score


def test_dense_retriever_save_load_roundtrip(tmp_path):
    retriever = DenseRetriever(encoder=_hash_encoder).build(_corpus())
    retriever.save(tmp_path / "index")

    loaded = DenseRetriever.load(tmp_path / "index", encoder=_hash_encoder)
    assert len(loaded) == len(retriever)
    assert loaded.search("mountain Japan Fuji", top_k=1)[0].doc_id == "d1"


def test_dense_retriever_search_before_build_raises():
    with pytest.raises(RuntimeError):
        DenseRetriever(encoder=_hash_encoder).search("anything")


def test_dense_retriever_empty_corpus_raises():
    with pytest.raises(ValueError):
        DenseRetriever(encoder=_hash_encoder).build([])


def test_clip_image_retriever_finds_matching_image(tmp_path):
    import numpy as np
    from PIL import Image

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    Image.new("RGB", (8, 8), "red").save(red)
    Image.new("RGB", (8, 8), "blue").save(blue)

    def image_encoder(images):
        return np.asarray(
            [
                np.asarray(image, dtype="float32").mean(axis=(0, 1))
                for image in images
            ]
        )

    corpus = [
        Document(doc_id="red", text="red image", image_path=str(red)),
        Document(doc_id="blue", text="blue image", image_path=str(blue)),
    ]
    retriever = ClipImageRetriever(
        encoder=_hash_encoder, image_encoder=image_encoder
    ).build(corpus)

    assert retriever.search_image(str(red), top_k=1)[0].doc_id == "red"


class _FakeChromaCollection:
    def __init__(self, metadata):
        self.metadata = metadata
        self.last_query = None

    def query(self, **kwargs):
        self.last_query = kwargs
        return {
            "ids": [["item-1"]],
            "metadatas": [[self.metadata]],
            "distances": [[0.2]],
        }


def test_crag_web_retriever_formats_official_index_metadata():
    import numpy as np

    collection = _FakeChromaCollection(
        {
            "page_name": "Example page",
            "page_snippet": "The indexed fact.",
            "page_url": "https://example.test/page",
        }
    )
    retriever = CragWebRetriever(
        index_path="unused",
        model_name="unused",
        encoder=lambda texts: np.ones((len(texts), 4), dtype="float32"),
        collection=collection,
    )
    hit = retriever.search("indexed fact", top_k=1)[0]

    assert hit.doc_id == "crag-web:item-1"
    assert hit.title == "Example page"
    assert "The indexed fact." in hit.text
    assert hit.score == pytest.approx(0.8)


def test_crag_image_retriever_formats_entity_metadata(tmp_path):
    import numpy as np
    from PIL import Image

    query = tmp_path / "query.png"
    Image.new("RGB", (8, 8), "green").save(query)
    collection = _FakeChromaCollection(
        {
            "entities": '[{"name":"Eiffel Tower"}]',
            "info": '{"city":"Paris"}',
            "image_url": "https://example.test/image.jpg",
        }
    )
    retriever = CragImageRetriever(
        index_path="unused",
        model_name="unused",
        image_encoder=lambda images: np.ones((len(images), 4), dtype="float32"),
        collection=collection,
    )
    hit = retriever.search_image(str(query), top_k=1)[0]

    assert hit.doc_id == "crag-image:item-1"
    assert "Eiffel Tower" in hit.text
    assert "Paris" in hit.text
    assert hit.image_path is None
