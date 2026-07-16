"""UnifiedClipRetriever: ONE CLIP space, unit-level mixed-modality retrieval.

Deterministic injected encoders (no downloads): a red/blue axis shared by the
text and image towers, so cross-modal similarity is real and predictable.
"""

import numpy as np
import pytest

from evidencetree.actions import (
    ActionExecutor,
    SearchState,
    TextSearchAction,
    UnifiedClipRetriever,
)
from evidencetree.actions.action_space import ImageSearchAction
from evidencetree.eval.benchmarks import Document


def _txt_enc(texts):
    return np.array(
        [[1.0, 0.0] if "red" in str(t).lower() else [0.0, 1.0] for t in texts],
        dtype="float32",
    )


def _img_enc(images):
    out = []
    for im in images:
        r, _, b = im.getpixel((0, 0))
        out.append([1.0, 0.0] if r > b else [0.0, 1.0])
    return np.array(out, dtype="float32")


@pytest.fixture
def images(tmp_path):
    from PIL import Image

    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    return str(red), str(blue)


@pytest.fixture
def corpus(images):
    red, blue = images
    return [
        # One doc with text AND image -> TWO independent units.
        Document(doc_id="rose", title="red rose",
                 text="a red rose is a red flower", image_path=red),
        Document(doc_id="sea", title="blue sea",
                 text="the blue sea is blue water", image_path=blue),
        Document(doc_id="note", title="", text="the blue whale is huge"),
    ]


def _build(corpus, **kwargs):
    return UnifiedClipRetriever(
        text_encoder=_txt_enc, image_encoder=_img_enc, **kwargs
    ).build(corpus)


def test_one_doc_yields_independent_text_and_image_units(corpus):
    r = _build(corpus)
    # 3 text units (all docs have text) + 2 image units.
    assert len(r) == 5
    assert r.text_unit_count == 3
    assert r.image_unit_count == 2


def test_text_search_returns_mixed_modalities_from_one_index(corpus):
    r = _build(corpus)
    hits = r.search_text("red flower", top_k=3)
    modalities = {h.result_modality for h in hits}
    assert modalities == {"text", "image"}
    # Top hits are the red doc's units (text and image score equally on the
    # red axis); every hit carries its unit modality.
    assert {h.doc_id for h in hits[:2]} == {"rose"}
    text_hits = [h for h in hits if h.result_modality == "text"]
    image_hits = [h for h in hits if h.result_modality == "image"]
    assert all(h.image_path is None for h in text_hits)
    assert all(h.image_path and h.text == "" for h in image_hits)


def test_image_search_returns_mixed_modalities(corpus, images):
    red, _ = images
    r = _build(corpus)
    hits = r.search_image(red, top_k=3)
    assert {h.result_modality for h in hits} == {"text", "image"}
    assert {h.doc_id for h in hits[:2]} == {"rose"}


def test_image_unit_text_is_not_a_bound_document_text(corpus, images):
    """Image units must NOT carry the doc's text (P1: inflated grounding)."""
    red, _ = images
    r = _build(corpus)
    image_hits = [
        h for h in r.search_image(red, top_k=5) if h.result_modality == "image"
    ]
    assert image_hits and all(h.text == "" for h in image_hits)


def test_entity_dedupe_collapses_repeated_entity_before_top_k(images):
    red, blue = images
    # Same entity image attached to many chunks (P3).
    corpus = [
        Document(doc_id=f"e1_{i}", title="Red Tower",
                 text=f"red tower chunk {i}", image_path=red, entity_id="E1")
        for i in range(4)
    ] + [
        Document(doc_id="e2", title="Blue Lake",
                 text="the blue lake", image_path=blue, entity_id="E2"),
    ]
    no_dedupe = _build(corpus).search_image(red, top_k=3)
    assert all("e1" in h.doc_id for h in no_dedupe)  # top-k all one entity

    deduped = _build(corpus, dedupe_key="entity").search_image(red, top_k=3)
    entities = ["E1" if "e1" in h.doc_id else "E2" for h in deduped]
    assert entities.count("E1") == 1
    assert "E2" in entities


def test_doc_dedupe_key(corpus):
    r = _build(corpus, dedupe_key="doc")
    hits = r.search_text("red flower", top_k=4)
    doc_ids = [h.doc_id for h in hits]
    assert len(doc_ids) == len(set(doc_ids))  # one unit per doc survives


def test_dedupe_key_validation():
    with pytest.raises(ValueError):
        UnifiedClipRetriever(dedupe_key="banana")


def test_search_before_build_raises():
    r = UnifiedClipRetriever(text_encoder=_txt_enc, image_encoder=_img_enc)
    with pytest.raises(RuntimeError):
        r.search_text("anything")
    with pytest.raises(RuntimeError):
        r.search_image("nope.png")


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        _build([])


def test_numpy_fallback_matches_faiss(corpus, monkeypatch):
    r_faiss = _build(corpus)
    monkeypatch.setattr(UnifiedClipRetriever, "_try_faiss",
                        staticmethod(lambda vecs: None))
    r_numpy = _build(corpus)
    q = "red flower"
    faiss_hits = [(h.doc_id, h.result_modality) for h in r_faiss.search_text(q, top_k=5)]
    numpy_hits = [(h.doc_id, h.result_modality) for h in r_numpy.search_text(q, top_k=5)]
    assert set(faiss_hits[:2]) == set(numpy_hits[:2])
    assert {m for _, m in faiss_hits} == {m for _, m in numpy_hits}


def test_save_load_roundtrip(corpus, tmp_path, images):
    red, _ = images
    r = _build(corpus)
    r.save(tmp_path / "unified")
    loaded = UnifiedClipRetriever.load(
        tmp_path / "unified", text_encoder=_txt_enc, image_encoder=_img_enc
    )
    assert len(loaded) == len(r)
    assert loaded.n_docs == 3
    hits = loaded.search_image(red, top_k=2)
    assert {h.doc_id for h in hits} == {"rose"}


def test_executor_text_search_over_unified_space_tags_evidence(corpus):
    """text_search via the unified index yields modality-tagged Evidence."""
    r = _build(corpus)
    executor = ActionExecutor(text_retriever=r, image_retriever=r, top_k=3)
    state = SearchState(question="which flower is red?")
    state = executor.execute(state, TextSearchAction(query="red flower"))
    modalities = {e.result_modality for e in state.evidence}
    assert modalities == {"text", "image"}
    assert all(e.source_action == "text_search" for e in state.evidence)


def test_executor_image_search_over_unified_space(corpus, images):
    red, _ = images
    r = _build(corpus)
    executor = ActionExecutor(text_retriever=r, image_retriever=r, top_k=3)
    state = SearchState(question="what is this?", image_path=red)
    state = executor.execute(state, ImageSearchAction())
    assert {e.result_modality for e in state.evidence} == {"text", "image"}
