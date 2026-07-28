"""Tests for the typed action space: text_search / image_search / answer.

Both search actions query ONE unified CLIP index (corpus text chunks and
corpus images are independent units). Fake encoders (red/blue axis) keep the
tests deterministic and download-free.
"""

import numpy as np
import pytest

from evidencetree.actions import (
    ACTION_REGISTRY,
    ActionExecutor,
    AnswerAction,
    BM25Retriever,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
    UnifiedClipRetriever,
)
from evidencetree.eval.benchmarks import Document


def _fake_text_encoder(texts):
    return np.array(
        [[1.0, 0.0] if "red" in str(t).lower() else [0.0, 1.0] for t in texts],
        dtype="float32",
    )


def _fake_image_encoder(images):
    out = []
    for im in images:
        r, _, b = im.getpixel((0, 0))
        out.append([1.0, 0.0] if r > b else [0.0, 1.0])
    return np.array(out, dtype="float32")


def _unified(corpus):
    return UnifiedClipRetriever(
        text_encoder=_fake_text_encoder, image_encoder=_fake_image_encoder
    ).build(corpus)


@pytest.fixture
def corpus(tmp_path):
    from PIL import Image

    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    return [
        Document(doc_id="img_red", text="", title="red pic", image_path=str(red)),
        Document(doc_id="img_blue", text="", title="blue pic", image_path=str(blue)),
        Document(doc_id="txt_red", text="a red rose is a red flower", title="red text"),
        Document(doc_id="txt_blue", text="the blue sea is blue water", title="blue text"),
    ]


@pytest.fixture
def retriever(corpus):
    return _unified(corpus)


# --------------------------------------------------------------------------- #
# Action registry: only the three typed actions exist
# --------------------------------------------------------------------------- #
def test_only_three_actions_registered():
    # Other tests may dynamically register extra types into the global
    # registry, so check membership rather than exact equality.
    assert {"text_search", "image_search", "answer"} <= set(ACTION_REGISTRY)
    assert "text_to_image" not in ACTION_REGISTRY
    assert "image_to_text" not in ACTION_REGISTRY
    assert "ocr" not in ACTION_REGISTRY


# --------------------------------------------------------------------------- #
# Both actions query the same unified index
# --------------------------------------------------------------------------- #
def test_image_search_queries_the_unified_index(corpus, retriever):
    """A red query image retrieves red units, image- or text-side."""
    hits = retriever.search_image(corpus[0].image_path, top_k=2)
    assert hits and all(h.doc_id in {"img_red", "txt_red"} for h in hits)


def test_text_search_queries_the_same_index(retriever):
    hits = retriever.search("a red thing", top_k=2)
    assert hits and all(h.doc_id in {"img_red", "txt_red"} for h in hits)


def test_region_crop_flips_the_result(tmp_path):
    from PIL import Image

    # query image: left half red, right half blue
    query = Image.new("RGB", (8, 8), (0, 0, 255))
    for x in range(4):
        for y in range(8):
            query.putpixel((x, y), (255, 0, 0))
    query_path = tmp_path / "half.jpg"
    query.save(query_path)

    Image.new("RGB", (8, 8), (255, 0, 0)).save(tmp_path / "r.png")
    Image.new("RGB", (8, 8), (0, 0, 255)).save(tmp_path / "b.png")
    retriever = _unified([
        Document(doc_id="img_red", text="", title="red pic",
                 image_path=str(tmp_path / "r.png")),
        Document(doc_id="img_blue", text="", title="blue pic",
                 image_path=str(tmp_path / "b.png")),
    ])

    # whole image (top-left pixel is red) -> red; right-half region -> blue
    assert retriever.search_image(
        str(query_path), region=None, top_k=1
    )[0].doc_id == "img_red"
    assert retriever.search_image(
        str(query_path), region=(0.5, 0, 1, 1), top_k=1
    )[0].doc_id == "img_blue"


# --------------------------------------------------------------------------- #
# Executor: three handlers, correct evidence tagging
# --------------------------------------------------------------------------- #
def test_executor_runs_three_actions(corpus, retriever):
    text_ret = BM25Retriever().build(corpus)
    ex = ActionExecutor(text_retriever=text_ret, image_retriever=retriever, top_k=2)

    s0 = SearchState(question="what is this red rose?", image_path=corpus[0].image_path)
    s1 = ex.execute(s0, TextSearchAction(query="red rose flower"))
    assert s1.evidence[0].source_action == "text_search"

    s2 = ex.execute(s1, ImageSearchAction())
    assert any(e.source_action == "image_search" for e in s2.evidence)

    s3 = ex.execute(s2, AnswerAction(text="a rose"))
    assert s3.is_terminal and s3.final_answer == "a rose"


def test_image_search_without_retriever_raises():
    ex = ActionExecutor(text_retriever=None, image_retriever=None)
    with pytest.raises(RuntimeError):
        ex.execute(SearchState(question="q", image_path="x.jpg"), ImageSearchAction())
