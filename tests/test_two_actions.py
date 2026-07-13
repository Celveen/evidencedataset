"""Tests for the compressed action space: text_search / image_search / answer."""

import numpy as np
import pytest

from evidencetree.actions import (
    ACTION_REGISTRY,
    ActionExecutor,
    AnswerAction,
    BM25Retriever,
    ClipImageRetriever,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
)
from evidencetree.eval.benchmarks import Document


def _fake_text_encoder(texts):
    return np.array(
        [[1.0, 0.0] if "red" in str(text).lower() else [0.0, 1.0] for text in texts],
        dtype="float32",
    )


def _fake_image_encoder(images):
    out = []
    for image in images:
        red, _, blue = image.getpixel((0, 0))
        out.append([1.0, 0.0] if red > blue else [0.0, 1.0])
    return np.array(out, dtype="float32")


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
def clip_retriever(corpus):
    return ClipImageRetriever(
        encoder=_fake_text_encoder,
        image_encoder=_fake_image_encoder,
    ).build(corpus)


def test_only_three_actions_registered():
    assert {"text_search", "image_search", "answer"} <= set(ACTION_REGISTRY)
    assert "text_to_image" not in ACTION_REGISTRY
    assert "image_to_text" not in ACTION_REGISTRY


def test_image_search_queries_shared_corpus(corpus, clip_retriever):
    hits = clip_retriever.search_image(corpus[0].image_path, top_k=2)
    assert hits and all(hit.doc_id in {"img_red", "txt_red"} for hit in hits)


def test_text_query_hits_shared_corpus(clip_retriever):
    hits = clip_retriever.search("a red thing", top_k=2)
    assert hits and all(hit.doc_id in {"img_red", "txt_red"} for hit in hits)


def test_region_crop_flips_result(tmp_path):
    from PIL import Image

    query = Image.new("RGB", (8, 8), (0, 0, 255))
    for x in range(4):
        for y in range(8):
            query.putpixel((x, y), (255, 0, 0))
    query_path = tmp_path / "half.jpg"
    query.save(query_path)

    red = tmp_path / "r.png"
    blue = tmp_path / "b.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    corpus = [
        Document(doc_id="img_red", text="", title="r", image_path=str(red)),
        Document(doc_id="img_blue", text="", title="b", image_path=str(blue)),
    ]
    retriever = ClipImageRetriever(
        encoder=_fake_text_encoder,
        image_encoder=_fake_image_encoder,
    ).build(corpus)
    assert retriever.search_image(str(query_path), region=None, top_k=1)[0].doc_id == "img_red"
    assert (
        retriever.search_image(str(query_path), region=(0.5, 0, 1, 1), top_k=1)[0].doc_id
        == "img_blue"
    )


def test_executor_runs_three_actions(corpus, clip_retriever):
    text_ret = BM25Retriever().build(corpus)
    ex = ActionExecutor(text_retriever=text_ret, image_retriever=clip_retriever, top_k=2)

    s0 = SearchState(question="what is this red rose?", image_path=corpus[0].image_path)
    s1 = ex.execute(s0, TextSearchAction(query="red rose flower"))
    assert s1.evidence[0].source_action == "text_search"

    s2 = ex.execute(s1, ImageSearchAction())
    assert any(e.source_action == "image_search" for e in s2.evidence)

    s3 = ex.execute(s2, AnswerAction(text="a rose"))
    assert s3.is_terminal and s3.final_answer == "a rose"
