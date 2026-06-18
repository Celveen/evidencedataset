"""Tests for cross-modal actions: CrossModalCLIPRetriever + executor dispatch.

Uses a deterministic fake CLIP (red/blue 2-D axis) so text and image queries
land in a shared space without any model download.
"""

import numpy as np
import pytest

from evidencetree.actions import (
    ActionExecutor,
    CrossModalCLIPRetriever,
    ImageSearchAction,
    ImageToTextAction,
    SearchState,
    TextSearchAction,
    TextToImageAction,
)
from evidencetree.actions.action_space import ACTION_REGISTRY
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
    return CrossModalCLIPRetriever(
        text_encoder=_fake_text_encoder, image_encoder=_fake_image_encoder
    ).build(corpus)


# --------------------------------------------------------------------------- #
# Retriever: the cross-modal matrix
# --------------------------------------------------------------------------- #
def test_text_to_image_finds_matching_image(retriever):
    hits = retriever.text_to_image("a red object", top_k=1)
    assert hits and hits[0].doc_id == "img_red"


def test_image_to_text_finds_matching_text(retriever, corpus):
    red_img = corpus[0].image_path
    hits = retriever.image_to_text(red_img, top_k=1)
    assert hits and hits[0].doc_id == "txt_red"


def test_image_search_finds_similar_image(retriever, corpus):
    blue_img = corpus[1].image_path
    hits = retriever.search_image(blue_img, top_k=1)
    assert hits and hits[0].doc_id == "img_blue"


def test_region_crop_flips_image_to_text(tmp_path):
    """Left half red / right half blue: region selects which text is retrieved."""
    from PIL import Image

    img = Image.new("RGB", (8, 8), (0, 0, 255))
    for x in range(4):
        for y in range(8):
            img.putpixel((x, y), (255, 0, 0))
    p = tmp_path / "half.jpg"
    img.save(p)
    corpus = [
        Document(doc_id="txt_red", text="red red red", title="r"),
        Document(doc_id="txt_blue", text="blue blue blue", title="b"),
    ]
    r = CrossModalCLIPRetriever(
        text_encoder=_fake_text_encoder, image_encoder=_fake_image_encoder
    ).build(corpus)
    assert r.image_to_text(str(p), region=None, top_k=1)[0].doc_id == "txt_red"
    assert r.image_to_text(str(p), region=(0.5, 0, 1, 1), top_k=1)[0].doc_id == "txt_blue"


def test_text_only_corpus_returns_empty_for_image_targets():
    """No images in the corpus -> image-target actions return [] (the documented
    text-only limitation). Text targets still work."""
    r = CrossModalCLIPRetriever(
        text_encoder=_fake_text_encoder, image_encoder=_fake_image_encoder
    ).build([Document(doc_id="t0", text="some text", title="t")])
    assert r.text_to_image("red", top_k=3) == []   # no corpus images
    assert r.search_image("/x.jpg", top_k=3) == []  # no corpus images


# --------------------------------------------------------------------------- #
# Executor dispatch
# --------------------------------------------------------------------------- #
def test_executor_runs_new_actions(retriever, corpus):
    ex = ActionExecutor(image_retriever=retriever, top_k=2)
    red_img = corpus[0].image_path

    s0 = SearchState(question="what is this red thing?", image_path=red_img)
    s1 = ex.execute(s0, TextToImageAction(query="red object"))
    assert s1.evidence and s1.evidence[0].source_action == "text_to_image"

    s2 = ex.execute(s1, ImageToTextAction())
    itt = [e for e in s2.evidence if e.source_action == "image_to_text"]
    assert itt and itt[0].doc_id == "txt_red"  # red query image -> red text (rank 1)


def test_new_actions_registered_and_described():
    assert {"text_to_image", "image_to_text"} <= set(ACTION_REGISTRY)
    assert ImageToTextAction(region=(0.1, 0.1, 0.5, 0.5)).granularity == "region"
    assert ImageToTextAction().granularity == "whole"
    assert TextToImageAction(query="x").modality == "text"
    assert "text_to_image" in TextToImageAction(query="x").describe()


def test_text_to_image_needs_cross_modal_retriever():
    # An executor without a cross-modal retriever rejects the action clearly.
    ex = ActionExecutor(text_retriever=None, image_retriever=None)
    with pytest.raises(RuntimeError):
        ex.execute(SearchState(question="q"), TextToImageAction(query="x"))


def test_image_search_still_works(retriever, corpus):
    """Existing image_search action unaffected by the additions."""
    ex = ActionExecutor(image_retriever=retriever, top_k=1)
    s = ex.execute(
        SearchState(question="q", image_path=corpus[1].image_path), ImageSearchAction()
    )
    assert s.evidence[0].source_action == "image_search"
