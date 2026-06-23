"""Tests for unified CLIP grounding (text + image results in one space) and
GroundingVerifier dispatch. Fake encoders -> no downloads."""

import numpy as np
import pytest

from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier


# Deterministic 2-D fake CLIP: "red" axis vs "blue" axis, shared text+image.
def _fake_text_encoder(texts):
    return np.array(
        [[1.0, 0.0] if "red" in t.lower() else [0.0, 1.0] for t in texts],
        dtype="float32",
    )


def _fake_image_encoder(images):
    out = []
    for im in images:
        r, _, b = im.getpixel((0, 0))
        out.append([1.0, 0.0] if r > b else [0.0, 1.0])
    return np.array(out, dtype="float32")


def _scorer():
    return ClipGroundingScorer(
        text_encoder=_fake_text_encoder, image_encoder=_fake_image_encoder
    )


@pytest.fixture
def red_blue(tmp_path):
    from PIL import Image

    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    return str(red), str(blue)


# --------------------------------------------------------------------------- #
# Text results (CLIP text-text)
# --------------------------------------------------------------------------- #
def test_text_results_relevant_vs_irrelevant():
    s = _scorer()
    assert s.score_text_results("a red thing", ["red rose text"]) > 0.9
    assert s.score_text_results("a red thing", ["blue sea text"]) < 0.1
    assert s.score_text_results("a red thing", []) == 0.0


# --------------------------------------------------------------------------- #
# Image results (CLIP text-image)
# --------------------------------------------------------------------------- #
def test_image_results_relevant_vs_irrelevant(red_blue):
    red, blue = red_blue
    s = _scorer()
    assert s.score_image_results("a red object", [red]) > 0.9
    assert s.score_image_results("a red object", [blue]) < 0.1
    assert s.score_image_results("a red object", [blue, red]) > 0.9  # best wins
    assert s.score_image_results("a red object", []) == 0.0


def test_image_results_skip_unreadable(red_blue):
    red, _ = red_blue
    assert _scorer().score_image_results("a red object", ["/no/such.jpg", red]) > 0.9


def test_unified_space_same_anchor(red_blue):
    """text and image grounding share one CLIP model + the same question anchor,
    so a relevant text result and a relevant image result both score high."""
    red, _ = red_blue
    s = _scorer()
    t = s.score_text_results("a red thing", ["red rose"])
    i = s.score_image_results("a red thing", [red])
    assert t > 0.9 and i > 0.9


def test_band_validation():
    with pytest.raises(ValueError):
        ClipGroundingScorer(text_band=(0.9, 0.5))
    with pytest.raises(ValueError):
        ClipGroundingScorer(image_band=(0.4, 0.3))


# --------------------------------------------------------------------------- #
# GroundingVerifier dispatch by result modality
# --------------------------------------------------------------------------- #
def test_verifier_dispatches_by_result_modality(red_blue):
    red, _ = red_blue
    v = GroundingVerifier(clip_scorer=_scorer())
    assert v.score(question="red flower", action_type="text_search",
                   result_texts=["a red rose"]) > 0.9
    assert v.score(question="red flower", action_type="image_to_text",
                   result_texts=["a red rose"]) > 0.9
    assert v.score(question="a red object", action_type="text_to_image",
                   result_image_paths=[red]) > 0.9
    assert v.score(question="a red object", action_type="image_search",
                   result_image_paths=[red]) > 0.9
    assert v.score(question="q", action_type="answer", result_texts=["x"]) is None


def test_verifier_lexical_fallback_without_clip():
    """clip_scorer=None (mock/offline): text -> lexical recall, image -> neutral."""
    v = GroundingVerifier(clip_scorer=None)
    q = "How heavy is the bird in grams?"
    assert v.score(question=q, action_type="text_search",
                   result_texts=["the bird is heavy, 400 grams"]) > \
           v.score(question=q, action_type="text_search", result_texts=["Paris France"])
    assert v.score(question=q, action_type="text_search", result_texts=[]) == 0.0
    assert v.score(question=q, action_type="image_search", result_image_paths=["/i.jpg"]) == 0.5
    assert v.score(question=q, action_type="image_search", result_image_paths=[]) == 0.0
