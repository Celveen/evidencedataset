"""Tests for the CLIP image_search grounding verifier (fake encoders — no DL)."""

import numpy as np
import pytest

from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier


# Deterministic 2-D fake CLIP: "red" axis vs "blue" axis.
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
        image_encoder=_fake_image_encoder, text_encoder=_fake_text_encoder
    )


@pytest.fixture
def red_image(tmp_path):
    from PIL import Image

    p = tmp_path / "red.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(p)
    return str(p)


def test_alignment_high_when_image_matches_question(red_image):
    g = _scorer().score_image_query(red_image, None, "What is this red object?")
    assert g > 0.9  # cos=1 -> above cos_hi -> clamped to 1.0


def test_alignment_low_when_image_mismatches_question(red_image):
    g = _scorer().score_image_query(red_image, None, "What is this blue object?")
    assert g < 0.1  # cos=0 -> below cos_lo -> clamped to 0.0


def test_region_crop_changes_score(tmp_path):
    """Left half red, right half blue: focusing the right region flips alignment."""
    from PIL import Image

    img = Image.new("RGB", (8, 8), (0, 0, 255))
    for x in range(4):
        for y in range(8):
            img.putpixel((x, y), (255, 0, 0))
    p = tmp_path / "half.jpg"
    img.save(p)

    scorer = _scorer()
    # whole image: pixel (0,0) is red -> aligns with a red question
    assert scorer.score_image_query(str(p), None, "red object") > 0.9
    # right half (region) is blue -> does NOT align with a red question
    assert scorer.score_image_query(str(p), (0.5, 0.0, 1.0, 1.0), "red object") < 0.1


def test_missing_file_returns_neutral():
    assert _scorer().score_image_query("/no/such/file.jpg", None, "red") == 0.5


def test_cos_band_validation():
    with pytest.raises(ValueError):
        ClipGroundingScorer(cos_lo=0.4, cos_hi=0.3)


# --------------------------------------------------------------------------- #
# GroundingVerifier dispatch
# --------------------------------------------------------------------------- #
def test_verifier_dispatches_image_search_to_clip(red_image):
    v = GroundingVerifier(backend="lexical", image_scorer=_scorer())
    g = v.score(
        question="this red thing", action_type="image_search",
        action_input="image_search(<state image>, region=None)", image_path=red_image,
    )
    assert g > 0.9


def test_verifier_image_search_neutral_without_image():
    v = GroundingVerifier(backend="lexical", image_scorer=_scorer())
    g = v.score(question="q", action_type="image_search", action_input="x", image_path=None)
    assert g == 0.5


def test_verifier_image_search_neutral_without_scorer(red_image):
    """No CLIP scorer wired -> falls back to the documented 0.5 placeholder."""
    v = GroundingVerifier(backend="lexical")
    g = v.score(question="q", action_type="image_search", action_input="x", image_path=red_image)
    assert g == 0.5


def test_text_search_unaffected_by_image_scorer():
    v = GroundingVerifier(backend="lexical", image_scorer=_scorer())
    g = v.score(question="Where is the Eiffel Tower?", action_type="text_search",
                action_input="Eiffel Tower")
    assert g == pytest.approx(1.0)
    assert v.score(question="q?", action_type="answer", action_input="x") is None
