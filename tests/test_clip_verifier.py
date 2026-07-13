"""Tests for unified CLIP grounding over retrieved results."""

import numpy as np
import pytest

from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier


def _fake_text_encoder(texts):
    return np.array(
        [[1.0, 0.0] if "red" in str(text).lower() else [0.0, 1.0] for text in texts],
        dtype="float32",
    )


def _fake_image_encoder(images):
    vectors = []
    for image in images:
        red, _, blue = image.getpixel((0, 0))
        vectors.append([1.0, 0.0] if red > blue else [0.0, 1.0])
    return np.array(vectors, dtype="float32")


def _scorer():
    return ClipGroundingScorer(
        text_encoder=_fake_text_encoder,
        image_encoder=_fake_image_encoder,
        text_band=(0.0, 1.0),
        image_band=(0.0, 1.0),
    )


@pytest.fixture
def red_blue(tmp_path):
    from PIL import Image

    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    return str(red), str(blue)


def test_text_result_score(red_blue):
    scorer = _scorer()
    assert scorer.score_text_results("red flower", ["a red rose"]) > 0.9
    assert scorer.score_text_results("red flower", ["blue sea"]) < 0.1


def test_image_result_score(red_blue):
    red, blue = red_blue
    scorer = _scorer()
    assert scorer.score_image_results("red object", [red]) > 0.9
    assert scorer.score_image_results("red object", [blue]) < 0.1


def test_cos_band_validation():
    with pytest.raises(ValueError):
        ClipGroundingScorer(text_band=(0.4, 0.3))


def test_verifier_scores_by_result_modality(red_blue):
    red, _ = red_blue
    verifier = GroundingVerifier(clip_scorer=_scorer())
    assert verifier.score(
        question="red flower",
        action_type="text_search",
        result_texts=["a red rose"],
    ) > 0.9
    assert verifier.score(
        question="a red object",
        action_type="image_search",
        result_image_paths=[red],
    ) > 0.9
    assert verifier.score(
        question="a red object",
        action_type="image_search",
        result_texts=["blue sea text"],
        result_image_paths=[red],
    ) > 0.9
    assert verifier.score(question="q", action_type="answer", result_texts=["x"]) is None


def test_offline_fallback_scores_text_and_neutral_image(red_blue):
    red, _ = red_blue
    verifier = GroundingVerifier(clip_scorer=None)
    assert verifier.score(
        question="How heavy is this bird?",
        action_type="text_search",
        result_texts=["This bird is heavy."],
    ) > 0.0
    assert verifier.score(
        question="q?",
        action_type="image_search",
        result_image_paths=[red],
    ) == 0.5
    assert verifier.score(question="q?", action_type="image_search") == 0.0
