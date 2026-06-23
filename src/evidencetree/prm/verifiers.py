"""Grounding verifiers: g(action, results) -> float in [0, 1], graded (report §3.2).

Grounding measures **how relevant the action's RETRIEVED RESULT is to the
question** (not whether the action's input aligns with the question — that
cannot rate image_search / image_to_text whose query image is fixed, and
anti-fabrication is already handled by the policy prompt). Empty results -> 0.

UNIFIED CLIP SCORING (so scores are comparable across action types):
all grounding is a cosine in ONE CLIP embedding space against the SAME anchor
``CLIP_text(question)``. A text result is embedded with the CLIP text tower, an
image result with the CLIP image tower. If text and image grounding used two
different models (e.g. a sentence-transformer for text + CLIP for images), their
cosine scales would differ and the PRM would mistake the scale gap for a quality
gap, biasing it toward whichever action type scores systematically higher.

Because same-modality (text-text) cosine runs higher than cross-modality
(text-image) cosine, each is mapped to [0, 1] by its own ``[cos_lo, cos_hi]``
band — same model, same anchor, the band just calibrates both cosine
distributions onto the same "relevance" meaning. Bands are empirical; recalibrate
from the real cosine distribution on the server.

Dispatch by the action's RESULT modality:
    text_search, image_to_text   -> text results  -> CLIP text-text
    text_to_image, image_search  -> image results -> CLIP text-image
    answer                       -> None (outcome decides)

A ``clip_scorer=None`` verifier (mock / offline tests) falls back to lexical
question-word recall for text results and a neutral 0.5 for image results, so
the pipeline runs without any model download.
"""

from __future__ import annotations

import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "are", "was", "were",
    "which", "what", "who", "whom", "whose", "where", "when", "how", "name",
    "can", "be", "found", "located", "does", "do", "did", "this", "that", "it",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _lexical_recall(question: str, texts: Sequence[str]) -> float:
    """Offline fallback: question content-word recall in the best result."""
    valid = [t for t in texts if t]
    if not valid:
        return 0.0
    q = _content_tokens(question)
    if not q:
        return 0.0
    return max(len(q & _content_tokens(t)) / len(q) for t in valid)


# --------------------------------------------------------------------------- #
# Unified CLIP grounding scorer
# --------------------------------------------------------------------------- #
class ClipGroundingScorer:
    """Scores text AND image results against the question in one CLIP space.

    Encoders are injectable (``text_encoder`` / ``image_encoder``: list -> (n,d)
    array) for tests; the default lazily loads one CLIP model for both towers.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        text_encoder=None,
        image_encoder=None,
        device: str | None = None,
        text_band: tuple[float, float] = (0.5, 0.9),
        image_band: tuple[float, float] = (0.15, 0.32),
    ) -> None:
        for lo, hi in (text_band, image_band):
            if hi <= lo:
                raise ValueError("each cos band needs hi > lo.")
        self.model_name = model_name
        self.device = device
        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self._model = None
        self.text_band = text_band
        self.image_band = image_band

    # ------------------------------------------------------------------ #
    def score_text_results(self, question: str, result_texts: Sequence[str]) -> float:
        texts = [t for t in result_texts if t]
        if not texts:
            return 0.0
        qv = self._encode_text([question])[0]
        rv = self._encode_text(texts)
        cos = float(max((rv @ qv).tolist()))
        return self._map(cos, self.text_band)

    def score_image_results(self, question: str, image_paths: Sequence[str]) -> float:
        paths = [p for p in image_paths if p]
        if not paths:
            return 0.0
        qv = self._encode_text([question])[0]
        best = 0.0
        found = False
        for p in paths:
            try:
                iv = self._encode_images([self._open_image(p)])[0]
            except Exception:  # noqa: BLE001 - skip unreadable image
                continue
            found = True
            best = max(best, float((iv * qv).sum()))
        return self._map(best, self.image_band) if found else 0.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _map(cos: float, band: tuple[float, float]) -> float:
        lo, hi = band
        return min(1.0, max(0.0, (cos - lo) / (hi - lo)))

    def _encode_text(self, texts):
        return self._normalize(self._text_encoder, texts)

    def _encode_images(self, images):
        return self._normalize(self._image_encoder, images)

    def _normalize(self, encoder, items):
        import numpy as np

        if encoder is not None:
            vecs = np.asarray(encoder(items), dtype="float32")
        else:
            vecs = self._st_model().encode(
                list(items), convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    def _st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @staticmethod
    def _open_image(path: str):
        from PIL import Image

        return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
# Verifier: dispatch by result modality
# --------------------------------------------------------------------------- #
_IMAGE_RESULT_ACTIONS = {"text_to_image", "image_search"}   # results are images
_TEXT_RESULT_ACTIONS = {"text_search", "image_to_text"}     # results are text


class GroundingVerifier:
    """g(action, results) -> graded relevance of the retrieved result to the
    question, scored in one unified CLIP space (or lexical/neutral fallback when
    no clip_scorer is wired)."""

    def __init__(self, clip_scorer: ClipGroundingScorer | None = None):
        self.clip_scorer = clip_scorer

    def score(
        self,
        *,
        question: str,
        action_type: str,
        result_texts: Sequence[str] = (),
        result_image_paths: Sequence[str] = (),
    ) -> float | None:
        """Graded grounding score, or None for actions outcome decides."""
        if action_type == "answer":
            return None
        if action_type in _IMAGE_RESULT_ACTIONS:
            if self.clip_scorer is None:
                return 0.5 if [p for p in result_image_paths if p] else 0.0
            return self.clip_scorer.score_image_results(question, result_image_paths)
        # text-result actions (and any unknown type) -> text relevance
        if self.clip_scorer is None:
            return _lexical_recall(question, result_texts)
        return self.clip_scorer.score_text_results(question, result_texts)
