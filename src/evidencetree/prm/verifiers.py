"""Modality-specific grounding verifiers: g(action, state) -> float in [0, 1].

All grounding scores are GRADED (continuous 0-1), never binary (report §3.2).

Backends:
    lexical — deterministic token-overlap alignment. Offline, free; used by
              mock smoke runs and as a sanity baseline.
    api     — LLM judge through the generation API backend, returns a graded
              score. Used by DatasetConstruct when no local verifier is up.

Per-action-type semantics (report §Stage 2):
    text_search  -> alignment of the search query with the original question
                    (lexical now; local cross-encoder lands with full Stage 2)
    image_search -> CLIP image-text alignment between the query image (or its
                    region) and the question's visual entities, via the
                    injectable ``ClipGroundingScorer`` (report §3.2). If no
                    scorer is wired or no image is available, falls back to a
                    neutral 0.5.
    answer       -> None: not locally scored, outcome decides.

``ClipGroundingScorer`` loads a CLIP model (sentence-transformers) lazily; the
encoders are injectable so tests run deterministically without downloads. It
can share one CLIP instance with ``ClipImageRetriever`` on the GPU server.
"""

from __future__ import annotations

import re
from typing import Sequence

from evidencetree.generation.base import Generator

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "are", "was", "were",
    "which", "what", "who", "whom", "whose", "where", "when", "how", "name",
    "can", "be", "found", "located", "does", "do", "did",
}

_JUDGE_PROMPT = """You are grading one retrieval action inside a multi-step \
retrieval trajectory.

Question: {question}
Evidence already collected before this action:
{evidence}

Proposed action: {action_type}({action_input})

Rate how well-grounded this action is: does it plausibly move toward answering
the question given what is already known? 0.0 = clearly unjustified or
redundant, 1.0 = clearly the right move. Reply with ONLY a number in [0, 1]."""

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    }


# --------------------------------------------------------------------------- #
# image_search grounding — CLIP image-text alignment (report §3.2)
# --------------------------------------------------------------------------- #
class ClipGroundingScorer:
    """Graded grounding for image_search: how well the query image (or a region
    of it) aligns with the question's visual entities.

    ``g = clamp((cos - cos_lo) / (cos_hi - cos_lo), 0, 1)`` where ``cos`` is the
    cosine between the CLIP image embedding of the query crop and the CLIP text
    embedding of the question. The ``[cos_lo, cos_hi]`` band maps a
    model-specific cosine range to [0, 1] — CLIP ViT-B/32 image-text cosine sits
    around ~0.15 (random) to ~0.30 (matched), so the default band is a coarse
    *empirical* calibration; tune it or recalibrate in PRM Stage 4.3.

    Encoders are injectable (``image_encoder`` / ``text_encoder``: list -> (n, d)
    array) so tests avoid downloads; the default lazily loads one CLIP model for
    both towers. Pass a shared ``ClipImageRetriever``'s encoders to reuse weights.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        image_encoder=None,
        text_encoder=None,
        device: str | None = None,
        cos_lo: float = 0.15,
        cos_hi: float = 0.32,
    ) -> None:
        if cos_hi <= cos_lo:
            raise ValueError("cos_hi must be > cos_lo.")
        self.model_name = model_name
        self.device = device
        self._image_encoder = image_encoder
        self._text_encoder = text_encoder
        self._model = None
        self.cos_lo = float(cos_lo)
        self.cos_hi = float(cos_hi)

    def score_image_query(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None,
        question: str,
    ) -> float:
        """Alignment in [0, 1]; 0.5 on any failure (missing file / model error)."""
        try:
            img = self._open_image(image_path)
            if region is not None:
                w, h = img.size
                x1, y1, x2, y2 = region
                img = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            iv = self._encode_images([img])[0]
            tv = self._encode_text([question])[0]
            cos = float((iv * tv).sum())  # both L2-normalized -> dot == cosine
            return min(1.0, max(0.0, (cos - self.cos_lo) / (self.cos_hi - self.cos_lo)))
        except Exception:  # noqa: BLE001 - grounding must never crash labelling
            return 0.5

    # ------------------------------------------------------------------ #
    def _encode_images(self, images):
        return self._normalize(self._image_encoder, images)

    def _encode_text(self, texts):
        return self._normalize(self._text_encoder, texts)

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


class GroundingVerifier:
    """g(action, state) with a pluggable backend.

    ``backend`` handles text_search (and is the api LLM judge fallback for
    image_search when no ``image_scorer`` is wired); ``image_scorer`` (a
    :class:`ClipGroundingScorer`) handles image_search via CLIP when present.
    """

    def __init__(
        self,
        backend: str = "lexical",
        generator: Generator | None = None,
        image_scorer: "ClipGroundingScorer | None" = None,
    ):
        backend = backend.lower()
        if backend not in {"lexical", "api"}:
            raise ValueError(f"Unknown verifier backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api verifier backend needs a generator.")
        self.backend = backend
        self.generator = generator
        self.image_scorer = image_scorer

    # ------------------------------------------------------------------ #
    def score(
        self,
        *,
        question: str,
        action_type: str,
        action_input: str,
        state_evidence_texts: Sequence[str] = (),
        image_path: str | None = None,
        region: tuple[float, float, float, float] | None = None,
    ) -> float | None:
        """Graded grounding score, or None for actions outcome decides."""
        if action_type == "answer":
            return None
        if action_type == "image_search" and self.image_scorer is not None:
            if image_path is None:
                return 0.5  # no image available (text-only run) -> neutral
            return self.image_scorer.score_image_query(image_path, region, question)
        if self.backend == "lexical":
            return self._lexical(question, action_type, action_input)
        return self._api(question, action_type, action_input, state_evidence_texts)

    # ------------------------------------------------------------------ #
    def _lexical(self, question: str, action_type: str, action_input: str) -> float:
        if action_type == "image_search":
            # Lexical features cannot judge an image query; neutral score.
            # The CLIP-based verifier (Stage 2, GPU server) replaces this.
            return 0.5
        q_tokens = _content_tokens(question)
        a_tokens = _content_tokens(action_input)
        if not a_tokens or not q_tokens:
            return 0.0
        # Precision of the query w.r.t. the question: drifting queries score low.
        return len(q_tokens & a_tokens) / len(a_tokens)

    def _api(
        self,
        question: str,
        action_type: str,
        action_input: str,
        state_evidence_texts: Sequence[str],
    ) -> float:
        evidence = "\n".join(state_evidence_texts) or "(none)"
        prompt = _JUDGE_PROMPT.format(
            question=question,
            evidence=evidence,
            action_type=action_type,
            action_input=action_input,
        )
        try:
            raw = self.generator.generate(prompt, [])
            match = _NUMBER_RE.search(raw)
            if match is None:
                return 0.5
            return min(1.0, max(0.0, float(match.group(1))))
        except Exception:
            return 0.5  # judge failure -> neutral, never crash the pipeline


def grounding_score(action, state, verifier: GroundingVerifier | None = None) -> float | None:
    """Convenience wrapper over typed Action/SearchState objects."""
    verifier = verifier or GroundingVerifier()
    action_input = (
        getattr(action, "query", None) or getattr(action, "text", "") or action.describe()
    )
    return verifier.score(
        question=state.question,
        action_type=action.action_type,
        action_input=str(action_input),
        state_evidence_texts=state.evidence_texts(),
        image_path=getattr(action, "image_path", None) or state.image_path,
        region=getattr(action, "region", None),
    )
