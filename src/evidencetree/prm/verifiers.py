"""Grounding verifiers: g(action, results) -> float in [0, 1], graded.

Grounding measures how relevant the action's RETRIEVED RESULT is to the
question. It does not judge whether the action input itself overlaps the
question; query anti-fabrication belongs to the policy prompt.

Unified CLIP scoring:
all grounding is a cosine in one CLIP embedding space against the same anchor,
``CLIP_text(question)``. A text result is embedded with the CLIP text tower; an
image result is embedded with the CLIP image tower. Text and image cosine bands
are calibrated separately onto [0, 1].

Both text_search and image_search may return text-side or image-side documents
when image_search queries one shared CLIP corpus. Each modality present is
scored against the question and the best relevance is used. ``answer`` returns
None because outcome credit decides it.

With ``clip_scorer=None`` (mock/offline tests), text results fall back to
lexical question-word recall and image results receive neutral 0.5 when present.

``AnswerSupportVerifier`` is used only for answer steps. Grounding never scores
answers, so outcome-only labels can accidentally reward a correct but
unsupported parametric guess. The support verifier checks whether the proposed
answer is actually backed by the accumulated evidence bundle.
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


class ClipGroundingScorer:
    """Scores text and image results against the question in one CLIP space.

    Encoders are injectable for deterministic tests; the default lazily loads a
    sentence-transformers CLIP model for both towers.
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
        for path in paths:
            try:
                iv = self._encode_images([self._open_image(path)])[0]
            except Exception:  # noqa: BLE001 - skip unreadable image
                continue
            found = True
            best = max(best, float((iv * qv).sum()))
        return self._map(best, self.image_band) if found else 0.0

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


_UNVERIFIABLE_ANSWERS = {"yes", "no", "none", "true", "false", "unknown", "n/a"}
_NUM_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")

_SUPPORT_PROMPT = """Question: {question}

Evidence collected so far:
{evidence}

Proposed final answer: {answer}

Classify how this answer relates to the evidence. Reply with exactly one word:

- SUPPORTED: the evidence explicitly contains or entails the answer. For a
  multiple-choice answer it is enough that the evidence entails the CONTENT of
  the chosen option (verbatim wording is not required). An absence answer like
  "no" counts as supported when the evidence describes the relevant
  scene/content and it lacks the asked-about item.
- PARTIAL: the evidence points toward the answer but leaves a substantive part
  of it unconfirmed.
- UNSUPPORTED: the answer relies on external factual knowledge (entity facts,
  numbers, dates, names, statistics) that the evidence does not contain.
- NOT_REQUIRED: the answer is derivable from the question text and the visible
  image content alone, by perception, logic, or everyday commonsense — no
  external factual knowledge is needed. Do NOT use this for fine-grained
  entity facts (species data, dates, measurements, biographies): those always
  require evidence."""


class AnswerSupportVerifier:
    """Check whether an answer is supported by the accumulated evidence.

    ``lexical`` backend is offline and deterministic: answer content words and
    numbers must appear in the evidence, with unmatched numbers forcing the
    score down. Bare yes/no answers return None because containment cannot judge
    absence claims.

    ``api`` backend asks a judge model for SUPPORTED/PARTIAL/UNSUPPORTED/
    NOT_REQUIRED and maps those to 1.0/0.5/0.0/None. ``NOT_REQUIRED`` means the
    question is answerable from the question/image alone, so evidence support
    should not gate the answer score. ``classify`` returns the raw label for
    diagnostics; ``score`` keeps the older float-or-None API.
    """

    def __init__(self, backend: str = "lexical", generator=None) -> None:
        backend = backend.lower()
        if backend not in {"lexical", "api"}:
            raise ValueError(f"Unknown support backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api support backend needs a generator.")
        self.backend = backend
        self.generator = generator

    def score(
        self,
        *,
        question: str,
        answer: str,
        evidence_texts: Sequence[str] = (),
    ) -> float | None:
        return self.classify(
            question=question, answer=answer, evidence_texts=evidence_texts
        )[1]

    def classify(
        self,
        *,
        question: str,
        answer: str,
        evidence_texts: Sequence[str] = (),
    ) -> tuple[str, float | None]:
        """Return (label, support value); value None = fall back to outcome."""
        texts = [t for t in evidence_texts if t]
        if not texts:
            return ("no_evidence", 0.0)
        if self.backend == "lexical":
            return self._lexical(answer, texts)
        return self._api(question, answer, texts)

    def _lexical(self, answer: str, texts: Sequence[str]) -> tuple[str, float | None]:
        if answer.strip().lower() in _UNVERIFIABLE_ANSWERS:
            return ("unverifiable", None)
        tokens = _content_tokens(answer)
        if not tokens:
            return ("unverifiable", None)
        evidence_tokens: set[str] = set()
        for text in texts:
            evidence_tokens |= _content_tokens(text)
        nums = {t for t in tokens if _NUM_TOKEN_RE.match(t)}
        words = tokens - nums
        word_score = len(words & evidence_tokens) / len(words) if words else 1.0
        if not nums:
            return ("lexical", word_score)
        num_score = len(nums & evidence_tokens) / len(nums)
        return ("lexical", min(num_score, word_score) if words else num_score)

    def _api(self, question: str, answer: str, texts: Sequence[str]) -> tuple[str, float | None]:
        evidence = "\n".join(f"- {t}" for t in texts)
        prompt = _SUPPORT_PROMPT.format(
            question=question, evidence=evidence, answer=answer
        )
        try:
            reply = self.generator.generate(prompt, []).strip().upper()
        except Exception:  # noqa: BLE001 - keep large batches resumable
            return ("unparseable", None)
        if "NOT_REQUIRED" in reply or "NOT REQUIRED" in reply:
            return ("not_required", None)
        if "UNSUPPORTED" in reply:
            return ("unsupported", 0.0)
        if "PARTIAL" in reply:
            return ("partial", 0.5)
        if "SUPPORTED" in reply:
            return ("supported", 1.0)
        return ("unparseable", None)


class GroundingVerifier:
    """g(action, results) -> relevance of retrieved results to the question."""

    def __init__(self, clip_scorer: ClipGroundingScorer | None = None):
        self.clip_scorer = clip_scorer

    def score(
        self,
        *,
        question: str,
        action_type: str,
        result_texts: Sequence[str] = (),
        result_image_paths: Sequence[str] = (),
        **_: object,
    ) -> float | None:
        """Graded grounding score, or None for actions outcome decides."""
        if action_type == "answer":
            return None
        texts = [t for t in result_texts if t]
        images = [p for p in result_image_paths if p]
        if not texts and not images:
            return 0.0
        parts: list[float] = []
        if self.clip_scorer is None:
            if texts:
                parts.append(_lexical_recall(question, texts))
            if images:
                parts.append(0.5)
        else:
            if texts:
                parts.append(self.clip_scorer.score_text_results(question, texts))
            if images:
                parts.append(self.clip_scorer.score_image_results(question, images))
        return max(parts) if parts else 0.0


def grounding_score(action, state, verifier: GroundingVerifier | None = None) -> float | None:
    """Convenience wrapper over typed Action/SearchState objects."""
    verifier = verifier or GroundingVerifier()
    result_texts = [
        f"{e.title}: {e.text}" if e.title else e.text
        for e in state.evidence
        if e.source_action == action.action_type
    ]
    result_image_paths = [
        e.image_path
        for e in state.evidence
        if e.source_action == action.action_type and e.image_path
    ]
    return verifier.score(
        question=state.question,
        action_type=action.action_type,
        result_texts=result_texts,
        result_image_paths=result_image_paths,
    )
