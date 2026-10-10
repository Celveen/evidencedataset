"""Grounding verifiers: g(action, results) -> float in [0, 1], graded.

Grounding measures how relevant the action's RETRIEVED RESULT is to the
question. It does not judge whether the action input itself overlaps the
question; query anti-fabrication belongs to the policy prompt.

Unified CLIP scoring:
all grounding is a cosine in one CLIP embedding space against the same anchor,
``CLIP_text(question)``. A text result is embedded with the CLIP text tower; an
image result is embedded with the CLIP image tower. Text and image cosine bands
are calibrated separately onto [0, 1].

CLIP's text tower reads at most 77 tokens, while a corpus chunk holds ~200
words. A text result is therefore scored as the MAX over short windows of the
chunk (each prefixed with the page title), so relevance anywhere in the chunk
counts — not only in its first sentence.

Both text_search and image_search may return text-side or image-side units
from the one shared CLIP corpus. Each result is scored against the question and
the best relevance is used; :class:`GroundingDetail` also records which result
won, its modality and its raw (uncalibrated) cosine, so the calibration bands
can be refit from pilot data without re-encoding anything. ``answer`` returns
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
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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


def _titled(title: str, text: str) -> str:
    return f"{title}: {text}" if title else text


def text_windows(text: str, title: str = "", words: int = 48, stride: int = 40) -> list[str]:
    """Split a chunk into overlapping word windows, each prefixed with its title.

    ``words <= 0`` disables windowing (one window, the whole titled text).
    """
    tokens = text.split()
    if words <= 0 or len(tokens) <= words:
        return [_titled(title, text)]
    stride = max(1, min(stride, words))
    out = []
    for start in range(0, len(tokens), stride):
        out.append(_titled(title, " ".join(tokens[start:start + words])))
        if start + words >= len(tokens):
            break
    return out


@dataclass(frozen=True)
class GroundingDetail:
    """Grounding of one step: calibrated score plus what produced it."""

    score: float | None          # calibrated [0, 1]; None for answer steps
    modality: str | None = None  # modality of the best-scoring result
    raw_cos: float | None = None  # its uncalibrated cosine (None in lexical mode)
    best_index: int | None = None  # its position in the results list


class ClipGroundingScorer:
    """Scores text and image results against the question in one CLIP space.

    Encoders are injectable for deterministic tests; the default lazily loads a
    sentence-transformers CLIP model for both towers (shared with the unified
    retriever when both run in one process). Embeddings are cached, since the
    same chunks recur across the steps of many trajectories.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        text_encoder=None,
        image_encoder=None,
        device: str | None = None,
        text_band: tuple[float, float] = (0.5, 0.9),
        image_band: tuple[float, float] = (0.15, 0.32),
        window_words: int = 48,
        window_stride: int = 40,
        cache_size: int = 200_000,
        image_resolver=None,
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
        self.window_words = window_words
        self.window_stride = window_stride
        self.cache_size = cache_size
        self.image_resolver = image_resolver  # portable key -> readable path
        self._text_cache: dict[str, Any] = {}
        self._image_cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Backward-compatible scalar API
    # ------------------------------------------------------------------ #
    def score_text_results(
        self, question: str, result_texts: Sequence[str], result_titles: Sequence[str] = ()
    ) -> float:
        cos = [c for c in self.text_cosines(question, result_texts, result_titles) if c is not None]
        return self.calibrate(max(cos), "text") if cos else 0.0

    def score_image_results(self, question: str, image_paths: Sequence[str]) -> float:
        cos = [c for c in self.image_cosines(question, image_paths) if c is not None]
        return self.calibrate(max(cos), "image") if cos else 0.0

    # ------------------------------------------------------------------ #
    def text_cosines(
        self, question: str, texts: Sequence[str], titles: Sequence[str] = ()
    ) -> list[float | None]:
        """Raw question-result cosine per text (max over windows); None if empty."""
        import numpy as np

        titles = list(titles) + [""] * (len(texts) - len(titles))
        windows: list[list[str]] = [
            text_windows(t, ti, self.window_words, self.window_stride) if t else []
            for t, ti in zip(texts, titles)
        ]
        flat = [w for ws in windows for w in ws]
        if not flat:
            return [None] * len(texts)
        qv = self._embed_texts([question])[0]
        sims = self._embed_texts(flat) @ qv
        out: list[float | None] = []
        pos = 0
        for ws in windows:
            if not ws:
                out.append(None)
                continue
            out.append(float(np.max(sims[pos:pos + len(ws)])))
            pos += len(ws)
        return out

    def image_cosines(self, question: str, image_paths: Sequence[str]) -> list[float | None]:
        """Raw question-image cosine per image; None if missing/unreadable."""
        qv = None
        out: list[float | None] = []
        for path in image_paths:
            iv = self._embed_image(path) if path else None
            if iv is None:
                out.append(None)
                continue
            if qv is None:
                qv = self._embed_texts([question])[0]
            out.append(float((iv * qv).sum()))
        return out

    def calibrate(self, cos: float, modality: str) -> float:
        band = self.image_band if modality == "image" else self.text_band
        return self._map(cos, band)

    @staticmethod
    def _map(cos: float, band: tuple[float, float]) -> float:
        lo, hi = band
        return min(1.0, max(0.0, (cos - lo) / (hi - lo)))

    # ------------------------------------------------------------------ #
    # Result-to-result similarity (evidence redundancy)
    # ------------------------------------------------------------------ #
    def text_similarity(self, a: tuple[str, str], b: tuple[str, str]) -> float:
        """Cosine of two (title, text) results, each read as its first window."""
        va, vb = self._embed_texts([
            text_windows(a[1], a[0], self.window_words, self.window_stride)[0],
            text_windows(b[1], b[0], self.window_words, self.window_stride)[0],
        ])
        return float((va * vb).sum())

    def image_similarity(self, a: str, b: str) -> float | None:
        va, vb = self._embed_image(a), self._embed_image(b)
        if va is None or vb is None:
            return None
        return float((va * vb).sum())

    # ------------------------------------------------------------------ #
    # Encoding with caches
    # ------------------------------------------------------------------ #
    def _embed_texts(self, texts: Sequence[str]):
        import numpy as np

        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            vecs = self._normalize(self._text_encoder, missing, kind="text")
            with self._lock:
                if len(self._text_cache) + len(missing) > self.cache_size:
                    self._text_cache.clear()
                self._text_cache.update(zip(missing, vecs))
                local = {t: self._text_cache[t] for t in texts}
        else:
            with self._lock:
                local = {t: self._text_cache[t] for t in texts}
        return np.stack([local[t] for t in texts])

    def _embed_image(self, path: str):
        with self._lock:
            if path in self._image_cache:
                return self._image_cache[path]
        try:
            vec = self._normalize(self._image_encoder, [self._open_image(path)], kind="image")[0]
        except Exception:  # noqa: BLE001 - unreadable image counts as absent
            vec = None
        with self._lock:
            if len(self._image_cache) >= self.cache_size:
                self._image_cache.clear()
            self._image_cache[path] = vec
        return vec

    def _normalize(self, encoder, items, kind: str):
        import numpy as np

        if encoder is not None:
            vecs = np.asarray(encoder(list(items)), dtype="float32")
        else:
            from evidencetree.actions.retrievers import _MODEL_LOCK

            model = self._st_model()
            with _MODEL_LOCK:
                vecs = model.encode(
                    list(items), convert_to_numpy=True, show_progress_bar=False
                ).astype("float32")
        return vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)

    def _st_model(self):
        if self._model is None:
            from evidencetree.actions.retrievers import shared_sentence_model

            self._model = shared_sentence_model(self.model_name, self.device)
        return self._model

    def _open_image(self, path: str):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.image_resolver is not None:
            path = self.image_resolver(path)
        return Image.open(path).convert("RGB")


_UNVERIFIABLE_ANSWERS = {"yes", "no", "none", "true", "false", "unknown", "n/a"}
_NUM_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)?$")

_SUPPORT_PROMPT = """Question: {question}

Evidence collected so far (lines marked [image] are retrieved pictures, shown
only by the title of the page they belong to; you cannot see the question's
own image):
{evidence}

Proposed final answer: {answer}

Classify how this answer relates to the evidence:

- SUPPORTED: the evidence explicitly contains or entails the answer. For a
  multiple-choice answer it is enough that the evidence entails the CONTENT of
  the chosen option (verbatim wording is not required). An absence answer like
  "no" counts as supported when the evidence describes the relevant
  scene/content and it lacks the asked-about item.
- PARTIAL: the evidence points toward the answer but leaves a substantive part
  of it unconfirmed.
- UNSUPPORTED: the answer relies on external factual knowledge (entity facts,
  numbers, dates, names, statistics) that the evidence does not contain.
- NOT_REQUIRED: the question can be answered from the question text, its
  answer options and its image by reasoning or perception alone — grammar,
  logic, arithmetic, reading a given passage, sentence or diagram, comparing
  the options, or everyday commonsense — so no retrieved knowledge is needed.
  You cannot see the image: judge from what the question asks. Do NOT use this
  for facts about specific entities (capitals, species data, dates,
  measurements, biographies): those always require evidence.

Give your reason in one or two sentences. Then write a last line holding only
the label: SUPPORTED, PARTIAL, UNSUPPORTED or NOT_REQUIRED."""

# Verdict words on the judge's last line. "NOT SUPPORTED" is read as UNSUPPORTED;
# earlier lines are the judge's reason and may use these words in prose.
_VERDICT_RE = re.compile(r"\b(NOT[_ ]REQUIRED|UNSUPPORTED|NOT SUPPORTED|PARTIAL|SUPPORTED)\b")
_VERDICTS: dict[str, tuple[str, float | None]] = {
    "NOT_REQUIRED": ("not_required", None),
    "NOT REQUIRED": ("not_required", None),
    "UNSUPPORTED": ("unsupported", 0.0),
    "NOT SUPPORTED": ("unsupported", 0.0),
    "PARTIAL": ("partial", 0.5),
    "SUPPORTED": ("supported", 1.0),
}


def parse_support_verdict(reply: str) -> tuple[str, float | None, str]:
    """(label, support value, reason) from a judge reply: the verdict is the
    last verdict word on the last non-empty line; the lines before it are the
    reason. A reply whose last line holds no verdict is ``unparseable``."""
    lines = [ln.strip() for ln in str(reply).strip().splitlines() if ln.strip()]
    if not lines:
        return ("unparseable", None, "")
    found = _VERDICT_RE.findall(lines[-1].upper())
    if not found:
        return ("unparseable", None, " ".join(lines))
    label, value = _VERDICTS[found[-1]]
    reason = " ".join(lines[:-1])
    if not reason:  # "SUPPORTED: the evidence states ..." on a single line
        reason = _VERDICT_RE.sub("", lines[-1], count=1).strip(" :.-*")
    return (label, value, reason)


class AnswerSupportVerifier:
    """Check whether an answer is supported by the accumulated evidence.

    ``evidence_texts`` are evidence lines as formatted by
    :func:`evidencetree.prm.data_gen.support_evidence_lines` — id, page title
    and text, with image units shown as ``[image] <title>`` — so the judge can
    tell which entity a chunk belongs to.

    ``lexical`` backend is offline and deterministic: answer content words and
    numbers must appear in the evidence, with unmatched numbers forcing the
    score down. Bare yes/no answers return None because containment cannot judge
    absence claims.

    ``api`` backend asks a judge model for a one- or two-sentence reason and
    then SUPPORTED/PARTIAL/UNSUPPORTED/NOT_REQUIRED, mapped to 1.0/0.5/0.0/None.
    ``NOT_REQUIRED`` means the question is answerable from the question/image
    alone, so evidence support should not gate the answer score. ``judge``
    returns (label, value, reason); ``classify`` drops the reason; ``score``
    keeps the older float-or-None API.
    """

    def __init__(
        self, backend: str = "lexical", generator=None, max_line_chars: int = 1500
    ) -> None:
        backend = backend.lower()
        if backend not in {"lexical", "api"}:
            raise ValueError(f"Unknown support backend {backend!r}.")
        if backend == "api" and generator is None:
            raise ValueError("api support backend needs a generator.")
        self.backend = backend
        self.generator = generator
        self.max_line_chars = max_line_chars

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
        label, value, _ = self.judge(
            question=question, answer=answer, evidence_texts=evidence_texts
        )
        return label, value

    def judge(
        self,
        *,
        question: str,
        answer: str,
        evidence_texts: Sequence[str] = (),
    ) -> tuple[str, float | None, str]:
        """Return (label, support value, the judge's reason; "" if none)."""
        texts = [t for t in evidence_texts if t]
        if not texts:
            return ("no_evidence", 0.0, "")
        if self.backend == "lexical":
            return (*self._lexical(answer, texts), "")
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

    def _api(self, question: str, answer: str, texts: Sequence[str]) -> tuple[str, float | None, str]:
        evidence = "\n".join(
            f"- {t if len(t) <= self.max_line_chars else t[:self.max_line_chars] + ' …'}"
            for t in texts
        )
        prompt = _SUPPORT_PROMPT.format(
            question=question, evidence=evidence, answer=answer
        )
        try:
            reply = self.generator.generate(prompt, [])
        except Exception:  # noqa: BLE001 - keep large batches resumable
            return ("unparseable", None, "")
        return parse_support_verdict(reply)


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
        result_titles: Sequence[str] = (),
        **_: object,
    ) -> float | None:
        """Graded grounding score, or None for actions outcome decides."""
        titles = list(result_titles) + [""] * (len(result_texts) - len(result_titles))
        results = [
            {"text": t, "title": ti, "image_path": None}
            for t, ti in zip(result_texts, titles)
        ] + [{"text": "", "title": "", "image_path": p} for p in result_image_paths]
        return self.score_results(
            question=question, action_type=action_type, results=results
        ).score

    def score_results(
        self,
        *,
        question: str,
        action_type: str,
        results: Sequence[Mapping[str, Any]],
    ) -> GroundingDetail:
        """Grounding of one step from its result records (evidence dicts)."""
        if action_type == "answer":
            return GroundingDetail(score=None)
        texts = [(i, r) for i, r in enumerate(results) if r.get("text")]
        images = [(i, r) for i, r in enumerate(results) if r.get("image_path")]
        if not texts and not images:
            return GroundingDetail(score=0.0)
        candidates: list[tuple[float, str, float | None, int]] = []
        if self.clip_scorer is None:
            for i, r in texts:
                s = _lexical_recall(question, [_titled(r.get("title", ""), r["text"])])
                candidates.append((s, "text", None, i))
            for i, _ in images:
                candidates.append((0.5, "image", None, i))
        else:
            cos_t = self.clip_scorer.text_cosines(
                question, [r["text"] for _, r in texts], [r.get("title", "") for _, r in texts]
            )
            for (i, _), c in zip(texts, cos_t):
                if c is not None:
                    candidates.append((self.clip_scorer.calibrate(c, "text"), "text", c, i))
            cos_i = self.clip_scorer.image_cosines(question, [r["image_path"] for _, r in images])
            for (i, _), c in zip(images, cos_i):
                if c is not None:
                    candidates.append((self.clip_scorer.calibrate(c, "image"), "image", c, i))
        if not candidates:
            return GroundingDetail(score=0.0)
        # Highest calibrated score wins; ties go to the higher raw cosine, then order.
        best = max(candidates, key=lambda c: (c[0], c[2] if c[2] is not None else -1.0, -c[3]))
        return GroundingDetail(score=best[0], modality=best[1], raw_cos=best[2], best_index=best[3])

    def redundancy(
        self, result: Mapping[str, Any], prior: Sequence[Mapping[str, Any]]
    ) -> float:
        """Max similarity of ``result`` to the same-modality evidence in ``prior``.

        A result whose ``doc_id`` is already in ``prior`` counts as fully
        redundant (1.0). An empty bundle gives 0.0. With CLIP this is a raw
        cosine; offline it is content-word Jaccard for text and doc identity
        for images.
        """
        if not prior:
            return 0.0
        doc_id = result.get("doc_id")
        modality = "image" if result.get("image_path") and not result.get("text") else "text"
        best = 0.0
        for e in prior:
            if doc_id and e.get("doc_id") == doc_id:
                e_mod = "image" if e.get("image_path") and not e.get("text") else "text"
                if e_mod == modality:
                    return 1.0
            e_mod = "image" if e.get("image_path") and not e.get("text") else "text"
            if e_mod != modality:
                continue
            if modality == "text":
                if self.clip_scorer is not None:
                    sim = self.clip_scorer.text_similarity(
                        (result.get("title", ""), result.get("text", "")),
                        (e.get("title", ""), e.get("text", "")),
                    )
                else:
                    a = _content_tokens(_titled(result.get("title", ""), result.get("text", "")))
                    b = _content_tokens(_titled(e.get("title", ""), e.get("text", "")))
                    sim = len(a & b) / len(a | b) if a | b else 0.0
            else:
                sim = (
                    self.clip_scorer.image_similarity(result["image_path"], e["image_path"])
                    if self.clip_scorer is not None
                    else None
                ) or 0.0
            best = max(best, float(sim))
        return min(1.0, max(0.0, best))


def grounding_score(action, state, verifier: GroundingVerifier | None = None) -> float | None:
    """Convenience wrapper over typed Action/SearchState objects."""
    verifier = verifier or GroundingVerifier()
    results = [
        {"text": e.text, "title": e.title, "image_path": e.image_path}
        for e in state.evidence
        if e.source_action == action.action_type
    ]
    return verifier.score_results(
        question=state.question, action_type=action.action_type, results=results
    ).score
