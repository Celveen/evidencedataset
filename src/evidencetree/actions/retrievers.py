"""Retriever wrappers.

* ``BM25Retriever`` — lexical retrieval over the local corpus (``rank-bm25``).
* ``UnifiedClipRetriever`` — the retriever the method is built on: ONE CLIP
  space in which corpus text chunks and corpus images are independent
  retrieval units. ``search_text`` and ``search_image`` both return mixed
  top-k over all units, each hit tagged with ``result_modality``, with
  optional entity/doc dedupe before truncation.
* ``HybridTextRetriever`` — rank-interleaves BM25 with unified text-query hits.

All retrieval is OFFLINE over the benchmark's own corpus — no web API. Heavy
dependencies (faiss / sentence-transformers / PIL) are imported lazily so the
mock smoke tests never need them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from rank_bm25 import BM25Okapi

from ..eval.benchmarks import Document
from ..utils.image_regions import safe_crop_box

# An encoder maps a list of inputs (str or PIL.Image) to a 2D array (n, dim).
Encoder = Callable[[Sequence], "object"]
_SENTENCE_MODEL_CACHE: dict[tuple[str, str | None], object] = {}


@dataclass
class RetrievalHit:
    """A single retrieval result.

    ``result_modality`` tags the modality of the retrieved UNIT ("text" |
    "image"); ``None`` means the retriever predates unit-level retrieval and
    returned a whole document.
    """

    doc_id: str
    text: str
    score: float
    title: str = ""
    image_path: str | None = None
    result_modality: str | None = None




_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lexical BM25 retriever over an in-memory corpus."""

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def build(self, corpus: Sequence[Document]) -> "BM25Retriever":
        """Index a corpus. Must be called before :meth:`search`."""
        if not corpus:
            raise ValueError("Cannot build BM25 index from an empty corpus.")
        self._docs = list(corpus)
        tokenized = [_tokenize(f"{d.title} {d.text}") for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        return self

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Return the top-k documents for a text query."""
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.search called before build().")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self._docs)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            _hit(self._docs[i], float(scores[i]), modality="text") for i in ranked
        ]

    def __len__(self) -> int:
        return len(self._docs)


def _hit(doc: Document, score: float, modality: str | None = None) -> RetrievalHit:
    return RetrievalHit(
        doc_id=doc.doc_id,
        text=doc.text,
        score=score,
        title=doc.title,
        image_path=doc.image_path,
        result_modality=modality,
    )






@dataclass
class _RetrievalUnit:
    """One independently retrievable unit in the unified CLIP index."""

    doc_id: str
    modality: str  # "text" | "image"
    text: str
    title: str
    image_path: str | None
    entity_key: str


class UnifiedClipRetriever:
    """ONE unified CLIP space over independent text and image retrieval units.

    Design (implementation report §3.1/§3.2): corpus text chunks AND corpus
    images are INDEPENDENT retrieval units, each with its own normalized vector
    in a single index. ``search_text`` embeds a text query with the CLIP text
    tower; ``search_image`` embeds the query image (optionally a region) with
    the image tower. BOTH retrieve a mixed top-k over ALL units, and every hit
    carries ``result_modality`` ("text" | "image").

    A Document therefore contributes up to TWO units: one text unit (title +
    text, no image binding) and one image unit (image only — its hit text is
    empty so downstream grounding scores the image itself, never a bound text).

    ``dedupe_key`` ("entity" | "doc" | None) collapses near-duplicate units
    BEFORE truncating to top_k, so a corpus that attaches the same entity image
    to many chunks cannot fill the whole top-k with one entity. The entity key
    is ``Document.entity_id`` when present, else title, else doc_id.

    Encoders are injectable for deterministic tests (same pattern as
    ClipGroundingScorer / ClipImageRetriever); the default lazily loads one
    sentence-transformers CLIP model for both towers. The index is FAISS
    IndexFlatIP when faiss is importable, else a numpy matmul fallback.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        text_encoder: Encoder | None = None,
        image_encoder: Encoder | None = None,
        device: str | None = None,
        batch_size: int = 32,
        dedupe_key: str | None = None,
    ) -> None:
        if dedupe_key not in (None, "entity", "doc"):
            raise ValueError(
                f"dedupe_key must be 'entity', 'doc' or None, got {dedupe_key!r}."
            )
        self.model_name = model_name
        self.device = device
        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self._model = None
        self.batch_size = max(1, batch_size)
        self.dedupe_key = dedupe_key
        self._units: list[_RetrievalUnit] = []
        self._vecs = None      # (n_units, dim) float32, L2-normalized
        self._index = None     # faiss.IndexFlatIP | None (numpy fallback)
        self._n_docs = 0

    # ------------------------------------------------------------------ #
    def build(self, corpus: Sequence[Document]) -> "UnifiedClipRetriever":
        """Index a corpus as unit-level vectors. Must precede search."""
        if not corpus:
            raise ValueError("Cannot build a unified CLIP index from an empty corpus.")
        import numpy as np

        self._n_docs = len(corpus)
        units: list[_RetrievalUnit] = []
        for doc in corpus:
            entity = str(
                getattr(doc, "entity_id", None) or doc.title or doc.doc_id
            )
            if f"{doc.title} {doc.text}".strip():
                units.append(_RetrievalUnit(
                    doc_id=doc.doc_id, modality="text", text=doc.text,
                    title=doc.title, image_path=None, entity_key=entity,
                ))
            if doc.image_path:
                units.append(_RetrievalUnit(
                    doc_id=doc.doc_id, modality="image", text="",
                    title=doc.title, image_path=doc.image_path,
                    entity_key=entity,
                ))
        if not units:
            raise ValueError("Corpus yielded no retrieval units (no text, no images).")

        vectors: list = [None] * len(units)
        text_ids = [i for i, u in enumerate(units) if u.modality == "text"]
        image_ids = [i for i, u in enumerate(units) if u.modality == "image"]

        for start in range(0, len(text_ids), self.batch_size):
            ids = text_ids[start:start + self.batch_size]
            encoded = self._encode_text(
                [f"{units[i].title} {units[i].text}".strip() for i in ids]
            )
            for i, vec in zip(ids, encoded):
                vectors[i] = vec

        bad: list[int] = []
        for start in range(0, len(image_ids), self.batch_size):
            ids = image_ids[start:start + self.batch_size]
            ok_ids, images = [], []
            for i in ids:
                try:
                    images.append(self._open_image(units[i].image_path))
                    ok_ids.append(i)
                except Exception:  # noqa: BLE001 - unreadable corpus image: drop unit
                    bad.append(i)
            if images:
                encoded = self._encode_images(images)
                for i, vec in zip(ok_ids, encoded):
                    vectors[i] = vec

        if bad:  # drop units whose image could not be read
            keep = [i for i in range(len(units)) if i not in set(bad)]
            units = [units[i] for i in keep]
            vectors = [vectors[i] for i in keep]
        if not units:
            raise ValueError("No retrieval unit could be encoded.")

        self._units = units
        self._vecs = np.asarray(vectors, dtype="float32")
        self._index = self._try_faiss(self._vecs)
        return self

    # ------------------------------------------------------------------ #
    def search_text(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Text query -> mixed top-k over ALL units in the unified space."""
        if self._vecs is None:
            raise RuntimeError("UnifiedClipRetriever.search_text called before build().")
        return self._search_vec(self._encode_text([query]), top_k)

    # TextRetriever protocol alias (so the executor can use this retriever
    # directly as its text_search backend).
    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        return self.search_text(query, top_k=top_k)

    def search_image(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Image (or region) query -> mixed top-k over ALL units."""
        if self._vecs is None:
            raise RuntimeError("UnifiedClipRetriever.search_image called before build().")
        img = self._open_image(image_path)
        if region is not None:
            box = safe_crop_box(img.size, region)
            if box is not None:
                img = img.crop(box)
        return self._search_vec(self._encode_images([img]), top_k)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._units)

    @property
    def n_docs(self) -> int:
        return self._n_docs

    @property
    def text_unit_count(self) -> int:
        return sum(1 for u in self._units if u.modality == "text")

    @property
    def image_unit_count(self) -> int:
        return sum(1 for u in self._units if u.modality == "image")

    # ------------------------------------------------------------------ #
    # Persistence (offline index caching; vectors stored as .npy so loading
    # never requires faiss)
    # ------------------------------------------------------------------ #
    def save(self, dir_path: str | Path) -> Path:
        import numpy as np

        if self._vecs is None:
            raise RuntimeError("Nothing to save: build() the index first.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "vectors.npy", self._vecs)
        with (dir_path / "units.jsonl").open("w", encoding="utf-8") as f:
            for u in self._units:
                f.write(json.dumps({
                    "doc_id": u.doc_id, "modality": u.modality, "text": u.text,
                    "title": u.title, "image_path": u.image_path,
                    "entity_key": u.entity_key,
                }, ensure_ascii=False) + "\n")
        (dir_path / "meta.json").write_text(
            json.dumps({
                "model_name": self.model_name,
                "n_docs": self._n_docs,
                "n_units": len(self._units),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return dir_path

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        model_name: str = "clip-ViT-B-32",
        text_encoder: Encoder | None = None,
        image_encoder: Encoder | None = None,
        device: str | None = None,
        batch_size: int = 32,
        dedupe_key: str | None = None,
    ) -> "UnifiedClipRetriever":
        import numpy as np

        dir_path = Path(dir_path)
        inst = cls(
            model_name=model_name, text_encoder=text_encoder,
            image_encoder=image_encoder, device=device,
            batch_size=batch_size, dedupe_key=dedupe_key,
        )
        inst._vecs = np.load(dir_path / "vectors.npy").astype("float32")
        with (dir_path / "units.jsonl").open("r", encoding="utf-8") as f:
            inst._units = [
                _RetrievalUnit(**json.loads(line)) for line in f if line.strip()
            ]
        meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        inst._n_docs = int(meta.get("n_docs", 0))
        inst._index = inst._try_faiss(inst._vecs)
        return inst

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _search_vec(self, qv, top_k: int) -> list[RetrievalHit]:
        n = len(self._units)
        top_k = max(1, top_k)
        # Over-fetch when deduping so unique keys can still fill top_k.
        fetch = n if self.dedupe_key else min(n, top_k)
        scores, ids = self._raw_search(qv, fetch)
        hits: list[RetrievalHit] = []
        seen_keys: set[str] = set()
        for score, i in zip(scores, ids):
            if i < 0:
                continue
            u = self._units[int(i)]
            if self.dedupe_key:
                key = u.entity_key if self.dedupe_key == "entity" else u.doc_id
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            hits.append(RetrievalHit(
                doc_id=u.doc_id, text=u.text, score=float(score),
                title=u.title, image_path=u.image_path,
                result_modality=u.modality,
            ))
            if len(hits) >= top_k:
                break
        return hits

    def _raw_search(self, qv, fetch: int):
        import numpy as np

        if self._index is not None:
            scores, ids = self._index.search(qv, fetch)
            return scores[0], ids[0]
        sims = (self._vecs @ np.asarray(qv, dtype="float32")[0])
        order = np.argsort(-sims)[:fetch]
        return sims[order], order

    @staticmethod
    def _try_faiss(vecs):
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index

    def _encode_text(self, texts: Sequence[str]):
        return self._normalize(self._text_encoder, texts)

    def _encode_images(self, images: Sequence):
        return self._normalize(self._image_encoder, images)

    def _normalize(self, encoder: Encoder | None, items: Sequence):
        import numpy as np

        if encoder is not None:
            vecs = np.asarray(encoder(items), dtype="float32")
        else:
            vecs = self._st_model().encode(
                list(items), convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    def _st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            key = (self.model_name, self.device)
            if key not in _SENTENCE_MODEL_CACHE:
                _SENTENCE_MODEL_CACHE[key] = SentenceTransformer(
                    self.model_name, device=self.device
                )
            self._model = _SENTENCE_MODEL_CACHE[key]
        return self._model

    @staticmethod
    def _open_image(path: str):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return Image.open(path).convert("RGB")


class HybridTextRetriever:
    """Rank-interleave BM25 lexical hits with unified CLIP text-query hits.

    BM25 scores and CLIP cosines are not on a comparable scale, so the merge is
    rank-based: alternate lists (lexical first), dropping duplicate units.
    """

    def __init__(self, bm25: BM25Retriever, unified: UnifiedClipRetriever) -> None:
        self.bm25 = bm25
        self.unified = unified

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        lex = self.bm25.search(query, top_k=top_k)
        sem = self.unified.search_text(query, top_k=top_k)
        merged: list[RetrievalHit] = []
        seen: set[tuple[str, str | None]] = set()
        for pair in zip(lex + [None] * len(sem), sem + [None] * len(lex)):
            for hit in pair:
                if hit is None:
                    continue
                key = (hit.doc_id, hit.result_modality)
                if key in seen or (hit.doc_id, None) in seen:
                    continue
                seen.add(key)
                merged.append(hit)
                if len(merged) >= top_k:
                    return merged
        return merged












