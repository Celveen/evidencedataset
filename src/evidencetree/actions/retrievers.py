"""Retriever wrappers.

* ``BM25Retriever`` — lexical retrieval over the local corpus (``rank-bm25``).
* ``DenseRetriever`` — dense retrieval: encoder embeddings + FAISS inner
  product. The encoder is injectable (tests pass a deterministic fake); the
  default lazily loads a sentence-transformers model.
* ``UnifiedClipRetriever`` — THE design-intended retriever (report §3.1/§3.2):
  one unified CLIP space where corpus text chunks and corpus images are
  independent retrieval units; ``search_text`` / ``search_image`` both return
  mixed top-k over all units, each hit tagged with ``result_modality``.
  Optional entity/doc dedupe before top-k truncation.
* ``HybridTextRetriever`` — rank-interleaves BM25 with unified text-query hits.
* ``ClipImageRetriever`` — legacy doc-level image->document retrieval (one
  vector per Document: image if present, else text). Kept for offline index
  scripts; the pipeline uses ``UnifiedClipRetriever``.
* ``CrossModalCLIPRetriever`` — separate CLIP image/text sub-indexes for
  text->image, image->text, and image->image retrieval.
* ``CragWebRetriever`` / ``CragImageRetriever`` — query the official local
  CRAG-MM validation Chroma indexes with their matching embedding models.
* ``TesseractOcrEngine`` — read text from one image/region; not a Top-K
  retriever.

All retrieval is OFFLINE on the benchmark's own corpus — no web API.
Heavy deps (faiss / sentence-transformers / PIL) are imported lazily so the
mock smoke tests never need them.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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


class TesseractOcrEngine:
    """Thin wrapper around the ``tesseract`` CLI.

    OCR is modelled as a one-image reader, not a retriever: ``read_text``
    returns recognized text for the selected image or region.
    """

    def __init__(
        self,
        language: str = "eng",
        psm: int = 3,
        command: str = "tesseract",
    ) -> None:
        self.language = language
        self.psm = int(psm)
        self.command = command

    def read_text(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
    ) -> str:
        source = Path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"OCR image not found: {source}")
        run_path = source
        tmp_name = None
        try:
            if region is not None:
                cropped = self._crop_to_temp(source, region)
                if cropped is not None:
                    run_path, tmp_name = cropped
            proc = subprocess.run(
                [
                    self.command,
                    str(run_path),
                    "stdout",
                    "-l",
                    self.language,
                    "--psm",
                    str(self.psm),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "tesseract failed")
            return proc.stdout
        finally:
            if tmp_name:
                Path(tmp_name).unlink(missing_ok=True)

    @staticmethod
    def _crop_to_temp(
        image_path: Path,
        region: tuple[float, float, float, float],
    ) -> tuple[Path, str] | None:
        from PIL import Image

        with Image.open(image_path).convert("RGB") as image:
            box = safe_crop_box(image.size, region)
            if box is None:
                return None
            crop = image.crop(box)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_name = tmp.name
            tmp.close()
            crop.save(tmp_name)
        return Path(tmp_name), tmp_name


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


class DenseRetriever:
    """Dense retriever: encoder embeddings + FAISS inner-product index.

    Args:
        model_name: sentence-transformers model id (used only when no custom
            ``encoder`` is given; loaded lazily on first encode).
        encoder: optional callable ``(list[str]) -> (n, dim) array``. Inject a
            deterministic fake in tests to avoid model downloads.
        device: forwarded to sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Encoder | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._encoder = encoder
        self._model = None
        self._docs: list[Document] = []
        self._index = None  # faiss.IndexFlatIP

    # ------------------------------------------------------------------ #
    def build(self, corpus: Sequence[Document]) -> "DenseRetriever":
        """Encode and index a corpus. Must be called before :meth:`search`."""
        if not corpus:
            raise ValueError("Cannot build a dense index from an empty corpus.")
        self._docs = list(corpus)
        vecs = self._encode([f"{d.title} {d.text}".strip() for d in self._docs])
        self._index = self._new_index(vecs)
        return self

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("DenseRetriever.search called before build()/load().")
        qv = self._encode([query])
        return self._search_vec(qv, top_k)

    # ------------------------------------------------------------------ #
    # Persistence (used by scripts/build_index.py — offline, one-time)
    # ------------------------------------------------------------------ #
    def save(self, dir_path: str | Path) -> Path:
        """Write ``index.faiss`` + ``docs.jsonl`` into ``dir_path``."""
        import faiss

        if self._index is None:
            raise RuntimeError("Nothing to save: build() the index first.")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "index.faiss"))
        with (dir_path / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps({
                    "doc_id": d.doc_id, "text": d.text, "title": d.title,
                    "image_path": d.image_path,
                }, ensure_ascii=False) + "\n")
        return dir_path

    @classmethod
    def load(
        cls,
        dir_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Encoder | None = None,
        device: str | None = None,
    ) -> "DenseRetriever":
        import faiss

        dir_path = Path(dir_path)
        inst = cls(model_name=model_name, encoder=encoder, device=device)
        inst._index = faiss.read_index(str(dir_path / "index.faiss"))
        with (dir_path / "docs.jsonl").open("r", encoding="utf-8") as f:
            inst._docs = [Document(**json.loads(line)) for line in f if line.strip()]
        return inst

    def __len__(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _encode(self, items: Sequence):
        import numpy as np

        if self._encoder is not None:
            vecs = np.asarray(self._encoder(items), dtype="float32")
        else:
            vecs = self._st_model().encode(
                list(items), convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
        # L2-normalize so inner product == cosine similarity.
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
    def _new_index(vecs):
        import faiss

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index

    def _search_vec(self, qv, top_k: int) -> list[RetrievalHit]:
        scores, ids = self._index.search(qv, min(top_k, len(self._docs)))
        return [
            _hit(self._docs[i], float(s))
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]


class ClipImageRetriever(DenseRetriever):
    """Image->document retrieval in CLIP's shared text/image embedding space.

    Corpus side: a doc embeds via its image when ``image_path`` is set, else
    via its text (CLIP text tower, truncated to its token limit). Query side:
    an image (optionally cropped to a normalized bbox region first).

    For tests, inject ``encoder`` (used for text) and ``image_encoder``; the
    default lazily loads one CLIP model for both.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        encoder: Encoder | None = None,
        image_encoder: Encoder | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__(model_name=model_name, encoder=encoder, device=device)
        self._image_encoder = image_encoder
        self.batch_size = max(1, batch_size)

    def build(self, corpus: Sequence[Document]) -> "ClipImageRetriever":
        if not corpus:
            raise ValueError("Cannot build a CLIP index from an empty corpus.")
        import numpy as np

        self._docs = list(corpus)
        vectors: list[Any | None] = [None] * len(self._docs)
        image_indices = [i for i, doc in enumerate(self._docs) if doc.image_path]
        text_indices = [i for i, doc in enumerate(self._docs) if not doc.image_path]

        for start in range(0, len(image_indices), self.batch_size):
            indices = image_indices[start:start + self.batch_size]
            valid_indices = []
            images = []
            fallback_indices = []
            for index in indices:
                try:
                    images.append(self._open_image(self._docs[index].image_path))
                    valid_indices.append(index)
                except Exception:  # noqa: BLE001 - bad corpus image falls back to text
                    fallback_indices.append(index)
            if images:
                encoded = self._encode_images(images)
                for index, vector in zip(valid_indices, encoded):
                    vectors[index] = vector
            if fallback_indices:
                texts = [
                    f"{self._docs[i].title} {self._docs[i].text}".strip()
                    for i in fallback_indices
                ]
                encoded = self._encode(texts)
                for index, vector in zip(fallback_indices, encoded):
                    vectors[index] = vector

        for start in range(0, len(text_indices), self.batch_size):
            indices = text_indices[start:start + self.batch_size]
            texts = [
                f"{self._docs[i].title} {self._docs[i].text}".strip()
                for i in indices
            ]
            encoded = self._encode(texts)
            for index, vector in zip(indices, encoded):
                vectors[index] = vector

        self._index = self._new_index(np.asarray(vectors, dtype="float32"))
        return self

    def search_image(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        """Retrieve with an image (or a normalized-bbox region of it) as query."""
        if self._index is None:
            raise RuntimeError("ClipImageRetriever.search_image called before build().")
        img = self._open_image(image_path)
        if region is not None:
            box = safe_crop_box(img.size, region)
            if box is not None:
                img = img.crop(box)
        return self._search_vec(self._encode_images([img]), top_k)

    # ------------------------------------------------------------------ #
    def _encode_images(self, images: Sequence):
        import numpy as np

        if self._image_encoder is not None:
            vecs = np.asarray(self._image_encoder(images), dtype="float32")
        else:
            vecs = self._st_model().encode(
                list(images), convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    @staticmethod
    def _open_image(path: str):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return Image.open(path).convert("RGB")


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


class CrossModalCLIPRetriever:
    """CLIP cross-modal retrieval over separate image and text sub-indexes.

    Supported directions:

    * ``text_to_image(query)``: text query -> corpus images.
    * ``image_to_text(image, region)``: image query -> corpus text docs.
    * ``search_image(image, region)``: image query -> corpus images.

    The image sub-index contains docs with ``image_path``. The text sub-index
    contains any doc with meaningful ``text``. A document may therefore appear
    in both sub-indexes when it has an image plus weak labels/captions.
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        text_encoder: Encoder | None = None,
        image_encoder: Encoder | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self._model = None
        self.batch_size = max(1, batch_size)
        self._image_docs: list[Document] = []
        self._text_docs: list[Document] = []
        self._image_index = None
        self._text_index = None

    def build(self, corpus: Sequence[Document]) -> "CrossModalCLIPRetriever":
        if not corpus:
            raise ValueError("Cannot build a cross-modal index from an empty corpus.")
        self._image_docs = [doc for doc in corpus if doc.image_path]
        self._text_docs = [doc for doc in corpus if doc.text]
        if self._image_docs:
            vecs = []
            for start in range(0, len(self._image_docs), self.batch_size):
                docs = self._image_docs[start:start + self.batch_size]
                images = [self._open_image(doc.image_path) for doc in docs]
                vecs.append(self._encode_images(images))
            self._image_index = self._new_index_from_batches(vecs)
        if self._text_docs:
            vecs = []
            for start in range(0, len(self._text_docs), self.batch_size):
                docs = self._text_docs[start:start + self.batch_size]
                texts = [f"{doc.title} {doc.text}".strip() for doc in docs]
                vecs.append(self._encode_text(texts))
            self._text_index = self._new_index_from_batches(vecs)
        return self

    def text_to_image(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if self._image_index is None:
            return []
        return self._search(
            self._image_index,
            self._image_docs,
            self._encode_text([query]),
            top_k,
        )

    def image_to_text(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if self._text_index is None:
            return []
        return self._search(
            self._text_index,
            self._text_docs,
            self._encode_images([self._crop(image_path, region)]),
            top_k,
        )

    def search_image(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        if self._image_index is None:
            return []
        return self._search(
            self._image_index,
            self._image_docs,
            self._encode_images([self._crop(image_path, region)]),
            top_k,
        )

    def __len__(self) -> int:
        return len({doc.doc_id for doc in [*self._image_docs, *self._text_docs]})

    @property
    def has_image_index(self) -> bool:
        """Whether text->image / image->image CLIP retrieval has targets."""
        return self._image_index is not None and bool(self._image_docs)

    @property
    def has_text_index(self) -> bool:
        """Whether image->text CLIP retrieval has targets."""
        return self._text_index is not None and bool(self._text_docs)

    @property
    def image_doc_count(self) -> int:
        return len(self._image_docs)

    @property
    def text_doc_count(self) -> int:
        return len(self._text_docs)

    def save(self, dir_path: str | Path) -> Path:
        """Write cross-modal CLIP sub-indexes and docs into ``dir_path``."""
        import faiss

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        if self._image_index is not None:
            faiss.write_index(self._image_index, str(dir_path / "image_index.faiss"))
        if self._text_index is not None:
            faiss.write_index(self._text_index, str(dir_path / "text_index.faiss"))
        self._write_docs(dir_path / "image_docs.jsonl", self._image_docs)
        self._write_docs(dir_path / "text_docs.jsonl", self._text_docs)
        (dir_path / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "image_docs": len(self._image_docs),
                    "text_docs": len(self._text_docs),
                },
                ensure_ascii=False,
                indent=2,
            ),
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
    ) -> "CrossModalCLIPRetriever":
        import faiss

        dir_path = Path(dir_path)
        inst = cls(
            model_name=model_name,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            device=device,
            batch_size=batch_size,
        )
        image_index = dir_path / "image_index.faiss"
        text_index = dir_path / "text_index.faiss"
        if image_index.exists():
            inst._image_index = faiss.read_index(str(image_index))
        if text_index.exists():
            inst._text_index = faiss.read_index(str(text_index))
        inst._image_docs = cls._read_docs(dir_path / "image_docs.jsonl")
        inst._text_docs = cls._read_docs(dir_path / "text_docs.jsonl")
        return inst

    @staticmethod
    def _write_docs(path: Path, docs: Sequence[Document]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(
                    json.dumps(
                        {
                            "doc_id": doc.doc_id,
                            "text": doc.text,
                            "title": doc.title,
                            "image_path": doc.image_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    @staticmethod
    def _read_docs(path: Path) -> list[Document]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return [Document(**json.loads(line)) for line in f if line.strip()]

    @staticmethod
    def _search(index, docs: Sequence[Document], qv, top_k: int) -> list[RetrievalHit]:
        scores, ids = index.search(qv, min(top_k, len(docs)))
        return [_hit(docs[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    @staticmethod
    def _new_index_from_batches(batches):
        import numpy as np

        return DenseRetriever._new_index(np.vstack(batches))

    def _crop(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None,
    ):
        image = self._open_image(image_path)
        if region is not None:
            box = safe_crop_box(image.size, region)
            if box is not None:
                image = image.crop(box)
        return image

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
        from PIL import Image

        return Image.open(path).convert("RGB")


def _first_query_row(result: dict[str, Any], key: str) -> list[Any]:
    rows = result.get(key) or []
    return rows[0] if rows else []


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _compact_metadata(value: Any, limit: int = 1800) -> str:
    value = _json_value(value)
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return text[:limit]


class CragWebRetriever:
    """Text retrieval over CRAG-MM's official local web-search Chroma index."""

    def __init__(
        self,
        index_path: str | Path,
        model_name: str,
        device: str = "cpu",
        collection_name: str = "web_search_embeddings",
        encoder: Encoder | None = None,
        collection=None,
    ) -> None:
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.device = device
        self._encoder = encoder
        self._model = None
        self._tokenizer = None
        self._collection = collection or self._open_collection(collection_name)

    def _open_collection(self, name: str):
        import chromadb

        if not self.index_path.exists():
            raise FileNotFoundError(f"CRAG web index not found: {self.index_path}")
        return chromadb.PersistentClient(path=str(self.index_path)).get_collection(name)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        result = self._collection.query(
            query_embeddings=self._encode_texts([query]).tolist(),
            n_results=max(1, top_k),
            include=["metadatas", "distances"],
        )
        ids = _first_query_row(result, "ids")
        metadatas = _first_query_row(result, "metadatas")
        distances = _first_query_row(result, "distances")
        hits = []
        for item_id, metadata, distance in zip(ids, metadatas, distances):
            metadata = metadata or {}
            title = str(metadata.get("page_name") or "CRAG web result")
            snippet = str(metadata.get("page_snippet") or "")
            url = str(metadata.get("page_url") or "")
            text = snippet if not url else f"{snippet}\nSource: {url}"
            hits.append(
                RetrievalHit(
                    doc_id=f"crag-web:{item_id}",
                    title=title,
                    text=text,
                    score=1.0 - float(distance),
                )
            )
        return hits

    def _encode_texts(self, texts: Sequence[str]):
        import numpy as np

        if self._encoder is not None:
            vecs = np.asarray(self._encoder(texts), dtype="float32")
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self._model.eval()
            inputs = self._tokenizer(
                list(texts),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vecs = pooled.cpu().numpy().astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)


class CragImageRetriever:
    """Image retrieval over CRAG-MM's official local image Chroma index."""

    def __init__(
        self,
        index_path: str | Path,
        model_name: str,
        device: str = "cpu",
        collection_name: str = "image_embeddings",
        image_encoder: Encoder | None = None,
        collection=None,
    ) -> None:
        self.index_path = Path(index_path)
        self.model_name = model_name
        self.device = device
        self._image_encoder = image_encoder
        self._model = None
        self._processor = None
        self._collection = collection or self._open_collection(collection_name)

    def _open_collection(self, name: str):
        import chromadb

        if not self.index_path.exists():
            raise FileNotFoundError(f"CRAG image index not found: {self.index_path}")
        return chromadb.PersistentClient(path=str(self.index_path)).get_collection(name)

    def search_image(
        self,
        image_path: str,
        region: tuple[float, float, float, float] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        image = ClipImageRetriever._open_image(image_path)
        if region is not None:
            box = safe_crop_box(image.size, region)
            if box is not None:
                image = image.crop(box)
        result = self._collection.query(
            query_embeddings=self._encode_images([image]).tolist(),
            n_results=max(1, top_k),
            include=["metadatas", "distances"],
        )
        ids = _first_query_row(result, "ids")
        metadatas = _first_query_row(result, "metadatas")
        distances = _first_query_row(result, "distances")
        hits = []
        for item_id, metadata, distance in zip(ids, metadatas, distances):
            metadata = metadata or {}
            entities = _compact_metadata(metadata.get("entities"))
            info = _compact_metadata(metadata.get("info"))
            url = str(metadata.get("image_url") or "")
            pieces = [piece for piece in (entities, info) if piece]
            if url:
                pieces.append(f"Image source: {url}")
            hits.append(
                RetrievalHit(
                    doc_id=f"crag-image:{item_id}",
                    title="CRAG visual retrieval result",
                    text="\n".join(pieces) or "Visually similar indexed image.",
                    score=1.0 - float(distance),
                )
            )
        return hits

    def _encode_images(self, images: Sequence):
        import numpy as np

        if self._image_encoder is not None:
            vecs = np.asarray(self._image_encoder(images), dtype="float32")
        else:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            if self._model is None:
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self._model.eval()
            inputs = self._processor(images=list(images), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                vecs = self._model.get_image_features(pixel_values=pixel_values)
            vecs = vecs.cpu().numpy().astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)
