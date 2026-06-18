"""Retriever wrappers.

* ``BM25Retriever`` — lexical retrieval over the local corpus (``rank-bm25``).
* ``DenseRetriever`` — dense retrieval: encoder embeddings + FAISS inner
  product. The encoder is injectable (tests pass a deterministic fake); the
  default lazily loads a sentence-transformers model.
* ``ClipImageRetriever`` — image->document retrieval in CLIP's shared
  embedding space (query = image or image region; corpus side = doc image if
  present, else doc text).

All retrieval is OFFLINE on the benchmark's own corpus — no web API.
Heavy deps (faiss / sentence-transformers / PIL) are imported lazily so the
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

# An encoder maps a list of inputs (str or PIL.Image) to a 2D array (n, dim).
Encoder = Callable[[Sequence], "object"]


@dataclass
class RetrievalHit:
    """A single retrieval result."""

    doc_id: str
    text: str
    score: float
    title: str = ""
    image_path: str | None = None


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
        return [_hit(self._docs[i], float(scores[i])) for i in ranked]

    def __len__(self) -> int:
        return len(self._docs)


def _hit(doc: Document, score: float) -> RetrievalHit:
    return RetrievalHit(
        doc_id=doc.doc_id,
        text=doc.text,
        score=score,
        title=doc.title,
        image_path=doc.image_path,
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

            self._model = SentenceTransformer(self.model_name, device=self.device)
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
    ) -> None:
        super().__init__(model_name=model_name, encoder=encoder, device=device)
        self._image_encoder = image_encoder

    def build(self, corpus: Sequence[Document]) -> "ClipImageRetriever":
        if not corpus:
            raise ValueError("Cannot build a CLIP index from an empty corpus.")
        import numpy as np

        self._docs = list(corpus)
        parts = []
        for d in self._docs:
            if d.image_path:
                parts.append(self._encode_images([self._open_image(d.image_path)]))
            else:
                parts.append(self._encode([f"{d.title} {d.text}".strip()]))
        self._index = self._new_index(np.vstack(parts))
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
            w, h = img.size
            x1, y1, x2, y2 = region
            img = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
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
        from PIL import Image

        return Image.open(path).convert("RGB")


class CrossModalCLIPRetriever:
    """CLIP cross-modal retrieval over two separate sub-indexes.

    Builds an **image sub-index** (docs with ``image_path``, embedded via the
    CLIP image tower) and a **text sub-index** (docs with text, CLIP text
    tower). Because both towers share one embedding space, a text query can
    search images and an image query can search text:

        text_to_image(query)            text query  -> image sub-index
        image_to_text(image, region)    image query -> text  sub-index
        search_image(image, region)     image query -> image sub-index (=image_search)

    A doc that has an image is indexed on the image side; a text-only doc on the
    text side. On a text-only corpus the image sub-index is empty, so
    text_to_image / search_image return [] (documented limitation — needs
    images in the corpus). Encoders are injectable for tests (no downloads).
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        text_encoder: Encoder | None = None,
        image_encoder: Encoder | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._text_encoder = text_encoder
        self._image_encoder = image_encoder
        self._model = None
        self._image_docs: list[Document] = []
        self._text_docs: list[Document] = []
        self._image_index = None
        self._text_index = None

    # ------------------------------------------------------------------ #
    def build(self, corpus: Sequence[Document]) -> "CrossModalCLIPRetriever":
        if not corpus:
            raise ValueError("Cannot build a cross-modal index from an empty corpus.")
        self._image_docs = [d for d in corpus if d.image_path]
        self._text_docs = [d for d in corpus if d.text and not d.image_path]
        if self._image_docs:
            imgs = [self._open_image(d.image_path) for d in self._image_docs]
            self._image_index = self._new_index(self._encode_images(imgs))
        if self._text_docs:
            txts = [f"{d.title} {d.text}".strip() for d in self._text_docs]
            self._text_index = self._new_index(self._encode_text(txts))
        return self

    # ------------------------------------------------------------------ #
    def text_to_image(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if self._image_index is None:
            return []  # no images in the corpus
        return self._search(self._image_index, self._image_docs,
                            self._encode_text([query]), top_k)

    def image_to_text(self, image_path, region=None, top_k: int = 5) -> list[RetrievalHit]:
        if self._text_index is None:
            return []
        qv = self._encode_images([self._crop(image_path, region)])
        return self._search(self._text_index, self._text_docs, qv, top_k)

    def search_image(self, image_path, region=None, top_k: int = 5) -> list[RetrievalHit]:
        """image -> image (the image_search action)."""
        if self._image_index is None:
            return []
        qv = self._encode_images([self._crop(image_path, region)])
        return self._search(self._image_index, self._image_docs, qv, top_k)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _search(index, docs, qv, top_k: int) -> list[RetrievalHit]:
        scores, ids = index.search(qv, min(top_k, len(docs)))
        return [_hit(docs[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def _crop(self, image_path: str, region):
        img = self._open_image(image_path)
        if region is not None:
            w, h = img.size
            x1, y1, x2, y2 = region
            img = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        return img

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
    def _new_index(vecs):
        import faiss

        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index

    @staticmethod
    def _open_image(path: str):
        from PIL import Image

        return Image.open(path).convert("RGB")
