"""Retriever wrappers.

* ``BM25Retriever`` — real, lexical retrieval over the local corpus
  (``rank-bm25``). Used by Stage 0.1's vanilla RAG and by Stage 1.
* ``DenseRetriever`` — placeholder for Stage 1 (sentence-transformers / CLIP
  + FAISS). Raises ``NotImplementedError`` until then.

All retrieval is OFFLINE on the benchmark's own corpus — no web API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from ..eval.benchmarks import Document


@dataclass
class RetrievalHit:
    """A single retrieval result."""

    doc_id: str
    text: str
    score: float
    title: str = ""


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
            RetrievalHit(
                doc_id=self._docs[i].doc_id,
                text=self._docs[i].text,
                score=float(scores[i]),
                title=self._docs[i].title,
            )
            for i in ranked
        ]

    def __len__(self) -> int:
        return len(self._docs)


class DenseRetriever:
    """Dense / multimodal retriever (sentence-transformers / CLIP + FAISS).

    Placeholder — implemented in Stage 1 (see implementation report §Stage 1).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DenseRetriever is implemented in Stage 1 (dense/CLIP + FAISS). "
            "Stage 0.1 only uses BM25Retriever."
        )
