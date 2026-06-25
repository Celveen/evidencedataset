"""Retrieval action space: definitions, retrievers, and executor (Stage 1).

Two retrieval actions split by QUERY modality (+ answer):
    text_search(query)            — query the corpus with text  (BM25/dense)
    image_search(image, region?)  — query the corpus with the image (CLIP)
    answer(text)                  — terminal
"""

from .action_space import (
    ACTION_REGISTRY,
    Action,
    AnswerAction,
    Evidence,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
    register_action,
)
from .executor import ActionExecutor
from .retrievers import (
    BM25Retriever,
    ClipImageRetriever,
    DenseRetriever,
    RetrievalHit,
)

__all__ = [
    "ACTION_REGISTRY",
    "Action",
    "ActionExecutor",
    "AnswerAction",
    "BM25Retriever",
    "ClipImageRetriever",
    "DenseRetriever",
    "Evidence",
    "ImageSearchAction",
    "RetrievalHit",
    "SearchState",
    "TextSearchAction",
    "register_action",
]
