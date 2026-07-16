"""Retrieval action space: definitions, retrievers, and executor (Stage 1).

Compressed action space:
    text_search(query)            — query the corpus with text
    image_search(image, region?)  — query the shared CLIP corpus index
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
    CragImageRetriever,
    CragWebRetriever,
    DenseRetriever,
    HybridTextRetriever,
    RetrievalHit,
    UnifiedClipRetriever,
)

__all__ = [
    "ACTION_REGISTRY",
    "Action",
    "ActionExecutor",
    "AnswerAction",
    "BM25Retriever",
    "ClipImageRetriever",
    "CragImageRetriever",
    "CragWebRetriever",
    "DenseRetriever",
    "Evidence",
    "HybridTextRetriever",
    "ImageSearchAction",
    "RetrievalHit",
    "SearchState",
    "TextSearchAction",
    "UnifiedClipRetriever",
    "register_action",
]
