"""Action executor: (state, action) -> new state with an updated evidence bundle.

All retrieval is OFFLINE on local indices — no web API (implementation report
§4.1.4). The executor is the single place where actions touch retrievers, so
the search code never needs to know how an action is carried out.

Extensibility: new action types (crop / zoom / focus, pending Stage 0.4) plug
in via :meth:`ActionExecutor.register_handler` — no changes to existing code.
"""

from __future__ import annotations

from typing import Callable, Protocol

from .action_space import (
    Action,
    AnswerAction,
    Evidence,
    ImageSearchAction,
    ImageToTextAction,
    SearchState,
    TextSearchAction,
    TextToImageAction,
)


class TextRetriever(Protocol):
    """Anything with ``search(query, top_k) -> list[RetrievalHit]``."""

    def search(self, query: str, top_k: int = 5): ...


class ImageRetriever(Protocol):
    """CLIP retriever for image-involving actions.

    ``search_image`` (image->image) is required; ``text_to_image`` and
    ``image_to_text`` are needed only if those cross-modal actions are used
    (see :class:`~evidencetree.actions.retrievers.CrossModalCLIPRetriever`).
    """

    def search_image(self, image_path: str, region=None, top_k: int = 5): ...


Handler = Callable[[SearchState, Action], SearchState]


class ActionExecutor:
    """Executes typed actions against local retrievers."""

    def __init__(
        self,
        text_retriever: TextRetriever | None = None,
        image_retriever: ImageRetriever | None = None,
        top_k: int = 5,
    ) -> None:
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.top_k = top_k
        self._handlers: dict[type[Action], Handler] = {
            TextSearchAction: self._exec_text_search,
            ImageSearchAction: self._exec_image_search,
            TextToImageAction: self._exec_text_to_image,
            ImageToTextAction: self._exec_image_to_text,
            AnswerAction: self._exec_answer,
        }

    # ------------------------------------------------------------------ #
    def register_handler(self, action_cls: type[Action], handler: Handler) -> None:
        """Plug in a handler for a new action type (e.g. crop/zoom later)."""
        self._handlers[action_cls] = handler

    def execute(self, state: SearchState, action: Action) -> SearchState:
        """Run ``action`` from ``state``; return the successor state."""
        if state.is_terminal:
            raise ValueError("Cannot execute an action from a terminal state.")
        handler = self._handlers.get(type(action))
        if handler is None:
            raise TypeError(
                f"No handler registered for action type {type(action).__name__}. "
                "Use register_handler() to add one."
            )
        return handler(state, action)

    # ------------------------------------------------------------------ #
    # Built-in handlers
    # ------------------------------------------------------------------ #
    def _exec_text_search(self, state: SearchState, action: Action) -> SearchState:
        assert isinstance(action, TextSearchAction)
        if self.text_retriever is None:
            raise RuntimeError("ActionExecutor has no text_retriever configured.")
        hits = self.text_retriever.search(action.query, top_k=self.top_k)
        return state.advanced(action, self._hits_to_evidence(state, action, hits))

    def _exec_image_search(self, state: SearchState, action: Action) -> SearchState:
        assert isinstance(action, ImageSearchAction)
        if self.image_retriever is None:
            raise RuntimeError(
                "ActionExecutor has no image_retriever configured "
                "(image_search needs a CLIP-style index, see retrievers.py)."
            )
        image_path = action.image_path or state.image_path
        if image_path is None:
            raise ValueError("image_search with no image: neither the action nor "
                             "the state carries an image_path.")
        hits = self.image_retriever.search_image(
            image_path, region=action.region, top_k=self.top_k
        )
        return state.advanced(action, self._hits_to_evidence(state, action, hits))

    def _exec_text_to_image(self, state: SearchState, action: Action) -> SearchState:
        assert isinstance(action, TextToImageAction)
        retriever = self._require_cross_modal("text_to_image", "text_to_image")
        hits = retriever.text_to_image(action.query, top_k=self.top_k)
        return state.advanced(action, self._hits_to_evidence(state, action, hits))

    def _exec_image_to_text(self, state: SearchState, action: Action) -> SearchState:
        assert isinstance(action, ImageToTextAction)
        retriever = self._require_cross_modal("image_to_text", "image_to_text")
        image_path = action.image_path or state.image_path
        if image_path is None:
            raise ValueError("image_to_text with no image: neither the action nor "
                             "the state carries an image_path.")
        hits = retriever.image_to_text(image_path, region=action.region, top_k=self.top_k)
        return state.advanced(action, self._hits_to_evidence(state, action, hits))

    def _require_cross_modal(self, action_name: str, method: str):
        if self.image_retriever is None:
            raise RuntimeError(
                f"ActionExecutor has no image_retriever configured ({action_name} "
                "needs a CrossModalCLIPRetriever, see retrievers.py)."
            )
        if not hasattr(self.image_retriever, method):
            raise TypeError(
                f"{action_name} needs a retriever with a {method}() method "
                "(use CrossModalCLIPRetriever)."
            )
        return self.image_retriever

    def _exec_answer(self, state: SearchState, action: Action) -> SearchState:
        assert isinstance(action, AnswerAction)
        return state.answered(action)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hits_to_evidence(state: SearchState, action: Action, hits) -> list[Evidence]:
        base = len(state.evidence)
        step = len(state.actions_taken)  # index this action will occupy
        return [
            Evidence(
                evidence_id=f"e{base + i}",
                step_index=step,
                source_action=action.action_type,
                doc_id=h.doc_id,
                text=h.text,
                title=getattr(h, "title", ""),
                image_path=getattr(h, "image_path", None),
                score=float(getattr(h, "score", 0.0)),
            )
            for i, h in enumerate(hits)
        ]
