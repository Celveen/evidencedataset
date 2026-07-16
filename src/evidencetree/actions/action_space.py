"""Retrieval action space (Stage 1, v1.3 compressed set).

Typed actions over which the MCTS searches:

    text_search(query)              — text retrieval          [required]
    image_search(image, region?)    — image-query CLIP retrieval over a
                                      shared text/image corpus index
    answer(text)                    — terminal action         [required]

Deferred until Stage 0.4 statistics: crop / zoom / focus. New action types are
added by subclassing :class:`Action` with ``@register_action`` and registering
a handler on the executor — no rewrite of search code needed.

Do NOT add parallel_search: tree expansion already covers multi-candidate
exploration, and its backup semantics would conflict with tree-level credit
(implementation report §4.1.6).

This module also defines the data the tree carries:

* :class:`Evidence` — one retrieved item, tagged with the step that fetched it.
* :class:`SearchState` — immutable (question, image, evidence_bundle,
  actions_taken, final_answer). Executing an action returns a *new* state, so
  MCTS nodes can share ancestor states safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
ACTION_REGISTRY: dict[str, type["Action"]] = {}


def register_action(cls: type["Action"]) -> type["Action"]:
    """Class decorator: make an action type discoverable by name."""
    ACTION_REGISTRY[cls.action_type] = cls
    return cls


@dataclass(frozen=True)
class Action:
    """Base typed action. Subclasses are frozen dataclasses (hashable,
    comparable — expansion dedupes identical candidates via ``==``)."""

    action_type: ClassVar[str] = "abstract"
    modality: ClassVar[str] = "n/a"  # "text" | "image" | "answer"

    @property
    def granularity(self) -> str:
        """Visual granularity of the action ("whole" | "region" | "n/a")."""
        return "n/a"

    def describe(self) -> str:
        return self.action_type


@register_action
@dataclass(frozen=True)
class TextSearchAction(Action):
    """Retrieve documents from the local text index."""

    query: str = ""

    action_type: ClassVar[str] = "text_search"
    modality: ClassVar[str] = "text"

    def describe(self) -> str:
        return f"text_search({self.query!r})"


@register_action
@dataclass(frozen=True)
class ImageSearchAction(Action):
    """Retrieve from the corpus using an image (or region) as the query.

    The corpus side lives in one shared CLIP space: documents with images are
    embedded by the image tower; text-only documents are embedded by the text
    tower. Therefore this action may retrieve image-side or text-side docs.

    ``image_path=None`` means "use the state's own image". ``region`` is an
    optional normalized bbox (x1, y1, x2, y2) in [0, 1].
    """

    image_path: str | None = None
    region: tuple[float, float, float, float] | None = None

    action_type: ClassVar[str] = "image_search"
    modality: ClassVar[str] = "image"

    @property
    def granularity(self) -> str:
        return "region" if self.region is not None else "whole"

    def describe(self) -> str:
        src = self.image_path or "<state image>"
        return f"image_search({src}, region={self.region})"


@register_action
@dataclass(frozen=True)
class AnswerAction(Action):
    """Terminal action: commit to a final answer."""

    text: str = ""

    action_type: ClassVar[str] = "answer"
    modality: ClassVar[str] = "answer"

    def describe(self) -> str:
        return f"answer({self.text!r})"


# --------------------------------------------------------------------------- #
# Evidence & state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Evidence:
    """One retrieved item in the evidence bundle."""

    evidence_id: str            # "e0", "e1", ... unique within a trajectory
    step_index: int             # index into SearchState.actions_taken
    source_action: str          # action_type that fetched it
    doc_id: str
    text: str
    title: str = ""
    image_path: str | None = None
    score: float = 0.0
    result_modality: str | None = None  # modality of the retrieved unit
                                        # ("text" | "image"; None = legacy
                                        # doc-level retriever)


@dataclass(frozen=True)
class SearchState:
    """Immutable partial-retrieval state: an MCTS node's payload.

    The evidence bundle grows monotonically along a trajectory (search A, then
    search B -> bundle holds both), which is how serial trajectories express
    multi-source evidence fusion (no parallel_search).
    """

    question: str
    image_path: str | None = None
    evidence: tuple[Evidence, ...] = ()
    actions_taken: tuple[Action, ...] = ()
    final_answer: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.final_answer is not None

    @property
    def depth(self) -> int:
        return len(self.actions_taken)

    def evidence_texts(self) -> list[str]:
        return [
            f"{e.title}: {e.text}" if e.title else e.text for e in self.evidence
        ]

    def evidence_image_paths(self) -> list[str]:
        """Unique images returned by retrieval, in evidence order."""
        paths = []
        for evidence in self.evidence:
            if evidence.image_path and evidence.image_path not in paths:
                paths.append(evidence.image_path)
        return paths

    def advanced(self, action: Action, new_evidence: list[Evidence]) -> "SearchState":
        """New state after a retrieval action appended ``new_evidence``."""
        return SearchState(
            question=self.question,
            image_path=self.image_path,
            evidence=self.evidence + tuple(new_evidence),
            actions_taken=self.actions_taken + (action,),
            final_answer=self.final_answer,
        )

    def answered(self, action: AnswerAction) -> "SearchState":
        """Terminal state after an answer action."""
        return SearchState(
            question=self.question,
            image_path=self.image_path,
            evidence=self.evidence,
            actions_taken=self.actions_taken + (action,),
            final_answer=action.text,
        )
