"""Candidate-action proposers for MCTS expansion / simulation.

A proposer answers "from this state, what could we do next?". The searcher
then lets the PRM rank the candidates and keeps the top-k as children.

Two implementations:

* :class:`HeuristicProposer` — LLM-free templates (query reformulation,
  optional image search, answer from accumulated evidence). Enough for mock
  runs, unit tests, and Stage 0.2's simplified MCTS.
* :class:`LLMProposer` — prompts a policy model (any
  ``evidencetree.generation`` backend) to emit candidate actions as JSON;
  falls back to the heuristics whenever parsing fails. Used on the GPU server.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Protocol, Sequence

from evidencetree.actions.action_space import (
    Action,
    AnswerAction,
    ImageSearchAction,
    SearchState,
    TextSearchAction,
)
from evidencetree.generation.base import Generator


class ActionProposer(Protocol):
    """Interface the searcher depends on."""

    def propose(self, state: SearchState, k: int = 4) -> list[Action]: ...

    def propose_answer(self, state: SearchState) -> AnswerAction: ...


# --------------------------------------------------------------------------- #
# Heuristic proposer (no LLM calls for the search actions)
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "are", "was", "were",
    "which", "what", "who", "whom", "whose", "where", "when", "how", "name",
    "can", "be", "found", "located", "does", "do", "did",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def keyword_query(question: str) -> str:
    """Strip stopwords: 'In which city is the Eiffel Tower located?' ->
    'city Eiffel Tower'."""
    tokens = [t for t in _TOKEN_RE.findall(question) if t.lower() not in _STOPWORDS]
    return " ".join(tokens)


class HeuristicProposer:
    """Template-based proposer.

    Args:
        generator: generation backend used ONLY to draft answer text.
        enable_image_actions: propose image_search when the state carries a
            query image.
        answer_kwargs_fn: optional ``state -> kwargs`` passed to the generator
            when drafting answers. Mock runs use it to thread the eval-only
            ``reference`` hint; real runs leave it None.
    """

    def __init__(
        self,
        generator: Generator,
        enable_image_actions: bool = True,
        answer_kwargs_fn: Callable[[SearchState], dict] | None = None,
    ) -> None:
        self.generator = generator
        self.enable_image_actions = enable_image_actions
        self.answer_kwargs_fn = answer_kwargs_fn

    # ------------------------------------------------------------------ #
    def propose(self, state: SearchState, k: int = 4) -> list[Action]:
        taken = set(state.actions_taken)
        candidates: list[Action] = []

        direct = TextSearchAction(query=state.question)
        if direct not in taken:
            candidates.append(direct)

        kq = keyword_query(state.question)
        reformulated = TextSearchAction(query=kq)
        if kq and kq != state.question and reformulated not in taken:
            candidates.append(reformulated)

        # image_search queries the corpus with the image; only when the state
        # carries one (a text-only run never reaches here).
        if (
            self.enable_image_actions
            and state.image_path is not None
            and ImageSearchAction() not in taken
        ):
            candidates.append(ImageSearchAction())

        # Answering blind (no evidence) is allowed only when nothing else is
        # left — the PRM is supposed to punish it, but don't waste budget.
        # Draft the answer (a generator call) only when it actually fits in k.
        if (state.evidence or not candidates) and len(candidates) < k:
            candidates.append(self.propose_answer(state))

        return candidates[:k]

    def propose_answer(self, state: SearchState) -> AnswerAction:
        kwargs = self.answer_kwargs_fn(state) if self.answer_kwargs_fn else {}
        text = self.generator.generate(
            state.question, state.evidence_texts(), **kwargs
        )
        return AnswerAction(text=text)


# --------------------------------------------------------------------------- #
# LLM proposer (policy model; GPU server / API)
# --------------------------------------------------------------------------- #
_PROPOSE_PROMPT = """You are planning retrieval actions to answer a question \
about the given image.

Question: {question}
Evidence collected so far:
{evidence}

Propose up to {k} candidate next actions as JSON, one per line. Allowed:
  {{"type": "text_search", "query": "..."}}    # query the corpus with text
  {{"type": "image_search"}}                    # query the corpus with the image
  {{"type": "answer", "text": "..."}}

Rules:
- The question refers to the given image (e.g. "this bird", "this building").
  Do NOT invent or assume a specific entity identity (a concrete species,
  landmark, person, ...) that the image and evidence have not established.
  Guessing an identity and searching for it propagates errors through the
  whole trajectory.
- Using an entity name that ALREADY APPEARS IN THE EVIDENCE above is NOT
  guessing: once image_search (or an earlier text_search) has surfaced the
  entity, you SHOULD build text_search queries from that evidence-established
  name.
- Retrieval chain for entity questions: if the entity is not yet identified,
  FIRST use image_search to match the image against the corpus. Once the
  entity is identified but the ASKED ATTRIBUTE is still missing from the
  evidence, the next action MUST include a text_search combining the
  evidence-established entity name with the asked attribute (e.g.
  "<entity name> opening date") — do NOT answer from an entity match alone.
- Make the candidates genuinely different from each other: text_search
  queries within one proposal must not be near-duplicates — vary the angle
  (entity name + attribute, bare keywords, a rephrasing of the question), and
  never re-issue a query already executed on this path.
- Keep every search query faithful to the original question's intent; do not
  drift away from what is actually being asked.
- Only answer when the collected evidence actually supports the answer, OR
  when the question is answerable from the image and question alone by
  perception, logic, or everyday commonsense (no external facts needed).
Output ONLY the JSON lines."""


class LLMProposer:
    """Policy-LLM proposer with heuristic fallback."""

    def __init__(
        self,
        generator: Generator,
        fallback: HeuristicProposer | None = None,
    ) -> None:
        self.generator = generator
        self.fallback = fallback or HeuristicProposer(generator)

    def propose(self, state: SearchState, k: int = 4) -> list[Action]:
        evidence = "\n".join(state.evidence_texts()) or "(none)"
        prompt = _PROPOSE_PROMPT.format(
            question=state.question, evidence=evidence, k=k
        )
        try:
            raw = self.generator.generate(prompt, [], image_path=state.image_path)
            actions = self._parse(raw, state)
        except Exception:
            actions = []
        if not actions:
            return self.fallback.propose(state, k)
        taken = set(state.actions_taken)
        return [a for a in actions if a not in taken][:k]

    def propose_answer(self, state: SearchState) -> AnswerAction:
        text = self.generator.generate(
            state.question, state.evidence_texts(), image_path=state.image_path
        )
        return AnswerAction(text=text)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(raw: str, state: SearchState) -> list[Action]:
        actions: list[Action] = []
        for line in raw.splitlines():
            line = line.strip().strip("`")
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            region = obj.get("region")
            region = tuple(region) if region else None
            if t == "text_search" and obj.get("query"):
                actions.append(TextSearchAction(query=str(obj["query"])))
            elif t == "image_search" and state.image_path is not None:
                actions.append(ImageSearchAction(region=region))
            elif t == "answer" and obj.get("text"):
                actions.append(AnswerAction(text=str(obj["text"])))
        return actions
