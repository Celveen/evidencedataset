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


class GatedProposer:
    """Thin runtime gate over an existing proposer.

    The wrapped proposer may expose the full action space. This gate filters and
    ranks the candidate actions before MCTS expansion sees them. It is purposely
    rule-based: cheap enough for data construction and easy to audit.
    """

    def __init__(
        self,
        base: ActionProposer,
        benchmark: str = "",
        mode: str = "dataset",
        max_actions: int | None = None,
        available_actions: set[str] | Sequence[str] | None = None,
    ) -> None:
        self.base = base
        self.benchmark = benchmark.lower().replace("-", "").replace("_", "")
        self.mode = mode.lower()
        self.max_actions = max_actions
        self.available_actions = (
            {str(action) for action in available_actions}
            if available_actions is not None
            else None
        )

    def propose(self, state: SearchState, k: int = 4) -> list[Action]:
        # Ask the underlying proposer for extra candidates during expansion,
        # but keep simulation k=1 cheap; otherwise fallback proposers may draft
        # unnecessary answers and make the policy server much less stable.
        base_k = k if k <= 1 else max(k, self.max_actions or k, 6)
        candidates = self.base.propose(state, k=base_k)
        allowed = self._allowed_types(state)
        if self.available_actions is not None:
            allowed &= self.available_actions
            allowed.add("answer")
        filtered = [a for a in candidates if a.action_type in allowed]
        if not filtered:
            filtered = [a for a in candidates if isinstance(a, AnswerAction)]
        ranked = sorted(
            filtered,
            key=lambda action: self._priority(action, state),
            reverse=True,
        )
        limit = self.max_actions or k
        return ranked[: min(k, limit)]

    def propose_answer(self, state: SearchState) -> AnswerAction:
        return self.base.propose_answer(state)

    def _allowed_types(self, state: SearchState) -> set[str]:
        if self.mode == "off":
            return {"text_search", "image_search", "answer"}
        if self.mode == "dataset":
            return self._dataset_allowed()
        if self.mode in {"heuristic", "strict"}:
            return self._heuristic_allowed(state)
        return self._dataset_allowed()

    def _dataset_allowed(self) -> set[str]:
        if self.benchmark == "scienceqa":
            return {"text_search", "answer"}
        if self.benchmark == "cragmm":
            return {"text_search", "image_search", "answer"}
        if self.benchmark in {"infoseek", "mragbench"}:
            return {"text_search", "image_search", "answer"}
        return {"text_search", "image_search", "answer"}

    def _heuristic_allowed(self, state: SearchState) -> set[str]:
        q = state.question.lower()
        allowed = {"text_search", "answer"}
        has_image = state.image_path is not None
        has_choices = "choices:" in q or "choices" in q
        visual_terms = (
            "this building", "this aircraft", "this animal", "this car",
            "this model", "this image", "this object", "this bird",
            "this lake", "this mountain", "identify", "which animal",
            "what is shown", "package", "flower on the package",
        )
        text_reasoning_terms = (
            "passage", "weather", "climate", "expected ratio", "offspring",
            "experiment", "best answer", "select the best",
        )

        if self.benchmark == "cragmm":
            return {"text_search", "image_search", "answer"}

        if self.benchmark == "scienceqa":
            return allowed

        if has_image and any(term in q for term in visual_terms):
            allowed.add("image_search")

        if self.benchmark == "mragbench" and has_image:
            allowed.add("image_search")

        if any(term in q for term in text_reasoning_terms) and self.benchmark != "mragbench":
            allowed.discard("image_search")

        return allowed

    def _priority(self, action: Action, state: SearchState) -> float:
        q = state.question.lower()
        at = action.action_type
        if isinstance(action, AnswerAction):
            return 0.55 if state.evidence else -1.0
        if self.benchmark == "scienceqa":
            pri = {
                "text_search": 1.0,
                "image_search": 0.2,
            }
        elif self.benchmark == "mragbench":
            pri = {
                "image_search": 1.0,
                "text_search": 0.45,
            }
        elif self.benchmark == "infoseek":
            pri = {
                "image_search": 1.0,
                "text_search": 0.85,
            }
        else:
            pri = {
                "text_search": 0.9,
                "image_search": 0.75,
            }
        score = pri.get(at, 0.0)
        if self.mode == "strict":
            if at == "image_search" and self.benchmark == "scienceqa":
                score -= 0.4
        return score


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
        enable_image_actions: propose image_search when the state has an image.
        answer_kwargs_fn: optional ``state -> kwargs`` passed to the generator
            when drafting answers. Mock runs use it to thread the eval-only
            ``reference`` hint; real runs leave it None.
    """

    def __init__(
        self,
        generator: Generator,
        enable_image_actions: bool = True,
        use_reformulated_query: bool = True,
        answer_kwargs_fn: Callable[[SearchState], dict] | None = None,
    ) -> None:
        self.generator = generator
        self.enable_image_actions = enable_image_actions
        self.use_reformulated_query = use_reformulated_query
        self.answer_kwargs_fn = answer_kwargs_fn

    # ------------------------------------------------------------------ #
    def propose(self, state: SearchState, k: int = 4) -> list[Action]:
        taken = set(state.actions_taken)
        candidates: list[Action] = []

        direct = TextSearchAction(query=state.question)
        if direct not in taken:
            candidates.append(direct)

        kq = keyword_query(state.question)
        if self.use_reformulated_query:
            reformulated = TextSearchAction(query=kq)
            if kq and kq != state.question and reformulated not in taken:
                candidates.append(reformulated)

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

Propose exactly {k} DISTINCT candidate next actions as JSON, one per line.\nThe candidates are ALTERNATIVE branches of a search tree, not a sequence:\ncover different action types. Returning a single candidate is wrong. Allowed:
{allowed_actions}

Rules:
- The question may refer to the given image (e.g. "this bird", "this
  building"). Do NOT invent or assume a specific entity identity (a concrete
  species, landmark, person, ...) that the image and evidence have not
  established. Guessing an identity and searching for it propagates errors
  through the whole trajectory.
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
- Even when you propose answer, ALSO propose the best follow-up text_search
  as a separate candidate whenever the asked attribute is not literally
  present in the evidence above.
- Only answer when the collected evidence actually supports the answer, OR
  when the question is answerable from the image and question alone by
  perception, logic, or everyday commonsense (no external facts needed).
{query_rule}Output ONLY the JSON lines."""


class LLMProposer:
    """Policy-LLM proposer with heuristic fallback."""

    def __init__(
        self,
        generator: Generator,
        fallback: HeuristicProposer | None = None,
        enable_image_search: bool = True,
        freeze_action_queries: bool = False,
    ) -> None:
        self.generator = generator
        self.enable_image_search = enable_image_search
        self.freeze_action_queries = freeze_action_queries
        self.fallback = fallback or HeuristicProposer(
            generator,
            enable_image_actions=enable_image_search,
            use_reformulated_query=not freeze_action_queries,
        )

    def propose(self, state: SearchState, k: int = 4) -> list[Action]:
        evidence = "\n".join(state.evidence_texts()) or "(none)"
        query_hint = (
            "<copy exactly the original question or one existing evidence "
            "title/text>"
            if self.freeze_action_queries
            else "..."
        )
        allowed_actions = [f'  {{"type": "text_search", "query": "{query_hint}"}}']
        if self.enable_image_search:
            allowed_actions.append('  {"type": "image_search"}')
        allowed_actions.append('  {"type": "answer", "text": "..."}')
        prompt = _PROPOSE_PROMPT.format(
            question=state.question,
            evidence=evidence,
            k=k,
            allowed_actions="\n".join(allowed_actions),
            query_rule=(
                " For text_search, do not paraphrase or invent "
                "query strings; copy the original question or an existing "
                "evidence title/text exactly."
                if self.freeze_action_queries
                else ""
            ),
        )
        try:
            raw = self.generator.generate(
                prompt, [], image_paths=self._vision_paths(state)
            )
            actions = self._parse(raw, state)
        except Exception:
            actions = []
        if not actions:
            actions = self.fallback.propose(state, k)
        if not self.enable_image_search:
            actions = [a for a in actions if not isinstance(a, ImageSearchAction)]
        if not state.evidence:
            # Dataset construction requires evidence-backed, multi-step
            # trajectories. Ignore a premature direct answer when the policy
            # also proposed retrieval; if it proposed only an answer, fall
            # back to the retrieval-first heuristic below.
            actions = [a for a in actions if not isinstance(a, AnswerAction)]
        taken = set(state.actions_taken)
        image_action = ImageSearchAction()
        if (
            self.enable_image_search
            and state.image_path is not None
            and image_action not in taken
            and not any(isinstance(a, ImageSearchAction) for a in actions)
        ):
            # Keep the required visual branch available even when the policy
            # emits only text actions. UCT still decides whether to explore it.
            actions.insert(min(1, len(actions)), image_action)
        if not any(isinstance(a, TextSearchAction) for a in actions):
            # Symmetric guarantee for the TEXTUAL branch. The 40q ablation
            # showed the policy returns a single candidate per expansion
            # (root: image_search only; post-retrieval: answer only), which
            # collapses MCTS to branching factor 1 regardless of the scorer.
            # Never trust the prompt for structure: synthesize the attribute
            # lookup from the evidence-established title + question keywords
            # so search vs. answer is ALWAYS a real choice for UCT/the PRM.
            kq = keyword_query(state.question)
            titles = [e.title for e in state.evidence if e.title]
            for cand in (
                [TextSearchAction(query=f"{titles[0]} {kq}".strip())] if titles and kq else []
            ) + [TextSearchAction(query=kq or state.question)]:
                if cand not in taken and cand not in actions:
                    actions.append(cand)
                    break
        if not actions:
            return self.fallback.propose(state, k)
        return [a for a in actions if a not in taken][:k]

    def propose_answer(self, state: SearchState) -> AnswerAction:
        text = self.generator.generate(
            state.question,
            state.evidence_texts(),
            image_paths=self._vision_paths(state),
        )
        return AnswerAction(text=text)

    @staticmethod
    def _vision_paths(state: SearchState) -> list[str]:
        paths = [state.image_path] if state.image_path else []
        paths.extend(state.evidence_image_paths())
        return list(dict.fromkeys(paths))

    # ------------------------------------------------------------------ #
    def _parse(self, raw: str, state: SearchState) -> list[Action]:
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
            if t == "text_search" and obj.get("query"):
                actions.append(TextSearchAction(query=self._action_query(obj, state)))
            elif t == "image_search" and state.image_path is not None:
                region = obj.get("region")
                actions.append(
                    ImageSearchAction(region=tuple(region) if region else None)
                )
            elif t == "answer" and obj.get("text"):
                actions.append(AnswerAction(text=str(obj["text"])))
        return actions

    def _action_query(self, obj: dict, state: SearchState) -> str:
        query = str(obj.get("query", ""))
        if self.freeze_action_queries:
            return self._frozen_query(query, state)
        return query

    @staticmethod
    def _frozen_query(query: str, state: SearchState) -> str:
        """Allow only raw question/evidence strings as retriever queries.

        If the policy model invents a new query, fall back to the original
        question. Existing evidence may be reused only by exact string copy.
        """
        raw = query.strip()
        choices = [state.question]
        for ev in state.evidence:
            if ev.title:
                choices.append(ev.title)
            if ev.text:
                choices.append(ev.text)
        for choice in choices:
            if raw == choice.strip():
                return choice
        return state.question
