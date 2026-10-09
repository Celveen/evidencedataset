"""Single shared search-stack assembly.

``build_search_stack`` is the ONE place that wires retrievers -> executor ->
generator -> proposer (-> optional gate) -> scorer -> MCTSSearcher. Both
DatasetConstruct/step1_gen_trajectories.py (data construction) and
scripts/run_inference.py (inference) call it, so construction and inference
behave identically by construction — including the UNCONDITIONAL text-branch
guarantee inside LLMProposer (no enforce/branching flag exists on purpose).

Config keys consumed (superset of the two former call sites):

    retriever.backend      unified | bm25 | hybrid   (default unified)
    retriever.dedupe       entity | doc | none       (default entity)
    retriever.top_k / retriever.image_search / retriever.clip_model
    retriever.image.{enabled,model_name,device,batch_size,top_k,index_dir}
    retriever.image_top_k  (legacy inference alias of retriever.image.top_k)
    retriever.cache_size   memoized retrieval calls (default 50000; 0 = off)
    retriever.bm25_index_dir  on-disk BM25 cache (default <data_dir>/bm25_index)
    retriever.mock_encoders   mock runs over an image-bearing world use the
                           deterministic stand-in CLIP encoders (default true)
    policy | generation    proposer / answer-drafting backend
    action_gate.{enabled,mode,max_actions}
    prm.scorer             overlap | visualprm (default: overlap when mock,
                           else visualprm) + prm.model_name/device
    mcts | search          SearchConfig block (both spellings accepted)

Mock mode never downloads a model. When the mock queries carry images it
injects the deterministic encoders of ``evidencetree.utils.mock_encoders`` so
the full unified-CLIP mechanics (image_search, cross-modal hits) run offline;
a text-only mock world, or ``retriever.mock_encoders: false``, falls back to
BM25-only retrieval. Tests and demos may inject their own encoders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from evidencetree.actions import ActionExecutor, BM25Retriever, HybridTextRetriever
from evidencetree.actions.retrievers import Encoder, UnifiedClipRetriever, corpus_fingerprint
from evidencetree.eval.benchmarks import Document, Query
from evidencetree.generation import build_generator
from evidencetree.mcts import HeuristicProposer, LLMProposer, MCTSSearcher, SearchConfig
from evidencetree.mcts.proposer import GatedProposer
from evidencetree.prm.model import HeuristicOverlapScorer, VisualPRMScorer
from evidencetree.utils import get_logger

log = get_logger("pipeline.assembly")

_TEXT_BACKENDS = {"unified", "bm25", "hybrid"}


def build_search_stack(
    cfg: dict[str, Any],
    queries: Sequence[Query],
    corpus: Sequence[Document],
    mock: bool = False,
    benchmark: str | None = None,
    text_encoder: Encoder | None = None,
    image_encoder: Encoder | None = None,
) -> MCTSSearcher:
    """Wire the full MCTS search stack from config. See module docstring."""
    retriever_cfg = dict(cfg.get("retriever", {}))
    image_cfg = dict(retriever_cfg.get("image", {}))
    benchmark = str(
        benchmark
        or cfg.get("benchmark")
        or cfg.get("data", {}).get("benchmark")
        or "infoseek"
    )

    backend = str(retriever_cfg.get("backend", "unified")).lower()
    if backend not in _TEXT_BACKENDS:
        raise ValueError(
            f"retriever.backend must be one of {sorted(_TEXT_BACKENDS)}, "
            f"got {backend!r}."
        )
    has_query_images = any(q.image_path for q in queries)
    injected = text_encoder is not None or image_encoder is not None
    if (
        mock and not injected and has_query_images
        and bool(retriever_cfg.get("mock_encoders", True))
    ):
        from evidencetree.utils.mock_encoders import mock_image_encoder, mock_text_encoder

        text_encoder, image_encoder = mock_text_encoder, mock_image_encoder
        injected = True
        log.info("Mock mode: deterministic stand-in CLIP encoders (no downloads).")
    if mock and not injected and backend != "bm25":
        log.info(
            "Mock mode without encoders: retriever.backend %s -> bm25 "
            "(no CLIP model downloads in smoke runs).", backend,
        )
        backend = "bm25"

    image_requested = bool(
        retriever_cfg.get("image_search", image_cfg.get("enabled", False))
    )
    # image_search is possible whenever a unified index exists (an image query
    # over text-only units is legal cross-modal retrieval by design) and the
    # queries actually carry images; mock runs need injected encoders.
    image_possible = (
        image_requested and has_query_images and (not mock or injected)
    )

    unified = None
    if backend in ("unified", "hybrid") or image_possible:
        unified = _build_or_load_unified(
            cfg, corpus, retriever_cfg, image_cfg,
            text_encoder=text_encoder, image_encoder=image_encoder,
        )

    dedupe = _dedupe_key(retriever_cfg)
    bm25 = (
        _build_or_load_bm25(cfg, corpus, retriever_cfg, dedupe, cache=not (mock or injected))
        if backend in ("bm25", "hybrid") else None
    )
    if backend == "unified":
        text_retriever = unified
    elif backend == "hybrid":
        text_retriever = HybridTextRetriever(bm25, unified, dedupe_key=dedupe)
    else:
        text_retriever = bm25

    image_enabled = image_possible and unified is not None
    if image_requested and not image_enabled and not mock:
        log.info(
            "image_search disabled for %s: query_images=%s, unified_index=%s",
            benchmark, has_query_images, unified is not None,
        )
    if unified is not None:
        log.info(
            "Unified CLIP index for %s: %d text units + %d image units over "
            "%d docs (text backend=%s, image_search=%s, dedupe=%s)",
            benchmark, unified.text_unit_count, unified.image_unit_count,
            unified.n_docs, backend, image_enabled, unified.dedupe_key,
        )

    executor = ActionExecutor(
        text_retriever=text_retriever,
        image_retriever=unified if image_enabled else None,
        top_k=int(retriever_cfg.get("top_k", 5)),
        image_top_k=int(
            image_cfg.get("top_k", retriever_cfg.get("image_top_k", 3))
        ),
        cache_size=int(retriever_cfg.get("cache_size", 50_000)),
    )

    policy_cfg = dict(cfg.get("policy") or cfg.get("generation") or {})
    if mock:
        policy_cfg["backend"] = "mock"
    generator = build_generator(policy_cfg)

    if mock:
        # Mock runs thread the eval-only reference hint into answer drafting so
        # generator accuracy is controlled (same device as the Stage 0.1 pilot).
        gold_by_question = {q.question: q.gold_answers for q in queries}
        proposer = HeuristicProposer(
            generator,
            enable_image_actions=image_enabled,
            answer_kwargs_fn=lambda state: {
                "reference": gold_by_question.get(state.question)
            },
        )
    else:
        proposer = LLMProposer(
            generator,
            enable_image_search=image_enabled,
            freeze_action_queries=bool(
                policy_cfg.get("freeze_action_queries", False)
            ),
        )

    gate_cfg = dict(cfg.get("action_gate", {}))
    if bool(gate_cfg.get("enabled", False)):
        available = {"text_search", "answer"}
        if image_enabled and len(unified) > 0:
            available.add("image_search")
        proposer = GatedProposer(
            proposer,
            benchmark=benchmark,
            mode=str(gate_cfg.get("mode", "dataset")),
            max_actions=(
                int(gate_cfg["max_actions"])
                if gate_cfg.get("max_actions") is not None
                else None
            ),
            available_actions=available,
        )
        log.info(
            "Action gate enabled: mode=%s, max_actions=%s, available_actions=%s",
            gate_cfg.get("mode", "dataset"), gate_cfg.get("max_actions"),
            ",".join(sorted(available)),
        )

    prm_cfg = dict(cfg.get("prm", {}))
    scorer_kind = prm_cfg.get("scorer") or ("overlap" if mock else "visualprm")
    if scorer_kind == "overlap":
        scorer = HeuristicOverlapScorer()
    elif scorer_kind == "visualprm":
        scorer = VisualPRMScorer(
            model_name=prm_cfg.get("model_name", "OpenGVLab/VisualPRM-8B"),
            mock=mock,
            device=prm_cfg.get("device"),
        )
    else:
        raise ValueError(f"Unknown prm.scorer {scorer_kind!r}.")

    log.info(
        "Search stack: proposer=%s | scorer=%s | policy backend=%s | "
        "text backend=%s | image_search=%s | gate=%s",
        type(proposer).__name__, scorer_kind, policy_cfg.get("backend"),
        backend, image_enabled, bool(gate_cfg.get("enabled", False)),
    )

    search_cfg = cfg.get("search") or cfg.get("mcts") or {}
    return MCTSSearcher(
        executor=executor,
        proposer=proposer,
        scorer=scorer,
        config=SearchConfig.from_dict({"search": search_cfg}),
    )


# --------------------------------------------------------------------------- #
def _dedupe_key(retriever_cfg: dict[str, Any]) -> str | None:
    raw = retriever_cfg.get("dedupe", "entity")
    if raw in (None, False):
        return None
    raw = str(raw).lower()
    if raw in ("none", "off", "false", ""):
        return None
    if raw in ("entity", "doc"):
        return raw
    raise ValueError(f"retriever.dedupe must be entity|doc|none, got {raw!r}.")


def _build_or_load_bm25(
    cfg: dict[str, Any],
    corpus: Sequence[Document],
    retriever_cfg: dict[str, Any],
    dedupe: str | None,
    cache: bool,
) -> BM25Retriever:
    """Build the BM25 index, reusing the on-disk copy when the corpus matches.

    Tokenizing a million chunks takes minutes; every resumed step-1 run would
    otherwise pay it again.
    """
    if not cache:
        return BM25Retriever(dedupe_key=dedupe).build(corpus)
    data_dir = Path(cfg.get("data", {}).get("data_dir") or ".")
    index_dir = Path(retriever_cfg.get("bm25_index_dir") or data_dir / "bm25_index")
    fingerprint = corpus_fingerprint(corpus)
    loaded = BM25Retriever.load(index_dir, corpus, fingerprint, dedupe_key=dedupe)
    if loaded is not None:
        log.info("Loaded cached BM25 index: %s (%d docs)", index_dir, len(loaded))
        return loaded
    bm25 = BM25Retriever(dedupe_key=dedupe).build(corpus)
    try:
        bm25.save(index_dir, fingerprint)
        log.info("Built and cached BM25 index: %s (%d docs)", index_dir, len(bm25))
    except RuntimeError:  # rank-bm25 fallback cannot be saved; rebuild next time
        pass
    return bm25


def _build_or_load_unified(
    cfg: dict[str, Any],
    corpus: Sequence[Document],
    retriever_cfg: dict[str, Any],
    image_cfg: dict[str, Any],
    text_encoder: Encoder | None,
    image_encoder: Encoder | None,
) -> UnifiedClipRetriever:
    """Build the unified CLIP index (with on-disk caching for real encoders)."""
    model_name = str(
        image_cfg.get("model_name")
        or retriever_cfg.get("clip_model")
        or "clip-ViT-B-32"
    )
    kwargs = dict(
        model_name=model_name,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        device=image_cfg.get("device"),
        batch_size=int(image_cfg.get("batch_size", 32)),
        dedupe_key=_dedupe_key(retriever_cfg),
    )
    injected = text_encoder is not None or image_encoder is not None
    index_dir = None
    if not injected:  # caching only makes sense for the real CLIP encoders
        data_dir = Path(cfg.get("data", {}).get("data_dir") or ".")
        index_value = image_cfg.get("index_dir")
        index_dir = Path(index_value) if index_value else data_dir / "unified_clip_index"
        if (index_dir / "vectors.npy").exists() and (index_dir / "units.jsonl").exists():
            loaded = UnifiedClipRetriever.load(index_dir, **kwargs)
            if loaded.n_docs == len(corpus):
                log.info("Loaded cached unified CLIP index: %s", index_dir)
                return loaded
            log.warning(
                "Ignoring stale unified CLIP index (%d docs, corpus has %d): %s",
                loaded.n_docs, len(corpus), index_dir,
            )

    retriever = UnifiedClipRetriever(**kwargs).build(corpus)
    if index_dir is not None:
        retriever.save(index_dir)
        log.info("Built and cached unified CLIP index: %s", index_dir)
    return retriever
