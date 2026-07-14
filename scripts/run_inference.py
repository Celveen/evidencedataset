"""EvidenceTree inference: MCTS over retrieval actions, PRM-guided.

Runs the MCTS search loop (UCB1 over the PRM's Q) on a benchmark and reports EM
plus search-behavior stats (action-type usage, rollouts).

Scorer selection (``prm.scorer``):
    overlap    — heuristic evidence-overlap scorer (offline debug; mock default)
    visualprm  — frozen VisualPRM-8B (GPU server; real default until Stage 4)

Usage:
    python scripts/run_inference.py --config configs/mcts.yaml --mock
    python scripts/run_inference.py --config configs/mcts.yaml --n 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.actions import ActionExecutor, BM25Retriever  # noqa: E402
from evidencetree.actions.retrievers import ClipImageRetriever  # noqa: E402
from evidencetree.eval import benchmarks, metrics  # noqa: E402
from evidencetree.generation import build_generator  # noqa: E402
from evidencetree.mcts import (  # noqa: E402
    HeuristicProposer, LLMProposer, MCTSSearcher, SearchConfig,
)
from evidencetree.mcts.proposer import GatedProposer  # noqa: E402
from evidencetree.prm.model import HeuristicOverlapScorer, VisualPRMScorer  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("run_inference")


def build_components(cfg: dict[str, Any], mock: bool):
    data_cfg = cfg.get("data", {})
    queries, corpus = benchmarks.load_infoseek(
        n=int(cfg.get("n_queries", 20)),
        mock=mock,
        data_dir=data_cfg.get("data_dir"),
        seed=int(data_cfg.get("seed", 0)),
    )
    log.info("Loaded %d queries, %d docs.", len(queries), len(corpus))

    ret_cfg = dict(cfg.get("retriever", {}))
    retriever = BM25Retriever().build(corpus)

    # image_search needs a CLIP retriever AND images on both sides; otherwise
    # the action executes against nothing (same guard as DatasetConstruct step1).
    image_retriever = None
    image_enabled = bool(ret_cfg.get("image_search", False)) and not mock
    if image_enabled:
        has_query_images = any(q.image_path for q in queries)
        has_corpus_images = any(getattr(d, "image_path", None) for d in corpus)
        if has_query_images and has_corpus_images:
            image_retriever = ClipImageRetriever(
                model_name=ret_cfg.get("clip_model", "clip-ViT-B-32"),
            ).build(corpus)
        else:
            image_enabled = False
            log.info(
                "image_search disabled: query_images=%s, corpus_images=%s",
                has_query_images, has_corpus_images,
            )
    executor = ActionExecutor(
        text_retriever=retriever,
        image_retriever=image_retriever,
        top_k=int(ret_cfg.get("top_k", 5)),
        image_top_k=int(ret_cfg.get("image_top_k", 3)),
    )

    # Policy VLM proposes candidate actions (mirrors DatasetConstruct step1):
    # real mode = LLMProposer (Qwen decides what to do next, sees image +
    # evidence) with heuristic fallback on parse failure; mock = heuristic.
    policy_cfg = dict(cfg.get("policy") or cfg.get("generation", {}))
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
            freeze_action_queries=bool(policy_cfg.get("freeze_action_queries", False)),
        )

    gate_cfg = dict(cfg.get("action_gate", {}))
    if bool(gate_cfg.get("enabled", False)):
        available = {"text_search", "answer"} | ({"image_search"} if image_enabled else set())
        proposer = GatedProposer(
            proposer,
            benchmark=str(cfg.get("benchmark", "infoseek")),
            mode=str(gate_cfg.get("mode", "dataset")),
            max_actions=(int(gate_cfg["max_actions"])
                         if gate_cfg.get("max_actions") is not None else None),
            available_actions=available,
        )
    log.info(
        "Proposer: %s | image_search=%s | gate=%s",
        type(proposer).__name__, image_enabled, bool(gate_cfg.get("enabled", False)),
    )

    prm_cfg = dict(cfg.get("prm", {}))
    scorer_kind = prm_cfg.get("scorer") or ("overlap" if mock else "visualprm")
    if scorer_kind == "overlap":
        scorer = HeuristicOverlapScorer()
    else:
        scorer = VisualPRMScorer(
            model_name=prm_cfg.get("model_name", "OpenGVLab/VisualPRM-8B"),
            mock=mock,
            device=prm_cfg.get("device"),
        )
    log.info("Scorer: %s | policy backend: %s", scorer_kind, policy_cfg.get("backend"))

    searcher = MCTSSearcher(
        executor=executor,
        proposer=proposer,
        scorer=scorer,
        config=SearchConfig.from_dict(cfg),
    )
    return queries, searcher


def run(cfg: dict[str, Any], mock: bool) -> dict[str, Any]:
    queries, searcher = build_components(cfg, mock)
    outcome_metric = cfg.get("outcome", {}).get("metric", "exact_match")

    records: list[dict[str, Any]] = []
    action_usage: Counter[str] = Counter()
    for i, query in enumerate(queries):
        result = searcher.search(query.question, image_path=query.image_path)
        em = (
            metrics.f1_score(result.answer, query.gold_answers)
            if outcome_metric == "f1"
            else metrics.exact_match(result.answer, query.gold_answers)
        )
        for action in result.best_state.actions_taken:
            action_usage[action.action_type] += 1
        records.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "gold": query.gold_answers,
                "answer": result.answer,
                "outcome": em,
                "best_reward": result.best_reward,
                "rollouts": result.rollouts_run,
                "best_path": [
                    a.describe() for a in result.best_state.actions_taken
                ],
            }
        )
        if (i + 1) % max(1, len(queries) // 10) == 0:
            log.info("  searched %d/%d", i + 1, len(queries))

    n = len(records)
    summary = {
        "n": n,
        "outcome_metric": outcome_metric,
        "accuracy": sum(r["outcome"] for r in records) / n if n else 0.0,
        "mean_rollouts": sum(r["rollouts"] for r in records) / n if n else 0.0,
        "mean_best_reward": sum(r["best_reward"] for r in records) / n if n else 0.0,
        "action_usage": dict(action_usage),
    }
    return {"summary": summary, "records": records}


def write_report(result: dict[str, Any], cfg: dict[str, Any], mock: bool) -> Path:
    report_dir = Path(cfg.get("output", {}).get("report_dir", "data/inference_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "mock" if mock else "real"
    base = report_dir / f"inference_{tag}_{ts}"
    payload = {
        "mode": tag,
        "timestamp": ts,
        "config": cfg,
        "summary": result["summary"],
        "records": result["records"],
    }
    base.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return base


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EvidenceTree MCTS inference.")
    p.add_argument("--config", default="configs/mcts.yaml")
    p.add_argument("--mock", action="store_true", help="Mock data + offline scorer.")
    p.add_argument("--n", type=int, default=None, help="Override n_queries.")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override config, e.g. --set search.rollouts=20 (repeatable).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg["n_queries"] = args.n

    log.info("=== EvidenceTree MCTS inference (mock=%s) ===", args.mock)
    if args.mock:
        log.warning("MOCK 模式：合成数据 + 启发式 scorer，仅验证搜索框架，非真实结果。")
    result = run(cfg, mock=args.mock)
    base = write_report(result, cfg, mock=args.mock)

    s = result["summary"]
    log.info(
        "accuracy=%.3f | mean_rollouts=%.1f | action_usage=%s",
        s["accuracy"], s["mean_rollouts"], s["action_usage"],
    )
    log.info("Report written: %s.json", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
