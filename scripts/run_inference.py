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

from evidencetree.eval import benchmarks, metrics  # noqa: E402
from evidencetree.pipeline import build_search_stack  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("run_inference")


def build_components(cfg: dict[str, Any], mock: bool):
    """Load the benchmark and wire the shared search stack.

    The stack itself (retrievers -> executor -> proposer -> gate -> scorer)
    comes from ``evidencetree.pipeline.build_search_stack`` — the SAME assembly
    DatasetConstruct step1 uses, so inference cannot drift from construction.
    """
    data_cfg = cfg.get("data", {})
    benchmark = str(cfg.get("benchmark") or data_cfg.get("benchmark") or "infoseek")
    queries, corpus = benchmarks.load_benchmark(
        name=benchmark,
        n=int(cfg.get("n_queries", 20)),
        mock=mock,
        data_dir=data_cfg.get("data_dir"),
        seed=int(data_cfg.get("seed", 0)),
    )
    log.info("Loaded %d queries, %d docs.", len(queries), len(corpus))

    searcher = build_search_stack(
        cfg, queries, corpus, mock=mock, benchmark=benchmark
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
        log.warning("MOCK mode: synthetic data + heuristic scorer. This validates the search "
            "framework only; the numbers are not real results.")
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
