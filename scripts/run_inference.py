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
import copy
import json
import statistics
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


def seeded_config(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """Return a copy of cfg with every seed knob set to ``seed``.

    Search itself is deterministic given the policy; the run-to-run variance
    the paper reports comes from policy sampling (temperature > 0) and from
    which queries are drawn. Both are pinned here, so seeds 0/1/2 are genuine
    independent replicates rather than three identical runs.
    """
    out = copy.deepcopy(cfg)
    out.setdefault("search", {})["seed"] = seed
    out.setdefault("data", {})["seed"] = seed
    for block in ("policy", "generation"):
        if isinstance(out.get(block), dict):
            out[block]["seed"] = seed
            out[block]["mock_seed"] = seed
    out.setdefault("prm", {})["mock_seed"] = seed
    return out


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


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean +/- standard deviation over seed replicates."""
    accuracies = [r["summary"]["accuracy"] for r in runs]
    action_usage: Counter[str] = Counter()
    for r in runs:
        action_usage.update(r["summary"]["action_usage"])
    return {
        "summary": {
            "seeds": [r["summary"]["seed"] for r in runs],
            "n": runs[0]["summary"]["n"],
            "outcome_metric": runs[0]["summary"]["outcome_metric"],
            "accuracy": statistics.fmean(accuracies),
            "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "accuracy_per_seed": accuracies,
            "mean_rollouts": statistics.fmean(
                r["summary"]["mean_rollouts"] for r in runs
            ),
            "action_usage": dict(action_usage),
        },
        "records": [
            dict(rec, seed=r["summary"]["seed"]) for r in runs for rec in r["records"]
        ],
    }


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
        "--seeds", default="0",
        help="Comma-separated seeds to run, e.g. --seeds 0,1,2 for the "
             "three-seed protocol reported in the paper.",
    )
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

    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip() != ""]

    log.info("=== EvidenceTree MCTS inference (mock=%s, seeds=%s) ===", args.mock, seeds)
    if args.mock:
        log.warning("MOCK mode: synthetic data + heuristic scorer. This validates the search "
            "framework only; the numbers are not real results.")

    runs = []
    for seed in seeds:
        if len(seeds) > 1:
            log.info("--- seed %d ---", seed)
        result = run(seeded_config(cfg, seed), mock=args.mock)
        result["summary"]["seed"] = seed
        runs.append(result)
        s = result["summary"]
        log.info(
            "seed=%d | accuracy=%.3f | mean_rollouts=%.1f | action_usage=%s",
            seed, s["accuracy"], s["mean_rollouts"], s["action_usage"],
        )

    report = runs[0] if len(runs) == 1 else aggregate(runs)
    base = write_report(report, cfg, mock=args.mock)
    if len(runs) > 1:
        agg = report["summary"]
        log.info(
            "accuracy over %d seeds: %.3f +/- %.3f (%s)",
            len(runs), agg["accuracy"], agg["accuracy_std"],
            ", ".join(f"{a:.3f}" for a in agg["accuracy_per_seed"]),
        )
    log.info("Report written: %s.json", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
