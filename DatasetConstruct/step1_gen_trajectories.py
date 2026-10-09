"""Step 1 — run MCTS with the policy VLM on every seed query to generate trajectories.

The search stack is assembled by evidencetree.pipeline.build_search_stack, the
exact same path scripts/run_inference.py uses, so construction and inference
cannot drift apart. Every rollout's terminal trajectory is collected (not just
the best one); identical action sequences within a query are written once.

Leakage guard: real runs load only the ids in ``data.seed_ids`` and fail if any
query is listed in ``data.eval_ids`` (the held-out evaluation set); with
``data.require_splits: true`` both files are mandatory.

Outcome labels use each benchmark's official rule (``metrics.outcome_match``);
image paths are written relative to ``data.data_dir`` so records stay
re-distributable. ``step1.concurrency`` searches several queries at once — the
policy server batches the concurrent requests. Runs resume: query_ids already
present in the output file are skipped (--force regenerates all).

Usage:
    python DatasetConstruct/step1_gen_trajectories.py --mock
    python DatasetConstruct/step1_gen_trajectories.py --config DatasetConstruct/configs/infoseek.yaml --n 100
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import (
    append_jsonl,
    benchmark_name,
    data_dir,
    load_env,
    mock_asset_dir,
    portable_path,
    read_jsonl,
    resolve,
    stage_path,
    with_resolved_paths,
)

from evidencetree.data.splits import read_ids
from evidencetree.eval import benchmarks, metrics
from evidencetree.pipeline import build_search_stack
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step1")


def action_input_of(action) -> str:
    """Human-readable action input: the query, the answer text, or — for
    image_search — which picture was used (SCHEMA: "image-region description")."""
    if action.action_type == "image_search":
        region = getattr(action, "region", None)
        source = "query image" if getattr(action, "image_path", None) is None else "image"
        return source if region is None else f"{source} region {[round(v, 3) for v in region]}"
    return str(getattr(action, "query", None) or getattr(action, "text", "") or action.describe())


def trajectory_dict(
    query, record, t_index: int, benchmark: str = "", base: Path | None = None
) -> dict[str, Any]:
    """Serialize one rollout's terminal SearchState as a trajectory JSON row."""
    state = record.state
    steps = []
    for i, action in enumerate(state.actions_taken):
        action_input = action_input_of(action)
        step = {
            "step_index": i,
            "action_type": action.action_type,
            "action_input": str(action_input),
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "doc_id": e.doc_id,
                    "title": e.title,
                    "text": e.text,
                    "image_path": portable_path(e.image_path, base),
                    "score": e.score,
                    "result_modality": e.result_modality,
                }
                for e in state.evidence
                if e.step_index == i
            ],
        }
        if action.action_type == "image_search":
            region = getattr(action, "region", None)
            step["region"] = list(region) if region is not None else None
            step["image_path"] = portable_path(
                getattr(action, "image_path", None) or state.image_path, base
            )
        steps.append(step)
    final_answer = state.final_answer or ""
    return {
        "traj_id": f"{query.query_id}#t{t_index}",
        "query_id": query.query_id,
        "benchmark": benchmark,
        "question": query.question,
        "image_path": portable_path(query.image_path, base),
        "gold_answers": query.gold_answers,
        "answer_type": metrics.answer_type(query.gold_answers, query.metadata),
        "rollout_t": record.t,
        "gen_reward": record.reward,          # guidance score used during generation (not a label)
        "final_answer": final_answer,
        "outcome_em": metrics.outcome_match(
            final_answer, query.gold_answers, benchmark, query.metadata
        ),
        "steps": steps,
    }


def load_seed_queries(cfg: dict[str, Any], mock: bool):
    """Seed queries + corpus, with the held-out evaluation ids enforced."""
    data_cfg = cfg.get("data", {})
    include = exclude = None
    if not mock:
        seed_ids, eval_ids = data_cfg.get("seed_ids"), data_cfg.get("eval_ids")
        if data_cfg.get("require_splits", False) and not (seed_ids and eval_ids):
            raise ValueError(
                "data.seed_ids and data.eval_ids are required (data.require_splits); "
                "run the benchmark's data preparation to create them."
            )
        if seed_ids:
            include = read_ids(resolve(seed_ids))
        if eval_ids:
            exclude = read_ids(resolve(eval_ids))
    ddir = data_dir(cfg)
    queries, corpus = benchmarks.load_benchmark(
        name=benchmark_name(cfg),
        n=int(data_cfg.get("n_queries", 100)),
        mock=mock,
        data_dir=ddir,
        seed=int(data_cfg.get("seed", 0)),
        include_ids=include,
        exclude_ids=exclude,
        mock_asset_dir=(
            mock_asset_dir(cfg) if mock and data_cfg.get("mock_images", True) else None
        ),
    )
    if include is not None and len(queries) < min(len(include), int(data_cfg.get("n_queries", 100))):
        log.warning(
            "only %d of %d seed ids were found in the queries file",
            len(queries), len(include),
        )
    return queries, corpus


def _query_rows(searcher, query, dedupe: bool, benchmark: str, base) -> list[dict[str, Any]]:
    result = searcher.search(query.question, image_path=query.image_path)
    rows, seen_sigs = [], set()
    for record in result.rollout_log:
        if record.state is None:
            continue
        sig = tuple(
            (a.action_type, getattr(a, "query", None) or getattr(a, "text", ""))
            for a in record.state.actions_taken
        )
        if dedupe and sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        rows.append(trajectory_dict(query, record, len(rows), benchmark, base))
    return rows


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    cfg = with_resolved_paths(cfg)
    out = stage_path(cfg, "trajectories", mock)
    if force and out.exists():
        out.unlink()
    done_queries = (
        {row["query_id"] for row in read_jsonl(out)} if out.exists() else set()
    )

    queries, corpus = load_seed_queries(cfg, mock)
    benchmark = benchmark_name(cfg)
    searcher = build_search_stack(cfg, queries, corpus, mock=mock, benchmark=benchmark)
    dedupe = bool(cfg.get("quality", {}).get("dedupe_within_query", True))
    base = mock_asset_dir(cfg) if mock else data_dir(cfg)
    workers = 1 if mock else max(1, int(cfg.get("step1", {}).get("concurrency", 1)))

    todo = [q for q in queries if q.query_id not in done_queries]
    log.info(
        "step1: %d seed queries (%d already done), %d workers -> %s",
        len(queries), len(queries) - len(todo), workers, out,
    )
    n_new = n_failed = n_done = 0
    report_every = max(1, len(todo) // 20)

    def _record(query, rows_or_exc) -> None:
        nonlocal n_new, n_failed, n_done
        n_done += 1
        if isinstance(rows_or_exc, Exception):
            n_failed += 1
            log.warning(
                "  step1: query %s skipped (%s: %s)",
                query.query_id, type(rows_or_exc).__name__, str(rows_or_exc)[:160],
            )
        else:
            n_new += append_jsonl(out, rows_or_exc)
        if n_done % report_every == 0:
            log.info("  step1: %d/%d queries (%d failed)", n_done, len(todo), n_failed)

    if workers == 1:
        for query in todo:
            try:
                rows = _query_rows(searcher, query, dedupe, benchmark, base)
            except Exception as e:  # noqa: BLE001 - isolate per-query API/runtime failures
                rows = e
            _record(query, rows)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_query_rows, searcher, q, dedupe, benchmark, base): q
                for q in todo
            }
            for future in as_completed(futures):
                try:
                    rows = future.result()
                except Exception as e:  # noqa: BLE001
                    rows = e
                _record(futures[future], rows)

    all_rows = list(read_jsonl(out)) if out.exists() else []
    n_traj = len(all_rows)
    if n_traj:
        n_q = len({r["query_id"] for r in all_rows})
        log.info(
            "step1 done: %d trajectories over %d queries (+%d new, %d queries failed) | "
            "%.2f traj/query | mean steps %.2f | outcome rate %.3f -> %s",
            n_traj, n_q, n_new, n_failed, n_traj / n_q,
            sum(len(r["steps"]) for r in all_rows) / n_traj,
            sum(r["outcome_em"] for r in all_rows) / n_traj, out,
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 1: generate MCTS trajectories.")
    p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "infoseek.yaml"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Regenerate from scratch.")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg.setdefault("data", {})["n_queries"] = args.n
    run(cfg, mock=args.mock, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
