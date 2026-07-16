"""Step 1 — policy VLM 在每个 query 上跑 MCTS，生成 trajectories。

搜索栈由 evidencetree.pipeline.build_search_stack 统一装配（与
scripts/run_inference.py 完全同一条装配路径，杜绝两处漂移）。
每条 rollout 的终态轨迹都被收集（不止最优那条），写入 trajectories JSONL。
断点续跑：输出文件里已有的 query_id 自动跳过（--force 重跑全部）。

Usage:
    python DatasetConstruct/step1_gen_trajectories.py --mock
    python DatasetConstruct/step1_gen_trajectories.py --n 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_jsonl, load_env, read_jsonl, resolve, tagged

from evidencetree.eval import benchmarks, metrics
from evidencetree.pipeline import build_search_stack
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step1")


def trajectory_dict(query, record, t_index: int) -> dict[str, Any]:
    """把一条 rollout 的终态 SearchState 序列化为 trajectory JSON 行。"""
    state = record.state
    steps = []
    for i, action in enumerate(state.actions_taken):
        action_input = (
            getattr(action, "query", None)
            or getattr(action, "text", "")
            or action.describe()
        )
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
                    "image_path": e.image_path,
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
            step["image_path"] = getattr(action, "image_path", None) or state.image_path
        steps.append(step)
    final_answer = state.final_answer or ""
    return {
        "traj_id": f"{query.query_id}#t{t_index}",
        "query_id": query.query_id,
        "question": query.question,
        "image_path": query.image_path,
        "gold_answers": query.gold_answers,
        "rollout_t": record.t,
        "gen_reward": record.reward,          # 生成期引导分（非标签）
        "final_answer": final_answer,
        "outcome_em": metrics.exact_match(final_answer, query.gold_answers),
        "steps": steps,
    }


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    out = tagged(resolve(cfg["output"]["trajectories"]), mock)
    if force and out.exists():
        out.unlink()
    done_queries = (
        {row["query_id"] for row in read_jsonl(out)} if out.exists() else set()
    )

    data_cfg = cfg.get("data", {})
    queries, corpus = benchmarks.load_benchmark(
        name=str(cfg.get("benchmark", "infoseek")),
        n=int(data_cfg.get("n_queries", 100)),
        mock=mock,
        data_dir=data_cfg.get("data_dir"),
        seed=int(data_cfg.get("seed", 0)),
    )
    searcher = build_search_stack(
        cfg, queries, corpus, mock=mock, benchmark=str(cfg.get("benchmark", ""))
    )
    dedupe = bool(cfg.get("quality", {}).get("dedupe_within_query", True))

    n_new = n_failed = 0
    for qi, query in enumerate(queries):
        if query.query_id in done_queries:
            continue
        try:
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
                rows.append(trajectory_dict(query, record, t_index=len(rows)))
            n_new += append_jsonl(out, rows)
        except Exception as e:  # noqa: BLE001 - isolate per-query API/runtime failures
            n_failed += 1
            log.warning(
                "  step1: query %s skipped (%s: %s)",
                query.query_id,
                type(e).__name__,
                str(e)[:160],
            )
        if (qi + 1) % max(1, len(queries) // 10) == 0:
            log.info("  step1: %d/%d queries (%d skipped)", qi + 1, len(queries), n_failed)

    all_rows = list(read_jsonl(out))
    n_traj = len(all_rows)
    if n_traj:
        mean_steps = sum(len(r["steps"]) for r in all_rows) / n_traj
        outcome_rate = sum(r["outcome_em"] for r in all_rows) / n_traj
        log.info(
            "step1 done: %d trajectories (+%d new, %d queries skipped) | "
            "mean steps %.2f | outcome rate %.3f -> %s",
            n_traj, n_new, n_failed, mean_steps, outcome_rate, out,
        )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 1: generate MCTS trajectories.")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
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
