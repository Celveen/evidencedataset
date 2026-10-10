"""Step 3 — free filters, node collapse, and budgeted sampling (no API calls).

Applies every drop rule that needs no paid call BEFORE anything is paid for:

    1. trajectory length: drop whole trajectories outside [min_steps, max_steps]
       (their full length from step 1 — never a post-filter count)
    2. node collapse: one record per unique tree node (rollouts share prefixes,
       so the same state+action appears once per trajectory through it)
    3. grounding-outcome contradiction: drop non-answer nodes with
       |local - outcome| > max_grounding_outcome_gap

then samples sibling groups (all kept actions from one state) under the budget
``sampling.budget`` (or ``sampling.budget_per_query`` x the number of queries),
round-robin over queries, with each query's first group rotated over the
decision depths (``sampling.depth_cycle``). Only the sampled nodes reach the
support judge (step 4) and the rationale model (step 5).

The output is deterministic; an existing output is kept unless --force (so the
paid steps downstream never see the sample set change under them).

Usage:
    python DatasetConstruct/step3_sample.py --mock
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import load_env, read_jsonl, stage_path, write_json, write_jsonl

from evidencetree.prm.sampling import collapse_nodes, filter_gap, filter_trajectory_length, sample_groups
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step3")


def report_path(cfg: dict[str, Any], mock: bool) -> Path:
    return stage_path(cfg, "sampled", mock).with_suffix(".report.json")


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = stage_path(cfg, "labeled", mock)
    out = stage_path(cfg, "sampled", mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 2 output not found: {src} — run step 2 first.")
    if out.exists() and not force:
        log.info("step3: sample already drawn, keeping it (use --force to redraw): %s", out)
        return out

    q = cfg.get("quality", {})
    s = cfg.get("sampling", {})
    samples = list(read_jsonl(src))
    n_queries = len({x["query_id"] for x in samples})

    kept, n_len = filter_trajectory_length(
        samples, int(q.get("min_steps", 2)), int(q.get("max_steps", 8))
    )
    nodes = collapse_nodes(kept)
    n_nodes = len(nodes)
    nodes, n_gap = filter_gap(nodes, float(q.get("max_grounding_outcome_gap", 0.7)))

    per_query = float(s.get("budget_per_query", 3))
    budget = int(s["budget"]) if s.get("budget") else math.ceil(per_query * n_queries)
    max_per_query = s.get("max_per_query")
    if max_per_query is None:
        max_per_query = max(1, math.ceil(3 * per_query))
    selected, report = sample_groups(
        nodes, budget=budget, max_per_query=int(max_per_query),
        seed=int(s.get("seed", cfg.get("data", {}).get("seed", 0))),
        depth_cycle=int(s.get("depth_cycle") or 0),
    )
    write_jsonl(out, selected)
    report.update({
        "step_records_in": len(samples),
        "queries_in_data": n_queries,
        "dropped": {"traj_length_records": n_len, "grounding_outcome_gap_nodes": n_gap},
        "unique_nodes_after_length_filter": n_nodes,
    })
    write_json(report_path(cfg, mock), report)
    log.info(
        "step3 done: %d step records -> %d unique nodes -> %d after gap filter -> "
        "%d sampled (budget %d, %d queries covered, %d contrast groups) | "
        "actions %s -> %s",
        len(samples), n_nodes, len(nodes), len(selected), budget,
        report["queries_covered"], report["groups_with_contrast_selected"],
        report["action_types_selected"], out,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 3: filter, collapse nodes, sample.")
    p.add_argument("--config", default=str(Path(__file__).parent / "configs" / "infoseek.yaml"))
    p.add_argument("--mock", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    run(cfg, mock=args.mock, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
