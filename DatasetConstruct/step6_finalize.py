"""Step 6 — rationale-QC filter, query-disjoint splits, pairs, provenance, stats.

    1. drop samples whose rationale still fails QC after regeneration
    2. split by query hash into train / val (/ test when quality.test_fraction
       > 0) — no query spans two splits
    3. attach ``gen_provenance`` (policy / retriever / judge / rationale models,
       MCTS and sampling settings, seed, pipeline commit, dataset version)
    4. mine same-state preference pairs inside each split (pairs_<split>.jsonl)
    5. write stats.json (SCHEMA §4) and validate every split against SCHEMA §2

Output: <output.dataset_dir>/<benchmark>/{train,val[,test]}.jsonl,
pairs_*.jsonl, stats.json.

Usage:
    python DatasetConstruct/step6_finalize.py --mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    benchmark_name,
    build_provenance,
    dataset_dir,
    load_env,
    read_jsonl,
    stage_path,
    write_json,
    write_jsonl,
)

from evidencetree.data.splits import stable_rank
from evidencetree.prm.pairs import mine_pairs, pair_stats
from evidencetree.prm.stats import rationale_stats, split_stats, trajectory_stats
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step6")

_SPLIT_SALT = "etbench-open-split-v1"


def assign_split(query_id: str, val_fraction: float, test_fraction: float = 0.0) -> str:
    """Deterministic per-query split (stable across runs and machines)."""
    u = (stable_rank(query_id, _SPLIT_SALT) % 1_000_000) / 1_000_000
    if u < test_fraction:
        return "test"
    if u < test_fraction + val_fraction:
        return "val"
    return "train"


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = stage_path(cfg, "rationales", mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 5 output not found: {src} — run step 5 first.")
    out_dir = dataset_dir(cfg, mock)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = cfg.get("quality", {})
    p_cfg = cfg.get("pairs", {})
    val_fraction = float(q.get("val_fraction", 0.05))
    test_fraction = float(q.get("test_fraction", 0.0))

    rows = list(read_jsonl(src))
    provenance = build_provenance(cfg, mock)
    _models_from_rows(provenance, rows)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    if test_fraction > 0:
        splits["test"] = []
    n_qc_drop = 0
    for s in rows:
        if not s.get("rationale_qc_pass", False):
            n_qc_drop += 1
            continue
        s["gen_provenance"] = provenance
        splits[assign_split(s["query_id"], val_fraction, test_fraction)].append(s)

    pairs_by_split: dict[str, dict[str, int]] = {}
    for name, samples in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", samples)
        pairs = mine_pairs(
            samples,
            margin=float(p_cfg.get("margin", 0.3)),
            stop_pairs=bool(p_cfg.get("answer_now_vs_search_more", True)),
        )
        write_jsonl(out_dir / f"pairs_{name}.jsonl", pairs)
        pairs_by_split[name] = pair_stats(pairs)
    for stale in ("test",):  # a previous build may have written a test split
        if stale not in splits:
            for f in (out_dir / f"{stale}.jsonl", out_dir / f"pairs_{stale}.jsonl"):
                if f.exists():
                    f.unlink()

    sampling_report_path = stage_path(cfg, "sampled", mock).with_suffix(".report.json")
    sampling_report = (
        json.loads(sampling_report_path.read_text(encoding="utf-8"))
        if sampling_report_path.exists() else {}
    )
    traj_path = stage_path(cfg, "trajectories", mock)
    trajectories = list(read_jsonl(traj_path)) if traj_path.exists() else []

    validation = _validate(out_dir, list(splits), q)
    stats = {
        "benchmark": benchmark_name(cfg),
        "dataset_version": provenance["dataset_version"],
        "splits": {name: split_stats(samples) for name, samples in splits.items()},
        "construction": trajectory_stats(trajectories) if trajectories else {},
        "sampling": sampling_report,
        "rationales": rationale_stats(rows),
        "dropped": {
            "trajectory_length_step_records": (sampling_report.get("dropped") or {}).get("traj_length_records"),
            "grounding_outcome_gap_nodes": (sampling_report.get("dropped") or {}).get("grounding_outcome_gap_nodes"),
            "rationale_qc_nodes": n_qc_drop,
        },
        "pairs": pairs_by_split,
        "validation": validation,
        "provenance": provenance,
    }
    write_json(out_dir / "stats.json", stats)
    log.info(
        "step6 done: %s | dropped %d rationale-QC failures | pairs %s | validation %s -> %s",
        {k: len(v) for k, v in splits.items()}, n_qc_drop,
        {k: sum(v.values()) for k, v in pairs_by_split.items()},
        "CONFORMANT" if validation["violations"] == 0 else f"{validation['violations']} VIOLATIONS",
        out_dir,
    )
    return out_dir


def _models_from_rows(provenance: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """The models recorded on the samples win over the config: steps 4 and 5
    may have run on another machine with a different config."""
    for field, key in (("rationale_model", "rationale_model"),
                       ("support_judge_model", "support_judge_model")):
        seen = sorted({str(r[field]) for r in rows if r.get(field)})
        if seen:
            if seen != [str(provenance.get(key))]:
                provenance[f"{key}_from_config"] = provenance.get(key)
            provenance[key] = seen[0] if len(seen) == 1 else seen


def _validate(out_dir: Path, split_names: list[str], q: dict[str, Any]) -> dict[str, Any]:
    """Run the SCHEMA §2 validator over the written splits (release mode)."""
    import validate_dataset

    kw = dict(
        min_steps=int(q.get("min_steps", 2)),
        max_steps=int(q.get("max_steps", 8)),
        max_gap=float(q.get("max_grounding_outcome_gap", 0.7)),
        require_provenance=True,
    )
    results = [validate_dataset.validate_file(out_dir / f"{n}.jsonl", **kw) for n in split_names]
    violations = sum(len(r["violations"]) for r in results)
    overlap = 0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            overlap += len(results[i]["query_ids"] & results[j]["query_ids"])
    if violations or overlap:
        for r in results:
            for line in r["violations"][:10]:
                log.warning("  validation: %s", line)
    return {
        "violations": violations + overlap,
        "split_query_overlap": overlap,
        "per_split": {n: len(r["violations"]) for n, r in zip(split_names, results)},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 6: finalize ETBench-Open splits.")
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
