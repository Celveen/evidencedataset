"""Step 4 — quality filtering and train/val splitting -> ETBench-Open.

Three drop rules:
    1. trajectory length: drop the whole trajectory when its step count is
       outside [min_steps, max_steps]
    2. grounding-outcome contradiction: drop samples with |local - outcome| >
       gap as noisy labels
    3. rationale QC: drop samples with qc_pass=false (still invalid after
       regeneration)
The train/val split is by query_id, so no query spans both sides.

Usage:
    python DatasetConstruct/step4_quality_filter.py --mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import load_env, read_jsonl, resolve, write_jsonl

from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step4")


def _val_split(query_id: str, val_fraction: float) -> bool:
    """Deterministic per-query split (stable across runs/machines)."""
    h = int(hashlib.md5(query_id.encode()).hexdigest(), 16) % 10_000
    return h < int(val_fraction * 10_000)


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    from common import tagged

    src = tagged(resolve(cfg["output"]["rationales"]), mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 3 output not found: {src} — run step 3 first.")
    out_dir = resolve(cfg["output"]["dataset_dir"])
    if mock:
        out_dir = out_dir / "mock"
    out_dir.mkdir(parents=True, exist_ok=True)

    q = cfg.get("quality", {})
    min_steps = int(q.get("min_steps", 2))
    max_steps = int(q.get("max_steps", 8))
    max_gap = float(q.get("max_grounding_outcome_gap", 0.7))
    val_fraction = float(q.get("val_fraction", 0.05))

    samples = list(read_jsonl(src))

    # Rule 1 — trajectory length (count steps per traj from its samples).
    steps_per_traj = Counter(s["traj_id"] for s in samples)
    bad_trajs = {
        t for t, n in steps_per_traj.items() if n < min_steps or n > max_steps
    }

    dropped = Counter()
    train, val = [], []
    for s in samples:
        if s["traj_id"] in bad_trajs:
            dropped["traj_length"] += 1
            continue
        local = s["local_grounding"]
        if local is not None and abs(local - s["outcome_credit"]) > max_gap:
            dropped["grounding_outcome_gap"] += 1
            continue
        if not s.get("rationale_qc_pass", False):
            dropped["rationale_qc"] += 1
            continue
        (val if _val_split(s["query_id"], val_fraction) else train).append(s)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    kept = train + val
    stats = {
        "input_samples": len(samples),
        "input_trajectories": len(steps_per_traj),
        "dropped": dict(dropped),
        "kept": len(kept),
        "train": len(train),
        "val": len(val),
        "action_type_distribution": dict(
            Counter(s["action"]["type"] for s in kept)
        ),
        "mean_score": sum(s["score"] for s in kept) / len(kept) if kept else 0.0,
        "mean_outcome_credit": (
            sum(s["outcome_credit"] for s in kept) / len(kept) if kept else 0.0
        ),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "step4 done: kept %d/%d (train %d / val %d), dropped %s -> %s",
        len(kept), len(samples), len(train), len(val), dict(dropped), out_dir,
    )
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 4: quality filter -> ETBench-Open.")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
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
