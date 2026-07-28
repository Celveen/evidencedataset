"""ETBench-Open construction: run all four steps (or a chosen subset).

    Step 1  policy VLM(API) MCTS rollout -> trajectories
    Step 2  grounding verifier + tree-level credit -> score labels
    Step 3  rationale generation with a strong LLM
    Step 4  quality filtering -> data/etbench_open/{train,val}.jsonl

Usage:
    python DatasetConstruct/run_pipeline.py --mock            # offline smoke run, no API calls
    python DatasetConstruct/run_pipeline.py                   # real run (needs .env)
    python DatasetConstruct/run_pipeline.py --steps 3,4       # only the last two steps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_env  # noqa: E402

import step1_gen_trajectories  # noqa: E402
import step2_score_labels  # noqa: E402
import step3_gen_rationales  # noqa: E402
import step4_quality_filter  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("dataset.pipeline")

_STEPS = {
    1: ("generate trajectories", step1_gen_trajectories.run),
    2: ("score labels (grounding + tree credit)", step2_score_labels.run),
    3: ("generate rationales", step3_gen_rationales.run),
    4: ("quality filter -> ETBench-Open", step4_quality_filter.run),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ETBench-Open construction pipeline.")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--steps", default="1,2,3,4", help="e.g. 1,2 or 3,4.")
    p.add_argument("--mock", action="store_true", help="Offline smoke run (no API calls, no downloads).")
    p.add_argument("--n", type=int, default=None, help="Override data.n_queries.")
    p.add_argument("--force", action="store_true", help="Redo from scratch (ignore existing output).")
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed for query sampling and policy sampling (paper uses 0/1/2).",
    )
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg.setdefault("data", {})["n_queries"] = args.n
    if args.seed is not None:
        cfg.setdefault("data", {})["seed"] = args.seed
        cfg.setdefault("mcts", {})["seed"] = args.seed
        for block in ("policy", "generation"):
            if isinstance(cfg.get(block), dict):
                cfg[block]["seed"] = args.seed
                cfg[block]["mock_seed"] = args.seed

    selected = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    if args.mock:
        log.warning("MOCK mode: synthetic data + template rationales. This validates the "
            "pipeline only; it is not a real dataset.")
    for step in selected:
        name, fn = _STEPS[step]
        log.info("=== Step %d — %s ===", step, name)
        t0 = time.time()
        fn(cfg, mock=args.mock, force=args.force)
        log.info("=== Step %d done (%.1fs) ===", step, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
