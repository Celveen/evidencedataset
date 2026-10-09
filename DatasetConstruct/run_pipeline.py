"""ETBench-Open construction: run all six steps (or a chosen subset).

    Step 1  policy VLM MCTS rollouts on the seed queries -> trajectories
    Step 2  free labels: CLIP grounding + tree-level credit      (no API)
    Step 3  length/gap filters, node collapse, budgeted sampling  (no API)
    Step 4  answer-support judge on sampled correct answers        (API)
    Step 5  rationale generation + 5-rule QC                       (API)
    Step 6  QC filter, query-disjoint splits, pairs, provenance, stats

Usage:
    python DatasetConstruct/run_pipeline.py --mock                         # offline smoke run
    python DatasetConstruct/run_pipeline.py --benchmark infoseek --n 100   # pilot
    python DatasetConstruct/run_pipeline.py --benchmark gqa --steps 1,2,3  # free steps only
    python DatasetConstruct/run_pipeline.py --config DatasetConstruct/configs/scienceqa.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import apply_provider, load_env  # noqa: E402

import step1_gen_trajectories  # noqa: E402
import step2_label  # noqa: E402
import step3_sample  # noqa: E402
import step4_judge_support  # noqa: E402
import step5_gen_rationales  # noqa: E402
import step6_finalize  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("dataset.pipeline")

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_STEPS = {
    1: ("MCTS rollouts -> trajectories", step1_gen_trajectories.run),
    2: ("free labels (grounding + tree credit)", step2_label.run),
    3: ("filters + node collapse + sampling", step3_sample.run),
    4: ("answer-support judge (sampled answers)", step4_judge_support.run),
    5: ("rationales + QC", step5_gen_rationales.run),
    6: ("splits + pairs + provenance + stats", step6_finalize.run),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ETBench-Open construction pipeline.")
    p.add_argument("--config", default=None, help="Config file (default: configs/<benchmark>.yaml).")
    p.add_argument("--benchmark", default="infoseek", help="infoseek | scienceqa | gqa.")
    p.add_argument("--steps", default="1,2,3,4,5,6", help="e.g. 1,2,3 or 4,5,6.")
    p.add_argument("--mock", action="store_true", help="Offline smoke run (no API calls, no downloads).")
    p.add_argument("--n", type=int, default=None, help="Override data.n_queries.")
    p.add_argument("--force", action="store_true", help="Redo the selected steps from scratch.")
    p.add_argument(
        "--seed", type=int, default=None,
        help="Seed for query sampling, policy sampling and node sampling.",
    )
    p.add_argument("--rationale-provider", default=None,
                   help="step-5 endpoint preset from configs/providers/ (e.g. kimi_code)")
    p.add_argument("--judge-provider", default=None,
                   help="step-4 endpoint preset from configs/providers/")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    config = args.config or CONFIG_DIR / f"{args.benchmark}.yaml"
    cfg = cfgutil.load_config(config, overrides=args.overrides)
    cfg = apply_provider(cfg, "rationale", args.rationale_provider)
    cfg = apply_provider(cfg, "judge", args.judge_provider)
    if args.n is not None:
        cfg.setdefault("data", {})["n_queries"] = args.n
    if args.seed is not None:
        cfg.setdefault("data", {})["seed"] = args.seed
        cfg.setdefault("mcts", {})["seed"] = args.seed
        cfg.setdefault("sampling", {})["seed"] = args.seed
        for block in ("policy", "generation"):
            if isinstance(cfg.get(block), dict):
                cfg[block]["seed"] = args.seed
                cfg[block]["mock_seed"] = args.seed

    selected = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    if args.mock:
        log.warning("MOCK mode: synthetic data + template rationales. This validates the "
                    "pipeline only; it is not a real dataset.")
    log.info("Config %s | benchmark %s | steps %s", config, cfg.get("benchmark"), selected)
    for step in selected:
        name, fn = _STEPS[step]
        log.info("=== Step %d — %s ===", step, name)
        t0 = time.time()
        fn(cfg, mock=args.mock, force=args.force)
        log.info("=== Step %d done (%.1fs) ===", step, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
