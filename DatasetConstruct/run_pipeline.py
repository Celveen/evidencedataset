"""ETBench-Open 数据集构建：一键跑完 4 步（或指定子集）。

    Step 1  policy VLM(API) MCTS rollout -> trajectories
    Step 2  grounding verifier + tree-level credit -> score labels
    Step 3  强 LLM 生成 rationale
    Step 4  质量过滤 -> data/etbench_open/{train,val}.jsonl

Usage:
    python DatasetConstruct/run_pipeline.py --mock            # 本地冒烟，无 API
    python DatasetConstruct/run_pipeline.py                   # 真实运行（需 .env）
    python DatasetConstruct/run_pipeline.py --steps 3,4       # 只跑后两步
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
    1: ("生成 trajectories", step1_gen_trajectories.run),
    2: ("score labels（grounding + tree credit）", step2_score_labels.run),
    3: ("rationale 生成", step3_gen_rationales.run),
    4: ("质量过滤 → ETBench-Open", step4_quality_filter.run),
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ETBench-Open construction pipeline.")
    p.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--steps", default="1,2,3,4", help="如 1,2 或 3,4。")
    p.add_argument("--mock", action="store_true", help="本地冒烟（无 API、无下载）。")
    p.add_argument("--n", type=int, default=None, help="覆盖 data.n_queries。")
    p.add_argument("--force", action="store_true", help="重做（忽略已有输出）。")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg.setdefault("data", {})["n_queries"] = args.n

    selected = sorted({int(s) for s in args.steps.split(",") if s.strip()})
    if args.mock:
        log.warning("MOCK 模式：合成数据 + 模板 rationale，仅验证 pipeline，非真实数据集。")
    for step in selected:
        name, fn = _STEPS[step]
        log.info("=== Step %d — %s ===", step, name)
        t0 = time.time()
        fn(cfg, mock=args.mock, force=args.force)
        log.info("=== Step %d 完成（%.1fs）===", step, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
