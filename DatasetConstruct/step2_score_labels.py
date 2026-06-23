"""Step 2 — 给每条 trajectory 的每个 step 打 score label。

    local grounding  — Stage 2 verifier（lexical 或 API LLM judge），graded 0-1
    outcome credit   — tree-level credit：经过该节点（同 query 内相同动作前缀）
                       的所有 trajectory 的成功率（Monte Carlo），不是均摊
    score            — alpha*local + (1-alpha)*outcome（answer 步 outcome-only）

核心逻辑在 evidencetree.prm.data_gen.label_steps；本脚本只做 IO 与统计。

Usage:
    python DatasetConstruct/step2_score_labels.py --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_env, read_jsonl, resolve, tagged, write_jsonl

from evidencetree.prm.data_gen import label_steps
from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step2")


def build_verifier(cfg: dict[str, Any], mock: bool) -> GroundingVerifier:
    """Unified CLIP grounding (text + image results in one space). Mock/lexical
    backend uses no model (offline smoke tests)."""
    v_cfg = dict(cfg.get("verifier", {}))
    backend = "lexical" if mock else v_cfg.get("backend", "clip")
    if backend == "lexical":
        return GroundingVerifier(clip_scorer=None)
    scorer = ClipGroundingScorer(
        model_name=v_cfg.get("clip_model", "clip-ViT-B-32"),
        device=v_cfg.get("device"),
        text_band=(float(v_cfg.get("text_cos_lo", 0.5)), float(v_cfg.get("text_cos_hi", 0.9))),
        image_band=(float(v_cfg.get("image_cos_lo", 0.15)), float(v_cfg.get("image_cos_hi", 0.32))),
    )
    return GroundingVerifier(clip_scorer=scorer)


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = tagged(resolve(cfg["output"]["trajectories"]), mock)
    out = tagged(resolve(cfg["output"]["scored"]), mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 1 output not found: {src} — run step 1 first.")
    if out.exists() and not force:
        log.info("step2 output exists, skipping (use --force to redo): %s", out)
        return out

    trajectories = list(read_jsonl(src))
    verifier = build_verifier(cfg, mock)
    alpha = float(cfg.get("verifier", {}).get("alpha", 0.5))

    samples = label_steps(trajectories, verifier=verifier, alpha=alpha)
    n = write_jsonl(out, samples)

    locals_ = [s["local_grounding"] for s in samples if s["local_grounding"] is not None]
    log.info(
        "step2 done: %d samples from %d trajectories | mean local %.3f | "
        "mean outcome credit %.3f | mean score %.3f -> %s",
        n, len(trajectories),
        sum(locals_) / len(locals_) if locals_ else 0.0,
        sum(s["outcome_credit"] for s in samples) / n if n else 0.0,
        sum(s["score"] for s in samples) / n if n else 0.0,
        out,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 2: score labels (grounding + tree credit).")
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
