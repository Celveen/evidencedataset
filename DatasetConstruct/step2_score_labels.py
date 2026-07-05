"""Step 2 — 给每条 trajectory 的每个 step 打 score label。

    local grounding  — Stage 2 verifier（lexical 或 API LLM judge），graded 0-1
    outcome credit   — tree-level credit：经过该节点（同 query 内相同动作前缀）
                       的所有 trajectory 的成功率（Monte Carlo），不是均摊
    answer support   — answer 步专属：答案是否被已积累证据支撑（lexical/API judge）。
                       堵"答对但证据不支撑"（参数化蒙对）被打满分的口子。
    score            — 非 answer 步：alpha*local + (1-alpha)*outcome；
                       answer 步：outcome * (floor + (1-floor)*support)，
                       support 缺失时退化为 outcome-only

核心逻辑在 evidencetree.prm.data_gen.label_steps；本脚本只做 IO 与统计。

Usage:
    python DatasetConstruct/step2_score_labels.py --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import load_env, read_jsonl, resolve, tagged, write_jsonl

from evidencetree.generation import build_generator
from evidencetree.prm.data_gen import label_steps
from evidencetree.prm.verifiers import (
    AnswerSupportVerifier,
    ClipGroundingScorer,
    GroundingVerifier,
)
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


def build_support_verifier(
    cfg: dict[str, Any], mock: bool
) -> AnswerSupportVerifier | None:
    """Answer-support verifier (answer 步：答案 vs 已积累证据)。mock 强制 lexical；
    backend=off 关闭（answer 步退回 outcome-only）。"""
    s_cfg = dict(cfg.get("verifier", {}).get("support", {}))
    backend = "lexical" if mock else s_cfg.get("backend", "lexical")
    if backend == "off":
        return None
    generator = None
    if backend == "api":
        generator = build_generator(s_cfg.get("generation", {}))
    return AnswerSupportVerifier(backend=backend, generator=generator)


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
    v_cfg = cfg.get("verifier", {})
    alpha = float(v_cfg.get("alpha", 0.5))
    s_cfg = v_cfg.get("support", {})

    samples = label_steps(
        trajectories,
        verifier=verifier,
        alpha=alpha,
        support_verifier=build_support_verifier(cfg, mock),
        support_floor=float(s_cfg.get("floor", 0.3)),
        unsupported_threshold=float(s_cfg.get("unsupported_threshold", 0.3)),
    )
    n = write_jsonl(out, samples)

    locals_ = [s["local_grounding"] for s in samples if s["local_grounding"] is not None]
    supports = [s["answer_support"] for s in samples if s["answer_support"] is not None]
    n_unsup = sum(1 for s in samples if s.get("unsupported_correct"))
    from collections import Counter

    label_dist = Counter(
        s["answer_support_label"] for s in samples
        if s.get("answer_support_label") is not None
    )
    log.info(
        "step2 done: %d samples from %d trajectories | mean local %.3f | "
        "mean outcome credit %.3f | mean score %.3f | mean answer support %.3f "
        "(%d scored) | unsupported-correct answers %d | support labels %s -> %s",
        n, len(trajectories),
        sum(locals_) / len(locals_) if locals_ else 0.0,
        sum(s["outcome_credit"] for s in samples) / n if n else 0.0,
        sum(s["score"] for s in samples) / n if n else 0.0,
        sum(supports) / len(supports) if supports else 0.0,
        len(supports),
        n_unsup,
        dict(label_dist),
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
