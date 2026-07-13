"""Step 2 — 给每条 trajectory 的每个 step 打 score label。

    local grounding  — Stage 2 verifier（lexical 或 API LLM judge），graded 0-1
    outcome credit   — tree-level credit：经过该节点（同 query 内相同动作前缀）
                       的所有 trajectory 的成功率（Monte Carlo），不是均摊
    answer support   — answer 步专属：答案是否被已积累证据支撑（lexical/API judge）
    score            — 非 answer 步 alpha*local + (1-alpha)*outcome；
                       answer 步 outcome * (floor + (1-floor)*support)，
                       support 缺失时退回 outcome-only

核心逻辑在 evidencetree.prm.data_gen.label_steps；本脚本只做 IO 与统计。

Usage:
    python DatasetConstruct/step2_score_labels.py --mock
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import append_jsonl, load_env, read_jsonl, resolve, tagged

from evidencetree.generation import build_generator
from evidencetree.prm.data_gen import iter_label_steps
from evidencetree.prm.verifiers import (
    AnswerSupportVerifier,
    ClipGroundingScorer,
    GroundingVerifier,
)
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step2")


def build_verifier(cfg: dict[str, Any], mock: bool) -> GroundingVerifier:
    """Unified CLIP grounding over text and image results.

    Mock/lexical mode uses no model downloads. Real ``backend: clip`` mode
    scores retrieved text/image results against the original question in one
    CLIP space.
    """
    v_cfg = dict(cfg.get("verifier", {}))
    backend = "lexical" if mock else v_cfg.get("backend", "clip")
    if backend == "lexical":
        return GroundingVerifier(clip_scorer=None)
    scorer = ClipGroundingScorer(
        model_name=v_cfg.get("clip_model", "clip-ViT-B-32"),
        device=v_cfg.get("device"),
        text_band=(
            float(v_cfg.get("text_cos_lo", 0.5)),
            float(v_cfg.get("text_cos_hi", 0.9)),
        ),
        image_band=(
            float(v_cfg.get("image_cos_lo", 0.15)),
            float(v_cfg.get("image_cos_hi", 0.32)),
        ),
    )
    return GroundingVerifier(clip_scorer=scorer)


def build_support_verifier(
    cfg: dict[str, Any], mock: bool
) -> AnswerSupportVerifier | None:
    """Answer-support verifier.

    ``backend=off`` disables it. Mock mode forces lexical so smoke tests do not
    call external APIs.
    """
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
    if force and out.exists():
        out.unlink()

    trajectories = list(read_jsonl(src))
    total_steps = sum(len(traj.get("steps", [])) for traj in trajectories)
    done_ids: set[str] = set()
    if out.exists():
        done_ids = {
            row["sample_id"]
            for row in read_jsonl(out)
            if row.get("sample_id") is not None
        }
    if done_ids:
        log.info(
            "step2 resume: %d/%d samples already scored -> %s",
            len(done_ids),
            total_steps,
            out,
        )
    if total_steps and len(done_ids) >= total_steps:
        log.info("step2 output complete, skipping: %s", out)
        return out

    verifier = build_verifier(cfg, mock)
    v_cfg = cfg.get("verifier", {})
    alpha = float(v_cfg.get("alpha", 0.5))
    s_cfg = v_cfg.get("support", {})
    step2_cfg = cfg.get("step2", {})
    flush_every = int(step2_cfg.get("flush_every", 200))
    progress_every = int(step2_cfg.get("progress_every", flush_every))
    support_concurrency = int(step2_cfg.get("support_concurrency", 1))
    support_max_pending = int(
        step2_cfg.get("support_max_pending", max(1, support_concurrency * 4))
    )
    if support_concurrency > 1:
        log.info(
            "step2 answer-support concurrency enabled: workers=%d, max_pending=%d",
            support_concurrency,
            support_max_pending,
        )

    import time

    t0 = time.time()
    n_new = 0
    batch: list[dict[str, Any]] = []
    running_labels: Counter[str] = Counter()
    running_unsup = 0
    generator = iter_label_steps(
        trajectories,
        verifier=verifier,
        alpha=alpha,
        support_verifier=build_support_verifier(cfg, mock),
        support_floor=float(s_cfg.get("floor", 0.3)),
        unsupported_threshold=float(s_cfg.get("unsupported_threshold", 0.3)),
        skip_sample_ids=done_ids,
        support_concurrency=support_concurrency,
        support_max_pending=support_max_pending,
    )

    def _flush() -> None:
        nonlocal n_new, batch
        if not batch:
            return
        n_new += append_jsonl(out, batch)
        batch = []

    for sample in generator:
        batch.append(sample)
        label = sample.get("answer_support_label")
        if label:
            running_labels[str(label)] += 1
        running_unsup += int(bool(sample.get("unsupported_correct")))
        done_now = len(done_ids) + n_new + len(batch)
        if len(batch) >= flush_every:
            _flush()
        if done_now % max(1, progress_every) == 0:
            elapsed = max(1e-6, time.time() - t0)
            rate = n_new / elapsed if n_new else 0.0
            log.info(
                "  step2 progress: %d/%d samples (new %d, %.2f/s) | "
                "support labels %s | unsupported_correct %d -> %s",
                done_now,
                total_steps,
                n_new,
                rate,
                dict(running_labels),
                running_unsup,
                out,
            )
    _flush()

    samples = list(read_jsonl(out))
    n = len(samples)

    locals_ = [s["local_grounding"] for s in samples if s["local_grounding"] is not None]
    supports = [s["answer_support"] for s in samples if s["answer_support"] is not None]
    n_unsup = sum(1 for s in samples if s.get("unsupported_correct"))
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
