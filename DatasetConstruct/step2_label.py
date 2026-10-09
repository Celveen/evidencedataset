"""Step 2 — free step labels for every step of every trajectory (no API calls).

    local grounding  — unified CLIP grounding of what the step retrieved
                       (windowed text scoring; raw cosine + modality recorded)
    outcome credit   — tree-level credit: the Monte Carlo success rate of all
                       trajectories through this node (same action prefix within
                       a query), not an evenly shared outcome
    score            — non-answer steps: alpha*local + (1-alpha)*outcome;
                       answer steps: provisional outcome, marked ``pending`` —
                       the answer-support judge runs in step 4, after sampling,
                       so it is only paid for answers that are kept
    marginal gain    — g * (1 - redundancy of the best result w.r.t. the
                       evidence already held); recorded, not a label

The logic lives in evidencetree.prm.data_gen.iter_label_steps; this script only
does I/O and statistics. Runs resume by sample_id.

Usage:
    python DatasetConstruct/step2_label.py --mock
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import (
    append_jsonl,
    benchmark_name,
    data_dir,
    image_resolver,
    load_env,
    mock_asset_dir,
    read_jsonl,
    stage_path,
    with_resolved_paths,
)

from evidencetree.prm.data_gen import iter_label_steps
from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step2")

# Calibration bands for the deterministic mock encoders (smoke runs only).
_MOCK_TEXT_BAND = (0.05, 0.6)
_MOCK_IMAGE_BAND = (0.2, 0.9)


def build_verifier(cfg: dict[str, Any], mock: bool) -> GroundingVerifier:
    """Unified CLIP grounding over text and image results.

    ``backend: lexical`` is the offline fallback (no model). Mock runs use the
    deterministic stand-in CLIP encoders unless the config asks for lexical, so
    windowing, image grounding and redundancy all run in the smoke test.
    """
    v_cfg = dict(cfg.get("verifier", {}))
    backend = v_cfg.get("backend", "clip")
    if backend == "lexical":
        return GroundingVerifier(clip_scorer=None)
    windows = dict(
        window_words=int(v_cfg.get("window_words", 48)),
        window_stride=int(v_cfg.get("window_stride", 40)),
    )
    if mock:
        from evidencetree.utils.mock_encoders import mock_image_encoder, mock_text_encoder

        return GroundingVerifier(clip_scorer=ClipGroundingScorer(
            text_encoder=mock_text_encoder,
            image_encoder=mock_image_encoder,
            text_band=_MOCK_TEXT_BAND,
            image_band=_MOCK_IMAGE_BAND,
            image_resolver=image_resolver(mock_asset_dir(cfg)),
            **windows,
        ))
    scorer = ClipGroundingScorer(
        model_name=v_cfg.get("clip_model", "clip-ViT-B-32"),
        device=v_cfg.get("device"),
        text_band=(float(v_cfg.get("text_cos_lo", 0.5)), float(v_cfg.get("text_cos_hi", 0.9))),
        image_band=(float(v_cfg.get("image_cos_lo", 0.15)), float(v_cfg.get("image_cos_hi", 0.32))),
        image_resolver=image_resolver(data_dir(cfg)),
        **windows,
    )
    return GroundingVerifier(clip_scorer=scorer)


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    cfg = with_resolved_paths(cfg)
    src = stage_path(cfg, "trajectories", mock)
    out = stage_path(cfg, "labeled", mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 1 output not found: {src} — run step 1 first.")
    if force and out.exists():
        out.unlink()

    trajectories = list(read_jsonl(src))
    total_steps = sum(len(traj.get("steps", [])) for traj in trajectories)
    done_ids: set[str] = (
        {row["sample_id"] for row in read_jsonl(out)} if out.exists() else set()
    )
    if total_steps and len(done_ids) >= total_steps:
        log.info("step2 output complete, skipping: %s", out)
        return out
    if done_ids:
        log.info("step2 resume: %d/%d steps already labeled", len(done_ids), total_steps)

    v_cfg = cfg.get("verifier", {})
    flush_every = int(cfg.get("step2", {}).get("flush_every", 2000))
    t0 = time.time()
    n_new, batch = 0, []
    for sample in iter_label_steps(
        trajectories,
        verifier=build_verifier(cfg, mock),
        alpha=float(v_cfg.get("alpha", 0.5)),
        support_floor=float(v_cfg.get("support", {}).get("floor", 0.3)),
        skip_sample_ids=done_ids,
        defer_support=True,
        benchmark=benchmark_name(cfg),
    ):
        batch.append(sample)
        if len(batch) >= flush_every:
            n_new += append_jsonl(out, batch)
            batch = []
            log.info(
                "  step2 progress: %d/%d steps (%.1f/s)",
                len(done_ids) + n_new, total_steps, n_new / max(1e-6, time.time() - t0),
            )
    n_new += append_jsonl(out, batch)

    samples = list(read_jsonl(out))
    n = len(samples)
    locals_ = [s["local_grounding"] for s in samples if s["local_grounding"] is not None]
    gains = [s["marginal_gain"] for s in samples if s.get("marginal_gain") is not None]
    nodes = {(s["query_id"], s["node_key"]) for s in samples}
    log.info(
        "step2 done: %d step records (%d unique nodes) from %d trajectories | "
        "mean local %.3f | mean outcome credit %.3f | mean marginal gain %.3f -> %s",
        n, len(nodes), len(trajectories),
        sum(locals_) / len(locals_) if locals_ else 0.0,
        sum(s["outcome_credit"] for s in samples) / n if n else 0.0,
        sum(gains) / len(gains) if gains else 0.0,
        out,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 2: free step labels (grounding + tree credit).")
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
