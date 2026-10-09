"""Step 4 — answer-support judging on the sampled answer steps only.

Implements the answer-support gate of paper Eq. 6 for the nodes step 3 kept:

    score = outcome * (floor + (1 - floor) * support)

The judge (``verifier.support.generation``) classifies each answer against the
evidence bundle — page titles and retrieved images included — as SUPPORTED /
PARTIAL / UNSUPPORTED / NOT_REQUIRED (1 / 0.5 / 0 / no gating). Wrong answers
are not sent to the judge: their score is 0 whatever the verdict
(``skipped_incorrect``). Correct-but-unsupported answers are flagged
``unsupported_correct`` and kept as DPO rejected sides.

``verifier.support.backend``: ``api`` (default) | ``lexical`` (offline
containment check) | ``off`` (outcome-only). Mock runs use lexical. Runs
resume by sample_id.

Like step 5, this step needs only its input file and an API key, so it can run
on another machine (``--input sampled.jsonl --output scored.jsonl``). Judged
answers record ``support_judge_model``.

Usage:
    python DatasetConstruct/step4_judge_support.py --mock
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from common import append_jsonl, apply_provider, load_env, read_jsonl, stage_path

from evidencetree.generation import build_generator
from evidencetree.prm.data_gen import judge_answer_samples
from evidencetree.prm.verifiers import AnswerSupportVerifier
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step4")


def build_support_verifier(cfg: dict[str, Any], mock: bool) -> AnswerSupportVerifier | None:
    s_cfg = dict(cfg.get("verifier", {}).get("support", {}))
    backend = "lexical" if mock else s_cfg.get("backend", "api")
    if backend == "off":
        return None
    generator = build_generator(s_cfg.get("generation", {})) if backend == "api" else None
    return AnswerSupportVerifier(backend=backend, generator=generator)


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = stage_path(cfg, "sampled", mock)
    out = stage_path(cfg, "scored", mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 3 output not found: {src} — run step 3 first.")
    if force and out.exists():
        out.unlink()
    done = {row["sample_id"] for row in read_jsonl(out)} if out.exists() else set()
    todo = [s for s in read_jsonl(src) if s["sample_id"] not in done]
    if not todo:
        log.info("step4: nothing to do (%d already scored) -> %s", len(done), out)
        return out

    s_cfg = cfg.get("verifier", {}).get("support", {})
    concurrency = 1 if mock else int(s_cfg.get("concurrency", 8))
    n_answers = sum(1 for s in todo if s["action"]["type"] == "answer")
    n_to_judge = sum(
        1 for s in todo
        if s["action"]["type"] == "answer" and float(s["outcome_credit"]) > 0
    )
    log.info(
        "step4: %d sampled nodes, %d answers, %d correct answers to judge "
        "(concurrency %d)", len(todo), n_answers, n_to_judge, concurrency,
    )
    backend = "lexical" if mock else s_cfg.get("backend", "api")
    judge_label = (
        str((s_cfg.get("generation") or {}).get("model")) if backend == "api" else backend
    )
    labels: Counter[str] = Counter()
    batch: list[dict[str, Any]] = []
    n_new = 0
    for sample in judge_answer_samples(
        todo,
        build_support_verifier(cfg, mock),
        floor=float(s_cfg.get("floor", 0.3)),
        unsupported_threshold=float(s_cfg.get("unsupported_threshold", 0.3)),
        concurrency=concurrency,
    ):
        if sample["action"]["type"] == "answer":
            labels[str(sample.get("answer_support_label"))] += 1
            if sample.get("answer_support_label") not in (None, "skipped_incorrect"):
                sample["support_judge_model"] = judge_label
        batch.append(sample)
        if len(batch) >= 200:
            n_new += append_jsonl(out, batch)
            batch = []
    n_new += append_jsonl(out, batch)

    rows = list(read_jsonl(out))
    log.info(
        "step4 done: +%d nodes | answer labels %s | unsupported-correct %d -> %s",
        n_new, dict(labels), sum(1 for r in rows if r.get("unsupported_correct")), out,
    )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 4: answer-support judge on sampled answers.")
    p.add_argument("--config", default=None, help="default: configs/<benchmark>.yaml")
    p.add_argument("--benchmark", default="infoseek")
    p.add_argument("--input", default=None, help="sampled.jsonl (default: from the config)")
    p.add_argument("--output", default=None, help="scored.jsonl (default: from the config)")
    p.add_argument("--provider", default=None,
                   help="judge endpoint preset from configs/providers/ (e.g. kimi_code)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    config = args.config or Path(__file__).parent / "configs" / f"{args.benchmark}.yaml"
    cfg = apply_provider(cfgutil.load_config(config, overrides=args.overrides),
                         "judge", args.provider)
    if args.input:
        cfg.setdefault("output", {})["sampled"] = str(Path(args.input).resolve())
    if args.output:
        cfg.setdefault("output", {})["scored"] = str(Path(args.output).resolve())
    run(cfg, mock=args.mock, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
