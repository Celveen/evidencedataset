"""Step 3 — 强 LLM 看 (state, action, score)，生成解释 score 的 rationale。

项目里唯一被允许的外部 API 用途（离线一次性；检索绝不走 API）。
QC 不过的样本会重生成至多 max_attempts 次；仍不过的标记 qc_pass=false，
由 Step 4 丢弃。断点续跑：已有 sample_id 跳过。

Usage:
    python DatasetConstruct/step3_gen_rationales.py --mock
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import append_jsonl, load_env, read_jsonl, resolve, tagged

from evidencetree.generation import build_generator
from evidencetree.prm.rationale_gen import RationaleGenerator
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step3")


def build_rationale_generator(cfg: dict[str, Any], mock: bool) -> RationaleGenerator:
    r_cfg = dict(cfg.get("rationale", {}))
    backend = "mock" if mock else r_cfg.get("backend", "api")
    generator = None
    if backend == "api":
        generator = build_generator(r_cfg.get("generation", {}))
    return RationaleGenerator(
        backend=backend,
        generator=generator,
        max_attempts=int(r_cfg.get("max_attempts", 3)),
    )


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = tagged(resolve(cfg["output"]["scored"]), mock)
    out = tagged(resolve(cfg["output"]["rationales"]), mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 2 output not found: {src} — run step 2 first.")
    if force and out.exists():
        out.unlink()
    done = {row["sample_id"] for row in read_jsonl(out)} if out.exists() else set()

    gen = build_rationale_generator(cfg, mock)
    samples = [s for s in read_jsonl(src) if s["sample_id"] not in done]
    workers = 1 if mock else int(cfg.get("rationale", {}).get("concurrency", 8))

    def _process(sample: dict[str, Any]) -> dict[str, Any]:
        sample.update(gen.generate_for(sample))
        return sample

    n_new, n_pass, n_attempts, n_done = 0, 0, 0, 0
    batch: list[dict[str, Any]] = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sample in pool.map(_process, samples):
            n_pass += int(sample["rationale_qc_pass"])
            n_attempts += sample["rationale_attempts"]
            n_done += 1
            batch.append(sample)
            if len(batch) >= 50:  # flush regularly so interrupts lose little work
                n_new += append_jsonl(out, batch)
                batch = []
            if n_done % max(1, len(samples) // 10) == 0:
                log.info("  step3: %d/%d samples", n_done, len(samples))
    n_new += append_jsonl(out, batch)

    if samples:
        log.info(
            "step3 done: +%d rationales | QC pass %.1f%% | mean attempts %.2f -> %s",
            n_new, 100.0 * n_pass / len(samples), n_attempts / len(samples), out,
        )
    else:
        log.info("step3: nothing new to do (%d already done) -> %s", len(done), out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 3: rationale generation (strong LLM).")
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
