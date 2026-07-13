"""Step 3 — 强 LLM 看 (state, action, score)，生成解释 score 的 rationale。

项目里唯一被允许的外部 API 用途（离线一次性；检索绝不走 API）。
QC 不过的样本会重生成至多 max_attempts 次；仍不过的标记 qc_pass=false，
由 Step 4 丢弃。断点续跑：已有 sample_id 跳过。

Usage:
    python DatasetConstruct/step3_gen_rationales.py --mock
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from common import append_jsonl, load_env, read_jsonl, resolve, tagged, write_jsonl

from evidencetree.generation import build_generator
from evidencetree.prm.rationale_gen import RationaleGenerator
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step3")


def _is_retryable_generation_failure(row: dict[str, Any]) -> bool:
    """Failures caused by transient API/network issues should be regenerated.

    They are not true rationale QC failures and should not be counted as done.
    """
    if row.get("rationale_qc_pass"):
        return False
    reasons = row.get("rationale_qc_reasons") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    reason_text = "\n".join(str(r) for r in reasons).lower()
    retry_markers = (
        "generation exception:",
        "connection error",
        "timed out",
        "timeout",
        "insufficient balance",
        "error code: 402",
        "rate limit",
        "server disconnected",
    )
    if any(marker in reason_text for marker in retry_markers):
        return True
    # Empty generations are almost always transport/provider failures. Keep
    # non-empty QC failures for Step4 to filter as true rationale-quality cases.
    return not str(row.get("rationale") or "").strip()


def _drop_retryable_failures(path: Path) -> int:
    """Remove retryable rows from an existing rationale JSONL in-place.

    This keeps sample_id uniqueness for downstream Step4: regenerated rows will
    be appended later instead of coexisting with stale connection-error rows.
    """
    if not path.exists():
        return 0
    rows = list(read_jsonl(path))
    kept = [row for row in rows if not _is_retryable_generation_failure(row)]
    dropped = len(rows) - len(kept)
    if not dropped:
        return 0
    tmp = path.with_name(f"{path.name}.tmp")
    write_jsonl(tmp, kept)
    tmp.replace(path)
    return dropped


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

    gen = build_rationale_generator(cfg, mock)
    r_cfg = cfg.get("rationale", {})
    retry_connection_errors = bool(r_cfg.get("retry_connection_errors", True))
    if retry_connection_errors:
        n_retryable = _drop_retryable_failures(out)
        if n_retryable:
            log.info(
                "step3 retry: removed %d transient connection-error rows from %s",
                n_retryable,
                out,
            )

    done = {row["sample_id"] for row in read_jsonl(out)} if out.exists() else set()
    samples = [s for s in read_jsonl(src) if s["sample_id"] not in done]
    workers = 1 if mock else int(r_cfg.get("concurrency", 8))
    flush_every = int(r_cfg.get("flush_every", 50))
    progress_every = int(r_cfg.get("progress_every", max(1, len(samples) // 10)))
    max_pending = int(r_cfg.get("max_pending", max(1, workers * 4)))
    max_pending = max(1, max_pending)
    if workers > 1:
        log.info(
            "step3 rationale concurrency enabled: workers=%d, max_pending=%d",
            workers,
            max_pending,
        )

    def _process(sample: dict[str, Any]) -> dict[str, Any]:
        sample.update(gen.generate_for(sample))
        return sample

    n_new, n_pass, n_attempts, n_done = 0, 0, 0, 0
    batch: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal n_new, batch
        if not batch:
            return
        n_new += append_jsonl(out, batch)
        batch = []

    def _record(sample: dict[str, Any]) -> None:
        nonlocal n_pass, n_attempts, n_done
        n_pass += int(sample["rationale_qc_pass"])
        n_attempts += sample["rationale_attempts"]
        n_done += 1
        batch.append(sample)
        if len(batch) >= flush_every:
            _flush()
        if n_done % max(1, progress_every) == 0:
            log.info(
                "  step3 progress: %d/%d samples (new %d) | QC pass %.1f%% | "
                "mean attempts %.2f -> %s",
                n_done,
                len(samples),
                n_new + len(batch),
                100.0 * n_pass / n_done if n_done else 0.0,
                n_attempts / n_done if n_done else 0.0,
                out,
            )

    def _drain_done(
        pending: dict[Future[dict[str, Any]], dict[str, Any]],
        *,
        block: bool,
    ) -> None:
        if not pending:
            return
        if block:
            done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done_futures = {future for future in pending if future.done()}
        for future in done_futures:
            original = pending.pop(future)
            try:
                sample = future.result()
            except Exception as exc:  # noqa: BLE001 - keep long jobs resumable
                sample = dict(original)
                sample.update({
                    "rationale": "",
                    "rationale_verdict": None,
                    "rationale_backend": gen.backend,
                    "rationale_attempts": 1,
                    "rationale_qc_pass": False,
                    "rationale_qc_reasons": [f"generation exception: {exc}"],
                })
            _record(sample)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        for sample in samples:
            while len(pending) >= max_pending:
                _drain_done(pending, block=True)
            pending[pool.submit(_process, sample)] = sample
            _drain_done(pending, block=False)
        while pending:
            _drain_done(pending, block=True)
    _flush()

    if samples:
        log.info(
            "step3 done: +%d rationales | QC pass %.1f%% | mean attempts %.2f -> %s",
            n_new, 100.0 * n_pass / len(samples), n_attempts / len(samples), out,
        )
    else:
        log.info("step3: nothing new to do (%d already done) -> %s", len(done), out)

    if retry_connection_errors and out.exists():
        retryable_after = sum(
            1 for row in read_jsonl(out) if _is_retryable_generation_failure(row)
        )
        if retryable_after:
            raise RuntimeError(
                f"Step3 still has {retryable_after} transient connection-error "
                "rationales; retry this step."
            )
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
