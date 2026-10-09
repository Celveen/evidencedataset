"""Step 5 — a strong LLM reads (state, action, score) and writes a rationale for it.

Runs on the judged, sampled nodes only. Samples that fail the five-rule QC are
regenerated up to ``rationale.max_attempts`` times; those still failing are
marked qc_pass=false and dropped by step 6. Transient API failures (timeouts,
rate limits, empty replies) are not QC failures: they are removed from the
output and regenerated on the next run. Requests are sent grouped by query so a
provider's prefix cache can reuse the shared instruction block, question and
evidence. Runs resume: existing sample_ids are skipped.

Step 5 only needs ``scored.jsonl`` and an API key, so it can run on any machine
that reaches the rationale provider (e.g. when the construction server cannot):
copy the file there, run with ``--input`` / ``--output``, and copy the output
back. Each row records ``rationale_model``; step 6 builds the provenance from it.

Usage:
    python DatasetConstruct/step5_gen_rationales.py --mock
    python DatasetConstruct/step5_gen_rationales.py --benchmark infoseek \
        --input scored.jsonl --output rationales.jsonl
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from common import append_jsonl, apply_provider, load_env, read_jsonl, stage_path, write_jsonl

from evidencetree.generation import build_generator
from evidencetree.prm.rationale_gen import RationaleGenerator, group_samples
from evidencetree.utils import config as cfgutil
from evidencetree.utils import get_logger

log = get_logger("dataset.step5")


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
    # non-empty QC failures for step 6 to filter as true rationale-quality cases.
    return not str(row.get("rationale") or "").strip()


def _drop_retryable_failures(path: Path) -> int:
    """Remove retryable rows from an existing rationale JSONL in place, so the
    regenerated rows never coexist with stale connection-error rows."""
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
    generator = build_generator(r_cfg.get("generation", {})) if backend == "api" else None
    evidence = dict(r_cfg.get("evidence_chars", {}))
    return RationaleGenerator(
        backend=backend,
        generator=generator,
        max_attempts=int(r_cfg.get("max_attempts", 3)),
        before_chars=evidence.get("before", 300),
        before_chars_answer=evidence.get("before_answer", 1500),
        observation_chars=evidence.get("observation", 1500),
    )


def run(cfg: dict[str, Any], mock: bool = False, force: bool = False) -> Path:
    src = stage_path(cfg, "scored", mock)
    out = stage_path(cfg, "rationales", mock)
    if not src.exists():
        raise FileNotFoundError(f"Step 4 output not found: {src} — run step 4 first.")
    if force and out.exists():
        out.unlink()

    gen = build_rationale_generator(cfg, mock)
    r_cfg = cfg.get("rationale", {})
    retry_connection_errors = bool(r_cfg.get("retry_connection_errors", True))
    if retry_connection_errors:
        n_retryable = _drop_retryable_failures(out)
        if n_retryable:
            log.info("step5 retry: removed %d transient-failure rows from %s", n_retryable, out)

    done = {row["sample_id"] for row in read_jsonl(out)} if out.exists() else set()
    samples = [s for s in read_jsonl(src) if s["sample_id"] not in done]
    samples.sort(key=lambda s: (s["query_id"], s.get("parent_key", ""), s["sample_id"]))
    workers = 1 if mock else int(r_cfg.get("concurrency", 8))
    flush_every = int(r_cfg.get("flush_every", 50))
    progress_every = int(r_cfg.get("progress_every", max(1, len(samples) // 10)))
    max_pending = max(1, int(r_cfg.get("max_pending", max(1, workers * 4))))
    if workers > 1:
        log.info("step5 rationale concurrency: workers=%d, max_pending=%d", workers, max_pending)

    model_label = "mock" if gen.backend == "mock" else str(
        (r_cfg.get("generation") or {}).get("model")
    )
    batch_mode = str(r_cfg.get("batch", "group"))
    if batch_mode not in ("group", "node"):
        raise ValueError(f"rationale.batch must be 'group' or 'node', got {batch_mode!r}")
    units = (
        group_samples(samples, int(r_cfg.get("max_group", 6)))
        if batch_mode == "group" else [[s] for s in samples]
    )
    log.info("step5: %d samples in %d requests (batch=%s)", len(samples), len(units), batch_mode)

    def _process(unit: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fields = gen.generate_group(unit) if len(unit) > 1 else [gen.generate_for(unit[0])]
        for sample, f in zip(unit, fields):
            sample.update(f)
            sample["rationale_model"] = model_label
        return unit

    n_new, n_pass, n_attempts, n_done = 0, 0, 0, 0
    batch: list[dict[str, Any]] = []

    def _flush() -> None:
        nonlocal n_new, batch
        if batch:
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
                "  step5 progress: %d/%d samples | QC pass %.1f%% | mean attempts %.2f",
                n_done, len(samples),
                100.0 * n_pass / n_done if n_done else 0.0,
                n_attempts / n_done if n_done else 0.0,
            )

    def _drain_done(
        pending: dict[Future[list[dict[str, Any]]], list[dict[str, Any]]], *, block: bool
    ) -> None:
        if not pending:
            return
        done_futures = (
            wait(pending, return_when=FIRST_COMPLETED)[0] if block
            else {future for future in pending if future.done()}
        )
        for future in done_futures:
            unit = pending.pop(future)
            try:
                finished = future.result()
            except Exception as exc:  # noqa: BLE001 - keep long jobs resumable
                finished = []
                for original in unit:
                    sample = dict(original)
                    sample.update({
                        "rationale": "",
                        "rationale_verdict": None,
                        "rationale_backend": gen.backend,
                        "rationale_attempts": 1,
                        "rationale_qc_pass": False,
                        "rationale_qc_reasons": [f"generation exception: {exc}"],
                        "rationale_model": model_label,
                    })
                    finished.append(sample)
            for sample in finished:
                _record(sample)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[list[dict[str, Any]]], list[dict[str, Any]]] = {}
        for unit in units:
            while len(pending) >= max_pending:
                _drain_done(pending, block=True)
            pending[pool.submit(_process, unit)] = unit
            _drain_done(pending, block=False)
        while pending:
            _drain_done(pending, block=True)
    _flush()

    if samples:
        log.info(
            "step5 done: +%d rationales | QC pass %.1f%% | mean attempts %.2f -> %s",
            n_new, 100.0 * n_pass / len(samples), n_attempts / len(samples), out,
        )
    else:
        log.info("step5: nothing new to do (%d already done) -> %s", len(done), out)

    if retry_connection_errors and out.exists():
        retryable_after = sum(1 for row in read_jsonl(out) if _is_retryable_generation_failure(row))
        if retryable_after:
            raise RuntimeError(
                f"Step 5 still has {retryable_after} transient-failure rationales; "
                "rerun this step."
            )
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Step 5: rationale generation (strong LLM).")
    p.add_argument("--config", default=None, help="default: configs/<benchmark>.yaml")
    p.add_argument("--benchmark", default="infoseek")
    p.add_argument("--input", default=None, help="scored.jsonl (default: from the config)")
    p.add_argument("--output", default=None, help="rationales.jsonl (default: from the config)")
    p.add_argument("--provider", default=None,
                   help="endpoint preset from configs/providers/ (e.g. kimi_code)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    config = args.config or Path(__file__).parent / "configs" / f"{args.benchmark}.yaml"
    cfg = apply_provider(cfgutil.load_config(config, overrides=args.overrides),
                         "rationale", args.provider)
    if args.input:
        cfg.setdefault("output", {})["scored"] = str(Path(args.input).resolve())
    if args.output:
        cfg.setdefault("output", {})["rationales"] = str(Path(args.output).resolve())
    run(cfg, mock=args.mock, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
