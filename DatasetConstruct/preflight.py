"""Preflight check before a real construction run — nothing is generated.

Verifies, for one benchmark config:
    data      prepared files exist; seed and eval ids are disjoint; the seed
              queries file holds only seed ids; sampled image paths resolve
    models    local CLIP model directory present
    python    the packages the real run imports
    keys      API-key variables named by the config are set
    policy    every vLLM endpoint answers /v1/models with the configured model
    apis      (--check-apis) judge / rationale endpoints answer /models
    hardware  free GPU memory (nvidia-smi) and free disk for the outputs

Exit code 0 = ready (warnings allowed), 1 = something must be fixed.

Usage:
    python DatasetConstruct/preflight.py --benchmark infoseek
    python DatasetConstruct/preflight.py --benchmark gqa --check-apis
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from common import apply_provider, data_dir, dataset_dir, load_env, resolve, stage_path

from evidencetree.data.splits import read_ids
from evidencetree.utils import config as cfgutil

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_results: list[tuple[str, str, str]] = []


def report(status: str, item: str, detail: str = "") -> None:
    _results.append((status, item, detail))
    print(f"[{status:4s}] {item}" + (f" — {detail}" if detail else ""), flush=True)


def _jsonl_head(path: Path, n: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in itertools.islice(f, n) if line.strip()]


def check_data(cfg: dict[str, Any]) -> None:
    ddir = data_dir(cfg)
    if ddir is None or not ddir.exists():
        report("FAIL", "data_dir", f"{ddir} missing — run the benchmark's data preparation")
        return
    for name in ("queries.jsonl", "corpus.jsonl", "eval_queries.jsonl"):
        path = ddir / name
        report("OK" if path.exists() else "FAIL", name, str(path))
    d = cfg.get("data", {})
    seed_path, eval_path = d.get("seed_ids"), d.get("eval_ids")
    if not (seed_path and eval_path and resolve(seed_path).exists() and resolve(eval_path).exists()):
        report("FAIL", "split id files", f"{seed_path}, {eval_path}")
        return
    seeds, evals = set(read_ids(resolve(seed_path))), set(read_ids(resolve(eval_path)))
    overlap = seeds & evals
    report("OK" if not overlap else "FAIL", "seed/eval disjoint",
           f"{len(seeds)} seeds, {len(evals)} eval, overlap {len(overlap)}")
    if not (ddir / "queries.jsonl").exists():
        return
    with (ddir / "queries.jsonl").open(encoding="utf-8") as f:
        qids = [json.loads(line)["query_id"] for line in f if line.strip()]
    stray = set(qids) - seeds
    report("OK" if not stray and not set(qids) & evals else "FAIL", "queries.jsonl ids",
           f"{len(qids)} queries, {len(stray)} not in seed list")
    want = int(d.get("n_queries", 0))
    if len(seeds) < want:
        report("WARN", "seed count", f"{len(seeds)} seeds < data.n_queries {want}")
    for name in ("queries.jsonl", "corpus.jsonl"):
        rows = [r for r in _jsonl_head(ddir / name, 2000) if r.get("image_path")][:50]
        missing = [r["image_path"] for r in rows if not (ddir / r["image_path"]).exists()
                   and not Path(r["image_path"]).exists()]
        if rows:
            report("OK" if not missing else "FAIL", f"{name} image paths",
                   f"{len(rows) - len(missing)}/{len(rows)} sampled paths resolve")


def check_models(cfg: dict[str, Any]) -> None:
    names = {
        (cfg.get("retriever") or {}).get("clip_model"),
        ((cfg.get("retriever") or {}).get("image") or {}).get("model_name"),
        (cfg.get("verifier") or {}).get("clip_model"),
    } - {None}
    for name in sorted(names):
        path = resolve(name)
        if path.exists():
            report("OK", f"CLIP model {name}", str(path))
        elif "/" in name:
            report("FAIL", f"CLIP model {name}", "local directory missing (download_assets.sh models)")
        else:
            report("WARN", f"CLIP model {name}", "not a local path; sentence-transformers "
                   "would try to download it (HuggingFace is unreachable from the server)")


def check_python() -> None:
    for mod in ("numpy", "scipy", "sklearn", "PIL", "yaml", "openai", "httpx", "torch",
                "sentence_transformers", "pyarrow"):
        try:
            m = importlib.import_module(mod)
            report("OK", f"python: {mod}", getattr(m, "__version__", ""))
        except Exception as exc:  # noqa: BLE001
            report("FAIL", f"python: {mod}", f"not importable ({type(exc).__name__})")
    try:
        importlib.import_module("faiss")
        report("OK", "python: faiss")
    except Exception:  # noqa: BLE001
        report("WARN", "python: faiss", "missing — the CLIP index falls back to numpy (slower)")
    try:
        import torch

        report("OK" if torch.cuda.is_available() else "WARN", "CUDA",
               f"{torch.cuda.device_count()} visible device(s)")
    except Exception:  # noqa: BLE001
        pass


def _gen_blocks(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    v = cfg.get("verifier") or {}
    blocks = {"policy": cfg.get("policy") or {}}
    if (v.get("support") or {}).get("backend", "api") == "api":
        blocks["support judge"] = (v.get("support") or {}).get("generation") or {}
    if (cfg.get("rationale") or {}).get("backend", "api") == "api":
        blocks["rationale"] = (cfg.get("rationale") or {}).get("generation") or {}
    return blocks


def check_keys(cfg: dict[str, Any]) -> None:
    for name, gen in _gen_blocks(cfg).items():
        env = gen.get("api_key_env")
        if env:
            report("OK" if os.environ.get(env) else "FAIL", f"key {env} ({name})",
                   "set" if os.environ.get(env) else "not set — fill DatasetConstruct/.env")


def _get_models(base_url: str, key: str | None, timeout: float = 10.0) -> list[str]:
    import httpx

    local = any(h in base_url for h in ("127.0.0.1", "localhost"))
    with httpx.Client(trust_env=not local, timeout=timeout) as client:
        r = client.get(base_url.rstrip("/") + "/models",
                       headers={"Authorization": f"Bearer {key}"} if key else {})
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]


def check_endpoints(cfg: dict[str, Any], check_apis: bool) -> None:
    for name, gen in _gen_blocks(cfg).items():
        if name != "policy" and not check_apis:
            continue
        urls = gen.get("base_url") or []
        urls = urls if isinstance(urls, list) else [urls]
        key = os.environ.get(gen.get("api_key_env") or "")
        for url in urls:
            try:
                models = _get_models(url, key)
                ok = gen.get("model") in models
                report("OK" if ok else "FAIL", f"{name} endpoint {url}",
                       f"serves {models}" + ("" if ok else f"; config wants {gen.get('model')!r}"))
            except Exception as exc:  # noqa: BLE001
                report("FAIL", f"{name} endpoint {url}", f"unreachable ({type(exc).__name__}: {str(exc)[:80]})")


def check_hardware(cfg: dict[str, Any]) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()
        free = []
        for line in out:
            idx, used, total = (x.strip() for x in line.split(","))
            free.append(f"gpu{idx}:{(int(total) - int(used)) / 1024:.0f}G")
        report("OK", "GPU free memory", " ".join(free))
    except Exception:  # noqa: BLE001
        report("WARN", "GPU free memory", "nvidia-smi unavailable")
    target = stage_path(cfg, "trajectories").parent
    probe = target
    while not probe.exists():
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    report("OK" if free_gb > 50 else "WARN", "disk free", f"{free_gb:.0f} GB at {probe}")
    for p in (stage_path(cfg, "trajectories"), dataset_dir(cfg) / "train.jsonl"):
        if p.exists():
            report("WARN", "existing output", f"{p} (runs resume; --force starts over)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preflight check for a construction run.")
    p.add_argument("--benchmark", default="infoseek")
    p.add_argument("--config", default=None)
    p.add_argument("--check-apis", action="store_true", help="also query judge/rationale endpoints")
    p.add_argument("--rationale-provider", default=None, help="endpoint preset for step 5")
    p.add_argument("--judge-provider", default=None, help="endpoint preset for step 4")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config or CONFIG_DIR / f"{args.benchmark}.yaml",
                              overrides=args.overrides)
    cfg = apply_provider(cfg, "rationale", args.rationale_provider)
    cfg = apply_provider(cfg, "judge", args.judge_provider)
    print(f"== preflight: {cfg.get('benchmark')} ==")
    check_data(cfg)
    check_models(cfg)
    check_python()
    check_keys(cfg)
    check_endpoints(cfg, args.check_apis)
    check_hardware(cfg)
    n_fail = sum(1 for s, _, _ in _results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in _results if s == "WARN")
    print(f"== {n_fail} failure(s), {n_warn} warning(s): "
          f"{'NOT READY' if n_fail else 'READY'} ==")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
