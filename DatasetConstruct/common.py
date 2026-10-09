"""Shared helpers for the DatasetConstruct pipeline (paths, .env, JSONL IO,
provenance).

All algorithmic logic lives in the ``evidencetree`` package
(prm/verifiers.py, prm/data_gen.py, prm/sampling.py, prm/pairs.py,
prm/rationale_gen.py, mcts/*, data/splits.py); this folder only orchestrates
the six pipeline steps.

Output layout (``output`` block of the config):

    <output.dir>/<benchmark>/{trajectories,labeled,sampled,scored,rationales}.jsonl
    <output.dataset_dir>/<benchmark>/{train,val[,test]}.jsonl, pairs_*.jsonl, stats.json

Mock runs write under ``<...>/mock/`` so a smoke run never touches real data.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

STAGES = ("trajectories", "labeled", "sampled", "scored", "rationales")


def load_env(env_path: str | Path | None = None) -> None:
    """Load KEY=VALUE lines from DatasetConstruct/.env into the environment.

    Existing environment variables are never overridden, so an exported key
    always wins over the file.
    """
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ('"', "'"):
            value = value[1:].split(value[0], 1)[0]   # quoted: keep everything inside
        else:
            value = value.split(" #", 1)[0].strip()    # unquoted: drop an inline comment
        os.environ.setdefault(key.strip(), value)


def resolve(path: str | Path) -> Path:
    """Repo-root-relative path resolution."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def benchmark_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("benchmark", "infoseek"))


PROVIDER_DIR = Path(__file__).resolve().parent / "configs" / "providers"
# Keys that identify an endpoint; everything else (prompt, temperature, ...) stays.
_ENDPOINT_KEYS = ("provider", "model", "base_url", "api_key_env", "extra_body")


def provider_names() -> list[str]:
    return sorted(p.stem for p in PROVIDER_DIR.glob("*.yaml"))


def apply_provider(cfg: dict[str, Any], target: str, name: str | None) -> dict[str, Any]:
    """Copy of ``cfg`` whose ``rationale`` or ``judge`` (answer-support) generation
    block uses the endpoint preset ``configs/providers/<name>.yaml`` (or a path).

    Endpoint keys of the old block are dropped (so e.g. DeepSeek's ``extra_body``
    never leaks into a Kimi request); prompt and sampling settings are kept
    unless the preset sets them too.
    """
    if not name:
        return cfg
    import copy

    import yaml

    path = Path(name) if name.endswith((".yaml", ".yml")) else PROVIDER_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"provider preset {name!r} not found; available: {provider_names()}"
        )
    preset = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = copy.deepcopy(cfg)
    if target == "rationale":
        holder = out.setdefault("rationale", {})
    elif target == "judge":
        holder = out.setdefault("verifier", {}).setdefault("support", {})
    else:
        raise ValueError(f"target must be 'rationale' or 'judge', got {target!r}")
    block = {k: v for k, v in (holder.get("generation") or {}).items() if k not in _ENDPOINT_KEYS}
    block.update(preset)
    block["backend"] = "api"
    holder["generation"] = block
    holder["backend"] = "api"
    holder["provider_preset"] = path.stem
    return out


_PATH_KEYS = (
    ("data", "data_dir"),
    ("retriever", "clip_model"),
    ("retriever", "image", "model_name"),
    ("retriever", "image", "index_dir"),
    ("retriever", "bm25_index_dir"),
    ("verifier", "clip_model"),
)


def with_resolved_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``cfg`` whose repo-relative data / model paths are absolute, so
    a run behaves the same from any working directory. Model names that are
    not local paths (e.g. a hub id) are left untouched."""
    import copy

    out = copy.deepcopy(cfg)
    for keys in _PATH_KEYS:
        node = out
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict) or not node.get(keys[-1]):
            continue
        value = str(node[keys[-1]])
        if keys[-1] == "data_dir" or keys[-1].endswith("index_dir") or resolve(value).exists():
            node[keys[-1]] = str(resolve(value))
    return out


def stage_path(cfg: dict[str, Any], stage: str, mock: bool = False) -> Path:
    """Intermediate file of one pipeline stage for the configured benchmark."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    out = cfg.get("output", {})
    if out.get(stage):  # explicit per-stage path (--input/--output, ad-hoc runs): used as given
        return resolve(out[stage])
    base = resolve(out.get("dir", "data/construction"))
    if mock:
        base = base / "mock"
    return base / benchmark_name(cfg) / f"{stage}.jsonl"


def dataset_dir(cfg: dict[str, Any], mock: bool = False) -> Path:
    base = resolve(cfg.get("output", {}).get("dataset_dir", "data/etbench_open"))
    if mock:
        base = base / "mock"
    return base / benchmark_name(cfg)


def data_dir(cfg: dict[str, Any]) -> Path | None:
    value = cfg.get("data", {}).get("data_dir")
    return resolve(value) if value else None


def mock_asset_dir(cfg: dict[str, Any]) -> Path:
    """Where the mock world paints its images (inside the mock output tree)."""
    return resolve(cfg.get("output", {}).get("dir", "data/construction")) / "mock" / "assets"


def portable_path(path: str | None, base: Path | None) -> str | None:
    """Runtime media path -> key relative to the benchmark's data_dir.

    Released records must not carry machine-absolute paths (SCHEMA §2.9);
    paths outside ``base`` are kept as-is.
    """
    if not path:
        return None
    if base is None:
        return path
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path


def image_resolver(base: Path | None):
    """Portable media key -> readable path (inverse of :func:`portable_path`)."""
    def _resolve(key: str) -> str:
        p = Path(key)
        if p.is_absolute() or base is None:
            return str(p)
        return str(base / p)
    return _resolve


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """JSONL rows (gzip-compressed when the name ends in .gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance must never crash a run
        return None


def git_dirty() -> bool | None:
    """True when tracked source files differ from the recorded commit."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def _deployment(gen_cfg: dict[str, Any]) -> str:
    backend = str(gen_cfg.get("backend", "api"))
    if backend != "api":
        return backend
    urls = gen_cfg.get("base_url") or []
    urls = urls if isinstance(urls, (list, tuple)) else [urls]
    local = any(u and ("127.0.0.1" in u or "localhost" in u) for u in urls)
    return "local" if local else "api"


def build_provenance(cfg: dict[str, Any], mock: bool = False) -> dict[str, Any]:
    """``gen_provenance`` record (SCHEMA §1.4) from the resolved config."""
    from evidencetree.eval.metrics import outcome_metric_name

    policy = dict(cfg.get("policy") or {})
    rationale = dict((cfg.get("rationale") or {}).get("generation") or {})
    judge = dict(((cfg.get("verifier") or {}).get("support") or {}).get("generation") or {})
    retriever = dict(cfg.get("retriever") or {})
    mcts = dict(cfg.get("mcts") or cfg.get("search") or {})
    return {
        "benchmark": benchmark_name(cfg),
        "policy_model": "mock" if mock else policy.get("model"),
        "policy_deployment": "mock" if mock else _deployment(policy),
        "policy_snapshot": policy.get("snapshot"),
        "retriever": {
            "backend": retriever.get("backend"),
            "clip_model": retriever.get("clip_model"),
            "top_k": retriever.get("top_k"),
            "image_top_k": (retriever.get("image") or {}).get("top_k"),
            "dedupe": retriever.get("dedupe"),
        },
        "grounding": {
            k: (cfg.get("verifier") or {}).get(k)
            for k in ("backend", "clip_model", "text_cos_lo", "text_cos_hi",
                      "image_cos_lo", "image_cos_hi", "window_words", "alpha")
        },
        "support_judge_model": "mock" if mock else judge.get("model"),
        "rationale_model": "mock" if mock else rationale.get("model"),
        "rationale_deployment": "mock" if mock else _deployment(rationale),
        "mcts": {k: mcts.get(k) for k in ("rollouts", "max_depth", "top_k_children",
                                          "c_uct", "expansion")},
        "outcome_metric": outcome_metric_name(benchmark_name(cfg)),
        "seed": (cfg.get("data") or {}).get("seed"),
        "sampling": dict(cfg.get("sampling") or {}),
        "pipeline_commit": git_commit(),
        "pipeline_dirty": git_dirty(),
        "dataset_version": cfg.get("dataset_version", "v0.1"),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
