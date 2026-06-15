"""Shared helpers for the DatasetConstruct pipeline (paths, .env, JSONL IO).

All algorithmic logic lives in the ``evidencetree`` package
(prm/verifiers.py, prm/data_gen.py, prm/rationale_gen.py, mcts/*); this folder
only orchestrates the four pipeline steps.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def resolve(path: str | Path) -> Path:
    """Repo-root-relative path resolution."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def tagged(path: Path, mock: bool) -> Path:
    """Prefix the filename with ``mock_`` in mock mode so smoke runs never
    pollute real outputs."""
    return path.with_name(f"mock_{path.name}") if mock else path


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
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
