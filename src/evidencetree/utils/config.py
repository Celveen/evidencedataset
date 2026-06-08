"""Lightweight YAML config loading with dotted-key CLI overrides.

We deliberately avoid a hard dependency on hydra/omegaconf so the base install
stays light. A config is just a nested ``dict``; ``--set a.b=c`` style overrides
are applied on top.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """Load a YAML config file and apply dotted-key overrides.

    Args:
        path: path to a YAML file.
        overrides: iterable of ``"key.subkey=value"`` strings. Values are parsed
            as Python literals when possible (``true`` -> bool, ``5`` -> int,
            ``0.3`` -> float), otherwise kept as strings.

    Returns:
        The merged config as a nested dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key, raw_value = override.split("=", 1)
        _set_dotted(cfg, key.strip(), _parse_value(raw_value.strip()))
    return cfg


def _parse_value(raw: str) -> Any:
    """Parse a string into a Python literal, falling back to the raw string."""
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"] = value`` for ``dotted_key == "a.b"``, creating dicts."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        existing = node.get(key)
        if not isinstance(existing, dict):
            existing = {}
            node[key] = existing
        node = existing
    node[keys[-1]] = value


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read ``cfg["a"]["b"]`` for ``dotted_key == "a.b"``, returning default if missing."""
    node: Any = cfg
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
