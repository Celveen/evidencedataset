"""Lightweight YAML config loading with dotted-key CLI overrides.

We deliberately avoid a hard dependency on hydra/omegaconf so the base install
stays light. A config is just a nested ``dict``; ``--set a.b=c`` style overrides
are applied on top.

A config may inherit from another one with a top-level ``_base_: other.yaml``
key (resolved relative to the inheriting file). Nested dicts are merged key by
key; any other value in the inheriting file replaces the base value.
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
    cfg = _load_with_base(Path(path), chain=())

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key, raw_value = override.split("=", 1)
        _set_dotted(cfg, key.strip(), _parse_value(raw_value.strip()))
    return cfg


def _load_with_base(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    resolved = path.resolve()
    if resolved in chain:
        raise ValueError(f"Circular _base_ chain: {' -> '.join(map(str, chain + (resolved,)))}")
    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if not base:
        return cfg
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return deep_merge(_load_with_base(base_path, chain + (resolved,)), cfg)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated by ``override``; nested dicts merge key by key."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


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
