"""Config loading: `_base_` inheritance, deep merge, dotted overrides."""

from pathlib import Path

import pytest

from evidencetree.utils.config import deep_merge, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_base_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "a: 1\nnested:\n  x: 1\n  y: 2\nlist: [1, 2]\n"
    )
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\nnested:\n  y: 3\nlist: [9]\nb: 2\n"
    )
    cfg = load_config(tmp_path / "child.yaml", overrides=["nested.x=5", "c.d=true"])
    assert cfg == {"a": 1, "nested": {"x": 5, "y": 3}, "list": [9], "b": 2,
                   "c": {"d": True}}


def test_base_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"k": {"a": 1}}
    merged = deep_merge(base, {"k": {"b": 2}})
    assert merged == {"k": {"a": 1, "b": 2}} and base == {"k": {"a": 1}}


@pytest.mark.parametrize("bench", ["infoseek", "scienceqa", "gqa"])
def test_benchmark_configs_resolve(bench):
    cfg = load_config(ROOT / "DatasetConstruct" / "configs" / f"{bench}.yaml")
    assert cfg["benchmark"] == bench
    assert cfg["data"]["n_queries"] == 20000
    assert cfg["data"]["seed_ids"].endswith(f"splits/{bench}/seed_ids.txt")
    assert cfg["retriever"]["backend"] == "hybrid"
    assert cfg["retriever"]["top_k"] == 3 and cfg["retriever"]["image"]["top_k"] == 1
    assert cfg["mcts"]["expansion"] == "diverse"
    assert cfg["sampling"]["budget_per_query"] == 3
