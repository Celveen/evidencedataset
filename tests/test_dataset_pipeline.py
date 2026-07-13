"""End-to-end mock test of the DatasetConstruct pipeline (steps 1-4)."""

import json
import sys
from pathlib import Path

import pytest

_DC = Path(__file__).resolve().parents[1] / "DatasetConstruct"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

import step1_gen_trajectories  # noqa: E402
import step2_score_labels  # noqa: E402
import step3_gen_rationales  # noqa: E402
import step4_quality_filter  # noqa: E402
from common import read_jsonl  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return {
        "benchmark": "infoseek",
        "data": {"data_dir": None, "n_queries": 8, "seed": 0},
        "mcts": {"rollouts": 6, "max_depth": 3, "top_k_children": 3,
                 "lam": 0.3, "early_stop_q": 2.0, "seed": 0},
        "policy": {"backend": "mock", "mock_accuracy": 0.6, "mock_seed": 0},
        "retriever": {"top_k": 3},
        "verifier": {"backend": "lexical", "alpha": 0.5},
        "rationale": {"backend": "mock", "max_attempts": 3},
        "quality": {"min_steps": 2, "max_steps": 8,
                    "max_grounding_outcome_gap": 0.95,
                    "dedupe_within_query": True, "val_fraction": 0.2},
        "output": {
            "trajectories": str(tmp_path / "trajs.jsonl"),
            "scored": str(tmp_path / "scored.jsonl"),
            "rationales": str(tmp_path / "rationales.jsonl"),
            "dataset_dir": str(tmp_path / "etbench"),
        },
    }


def test_pipeline_end_to_end_mock(cfg):
    # Step 1 — trajectories
    p1 = step1_gen_trajectories.run(cfg, mock=True)
    trajs = list(read_jsonl(p1))
    assert trajs, "step 1 produced no trajectories"
    by_query = {}
    for t in trajs:
        by_query.setdefault(t["query_id"], []).append(t)
        assert t["steps"], "trajectory without steps"
        assert t["steps"][-1]["action_type"] == "answer"
        assert t["outcome_em"] in (0.0, 1.0)
        assert all("evidence" in s for s in t["steps"])

    # Step 1 resume: re-running adds nothing new
    n_before = len(trajs)
    step1_gen_trajectories.run(cfg, mock=True)
    assert len(list(read_jsonl(p1))) == n_before

    # Step 2 — score labels with tree-level credit
    p2 = step2_score_labels.run(cfg, mock=True)
    samples = list(read_jsonl(p2))
    assert len(samples) == sum(len(t["steps"]) for t in trajs)
    for s in samples:
        assert 0.0 <= s["score"] <= 1.0
        assert 0.0 <= s["outcome_credit"] <= 1.0
        if s["action"]["type"] == "answer":
            assert s["local_grounding"] is None
        else:
            assert 0.0 <= s["local_grounding"] <= 1.0

    # Tree-level credit sanity: per query, the trajectory counts of the
    # DISTINCT first nodes must add up to that query's trajectory total
    # (trajectories may diverge at the very first action).
    for q, ts in by_query.items():
        firsts = {}
        for s in samples:
            if s["query_id"] == q and not s["state"]["actions_before"]:
                key = (s["action"]["type"], s["action"]["input"])
                firsts[key] = s["n_traj_through"]
        assert sum(firsts.values()) == len(ts)

    # Step 3 — rationales (mock template passes QC)
    p3 = step3_gen_rationales.run(cfg, mock=True)
    rated = list(read_jsonl(p3))
    assert len(rated) == len(samples)
    assert all("rationale" in s for s in rated)
    pass_rate = sum(s["rationale_qc_pass"] for s in rated) / len(rated)
    assert pass_rate > 0.9

    # Step 4 — quality filter + split
    out_dir = step4_quality_filter.run(cfg, mock=True)
    train = list(read_jsonl(out_dir / "train.jsonl"))
    val = list(read_jsonl(out_dir / "val.jsonl"))
    stats = json.loads((out_dir / "stats.json").read_text())
    assert stats["kept"] == len(train) + len(val) > 0
    assert stats["input_samples"] == len(samples)
    # no query appears in both splits
    assert not ({s["query_id"] for s in train} & {s["query_id"] for s in val})
    # every kept sample passed QC
    assert all(s["rationale_qc_pass"] for s in train + val)


def test_step2_requires_step1_output(cfg):
    with pytest.raises(FileNotFoundError):
        step2_score_labels.run(cfg, mock=True)


def test_step1_loads_converted_benchmark(cfg, tmp_path):
    data_dir = tmp_path / "converted"
    data_dir.mkdir()
    (data_dir / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": "external-1",
                "question": "Which option is correct?",
                "gold_answers": ["Alpha"],
                "image_path": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "corpus.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "external-doc",
                "title": "Supporting note",
                "text": "Alpha is the correct option.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg["benchmark"] = "scienceqa"
    cfg["data"] = {"data_dir": str(data_dir), "n_queries": 1, "seed": 0}

    out = step1_gen_trajectories.run(cfg, mock=False, force=True)
    rows = list(read_jsonl(out))

    assert rows
    assert rows[0]["query_id"] == "external-1"
