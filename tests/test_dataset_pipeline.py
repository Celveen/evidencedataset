"""End-to-end mock test of the DatasetConstruct pipeline (steps 1-6)."""

import json
import sys
from pathlib import Path

import pytest

_DC = Path(__file__).resolve().parents[1] / "DatasetConstruct"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

import step1_gen_trajectories  # noqa: E402
import step2_label  # noqa: E402
import step3_sample  # noqa: E402
import step4_judge_support  # noqa: E402
import step5_gen_rationales  # noqa: E402
import step6_finalize  # noqa: E402
import validate_dataset  # noqa: E402
from common import read_jsonl, stage_path  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return {
        "benchmark": "infoseek",
        "data": {"data_dir": None, "n_queries": 40, "seed": 0},
        "mcts": {"rollouts": 6, "max_depth": 3, "top_k_children": 3,
                 "early_stop_q": 2.0, "seed": 0, "expansion": "diverse"},
        "policy": {"backend": "mock", "mock_accuracy": 0.6, "mock_seed": 0},
        "retriever": {"backend": "hybrid", "top_k": 3, "dedupe": "entity",
                      "image_search": True, "image": {"top_k": 1}},
        "prm": {"scorer": "overlap"},
        "verifier": {"backend": "clip", "alpha": 0.5},
        "rationale": {"backend": "mock", "max_attempts": 3},
        "quality": {"min_steps": 2, "max_steps": 8,
                    "max_grounding_outcome_gap": 0.7,
                    "dedupe_within_query": True, "val_fraction": 0.2},
        "sampling": {"budget_per_query": 3, "max_per_query": 9, "seed": 0},
        "pairs": {"margin": 0.3},
        "output": {"dir": str(tmp_path / "construction"),
                   "dataset_dir": str(tmp_path / "etbench")},
    }


def _run_all(cfg):
    for mod in (step1_gen_trajectories, step2_label, step3_sample,
                step4_judge_support, step5_gen_rationales):
        mod.run(cfg, mock=True)
    return step6_finalize.run(cfg, mock=True)


def test_pipeline_end_to_end_mock(cfg):
    out_dir = _run_all(cfg)

    # Step 1 — trajectories cover the full action space with portable paths.
    trajs = list(read_jsonl(stage_path(cfg, "trajectories", mock=True)))
    assert trajs
    for t in trajs:
        assert t["steps"][-1]["action_type"] == "answer"
        assert t["outcome_em"] in (0.0, 1.0)
        assert not str(t.get("image_path") or "").startswith("/")
    step_types = {s["action_type"] for t in trajs for s in t["steps"]}
    assert step_types == {"text_search", "image_search", "answer"}

    # Step 1 resume: re-running adds nothing new.
    n_before = len(trajs)
    step1_gen_trajectories.run(cfg, mock=True)
    assert len(list(read_jsonl(stage_path(cfg, "trajectories", mock=True)))) == n_before

    # Step 2 — every step labeled; answers pending the judge.
    labeled = list(read_jsonl(stage_path(cfg, "labeled", mock=True)))
    assert len(labeled) == sum(len(t["steps"]) for t in trajs)
    for s in labeled:
        if s["action"]["type"] == "answer":
            assert s["answer_support_label"] == "pending"
            assert s["local_grounding"] is None
        else:
            assert 0.0 <= s["local_grounding"] <= 1.0
            assert s["grounding_detail"]["modality"] in ("text", "image", None)

    # Tree-level credit sanity: per query, trajectory counts of the distinct
    # first nodes add up to that query's trajectory total.
    by_query = {}
    for t in trajs:
        by_query.setdefault(t["query_id"], []).append(t)
    for q, ts in by_query.items():
        firsts = {s["node_key"]: s["n_traj_through"] for s in labeled
                  if s["query_id"] == q and not s["state"]["actions_before"]}
        assert sum(firsts.values()) == len(ts)

    # Step 3 — unique nodes, within budget (+ one group of overshoot).
    sampled = list(read_jsonl(stage_path(cfg, "sampled", mock=True)))
    keys = [(s["query_id"], s["node_key"]) for s in sampled]
    assert len(keys) == len(set(keys))
    n_queries = len(by_query)
    assert 0 < len(sampled) <= 3 * n_queries + 8
    report = json.loads(step3_sample.report_path(cfg, True).read_text())
    assert report["queries_covered"] > 0.8 * n_queries

    # Step 4 — only correct answers reach the judge.
    scored = list(read_jsonl(stage_path(cfg, "scored", mock=True)))
    assert len(scored) == len(sampled)
    for s in scored:
        if s["action"]["type"] == "answer":
            assert s["answer_support_label"] != "pending"
            if s["outcome_credit"] == 0:
                assert s["answer_support_label"] == "skipped_incorrect"
                assert s["score"] == 0.0

    # Step 6 — conformant, query-disjoint splits with provenance, pairs, stats.
    train = list(read_jsonl(out_dir / "train.jsonl"))
    val = list(read_jsonl(out_dir / "val.jsonl"))
    assert train and val
    assert not ({s["query_id"] for s in train} & {s["query_id"] for s in val})
    assert all(s["gen_provenance"]["policy_model"] == "mock" for s in train + val)
    stats = json.loads((out_dir / "stats.json").read_text())
    assert stats["validation"]["violations"] == 0
    assert stats["splits"]["train"]["samples"] == len(train)
    assert len(stats["splits"]["train"]["hist_score"]) == 10
    pairs = list(read_jsonl(out_dir / "pairs_train.jsonl"))
    ids = {s["sample_id"] for s in train}
    assert pairs and all(p["chosen_sample_id"] in ids and p["rejected_sample_id"] in ids
                         for p in pairs)
    kw = dict(min_steps=2, max_steps=8, max_gap=0.7, require_provenance=True)
    assert validate_dataset.validate_file(out_dir / "train.jsonl", **kw)["violations"] == []


def test_step3_keeps_existing_sample_unless_forced(cfg):
    for mod in (step1_gen_trajectories, step2_label):
        mod.run(cfg, mock=True)
    out = step3_sample.run(cfg, mock=True)
    first = out.read_text()
    cfg["sampling"]["budget_per_query"] = 1
    step3_sample.run(cfg, mock=True)
    assert out.read_text() == first  # paid steps downstream never see it change
    step3_sample.run(cfg, mock=True, force=True)
    assert len(out.read_text().splitlines()) < len(first.splitlines())


def test_step2_requires_step1_output(cfg):
    with pytest.raises(FileNotFoundError):
        step2_label.run(cfg, mock=True)


def _write_benchmark(data_dir: Path, ids: list[str]) -> None:
    data_dir.mkdir()
    (data_dir / "queries.jsonl").write_text(
        "".join(
            json.dumps({"query_id": i, "question": f"Which option is correct for {i}?",
                        "gold_answers": ["Alpha"], "image_path": None}) + "\n"
            for i in ids
        ),
        encoding="utf-8",
    )
    (data_dir / "corpus.jsonl").write_text(
        json.dumps({"doc_id": "external-doc", "title": "Supporting note",
                    "text": "Alpha is the correct option."}) + "\n",
        encoding="utf-8",
    )


def test_step1_loads_only_seed_ids(cfg, tmp_path):
    data_dir = tmp_path / "converted"
    _write_benchmark(data_dir, ["s1", "s2", "other"])
    (tmp_path / "seed.txt").write_text("s1\ns2\n")
    (tmp_path / "eval.txt").write_text("e1\n")
    cfg["benchmark"] = "scienceqa"
    cfg["retriever"] = {"backend": "bm25", "top_k": 3}
    cfg["data"] = {"data_dir": str(data_dir), "n_queries": 10, "seed": 0,
                   "seed_ids": str(tmp_path / "seed.txt"),
                   "eval_ids": str(tmp_path / "eval.txt"), "require_splits": True}

    out = step1_gen_trajectories.run(cfg, mock=False, force=True)
    assert {r["query_id"] for r in read_jsonl(out)} == {"s1", "s2"}


def test_step1_refuses_eval_queries(cfg, tmp_path):
    data_dir = tmp_path / "converted"
    _write_benchmark(data_dir, ["s1", "e1"])
    (tmp_path / "eval.txt").write_text("e1\n")
    cfg["benchmark"] = "scienceqa"
    cfg["retriever"] = {"backend": "bm25", "top_k": 3}
    cfg["data"] = {"data_dir": str(data_dir), "n_queries": 10, "seed": 0,
                   "eval_ids": str(tmp_path / "eval.txt")}
    with pytest.raises(ValueError, match="held-out"):
        step1_gen_trajectories.run(cfg, mock=False, force=True)


def test_step1_requires_split_files_when_configured(cfg, tmp_path):
    data_dir = tmp_path / "converted"
    _write_benchmark(data_dir, ["s1"])
    cfg["benchmark"] = "scienceqa"
    cfg["data"] = {"data_dir": str(data_dir), "n_queries": 1, "require_splits": True}
    with pytest.raises(ValueError, match="seed_ids"):
        step1_gen_trajectories.run(cfg, mock=False, force=True)


def test_steps_4_5_run_from_explicit_files_and_provenance_follows_rows(cfg, tmp_path):
    """Steps 4-5 can run elsewhere from copied files; step 6 records the models
    the samples were actually made with, not the local config's."""
    for mod in (step1_gen_trajectories, step2_label, step3_sample):
        mod.run(cfg, mock=True)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    sampled = moved / "sampled.jsonl"
    sampled.write_text(stage_path(cfg, "sampled", mock=True).read_text())
    cfg_file = tmp_path / "cfg.yaml"
    import yaml

    cfg_file.write_text(yaml.safe_dump(cfg))
    assert step4_judge_support.main(["--config", str(cfg_file), "--mock",
                                     "--input", str(sampled),
                                     "--output", str(moved / "scored.jsonl")]) == 0
    assert step5_gen_rationales.main(["--config", str(cfg_file), "--mock",
                                      "--input", str(moved / "scored.jsonl"),
                                      "--output", str(moved / "rationales.jsonl")]) == 0
    rows = list(read_jsonl(moved / "rationales.jsonl"))
    assert rows and all(r["rationale_model"] == "mock" for r in rows)
    assert any(r.get("support_judge_model") == "lexical" for r in rows)

    for r in rows[: len(rows) // 2]:
        r["rationale_model"] = "kimi-test"
    prov = {"rationale_model": "deepseek-v4-pro", "support_judge_model": "deepseek-v4-pro"}
    step6_finalize._models_from_rows(prov, rows)
    assert prov["rationale_model"] == ["kimi-test", "mock"]
    assert prov["rationale_model_from_config"] == "deepseek-v4-pro"
    assert prov["support_judge_model"] == "lexical"


def test_load_env_strips_inline_comments(tmp_path, monkeypatch):
    from common import load_env

    env = tmp_path / ".env"
    env.write_text('A_KEY=sk-123   # comment\nB_KEY="sk #456"  # c\nC_KEY=plain\n# D_KEY=x\n')
    for k in ("A_KEY", "B_KEY", "C_KEY", "D_KEY"):
        monkeypatch.delenv(k, raising=False)
    load_env(env)
    import os

    assert os.environ["A_KEY"] == "sk-123" and os.environ["B_KEY"] == "sk #456"
    assert os.environ["C_KEY"] == "plain" and "D_KEY" not in os.environ


def test_try_rationale_model_mock_builtin_samples(capsys):
    import try_rationale_model

    assert try_rationale_model.main(["--mock", "--n", "3"]) == 0
    out = capsys.readouterr().out
    assert "QC pass 3/3" in out


def test_step5_group_mode_sends_fewer_requests(cfg):
    for mod in (step1_gen_trajectories, step2_label, step3_sample, step4_judge_support):
        mod.run(cfg, mock=True)
    scored = list(read_jsonl(stage_path(cfg, "scored", mock=True)))
    from evidencetree.prm.rationale_gen import group_samples

    n_requests = len(group_samples(scored, 6))
    assert n_requests < len(scored)
    step5_gen_rationales.run(cfg, mock=True)
    rows = list(read_jsonl(stage_path(cfg, "rationales", mock=True)))
    modes = {r["rationale_mode"] for r in rows}
    assert "group" in modes and all(r["rationale_qc_pass"] for r in rows)
    cfg["rationale"]["batch"] = "node"
    step5_gen_rationales.run(cfg, mock=True, force=True)
    rows = list(read_jsonl(stage_path(cfg, "rationales", mock=True)))
    assert {r["rationale_mode"] for r in rows} == {"single"}
