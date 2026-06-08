"""End-to-end smoke test for the Stage 0.1 pilot in mock mode."""

import importlib.util
import json
from pathlib import Path

import pytest

# Load the pilot script by path (it lives in pilot/, not in the package).
_PILOT_PATH = Path(__file__).resolve().parents[1] / "pilot" / "stage0_1_visualprm_diagnosis.py"
_spec = importlib.util.spec_from_file_location("stage0_1_pilot", _PILOT_PATH)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


@pytest.fixture
def mock_cfg(tmp_path):
    return {
        "n_queries": 20,
        "data": {"benchmark": "infoseek", "seed": 0},
        "retriever": {"type": "bm25", "top_k": 3},
        "generation": {"backend": "mock", "mock_accuracy": 0.6, "mock_seed": 0},
        "prm": {"mock_correlation": 0.9, "mock_seed": 0},
        "outcome": {"metric": "exact_match"},
        "output": {"report_dir": str(tmp_path / "reports")},
        "thresholds": {"needs_finetune_below": 0.3, "already_good_above": 0.6},
    }


def test_diagnose_runs_end_to_end(mock_cfg):
    result = pilot.diagnose(mock_cfg, mock=True)
    assert len(result["records"]) > 0
    a = result["analysis"]
    assert a["n"] == len(result["records"])
    for r in result["records"]:
        assert 0.0 <= r["prm_score"] <= 1.0
        assert r["outcome"] in (0.0, 1.0)


def test_high_mock_correlation_yields_high_spearman(mock_cfg):
    """With mock_correlation=0.9, PRM score should track outcome strongly."""
    result = pilot.diagnose(mock_cfg, mock=True)
    assert result["analysis"]["spearman"] is not None
    assert result["analysis"]["spearman"] > 0.5


def test_report_files_written(mock_cfg):
    result = pilot.diagnose(mock_cfg, mock=True)
    base = pilot.write_report(result, mock_cfg, mock=True)
    assert base.with_suffix(".json").exists()
    assert base.with_suffix(".md").exists()
    payload = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["verdict"]["decision"] in {"CONTINUE", "STOP", "GREY", "INCONCLUSIVE"}


def test_verdict_thresholds():
    th = {"needs_finetune_below": 0.3, "already_good_above": 0.6}
    assert pilot.verdict(0.1, th)[0] == "CONTINUE"
    assert pilot.verdict(0.8, th)[0] == "STOP"
    assert pilot.verdict(0.45, th)[0] == "GREY"
    assert pilot.verdict(None, th)[0] == "INCONCLUSIVE"


def test_main_mock_runs(monkeypatch, tmp_path):
    """The CLI entry point runs in mock mode and returns 0."""
    cfg_path = tmp_path / "pilot.yaml"
    cfg_path.write_text(
        "n_queries: 20\n"
        "data: {benchmark: infoseek, seed: 0}\n"
        "retriever: {top_k: 3}\n"
        "generation: {backend: mock, mock_accuracy: 0.6}\n"
        "prm: {mock_correlation: 0.5}\n"
        "outcome: {metric: exact_match}\n"
        f"output: {{report_dir: {tmp_path / 'reports'}}}\n"
        "thresholds: {needs_finetune_below: 0.3, already_good_above: 0.6}\n",
        encoding="utf-8",
    )
    rc = pilot.main(["--config", str(cfg_path), "--mock"])
    assert rc == 0
