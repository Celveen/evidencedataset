"""Tests for grounding verifier, tree-level credit, and rationale QC (Stage 2/3)."""

import pytest

from evidencetree.prm.data_gen import fuse_score, label_steps, tree_level_credit
from evidencetree.prm.rationale_gen import RationaleGenerator, check_rationale
from evidencetree.prm.verifiers import GroundingVerifier


# --------------------------------------------------------------------------- #
# Verifier (lexical backend)
# --------------------------------------------------------------------------- #
def test_lexical_verifier_grades_query_alignment():
    v = GroundingVerifier(backend="lexical")
    q = "In which city is the Eiffel Tower located?"
    aligned = v.score(question=q, action_type="text_search", action_input="Eiffel Tower city")
    drifting = v.score(question=q, action_type="text_search", action_input="Taylor Swift songs")
    assert aligned == pytest.approx(1.0)
    assert drifting == pytest.approx(0.0)
    partial = v.score(question=q, action_type="text_search", action_input="Eiffel Tower Japan")
    assert 0.0 < partial < 1.0  # graded, not binary


def test_verifier_answer_returns_none_and_image_neutral():
    v = GroundingVerifier(backend="lexical")
    assert v.score(question="q?", action_type="answer", action_input="Paris") is None
    assert v.score(question="q?", action_type="image_search", action_input="") == 0.5


def test_api_verifier_requires_generator():
    with pytest.raises(ValueError):
        GroundingVerifier(backend="api")


# --------------------------------------------------------------------------- #
# Tree-level credit (the §4.1.1 bug guard)
# --------------------------------------------------------------------------- #
def _traj(traj_id, query_id, steps, outcome):
    return {
        "traj_id": traj_id, "query_id": query_id, "question": "q?",
        "image_path": None, "outcome_em": outcome,
        "steps": [
            {"step_index": i, "action_type": t, "action_input": inp, "evidence": []}
            for i, (t, inp) in enumerate(steps)
        ],
    }


def test_tree_level_credit_is_per_node_not_per_trajectory():
    """Two trajectories share the first action, then diverge: the shared
    prefix gets the MEAN outcome, each divergent suffix keeps its own."""
    t1 = _traj("t1", "q1", [("text_search", "A"), ("answer", "right")], outcome=1.0)
    t2 = _traj("t2", "q1", [("text_search", "A"), ("answer", "wrong")], outcome=0.0)
    credit = tree_level_credit([t1, t2])

    shared = ("q1", (("text_search", "A"),))
    assert credit[shared] == (pytest.approx(0.5), 2)  # mean over both, NOT 1.0 or 0.0
    assert credit[("q1", (("text_search", "A"), ("answer", "right")))] == (1.0, 1)
    assert credit[("q1", (("text_search", "A"), ("answer", "wrong")))] == (0.0, 1)


def test_tree_level_credit_does_not_leak_across_queries():
    t1 = _traj("t1", "q1", [("text_search", "A")], outcome=1.0)
    t2 = _traj("t2", "q2", [("text_search", "A")], outcome=0.0)  # same action, other query
    credit = tree_level_credit([t1, t2])
    assert credit[("q1", (("text_search", "A"),))] == (1.0, 1)
    assert credit[("q2", (("text_search", "A"),))] == (0.0, 1)


def test_fuse_score_answer_steps_are_outcome_only():
    assert fuse_score(None, 0.8, alpha=0.5) == pytest.approx(0.8)
    assert fuse_score(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)
    assert fuse_score(1.0, 0.0, alpha=0.2) == pytest.approx(0.2)


def test_label_steps_snapshots_state_before_action():
    t1 = _traj("t1", "q1", [("text_search", "A"), ("answer", "x")], outcome=1.0)
    t1["steps"][0]["evidence"] = [
        {"evidence_id": "e0", "doc_id": "d0", "title": "T", "text": "ev text", "score": 1.0}
    ]
    samples = label_steps([t1], verifier=GroundingVerifier(), alpha=0.5)
    assert len(samples) == 2
    s0, s1 = samples
    assert s0["state"]["evidence_before"] == []          # before the first action
    assert s1["state"]["evidence_before"][0]["evidence_id"] == "e0"
    assert s1["state"]["actions_before"] == ["text_search(A)"]
    assert s0["observation_evidence"][0]["evidence_id"] == "e0"
    assert s1["local_grounding"] is None                 # answer step
    assert s1["score"] == pytest.approx(s1["outcome_credit"])


# --------------------------------------------------------------------------- #
# Rationale QC
# --------------------------------------------------------------------------- #
def _sample():
    return {
        "sample_id": "t1#s0", "question": "q?",
        "state": {"actions_before": [], "evidence_before": []},
        "action": {"type": "text_search", "input": "A"},
        "observation_evidence": [{"evidence_id": "e0", "title": "T", "text": "x"}],
        "score": 0.7,
    }


def test_check_rationale_rules():
    sample = _sample()
    filler = "This text_search action is appropriate because " * 9  # 54 tokens
    ok, reasons = check_rationale(filler + "see [e0].", sample)
    assert ok, reasons

    no_cite = filler + "no citation here."
    assert not check_rationale(no_cite, sample)[0]

    no_type = ("The action is appropriate because " * 11) + "see [e0]."
    assert not check_rationale(no_type, sample)[0]

    too_short = "text_search is fine [e0]."
    assert not check_rationale(too_short, sample)[0]

    too_long = filler * 4 + "see [e0]."
    assert not check_rationale(too_long, sample)[0]


def test_mock_rationale_generator_passes_qc():
    gen = RationaleGenerator(backend="mock")
    fields = gen.generate_for(_sample())
    assert fields["rationale_qc_pass"], fields["rationale_qc_reasons"]
    assert fields["rationale_attempts"] == 1
    assert "[e0]" in fields["rationale"]
