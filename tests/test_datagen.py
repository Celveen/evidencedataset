"""Tests for grounding verifier, tree-level credit, and rationale QC."""

import pytest

from evidencetree.prm.data_gen import (
    fuse_answer_score,
    fuse_score,
    label_steps,
    tree_level_credit,
)
from evidencetree.prm.rationale_gen import (
    RationaleGenerator,
    check_rationale,
    split_verdict,
)
from evidencetree.prm.verifiers import AnswerSupportVerifier, GroundingVerifier


def test_text_result_relevance_graded():
    verifier = GroundingVerifier(clip_scorer=None)
    q = "How heavy is this bird in grams?"
    relevant = verifier.score(
        question=q,
        action_type="text_search",
        result_texts=["The bird weighs about 400 grams when heavy."],
    )
    irrelevant = verifier.score(
        question=q,
        action_type="text_search",
        result_texts=["Paris is the capital of France."],
    )
    assert relevant > irrelevant
    assert relevant == pytest.approx(1.0)
    assert irrelevant == pytest.approx(0.0)


def test_empty_results_score_zero():
    verifier = GroundingVerifier(clip_scorer=None)
    assert verifier.score(question="q?", action_type="text_search", result_texts=[]) == 0.0
    assert verifier.score(question="q?", action_type="image_search", result_image_paths=[]) == 0.0


def test_verifier_answer_returns_none():
    verifier = GroundingVerifier(clip_scorer=None)
    assert verifier.score(question="q?", action_type="answer", result_texts=["x"]) is None


def test_image_result_action_without_scorer_is_neutral_when_results_exist():
    verifier = GroundingVerifier(clip_scorer=None)
    score = verifier.score(
        question="q?",
        action_type="image_search",
        result_image_paths=["/some/img.jpg"],
    )
    assert score == 0.5


def _traj(traj_id, query_id, steps, outcome):
    return {
        "traj_id": traj_id,
        "query_id": query_id,
        "question": "q?",
        "image_path": None,
        "outcome_em": outcome,
        "steps": [
            {"step_index": i, "action_type": t, "action_input": inp, "evidence": []}
            for i, (t, inp) in enumerate(steps)
        ],
    }


def test_tree_level_credit_is_per_node_not_per_trajectory():
    t1 = _traj("t1", "q1", [("text_search", "A"), ("answer", "right")], 1.0)
    t2 = _traj("t2", "q1", [("text_search", "A"), ("answer", "wrong")], 0.0)
    credit = tree_level_credit([t1, t2])

    shared = ("q1", (("text_search", "A"),))
    assert credit[shared] == (pytest.approx(0.5), 2)
    assert credit[("q1", (("text_search", "A"), ("answer", "right")))] == (1.0, 1)
    assert credit[("q1", (("text_search", "A"), ("answer", "wrong")))] == (0.0, 1)


def test_tree_level_credit_does_not_leak_across_queries():
    t1 = _traj("t1", "q1", [("text_search", "A")], 1.0)
    t2 = _traj("t2", "q2", [("text_search", "A")], 0.0)
    credit = tree_level_credit([t1, t2])
    assert credit[("q1", (("text_search", "A"),))] == (1.0, 1)
    assert credit[("q2", (("text_search", "A"),))] == (0.0, 1)


def test_fuse_score_answer_steps_are_outcome_only():
    assert fuse_score(None, 0.8, alpha=0.5) == pytest.approx(0.8)
    assert fuse_score(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)
    assert fuse_score(1.0, 0.0, alpha=0.2) == pytest.approx(0.2)


def test_answer_support_lexical_supported_vs_unsupported():
    verifier = AnswerSupportVerifier(backend="lexical")
    supported = verifier.score(
        question="When did it open?",
        answer="September 7, 1963",
        evidence_texts=["The Hall opened on September 7, 1963 in Canton."],
    )
    assert supported == pytest.approx(1.0)

    unsupported = verifier.score(
        question="How heavy in grams?",
        answer="Up to 145 grams",
        evidence_texts=["The white-winged dove is larger and heavier."],
    )
    assert unsupported == pytest.approx(0.0)


def test_answer_support_lexical_edge_cases():
    verifier = AnswerSupportVerifier(backend="lexical")
    assert verifier.score(question="q?", answer="No", evidence_texts=["some text"]) is None
    assert verifier.score(question="q?", answer="Paris", evidence_texts=[]) == 0.0


def test_fuse_answer_score_preserves_ordering():
    wrong = fuse_answer_score(0.0, 1.0)
    correct_unsupported = fuse_answer_score(1.0, 0.0)
    correct_supported = fuse_answer_score(1.0, 1.0)
    assert wrong < correct_unsupported < correct_supported
    assert correct_unsupported == pytest.approx(0.3)
    assert fuse_answer_score(0.8, None) == pytest.approx(0.8)


def test_label_steps_snapshots_state_before_action():
    t1 = _traj("t1", "q1", [("text_search", "A"), ("answer", "x")], 1.0)
    t1["steps"][0]["evidence"] = [
        {
            "evidence_id": "e0",
            "doc_id": "d0",
            "title": "T",
            "text": "ev text",
            "score": 1.0,
        }
    ]
    samples = label_steps([t1], verifier=GroundingVerifier(), alpha=0.5)
    assert len(samples) == 2
    s0, s1 = samples
    assert s0["state"]["evidence_before"] == []
    assert s1["state"]["evidence_before"][0]["evidence_id"] == "e0"
    assert s1["state"]["actions_before"] == ["text_search(A)"]
    assert s0["observation_evidence"][0]["evidence_id"] == "e0"
    assert s1["local_grounding"] is None
    assert s1["score"] == pytest.approx(s1["outcome_credit"])


def test_label_steps_flags_unsupported_correct_answer():
    traj = _traj(
        "t1",
        "q1",
        [("text_search", "A"), ("answer", "Up to 145 grams")],
        1.0,
    )
    traj["steps"][0]["evidence"] = [
        {
            "evidence_id": "e0",
            "doc_id": "d0",
            "title": "Doves",
            "text": "The white-winged dove is larger than the mourning dove.",
            "score": 1.0,
        }
    ]
    samples = label_steps(
        [traj],
        verifier=GroundingVerifier(),
        support_verifier=AnswerSupportVerifier(backend="lexical"),
    )
    answer = samples[-1]
    assert answer["action"]["type"] == "answer"
    assert answer["answer_support"] == pytest.approx(0.0)
    assert answer["unsupported_correct"] is True
    assert answer["score"] == pytest.approx(0.3)
    assert samples[0]["answer_support"] is None
    assert samples[0]["unsupported_correct"] is False


def _sample():
    return {
        "sample_id": "t1#s0",
        "question": "q?",
        "state": {"actions_before": [], "evidence_before": []},
        "action": {"type": "text_search", "input": "A"},
        "observation_evidence": [{"evidence_id": "e0", "title": "T", "text": "x"}],
        "score": 0.7,
        "gold_answers": ["Carson City"],
    }


def test_check_rationale_rules():
    sample = _sample()
    filler = "This text_search action is appropriate because " * 9
    ok, reasons = check_rationale(filler + "see [e0].\nVERDICT: good", sample)
    assert ok, reasons

    no_cite = filler + "no citation here.\nVERDICT: good"
    assert not check_rationale(no_cite, sample)[0]

    no_type = ("The action is appropriate because " * 11) + "see [e0].\nVERDICT: good"
    assert not check_rationale(no_type, sample)[0]

    too_short = "text_search is fine [e0].\nVERDICT: good"
    assert not check_rationale(too_short, sample)[0]


def test_check_rationale_requires_consistent_verdict():
    sample = _sample()
    body = ("This text_search action is appropriate because " * 9) + "see [e0]."
    ok, reasons = check_rationale(body, sample)
    assert not ok and any("VERDICT" in reason for reason in reasons)
    assert not check_rationale(body + "\nVERDICT: poor", sample)[0]
    assert check_rationale(body + "\nVERDICT: mixed", sample)[0]

    low = dict(sample, score=0.2)
    assert not check_rationale(body + "\nVERDICT: good", low)[0]
    assert check_rationale(body + "\nVERDICT: poor", low)[0]


def test_check_rationale_rejects_gt_leakage():
    sample = _sample()
    filler = "This text_search action is appropriate because " * 9
    leak_phrase = filler + "it matches the ground truth, see [e0].\nVERDICT: good"
    ok, reasons = check_rationale(leak_phrase, sample)
    assert not ok and any("evaluation phrasing" in reason for reason in reasons)

    leak_gold = filler + "it points to Carson City, see [e0].\nVERDICT: good"
    ok, reasons = check_rationale(leak_gold, sample)
    assert not ok and any("gold answer" in reason for reason in reasons)

    evidenced = dict(sample)
    evidenced["observation_evidence"] = [
        {"evidence_id": "e0", "title": "T", "text": "The capital is Carson City."}
    ]
    assert check_rationale(leak_gold, evidenced)[0]


def test_split_verdict():
    body, verdict = split_verdict("Reasoning text here.\nVERDICT: mixed")
    assert body == "Reasoning text here." and verdict == "mixed"
    assert split_verdict("no verdict at all") == ("no verdict at all", None)


def test_mock_rationale_generator_passes_qc():
    gen = RationaleGenerator(backend="mock")
    fields = gen.generate_for(_sample())
    assert fields["rationale_qc_pass"], fields["rationale_qc_reasons"]
    assert fields["rationale_attempts"] == 1
    assert "[e0]" in fields["rationale"]
    assert fields["rationale_verdict"] == "good"
    assert "VERDICT" not in fields["rationale"]
