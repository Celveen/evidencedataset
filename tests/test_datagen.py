"""Tests for grounding verifier, tree-level credit, and rationale QC (Stage 2/3)."""

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


# --------------------------------------------------------------------------- #
# Verifier — grounding = retrieved RESULT vs question (lexical fallback here)
# --------------------------------------------------------------------------- #
def test_text_result_relevance_graded():
    v = GroundingVerifier(clip_scorer=None)  # lexical fallback
    q = "How heavy is this bird in grams?"
    relevant = v.score(question=q, action_type="text_search",
                       result_texts=["The bird weighs about 400 grams when heavy."])
    irrelevant = v.score(question=q, action_type="text_search",
                         result_texts=["Paris is the capital of France."])
    assert relevant > irrelevant
    assert relevant == pytest.approx(1.0)      # all content words covered
    assert irrelevant == pytest.approx(0.0)


def test_empty_results_score_zero():
    v = GroundingVerifier(clip_scorer=None)
    assert v.score(question="q?", action_type="text_search", result_texts=[]) == 0.0
    # image_search with no images and no CLIP scorer -> 0.0
    assert v.score(question="q?", action_type="image_search", result_image_paths=[]) == 0.0


def test_verifier_answer_returns_none():
    v = GroundingVerifier(clip_scorer=None)
    assert v.score(question="q?", action_type="answer", result_texts=["x"]) is None


def test_image_result_action_without_scorer_is_neutral_when_results_exist():
    v = GroundingVerifier(clip_scorer=None)
    g = v.score(question="q?", action_type="image_search", result_image_paths=["/some/img.jpg"])
    assert g == 0.5  # can't judge images without CLIP, but something was retrieved


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
# Answer support (the "correct but unsupported" guard)
# --------------------------------------------------------------------------- #
def test_answer_support_lexical_supported_vs_unsupported():
    v = AnswerSupportVerifier(backend="lexical")
    ev = ["The Pro Football Hall of Fame opened on September 7, 1963 in Canton."]
    supported = v.score(question="When did it open?",
                        answer="September 7, 1963", evidence_texts=ev)
    assert supported == pytest.approx(1.0)
    # The mourning-dove case: generic words match but the number 145 is nowhere
    # in the evidence — the number gate must drive support to 0.
    ev2 = ["The white-winged dove is larger and heavier than the mourning dove."]
    unsupported = v.score(question="How heavy in grams?",
                          answer="Up to 145 grams", evidence_texts=ev2)
    assert unsupported == pytest.approx(0.0)


def test_answer_support_lexical_edge_cases():
    v = AnswerSupportVerifier(backend="lexical")
    # Bare yes/no cannot be verified by containment -> None (falls back to outcome)
    assert v.score(question="q?", answer="No", evidence_texts=["some text"]) is None
    # Empty evidence bundle -> answering is by definition unsupported
    assert v.score(question="q?", answer="Paris", evidence_texts=[]) == 0.0


def test_answer_support_api_parses_judge_reply():
    class FakeGen:
        def __init__(self, reply): self.reply = reply
        def generate(self, prompt, images): return self.reply

    def score(reply):
        v = AnswerSupportVerifier(backend="api", generator=FakeGen(reply))
        return v.score(question="q?", answer="a", evidence_texts=["ev"])

    assert score("SUPPORTED") == 1.0
    assert score("PARTIAL") == 0.5
    assert score("UNSUPPORTED") == 0.0
    assert score("I cannot tell") is None  # unparseable -> no fabricated value


def test_fuse_answer_score_preserves_ordering():
    wrong = fuse_answer_score(0.0, 1.0)
    correct_unsupported = fuse_answer_score(1.0, 0.0)
    correct_supported = fuse_answer_score(1.0, 1.0)
    assert wrong < correct_unsupported < correct_supported
    assert correct_unsupported == pytest.approx(0.3)   # lands at the floor
    assert fuse_answer_score(0.8, None) == pytest.approx(0.8)  # no verifier -> outcome


def test_label_steps_flags_unsupported_correct_answer():
    t = _traj("t1", "q1", [("text_search", "A"), ("answer", "Up to 145 grams")],
              outcome=1.0)
    t["steps"][0]["evidence"] = [
        {"evidence_id": "e0", "doc_id": "d0", "title": "Doves",
         "text": "The white-winged dove is larger than the mourning dove.",
         "score": 1.0}
    ]
    samples = label_steps([t], verifier=GroundingVerifier(),
                          support_verifier=AnswerSupportVerifier(backend="lexical"))
    ans = samples[-1]
    assert ans["action"]["type"] == "answer"
    assert ans["answer_support"] == pytest.approx(0.0)
    assert ans["unsupported_correct"] is True
    assert ans["score"] == pytest.approx(0.3)  # outcome 1.0 gated to the floor
    # Search steps never carry a support value
    assert samples[0]["answer_support"] is None
    assert samples[0]["unsupported_correct"] is False


def test_label_steps_without_support_verifier_keeps_outcome_only():
    t = _traj("t1", "q1", [("text_search", "A"), ("answer", "x")], outcome=1.0)
    samples = label_steps([t], verifier=GroundingVerifier())
    ans = samples[-1]
    assert ans["answer_support"] is None
    assert ans["score"] == pytest.approx(ans["outcome_credit"])


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
        "gold_answers": ["Carson City"],
    }


def test_check_rationale_rules():
    sample = _sample()
    filler = "This text_search action is appropriate because " * 9  # 54 tokens
    ok, reasons = check_rationale(filler + "see [e0].\nVERDICT: good", sample)
    assert ok, reasons

    no_cite = filler + "no citation here.\nVERDICT: good"
    assert not check_rationale(no_cite, sample)[0]

    no_type = ("The action is appropriate because " * 11) + "see [e0].\nVERDICT: good"
    assert not check_rationale(no_type, sample)[0]

    too_short = "text_search is fine [e0].\nVERDICT: good"
    assert not check_rationale(too_short, sample)[0]

    too_long = filler * 4 + "see [e0].\nVERDICT: good"
    assert not check_rationale(too_long, sample)[0]


def test_check_rationale_requires_consistent_verdict():
    sample = _sample()  # score = 0.7
    filler = "This text_search action is appropriate because " * 9
    body = filler + "see [e0]."
    ok, reasons = check_rationale(body, sample)  # no verdict line at all
    assert not ok and any("VERDICT" in r for r in reasons)
    # 'poor' on a 0.7 score is a hard contradiction
    assert not check_rationale(body + "\nVERDICT: poor", sample)[0]
    # 'mixed' is never a hard contradiction
    assert check_rationale(body + "\nVERDICT: mixed", sample)[0]
    # low score + 'good' is the symmetric contradiction
    low = dict(sample, score=0.2)
    assert not check_rationale(body + "\nVERDICT: good", low)[0]
    assert check_rationale(body + "\nVERDICT: poor", low)[0]


def test_check_rationale_rejects_gt_leakage():
    sample = _sample()  # gold = "Carson City", NOT in evidence text
    filler = "This text_search action is appropriate because " * 9
    leak_phrase = filler + "it matches the ground truth, see [e0].\nVERDICT: good"
    ok, reasons = check_rationale(leak_phrase, sample)
    assert not ok and any("evaluation phrasing" in r for r in reasons)

    leak_gold = filler + "it points to Carson City, see [e0].\nVERDICT: good"
    ok, reasons = check_rationale(leak_gold, sample)
    assert not ok and any("gold answer" in r for r in reasons)

    # Quoting the gold answer is legitimate when the evidence itself contains it
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
    assert fields["rationale_verdict"] == "good"          # score 0.7 -> good
    assert "VERDICT" not in fields["rationale"]           # stripped from the body
