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


# --------------------------------------------------------------------------- #
# Two-phase support judging, node keys, evidence formatting
# --------------------------------------------------------------------------- #
from evidencetree.prm.data_gen import (  # noqa: E402
    PENDING,
    SKIPPED_INCORRECT,
    iter_label_steps,
    judge_answer_samples,
    node_key,
    support_evidence_lines,
)


def _two_answer_trajs():
    ev = {"evidence_id": "e0", "doc_id": "d0", "title": "Hall of Fame",
          "text": "It opened on September 7, 1963.", "score": 1.0}
    right = _traj("t1", "q1", [("text_search", "A"), ("answer", "September 7, 1963")], 1.0)
    wrong = _traj("t2", "q1", [("text_search", "A"), ("answer", "1970")], 0.0)
    for t in (right, wrong):
        t["steps"][0]["evidence"] = [dict(ev)]
    return [right, wrong]


def test_node_keys_link_siblings_and_repeats():
    samples = label_steps(_two_answer_trajs(), verifier=GroundingVerifier(), defer_support=True)
    search_1, answer_1, search_2, answer_2 = samples
    assert search_1["node_key"] == search_2["node_key"]          # same node, two trajectories
    assert search_1["parent_key"] == node_key("q1", ())
    assert answer_1["parent_key"] == answer_2["parent_key"] == search_1["node_key"]
    assert answer_1["node_key"] != answer_2["node_key"]          # siblings
    assert answer_1["traj_len"] == 2 and answer_1["depth"] == 1


def test_deferred_support_then_judge_only_correct_answers():
    class CountingJudge(AnswerSupportVerifier):
        def __init__(self):
            super().__init__(backend="lexical")
            self.calls = 0

        def judge(self, **kw):
            self.calls += 1
            return super().judge(**kw)

    samples = label_steps(_two_answer_trajs(), verifier=GroundingVerifier(), defer_support=True)
    answers = [s for s in samples if s["action"]["type"] == "answer"]
    assert all(s["answer_support_label"] == PENDING for s in answers)
    judge = CountingJudge()
    done = list(judge_answer_samples(samples, judge))
    assert judge.calls == 1                                       # the wrong answer is skipped
    by_answer = {s["action"]["input"]: s for s in done if s["action"]["type"] == "answer"}
    assert by_answer["1970"]["answer_support_label"] == SKIPPED_INCORRECT
    assert by_answer["1970"]["score"] == 0.0
    right = by_answer["September 7, 1963"]
    assert right["answer_support"] == pytest.approx(1.0) and right["score"] == pytest.approx(1.0)
    assert right["score_source"] == "answer_support"


def test_api_judge_reads_the_last_line_verdict_and_keeps_the_reason():
    class Reply:
        def __init__(self, text):
            self.text, self.prompts = text, []

        def generate(self, prompt, docs):
            self.prompts.append(prompt)
            return self.text

    gen = Reply("The question is grammar only; the evidence is not supported prose.\nNOT_REQUIRED")
    judge = AnswerSupportVerifier(backend="api", generator=gen)
    label, value, reason = judge.judge(question="Which is a compound sentence?", answer="B",
                                       evidence_texts=["[e0] Lecture: clauses ..."])
    assert (label, value) == ("not_required", None) and reason.startswith("The question is grammar")
    assert gen.prompts[0].rstrip().endswith("SUPPORTED, PARTIAL, UNSUPPORTED or NOT_REQUIRED.")
    assert judge.classify(question="q", answer="a", evidence_texts=["[e0] x"]) == ("not_required", None)
    gen.text = "SUPPORTED, the passage states it\nmore text without a label"
    assert judge.judge(question="q", answer="a", evidence_texts=["[e0] x"])[0] == "unparseable"
    samples = label_steps(_two_answer_trajs(), verifier=GroundingVerifier(), defer_support=True)
    gen.text = "The evidence names the opening date.\nSUPPORTED"
    done = {s["action"]["input"]: s for s in judge_answer_samples(samples, judge)
            if s["action"]["type"] == "answer"}
    assert done["September 7, 1963"]["answer_support_reason"] == "The evidence names the opening date."
    assert done["1970"]["answer_support_reason"] is None          # skipped_incorrect: never judged


def test_support_lines_keep_titles_and_images():
    lines = support_evidence_lines([
        {"evidence_id": "e0", "title": "Pro Football Hall of Fame", "text": "", "image_path": "a.jpg"},
        {"evidence_id": "e1", "title": "History", "text": "Opened in 1963."},
        {"evidence_id": "e2", "title": "", "text": ""},
    ])
    assert lines == ["[e0] [image] Pro Football Hall of Fame", "[e1] History: Opened in 1963."]


def test_marginal_gain_penalizes_repeated_evidence():
    ev = {"evidence_id": "e0", "doc_id": "d0", "title": "T", "text": "how heavy bird grams 400"}
    t = _traj("t1", "q1", [("text_search", "A"), ("text_search", "B"), ("answer", "x")], 1.0)
    t["question"] = "How heavy is this bird in grams?"
    t["steps"][0]["evidence"] = [dict(ev)]
    t["steps"][1]["evidence"] = [dict(ev, evidence_id="e1")]
    first, second, _ = label_steps([t], verifier=GroundingVerifier(), defer_support=True)
    assert first["evidence_redundancy"] == 0.0 and first["marginal_gain"] == first["local_grounding"]
    assert second["n_repeat_docs"] == 1 and second["evidence_redundancy"] == 1.0
    assert second["marginal_gain"] == 0.0


# --------------------------------------------------------------------------- #
# Echo / verbatim-copy QC (found by probing local models)
# --------------------------------------------------------------------------- #
from evidencetree.prm.rationale_gen import copied_span, salvage_rationale  # noqa: E402


def _probe_sample():
    return {
        "sample_id": "t1#s1", "question": "When did this building open?",
        "state": {"actions_before": [], "evidence_before": []},
        "action": {"type": "text_search", "input": "opening date"},
        "observation_evidence": [{"evidence_id": "e1", "title": "Hall", "text":
            "The Pro Football Hall of Fame is the hall of fame for professional American "
            "football, located in Canton, Ohio, and it opened on September 7, 1963."}],
        "score": 0.9, "gold_answers": ["September 7, 1963"],
    }


def test_echoed_prompt_is_salvaged_or_rejected():
    good = ("This text_search step retrieves [e1], a page about the hall that states when it "
            "opened, which directly answers what the question asks and makes the next step an "
            "easy answer; the query stays faithful to the question and the evidence is on target.")
    echoed = ("Action taken: text_search(opening date)\nEvidence retrieved BY this action:\n[e1] Hall: ...\n"
              f"Rationale: {good}\nVERDICT: good")
    assert salvage_rationale(echoed).startswith("This text_search step")
    ok, reasons = check_rationale(echoed, _probe_sample())
    assert ok, reasons                                    # salvaged: only the rationale counts
    raw_echo = "Action taken: text_search(opening date) " + good + "\nVERDICT: good"
    ok, reasons = check_rationale(raw_echo, _probe_sample())
    assert not ok and any("echoes the prompt" in r for r in reasons)


def test_verbatim_copy_is_rejected():
    s = _probe_sample()
    copy = ("This text_search step retrieves [e1]: the hall of fame for professional american "
            "football located in canton ohio and it opened on september, which answers the question "
            "and gives the next step everything it needs to answer well.\nVERDICT: good")
    assert copied_span(copy, s) is not None
    ok, reasons = check_rationale(copy, s)
    assert not ok and any("verbatim" in r for r in reasons)
    assert copied_span("it opened on September 7, 1963 per [e1]", s) is None   # short quotes are fine
