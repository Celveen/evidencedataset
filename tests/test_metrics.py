"""Tests for EM / F1 metrics."""

from evidencetree.eval import metrics


def test_normalize_answer_strips_articles_punct_case():
    assert metrics.normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert metrics.normalize_answer("  A  CAT.  ") == "cat"


def test_exact_match_basic():
    assert metrics.exact_match("Paris", "paris") == 1.0
    assert metrics.exact_match("Paris", "London") == 0.0


def test_exact_match_against_multiple_golds():
    assert metrics.exact_match("NYC", ["New York", "NYC"]) == 1.0
    assert metrics.exact_match("Boston", ["New York", "NYC"]) == 0.0


def test_f1_partial_overlap():
    # 2 shared tokens out of pred=2, gold=3 -> P=1.0, R=2/3, F1=0.8
    f1 = metrics.f1_score("New York", "New York City")
    assert abs(f1 - 0.8) < 1e-6


def test_f1_no_overlap_is_zero():
    assert metrics.f1_score("apple", "banana") == 0.0


def test_infoseek_numeric_range_match():
    gold = "{'wikidata': 153.0, 'range': [137.7, 168.3]}"
    assert metrics.exact_match("150 g", [gold]) == 1.0
    assert metrics.f1_score("150 g", [gold]) == 1.0
    assert metrics.exact_match("46 g", [gold]) == 0.0


def test_aggregate():
    preds = ["Paris", "wrong"]
    refs = ["paris", "Tokyo"]
    out = metrics.aggregate(preds, refs)
    assert out["n"] == 2.0
    assert out["exact_match"] == 0.5


# --- InfoSeek relaxed accuracy (numeric range + string) --------------------- #
_NUMERIC_GOLD = ["{'wikidata': 153.0, 'range': [137.7, 168.3]}"]  # gold as str-repr


def test_infoseek_numeric_in_range_scores_one():
    # A correct number inside the range. exact_match is numeric-range-aware
    # too (server fix), so both metrics accept it.
    assert metrics.infoseek_accuracy("about 150 grams", _NUMERIC_GOLD) == 1.0
    assert metrics.exact_match("about 150 grams", _NUMERIC_GOLD) == 1.0


def test_infoseek_numeric_out_of_range_scores_zero():
    assert metrics.infoseek_accuracy("45 kg", _NUMERIC_GOLD) == 0.0
    assert metrics.infoseek_accuracy("no number here", _NUMERIC_GOLD) == 0.0


def test_infoseek_numeric_handles_commas_and_dict_gold():
    assert metrics.infoseek_accuracy("145,000", ["{'range': [144000, 146000]}"]) == 1.0
    # raw dict gold (not str-repr) also works
    assert metrics.infoseek_accuracy("160", [{"wikidata": 153.0, "range": [137.7, 168.3]}]) == 1.0


def test_infoseek_string_falls_back_to_exact_match():
    assert metrics.infoseek_accuracy("Aare", ["Lutschine", "Aare", "Aar"]) == 1.0
    assert metrics.infoseek_accuracy("Paris", ["Aare"]) == 0.0
