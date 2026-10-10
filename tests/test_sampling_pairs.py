"""Node collapse, free filters, budgeted group sampling, preference pairs."""

import pytest

from evidencetree.prm.pairs import mine_pairs, pair_stats
from evidencetree.prm.sampling import (
    collapse_nodes,
    filter_gap,
    filter_trajectory_length,
    sample_groups,
)


def _node(qid, parent, node, atype, score, **kw):
    base = {
        "sample_id": f"{qid}#{node}", "traj_id": f"{qid}#t0", "query_id": qid,
        "parent_key": parent, "node_key": node, "action": {"type": atype, "input": node},
        "score": score, "outcome_credit": kw.pop("outcome", score),
        "local_grounding": None if atype == "answer" else kw.pop("local", score),
        "n_traj_through": kw.pop("n", 1), "traj_len": kw.pop("traj_len", 3),
    }
    base.update(kw)
    return base


def test_collapse_keeps_first_record_and_counts_copies():
    a = _node("q", "root", "n1", "text_search", 0.5)
    b = dict(a, sample_id="q#t1#s0", traj_id="q#t1")
    c = _node("q", "root", "n2", "image_search", 0.2)
    nodes = collapse_nodes([a, b, c])
    assert [n["node_key"] for n in nodes] == ["n1", "n2"]
    assert nodes[0]["sample_id"] == a["sample_id"] and nodes[0]["n_step_records"] == 2


def test_free_filters():
    ok = _node("q", "r", "n1", "text_search", 0.5, traj_len=3)
    short = _node("q", "r", "n2", "text_search", 0.5, traj_len=1)
    kept, dropped = filter_trajectory_length([ok, short], 2, 8)
    assert kept == [ok] and dropped == 1
    noisy = _node("q", "r", "n3", "text_search", 0.5, local=0.0, outcome=0.9)
    ans = _node("q", "r", "n4", "answer", 0.0, outcome=1.0)
    kept, dropped = filter_gap([ok, noisy, ans], 0.7)
    assert [n["node_key"] for n in kept] == ["n1", "n4"] and dropped == 1


def test_sampling_is_query_balanced_and_prefers_contrast():
    nodes = []
    for q in range(10):
        qid = f"q{q}"
        # contrast group (2 children, spread 0.8) and a singleton group
        nodes.append(_node(qid, "root", f"{qid}a", "text_search", 0.9))
        nodes.append(_node(qid, "root", f"{qid}b", "image_search", 0.1))
        nodes.append(_node(qid, "deep", f"{qid}c", "answer", 1.0))
    selected, report = sample_groups(nodes, budget=20, max_per_query=3, seed=0)
    assert report["queries_covered"] == 10  # every query before any second group
    assert len(selected) == 20
    assert all(n["parent_key"] == "root" for n in selected)  # contrast groups first
    again, _ = sample_groups(list(reversed(nodes)), budget=20, max_per_query=3, seed=0)
    assert {n["node_key"] for n in again} == {n["node_key"] for n in selected}


def test_sampling_respects_per_query_cap_and_budget():
    nodes = [_node("q0", f"p{i}", f"n{i}", "text_search", 0.5) for i in range(10)]
    selected, _ = sample_groups(nodes, budget=100, max_per_query=4)
    assert len(selected) == 4
    selected, _ = sample_groups(nodes, budget=3, max_per_query=None)
    assert len(selected) == 3


def test_depth_cycle_rotates_the_first_group_over_depths():
    nodes = []
    for q in range(9):
        qid = f"q{q}"
        # root group: small spread, reliable credit; deeper groups: wide 0/1 spread
        nodes += [_node(qid, "root", f"{qid}r1", "text_search", 0.6, depth=0, n=3),
                  _node(qid, "root", f"{qid}r2", "image_search", 0.5, depth=0, n=3)]
        nodes += [_node(qid, "s1", f"{qid}d1", "answer", 1.0, depth=1),
                  _node(qid, "s1", f"{qid}d2", "answer", 0.0, depth=1)]
        nodes += [_node(qid, "s2", f"{qid}e1", "text_search", 0.9, depth=2),
                  _node(qid, "s2", f"{qid}e2", "answer", 0.1, depth=2)]
    plain, report = sample_groups(nodes, budget=18, max_per_query=2)
    assert report["depths_selected"] == {1: 18}                   # spread first: all depth 1
    rotated, report = sample_groups(nodes, budget=18, max_per_query=2, depth_cycle=3)
    assert report["depths_selected"] == {0: 6, 1: 6, 2: 6} and report["depth_cycle"] == 3
    assert report["queries_covered"] == 9


def test_pairs_types_and_margin():
    parent = "s"
    good_text = _node("q", parent, "t", "text_search", 0.9, outcome=0.8)
    bad_image = _node("q", parent, "i", "image_search", 0.2, outcome=0.8)
    supported = _node("q", "s2", "a1", "answer", 1.0, outcome=1.0, answer_support=1.0)
    lucky = _node("q", "s2", "a2", "answer", 0.3, outcome=1.0, answer_support=0.0,
                  unsupported_correct=True)
    stop_search = _node("q", "s2", "t2", "text_search", 0.85, outcome=0.8)
    pairs = mine_pairs([good_text, bad_image, supported, lucky, stop_search], margin=0.3)
    by_type = {p["pair_type"]: p for p in pairs}
    assert by_type["action_type"]["chosen_sample_id"] == good_text["sample_id"]
    assert by_type["answer_support"]["rejected_sample_id"] == lucky["sample_id"]
    stop = by_type["answer_now_vs_search_more"]
    assert stop["chosen_sample_id"] == supported["sample_id"]
    assert stop["rejected_sample_id"] == stop_search["sample_id"]
    assert all(p["margin"] >= 0.3 for p in pairs if p["pair_type"] != "answer_now_vs_search_more")
    assert pair_stats(pairs)["action_type"] == 1
    assert not [p for p in mine_pairs([good_text, bad_image], margin=0.9)]
    assert all(p["pair_type"] != "answer_now_vs_search_more"
               for p in mine_pairs([supported, stop_search], stop_pairs=False))


@pytest.mark.parametrize("budget", [1, 5])
def test_budget_overshoot_is_at_most_one_group(budget):
    nodes = [_node(f"q{i}", "root", f"n{i}{c}", "text_search", 0.5 + 0.1 * j)
             for i in range(3) for j, c in enumerate("abc")]
    selected, _ = sample_groups(nodes, budget=budget, max_per_query=None)
    assert budget <= len(selected) < budget + 3
