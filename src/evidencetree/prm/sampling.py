"""Node-level filtering and budgeted sampling of step labels (construction step 3).

MCTS rollouts share prefixes, so the same tree node — same state, same action,
same retrieved evidence, same label — appears once per trajectory through it.
Training on those copies would weight a node by its visit count, and paying a
rationale per copy would waste most of the budget. This module:

1. drops whole trajectories outside ``[min_steps, max_steps]`` (their full
   length, never a post-filter count);
2. collapses the remaining step records to one record per unique node;
3. drops noisy labels: non-answer nodes with ``|g - R_out| > max_gap``;
4. samples **sibling groups** — all kept children of one parent state — under a
   node budget, round-robin over queries so every query contributes before any
   contributes twice. Keeping siblings together preserves the same-state
   contrasts DPO needs. Within a query, groups are taken most-informative
   first: groups with a contrast (>= 2 children), then larger label spread,
   then more reliable credit (more trajectories through the least-visited
   child). That order alone favours deep groups — their credit rests on one
   trajectory, so it is 0/1 and spreads widely — and in the first pilot left
   root decisions at 4% of the sample. ``depth_cycle`` therefore rotates the
   depth of each query's FIRST group along the seeded query order (query i
   starts with its best contrast group at depth i mod depth_cycle, if it has
   one), so every decision depth is represented.

Everything is deterministic given the seed; answer scores are still the
provisional outcome-only values here (the support judge runs after sampling).
"""

from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from typing import Any, Iterable

from evidencetree.data.splits import stable_rank


def filter_trajectory_length(
    samples: Iterable[dict[str, Any]], min_steps: int, max_steps: int
) -> tuple[list[dict[str, Any]], int]:
    """Keep step records whose trajectory length is within bounds."""
    kept, dropped = [], 0
    for s in samples:
        if min_steps <= int(s["traj_len"]) <= max_steps:
            kept.append(s)
        else:
            dropped += 1
    return kept, dropped


def collapse_nodes(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per (query, node); the first record represents the node.

    ``n_step_records`` counts the trajectory copies that were merged.
    """
    nodes: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for s in samples:
        key = (s["query_id"], s["node_key"])
        if key in nodes:
            nodes[key]["n_step_records"] += 1
            continue
        node = dict(s)
        node["n_step_records"] = 1
        nodes[key] = node
    return list(nodes.values())


def filter_gap(
    nodes: Iterable[dict[str, Any]], max_gap: float
) -> tuple[list[dict[str, Any]], int]:
    """Drop non-answer nodes whose grounding and credit contradict each other."""
    kept, dropped = [], 0
    for n in nodes:
        local = n.get("local_grounding")
        if local is not None and abs(float(local) - float(n["outcome_credit"])) > max_gap:
            dropped += 1
        else:
            kept.append(n)
    return kept, dropped


def sibling_groups(nodes: Iterable[dict[str, Any]]) -> "OrderedDict[tuple[str, str], list[dict[str, Any]]]":
    groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for n in nodes:
        groups.setdefault((n["query_id"], n["parent_key"]), []).append(n)
    return groups


def _priority(group: list[dict[str, Any]]) -> tuple[int, float, int]:
    scores = [float(n["score"]) for n in group]
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    reliability = min(int(n["n_traj_through"]) for n in group)
    return (int(len(group) >= 2), round(spread, 6), reliability)


def _depth(group: list[dict[str, Any]]) -> int:
    return int(group[0].get("depth") or 0)


def sample_groups(
    nodes: list[dict[str, Any]],
    budget: int,
    max_per_query: int | None = None,
    seed: int = 0,
    depth_cycle: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Budgeted, query-balanced sampling of sibling groups.

    Args:
        nodes: collapsed, filtered nodes.
        budget: target number of nodes; the last group may overshoot it.
        max_per_query: cap on nodes per query (None = no cap).
        seed: fixes the round-robin order of queries.
        depth_cycle: rotate the depth of each query's first group over
            0..depth_cycle-1 (see the module docstring); None or 0 = off.

    Returns:
        (selected nodes ordered by query then parent state, report dict).
    """
    groups = sibling_groups(nodes)
    by_query: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for (qid, _), members in groups.items():
        by_query[qid].append(members)
    for qid in by_query:
        by_query[qid].sort(key=_priority, reverse=True)  # stable: ties keep tree order

    salt = f"sampling:{seed}"
    order = sorted(by_query, key=lambda q: (stable_rank(q, salt), q))
    if depth_cycle:
        for i, qid in enumerate(order):
            queue = by_query[qid]
            want = i % int(depth_cycle)
            first = next((k for k, g in enumerate(queue) if len(g) >= 2 and _depth(g) == want), None)
            if first:
                queue.insert(0, queue.pop(first))
    cursor = Counter()
    taken = Counter()
    chosen: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    total = 0
    progressed = True
    while total < budget and progressed:
        progressed = False
        for qid in order:
            if total >= budget:
                break
            queue = by_query[qid]
            while cursor[qid] < len(queue):
                group = queue[cursor[qid]]
                cursor[qid] += 1
                if max_per_query is not None and taken[qid] + len(group) > max_per_query:
                    continue  # a smaller group of this query may still fit
                chosen[qid].append(group)
                taken[qid] += len(group)
                total += len(group)
                progressed = True
                break

    selected = [n for qid in order for g in chosen.get(qid, []) for n in g]
    report = {
        "nodes_in": len(nodes),
        "groups_in": len(groups),
        "groups_with_contrast_in": sum(1 for g in groups.values() if len(g) >= 2),
        "queries_in": len(by_query),
        "budget": budget,
        "max_per_query": max_per_query,
        "nodes_selected": len(selected),
        "groups_selected": sum(len(v) for v in chosen.values()),
        "groups_with_contrast_selected": sum(
            1 for v in chosen.values() for g in v if len(g) >= 2
        ),
        "queries_covered": len(chosen),
        "action_types_in": dict(Counter(n["action"]["type"] for n in nodes)),
        "action_types_selected": dict(Counter(n["action"]["type"] for n in selected)),
        "depth_cycle": int(depth_cycle or 0),
        "depths_selected": dict(sorted(Counter(int(n.get("depth") or 0) for n in selected).items())),
    }
    return selected, report
