"""Evaluation holdout and construction-seed selection (stdlib only).

Every benchmark keeps a fixed evaluation set (~2,000 questions) that dataset
construction must never touch, and draws ~20K construction seeds from what is
left. The selection is a pure function of the candidate records and a salt —
no RNG state — so it reproduces bit-for-bit on any machine, including the
local fetch kit that selects InfoSeek queries before any image is downloaded.

Leakage rules, applied in this order:

1. **Group exclusivity.** Records sharing ``group_by`` (e.g. the same query
   image) land on one side only: a group that contributes an evaluation query
   contributes no seed.
2. **Entity disjointness.** For evaluation strata listed in
   ``entity_disjoint_strata`` (InfoSeek's unseen-entity split), every entity
   of those evaluation queries is removed from the seed pool, so "unseen
   entity" stays unseen by the PRM as well.
3. **Seed diversity.** ``max_seeds_per_group`` caps seeds per group (GQA: one
   question per image).

Records are plain dicts with ``query_id``, ``pool`` (``"eval"`` | ``"seed"`` |
``"both"``) and any fields named by the spec, looked up on the record first
and then in its ``metadata``.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

POOLS = ("eval", "seed", "both")


@dataclass(frozen=True)
class SplitSpec:
    eval_n: int = 2000
    seed_n: int = 20000
    eval_strata: str | None = None
    group_by: str | None = None
    entity_field: str | None = None
    entity_disjoint_strata: tuple[str, ...] = ()
    max_seeds_per_group: int | None = None
    seed_backup: int = 0
    salt: str = "etbench-open-v1"


@dataclass
class SplitResult:
    eval_ids: list[str]
    seed_ids: list[str]
    backup_seed_ids: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def stable_rank(key: str, salt: str) -> int:
    """Machine-independent pseudo-random rank of ``key`` under ``salt``."""
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big")


def _field(record: dict[str, Any], name: str | None) -> Any:
    if not name:
        return None
    if name in record:
        return record[name]
    return (record.get("metadata") or {}).get(name)


def select_splits(records: Iterable[dict[str, Any]], spec: SplitSpec) -> SplitResult:
    """Select the evaluation holdout and the construction seeds."""
    records = list(records)
    seen: set[str] = set()
    for rec in records:
        qid = str(rec["query_id"])
        if qid in seen:
            raise ValueError(f"duplicate query_id {qid!r} in split candidates")
        seen.add(qid)
        if rec.get("pool") not in POOLS:
            raise ValueError(f"record {qid!r} has pool {rec.get('pool')!r}; expected {POOLS}")

    def group_of(rec: dict[str, Any]) -> str:
        value = _field(rec, spec.group_by)
        return f"g:{value}" if value not in (None, "") else f"q:{rec['query_id']}"

    def stratum_of(rec: dict[str, Any]) -> str:
        value = _field(rec, spec.eval_strata)
        return str(value) if value not in (None, "") else "all"

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        groups[group_of(rec)].append(rec)

    # ---- 1. evaluation holdout: whole groups, stratified quotas ----------- #
    eval_groups: dict[str, list[dict[str, Any]]] = {}
    for key, members in groups.items():
        # Sorted, so the output never depends on the input order.
        eligible = sorted(
            (r for r in members if r["pool"] in ("eval", "both")),
            key=lambda r: str(r["query_id"]),
        )
        if eligible:
            eval_groups[key] = eligible
    strata = sorted({stratum_of(r) for rs in eval_groups.values() for r in rs})
    quotas: dict[str, int] = {}
    if strata:
        base, extra = divmod(spec.eval_n, len(strata))
        quotas = {s: base + (1 if i < extra else 0) for i, s in enumerate(strata)}
    taken: Counter[str] = Counter()
    chosen_groups: set[str] = set()
    eval_ids: list[str] = []
    salt_eval = f"{spec.salt}:eval"
    for key in sorted(eval_groups, key=lambda k: (stable_rank(k, salt_eval), k)):
        if all(taken[s] >= quotas[s] for s in quotas):
            break
        eligible = eval_groups[key]
        stratum = Counter(stratum_of(r) for r in eligible).most_common(1)[0][0]
        if taken[stratum] >= quotas[stratum]:
            continue
        chosen_groups.add(key)
        taken[stratum] += len(eligible)
        eval_ids.extend(str(r["query_id"]) for r in eligible)

    # ---- 2. entities that must stay unseen ------------------------------- #
    excluded_entities: set[str] = set()
    if spec.entity_field and spec.entity_disjoint_strata:
        protected = set(spec.entity_disjoint_strata)
        for key in chosen_groups:
            for rec in eval_groups[key]:
                entity = _field(rec, spec.entity_field)
                if entity not in (None, "") and stratum_of(rec) in protected:
                    excluded_entities.add(str(entity))

    # ---- 3. construction seeds ------------------------------------------- #
    dropped: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for key, members in groups.items():
        for rec in members:
            if rec["pool"] not in ("seed", "both"):
                continue
            if key in chosen_groups:
                dropped["shares_group_with_eval"] += 1
                continue
            entity = _field(rec, spec.entity_field)
            if entity not in (None, "") and str(entity) in excluded_entities:
                dropped["unseen_entity_protected"] += 1
                continue
            candidates.append(rec)
    salt_seed = f"{spec.salt}:seed"
    candidates.sort(key=lambda r: (stable_rank(str(r["query_id"]), salt_seed), str(r["query_id"])))
    per_group: Counter[str] = Counter()
    ranked: list[str] = []
    for rec in candidates:
        key = group_of(rec)
        if spec.max_seeds_per_group is not None and per_group[key] >= spec.max_seeds_per_group:
            dropped["group_cap"] += 1
            continue
        per_group[key] += 1
        ranked.append(str(rec["query_id"]))
    seed_ids = ranked[: spec.seed_n]
    backup = ranked[spec.seed_n: spec.seed_n + spec.seed_backup]

    report = {
        "spec": asdict(spec),
        "n_records": len(records),
        "pools": dict(Counter(r["pool"] for r in records)),
        "eval": {
            "n": len(eval_ids),
            "groups": len(chosen_groups),
            "per_stratum": dict(taken),
            "quotas": quotas,
        },
        "excluded_entities": len(excluded_entities),
        "seed_pool_after_exclusion": len(candidates),
        "seed_dropped": dict(dropped),
        "seeds": len(seed_ids),
        "seed_shortfall": max(0, spec.seed_n - len(seed_ids)),
        "backup_seeds": len(backup),
    }
    return SplitResult(eval_ids=eval_ids, seed_ids=seed_ids, backup_seed_ids=backup, report=report)


def write_ids(path: str | Path, ids: Iterable[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(ids)
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return len(ids)


def read_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
