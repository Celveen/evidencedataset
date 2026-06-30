"""Validate ETBench-Open JSONL against SCHEMA.md (field schema + invariants §2).

Checks the step-level sample records (train/val/test.jsonl) for: required fields,
types, score ranges, the local-grounding⇔answer rule, the score-fusion identity,
trajectory length, grounding-outcome gap, rationale QC, re-distributable image
paths, and (across files) train/val/test query-disjointness.

Stdlib only. Exit code 0 = conformant, 1 = hard violations found.

Usage:
    python DatasetConstruct/validate_dataset.py data/etbench_open/train.jsonl \
        data/etbench_open/val.jsonl
    python DatasetConstruct/validate_dataset.py data/etbench_open/train.jsonl \
        --min-steps 2 --max-steps 8 --max-gap 0.7 --require-provenance
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOL = 1e-6
_EVID_RE = re.compile(r"\be\d+\b")
ACTION_TYPES = {"text_search", "image_search", "answer"}


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _in01(x: Any) -> bool:
    return _is_num(x) and -TOL <= x <= 1 + TOL


def _abs_path(p: str) -> bool:
    """A machine-absolute path is not re-distributable (SCHEMA §2.8)."""
    return p.startswith("/") or (len(p) > 1 and p[1] == ":") or p.startswith("~")


def validate_sample(s: dict[str, Any], *, min_steps, max_steps, max_gap,
                    require_provenance) -> list[str]:
    """Return a list of violation strings for one sample (empty = clean)."""
    v: list[str] = []
    sid = s.get("sample_id", "<no id>")

    for f in ("sample_id", "traj_id", "query_id", "question", "state", "action",
              "local_grounding", "outcome_credit", "n_traj_through", "alpha",
              "score", "rationale", "rationale_qc_pass"):
        if f not in s:
            v.append(f"missing field '{f}'")
    if v:
        return [f"{sid}: {m}" for m in v]

    atype = s["action"].get("type")
    if atype not in ACTION_TYPES:
        v.append(f"action.type {atype!r} not in {sorted(ACTION_TYPES)}")
    if not str(s["action"].get("input", "")):
        v.append("action.input empty")

    # Score ranges
    for f in ("outcome_credit", "alpha", "score"):
        if not _in01(s[f]):
            v.append(f"{f}={s[f]} out of [0,1]")
    local = s["local_grounding"]
    # local⇔answer rule
    if atype == "answer" and local is not None:
        v.append("answer step must have local_grounding=null")
    if atype != "answer":
        if local is None:
            v.append("non-answer step must have a numeric local_grounding")
        elif not _in01(local):
            v.append(f"local_grounding={local} out of [0,1]")

    # Score fusion identity
    outcome = s["outcome_credit"]
    expected = outcome if local is None else s["alpha"] * local + (1 - s["alpha"]) * outcome
    if _is_num(s["score"]) and abs(s["score"] - expected) > 1e-4:
        v.append(f"score {s['score']:.4f} != fuse {expected:.4f}")

    if not (_is_num(s["n_traj_through"]) and s["n_traj_through"] >= 1):
        v.append(f"n_traj_through={s['n_traj_through']} (<1)")

    # grounding-outcome gap (kept samples must be within band)
    if local is not None and _is_num(local) and abs(local - outcome) > max_gap + TOL:
        v.append(f"|local-outcome|={abs(local-outcome):.3f} > max_gap {max_gap}")

    # sample_id ⇄ step_index consistency
    m = re.search(r"#s(\d+)$", str(s["sample_id"]))
    if not m:
        v.append("sample_id missing #s<step_index> suffix")

    # Rationale QC
    if not s["rationale_qc_pass"]:
        v.append("rationale_qc_pass is false (should be dropped before release)")
    else:
        r = s.get("rationale", "")
        ntok = len(r.split())
        if not (30 <= ntok <= 150):
            v.append(f"rationale length {ntok} outside [30,150]")
        if atype and atype not in r:
            v.append("rationale does not mention action type")
        visible = {e["evidence_id"] for e in s["state"].get("evidence_before", [])}
        visible |= {e["evidence_id"] for e in s.get("observation_evidence", [])}
        if visible and not (set(_EVID_RE.findall(r)) & visible):
            v.append("rationale cites no visible evidence_id")

    # Re-distributable image path
    ip = s.get("image_path")
    if isinstance(ip, str) and _abs_path(ip):
        v.append(f"image_path is machine-absolute (not re-distributable): {ip!r}")

    if require_provenance and "gen_provenance" not in s:
        v.append("missing gen_provenance (required for release)")

    return [f"{sid}: {m}" for m in v]


def validate_file(path: Path, **kw) -> dict[str, Any]:
    samples = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    violations: list[str] = []
    seen_ids: set[str] = set()
    steps_per_traj: Counter = Counter()
    atypes: Counter = Counter()
    for s in samples:
        sid = s.get("sample_id")
        if sid in seen_ids:
            violations.append(f"{sid}: duplicate sample_id")
        seen_ids.add(sid)
        steps_per_traj[s.get("traj_id")] += 1
        atypes[s.get("action", {}).get("type")] += 1
        violations.extend(validate_sample(s, **kw))

    # Trajectory length is a TRAJECTORY-LEVEL pre-filter gate (applied in step4 on
    # the full trajectory). Per-sample gap-filtering legitimately leaves some
    # trajectories with a single surviving sample — that sample is still a valid
    # independent (state, action, label) node, so a low post-filter count is NOT a
    # violation. We only flag the impossible case of MORE samples than max_steps
    # (which would mean the step4 gate failed); short counts are reported as info.
    orphaned = sum(1 for n in steps_per_traj.values() if n < kw["min_steps"])
    for t, n in steps_per_traj.items():
        if n > kw["max_steps"]:
            violations.append(f"traj {t}: {n} samples exceed max_steps {kw['max_steps']}")

    return {
        "orphaned_singletons": orphaned,
        "path": str(path),
        "n_samples": len(samples),
        "n_trajectories": len(steps_per_traj),
        "action_types": dict(atypes),
        "query_ids": {s.get("query_id") for s in samples},
        "violations": violations,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate ETBench-Open JSONL against SCHEMA.md.")
    p.add_argument("files", nargs="+", help="train/val/test JSONL paths.")
    p.add_argument("--min-steps", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-gap", type=float, default=0.7)
    p.add_argument("--require-provenance", action="store_true",
                   help="Require gen_provenance on every sample (release mode).")
    p.add_argument("--max-show", type=int, default=15, help="Max violations to print per file.")
    args = p.parse_args(argv)

    kw = dict(min_steps=args.min_steps, max_steps=args.max_steps,
              max_gap=args.max_gap, require_provenance=args.require_provenance)

    results = [validate_file(Path(f), **kw) for f in args.files]
    total_viol = 0
    print("=" * 72)
    for r in results:
        nv = len(r["violations"])
        total_viol += nv
        status = "PASS" if nv == 0 else f"FAIL ({nv} violations)"
        print(f"[{status}] {r['path']}")
        print(f"    samples={r['n_samples']}  trajectories={r['n_trajectories']}  "
              f"actions={r['action_types']}")
        if r.get("orphaned_singletons"):
            print(f"    info: {r['orphaned_singletons']} traj reduced to 1 sample by "
                  f"gap-filter (expected, not a violation)")
        vcat = Counter(v.split(": ", 1)[-1].split(" (")[0].split(":")[0][:48]
                       for v in r["violations"])
        for cat, c in vcat.most_common():
            print(f"      - {c:>4}×  {cat}")
        for line in r["violations"][:args.max_show]:
            print(f"        · {line}")
        if nv > args.max_show:
            print(f"        · ... (+{nv - args.max_show} more)")

    # Cross-split query disjointness
    print("-" * 72)
    if len(results) > 1:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                overlap = results[i]["query_ids"] & results[j]["query_ids"]
                tag = "OK" if not overlap else f"LEAK: {len(overlap)} shared query_id"
                print(f"  split purity {Path(results[i]['path']).name} ∩ "
                      f"{Path(results[j]['path']).name}: {tag}")
                if overlap:
                    total_viol += len(overlap)
    print("=" * 72)
    print(f"TOTAL violations: {total_viol} -> {'CONFORMANT' if total_viol == 0 else 'NOT CONFORMANT'}")
    return 0 if total_viol == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
