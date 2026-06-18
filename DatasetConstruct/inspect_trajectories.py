"""Human-readable inspector for the pipeline's JSONL outputs.

The raw JSONL is hard to eyeball (long evidence text, nested fields). This
renders a compact, truncated summary so you can verify each part at a glance:
which action types fired (text_search / text_to_image / image_to_text /
image_search / answer), how many hits each returned (with a snippet), the three
score labels, and the rationale + its QC status.

Auto-detects the file kind:
  * trajectories  (has "steps")            -> per-trajectory action timeline
  * scored / rationales (has "sample_id")  -> per-step labels + rationale

Usage:
    python DatasetConstruct/inspect_trajectories.py data/trajectories/mock_infoseek_rationales.jsonl
    python DatasetConstruct/inspect_trajectories.py <file> --n 5 --full
    python DatasetConstruct/inspect_trajectories.py <file> --query infoseek_val_00000000
    python DatasetConstruct/inspect_trajectories.py <file> --action image_to_text
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, resolve

_W = 100  # default snippet width


def trunc(s: str, n: int = _W) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _ev_line(ev: dict, width: int) -> str:
    title = ev.get("title", "")
    head = f"{ev['evidence_id']} " + (f"{title}: " if title else "")
    return "      " + trunc(head + ev.get("text", ""), width)


# --------------------------------------------------------------------------- #
def render_trajectory(t: dict, width: int) -> str:
    out = []
    mark = "✓" if t.get("outcome_em", 0) >= 0.5 else "✗"
    out.append(
        f"══ {t['traj_id']}  outcome={t.get('outcome_em')}{mark}  "
        f"λ={t.get('lambda')}  {len(t['steps'])} steps"
    )
    out.append(f"   Q: {trunc(t['question'], width)}")
    out.append(f"   gold: {t.get('gold_answers')}   image: {t.get('image_path')}")
    for st in t["steps"]:
        a_in = trunc(st.get("action_input", ""), 60)
        line = f"   [{st['step_index']}] {st['action_type']}: {a_in}"
        if st.get("region"):
            line += f"  region={st['region']}"
        out.append(line)
        ev = st.get("evidence", [])
        if ev:
            out.append(f"        -> {len(ev)} hits")
            for e in ev[:3]:
                out.append(_ev_line(e, width))
    out.append(f"   => answer: {trunc(t.get('final_answer', ''), width)} {mark}")
    return "\n".join(out)


def render_sample(s: dict, width: int) -> str:
    out = []
    a = s["action"]
    sc = s.get("score")
    local = s.get("local_grounding")
    local_s = "n/a" if local is None else f"{local:.2f}"
    out.append(
        f"── {s['sample_id']}  [{a['type']}]  "
        f"score={sc:.2f} (local={local_s}, outcome={s.get('outcome_credit', 0):.2f}, "
        f"n={s.get('n_traj_through')})"
    )
    out.append(f"   Q: {trunc(s['question'], width)}")
    out.append(f"   action: {a['type']}  {trunc(a.get('input', ''), 70)}")
    obs = s.get("observation_evidence", [])
    if obs:
        out.append(f"   observed: {len(obs)} hits")
        for e in obs[:2]:
            out.append(_ev_line(e, width))
    if "rationale" in s:
        qc = s.get("rationale_qc_pass")
        qc_mark = "✓" if qc else ("✗ " + ",".join(s.get("rationale_qc_reasons", [])))
        out.append(f"   rationale (qc={qc_mark}):")
        out.append("      " + trunc(s["rationale"], width if width > _W else 280))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render pipeline JSONL as readable summaries.")
    p.add_argument("path", help="trajectories / scored / rationales JSONL")
    p.add_argument("--n", type=int, default=10, help="max records to show (default 10)")
    p.add_argument("--full", action="store_true", help="don't truncate snippets")
    p.add_argument("--query", default=None, help="only this query_id")
    p.add_argument("--action", default=None, help="only records whose action is this type")
    args = p.parse_args(argv)

    rows = list(read_jsonl(resolve(args.path)))
    if not rows:
        print("(empty file)")
        return 0
    is_traj = "steps" in rows[0]
    width = 100_000 if args.full else _W

    if args.query:
        rows = [r for r in rows if r.get("query_id") == args.query]
    if args.action and not is_traj:
        rows = [r for r in rows if r.get("action", {}).get("type") == args.action]
    if args.action and is_traj:
        rows = [r for r in rows if any(s["action_type"] == args.action for s in r["steps"])]

    kind = "trajectories" if is_traj else "step samples"
    print(f"# {len(rows)} {kind} in {args.path} (showing up to {args.n})\n")
    for r in rows[: args.n]:
        print(render_trajectory(r, width) if is_traj else render_sample(r, width))
        print()

    # Aggregate action-type usage — quick sanity that each kind fires.
    from collections import Counter

    usage: Counter[str] = Counter()
    if is_traj:
        for r in rows:
            usage.update(s["action_type"] for s in r["steps"])
    else:
        usage.update(r["action"]["type"] for r in rows)
    print(f"action-type usage across {len(rows)} records: {dict(usage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
