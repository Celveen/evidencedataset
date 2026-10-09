"""Manual audit sheet for a construction stage (paper §4: "before each
construction stage we hand-inspect 100 sampled trajectories together with
their rationales; construction proceeds only once this audit passes").

Samples N queries (seeded), and for each renders the question, gold answer,
every sampled node grouped by its parent state (actions so far, the action,
what it retrieved, the three labels, the support verdict, the rationale and
its QC status) as one Markdown file with a checkbox per query. Read it, tick
what is right, note what is wrong, and only then scale up.

Usage:
    python DatasetConstruct/audit_report.py --benchmark infoseek --n 100
    python DatasetConstruct/audit_report.py --file data/construction/infoseek/rationales.jsonl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import read_jsonl, resolve, stage_path

from evidencetree.data.splits import stable_rank
from evidencetree.prm.data_gen import format_evidence_line
from evidencetree.utils import config as cfgutil

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _fmt(v: Any) -> str:
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def render(samples: list[dict[str, Any]], n: int, seed: int, evidence_chars: int = 240) -> str:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_query[s["query_id"]].append(s)
    chosen = sorted(by_query, key=lambda q: stable_rank(q, f"audit:{seed}"))[:n]
    lines = [
        f"# Audit sheet — {len(chosen)} of {len(by_query)} queries",
        "",
        "Check per query: the labels order the actions sensibly; retrieved evidence "
        "is what the action claims; the support verdict matches the evidence; the "
        "rationale argues from the evidence (no gold leakage) and agrees with the score.",
        "",
    ]
    for qid in chosen:
        rows = sorted(by_query[qid], key=lambda s: (s.get("depth", 0), s["sample_id"]))
        first = rows[0]
        lines += [
            f"## [ ] {qid}",
            f"**Q:** {first['question']}  ",
            f"**Gold:** {first.get('gold_answers')}  ·  **Image:** `{first.get('image_path')}`",
            "",
        ]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in rows:
            groups[s.get("parent_key", "")].append(s)
        for members in groups.values():
            before = members[0]["state"]["actions_before"]
            lines.append(f"**State:** {' → '.join(before) if before else '(start)'}")
            for s in members:
                a = s["action"]
                lines.append(
                    f"- `{a['type']}({a['input']})` — score **{_fmt(s['score'])}** "
                    f"(g={_fmt(s.get('local_grounding'))}, R_out={_fmt(s['outcome_credit'])} "
                    f"over {s.get('n_traj_through')} traj"
                    + (f", support={s.get('answer_support_label')}" if a["type"] == "answer" else "")
                    + (", **UNSUPPORTED-CORRECT**" if s.get("unsupported_correct") else "")
                    + ")"
                )
                for e in s.get("observation_evidence", [])[:3]:
                    lines.append(f"    - {format_evidence_line(e, evidence_chars)}")
                if s.get("rationale") is not None:
                    qc = "✓" if s.get("rationale_qc_pass") else "✗ " + "; ".join(
                        s.get("rationale_qc_reasons") or [])
                    lines.append(f"    - *rationale* ({s.get('rationale_verdict')}, QC {qc}): "
                                 f"{s['rationale']}")
            lines.append("")
        lines += ["Notes: ", "", "---", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Render a Markdown audit sheet.")
    p.add_argument("--benchmark", default="infoseek")
    p.add_argument("--config", default=None)
    p.add_argument("--file", default=None, help="any sample-level JSONL (default: rationales)")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.file:
        src = resolve(args.file)
    else:
        cfg = cfgutil.load_config(args.config or CONFIG_DIR / f"{args.benchmark}.yaml")
        src = stage_path(cfg, "rationales", args.mock)
    out = resolve(args.out) if args.out else src.with_name(f"audit_{src.stem}.md")
    out.write_text(render(list(read_jsonl(src)), args.n, args.seed), encoding="utf-8")
    print(f"audit sheet -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
