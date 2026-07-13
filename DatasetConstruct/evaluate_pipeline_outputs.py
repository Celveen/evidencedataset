"""Summarize EvidenceTree pipeline outputs with EM, F1, and soft match.

Usage:
    PYTHONPATH=src .venv-qwen25vl/bin/python \
      DatasetConstruct/evaluate_pipeline_outputs.py \
      --root data/pipeline_10q_r5 \
      --output data/pipeline_10q_r5/summary.md
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from evidencetree.eval.metrics import exact_match, f1_score, normalize_answer


DEFAULT_DATASETS = (
    "infoseek",
    "scienceqa",
    "gqa",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def soft_match(prediction: str, gold_answers: list[str], f1_threshold: float) -> bool:
    pred = normalize_answer(prediction)
    golds = [normalize_answer(gold) for gold in gold_answers]
    if not pred:
        return False
    if any(pred == gold for gold in golds):
        return True
    if any(_contained_answer(pred, gold) for gold in golds if gold):
        return True
    return f1_score(prediction, gold_answers) >= f1_threshold


def _contained_answer(pred: str, gold: str) -> bool:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return False
    if len(gold_tokens) == 1:
        token = gold_tokens[0]
        if len(token) <= 1:
            return False
        if token.isdigit():
            return token in pred
        return token in pred_tokens
    if len(pred_tokens) == 1:
        token = pred_tokens[0]
        if len(token) <= 1:
            return False
        if token.isdigit():
            return token in gold
        return token in gold_tokens
    return _has_subsequence(pred_tokens, gold_tokens) or _has_subsequence(
        gold_tokens, pred_tokens
    )


def _has_subsequence(tokens: list[str], phrase: list[str]) -> bool:
    if len(phrase) > len(tokens):
        return False
    width = len(phrase)
    return any(tokens[i:i + width] == phrase for i in range(len(tokens) - width + 1))


def summarize_dataset(root: Path, dataset: str, f1_threshold: float) -> dict[str, Any]:
    ds_root = root / dataset
    trajectories = read_jsonl(ds_root / "trajectories.jsonl")
    rationales = read_jsonl(ds_root / "rationales.jsonl")
    stats_path = ds_root / "dataset" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    by_query: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    actions = collections.Counter()
    f1_values: list[float] = []
    em_count = 0
    soft_count = 0
    near_misses = []

    for row in trajectories:
        by_query[row["query_id"]].append(row)
        gold = [str(item) for item in row.get("gold_answers", [])]
        pred = str(row.get("final_answer", ""))
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        soft = soft_match(pred, gold, f1_threshold)
        em_count += int(em)
        soft_count += int(soft)
        f1_values.append(f1)
        for step in row.get("steps", []):
            actions[step.get("action_type", "")] += 1
        if not em and soft:
            near_misses.append(
                {
                    "query_id": row["query_id"],
                    "question": row.get("question", ""),
                    "prediction": pred,
                    "gold_answers": gold,
                    "f1": f1,
                }
            )

    query_hit_em = sum(
        any(exact_match(row.get("final_answer", ""), row.get("gold_answers", [])) for row in rows)
        for rows in by_query.values()
    )
    query_hit_soft = sum(
        any(
            soft_match(row.get("final_answer", ""), row.get("gold_answers", []), f1_threshold)
            for row in rows
        )
        for rows in by_query.values()
    )
    qc_pass = sum(bool(row.get("rationale_qc_pass")) for row in rationales)
    return {
        "dataset": dataset,
        "queries": len(by_query),
        "query_hit_em": f"{query_hit_em}/{len(by_query)}",
        "query_hit_soft": f"{query_hit_soft}/{len(by_query)}",
        "traj_em": f"{em_count}/{len(trajectories)}",
        "traj_soft": f"{soft_count}/{len(trajectories)}",
        "mean_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "trajectories": len(trajectories),
        "steps": sum(1 for _ in (ds_root / "scored.jsonl").open()) if (ds_root / "scored.jsonl").exists() else 0,
        "kept": f"{stats.get('kept', 0)}/{stats.get('input_samples', 0)}",
        "train": stats.get("train", 0),
        "val": stats.get("val", 0),
        "mean_score": stats.get("mean_score", 0.0),
        "rationale_qc": f"{qc_pass}/{len(rationales)}",
        "actions": dict(actions),
        "near_misses": sorted(near_misses, key=lambda item: item["f1"], reverse=True),
    }


TABLE_HEADERS = [
    "dataset",
    "queries",
    "query_hit_em",
    "query_hit_soft",
    "traj_em",
    "traj_soft",
    "mean_f1",
    "trajectories",
    "steps",
    "kept",
    "train",
    "val",
    "mean_score",
    "rationale_qc",
    "actions",
]


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{key}:{value[key]}" for key in sorted(value))
    return str(value)


def format_table(rows: list[dict[str, Any]], *, markdown: bool = False) -> str:
    if markdown:
        lines = [
            "| " + " | ".join(TABLE_HEADERS) + " |",
            "| " + " | ".join("---" for _ in TABLE_HEADERS) + " |",
        ]
        for row in rows:
            lines.append(
                "| " + " | ".join(_cell(row[key]) for key in TABLE_HEADERS) + " |"
            )
        return "\n".join(lines)
    lines = ["\t".join(TABLE_HEADERS)]
    for row in rows:
        lines.append("\t".join(_cell(row[key]) for key in TABLE_HEADERS))
    return "\n".join(lines)


def print_table(rows: list[dict[str, Any]]) -> None:
    print(format_table(rows))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/pipeline_10q_r5"))
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--soft-threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional summary table path. .md writes Markdown; other suffixes write TSV.",
    )
    args = parser.parse_args(argv)

    rows = [
        summarize_dataset(args.root, dataset, args.soft_threshold)
        for dataset in args.datasets
    ]
    print_table(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table = format_table(rows, markdown=args.output.suffix.lower() == ".md")
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"\nSummary table written to: {args.output}")
    print("\nNear misses: EM=0 but soft_match=True")
    for row in rows:
        examples = row["near_misses"][: args.examples]
        if not examples:
            continue
        print(f"\n== {row['dataset']} ==")
        for item in examples:
            print(f"- {item['query_id']} | F1={item['f1']:.3f}")
            print(f"  Q: {item['question']}")
            print(f"  pred: {item['prediction']}")
            print(f"  gold: {item['gold_answers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
