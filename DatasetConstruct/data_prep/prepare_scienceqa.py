"""Prepare ScienceQA for ETBench-Open construction and evaluation.

Input: the ScienceQA parquet release (ModelScope ``AI-ModelScope/ScienceQA``,
mirroring ``derek-thomas/ScienceQA``) — ``train-*``, ``validation-*`` and
``test-*`` parquet files under ``--src`` (searched recursively).

Splits (leakage by construction is impossible across benchmark splits):
    seeds       train + validation (~17K questions; all are used when fewer
                than --seed-n)
    evaluation  a fixed sample of --eval-n questions from test

Corpus (paper Appendix E): the benchmark's ``lecture`` and ``hint`` fields
only — never ``solution``, which explains the gold answer. Lectures are
deduplicated (many questions share one lecture); each non-empty hint is one
document whose title names its question, so retrieval can reach the context
that belongs to a question; a question with neither field falls back to a
document holding its question text. Questions are posed WITHOUT their hint:
the context has to be retrieved.

Outputs (image paths relative to --out):
    data/corpus/scienceqa/{queries,eval_queries,corpus}.jsonl, images/, stats.json
    DatasetConstruct/splits/scienceqa/{seed_ids,eval_ids}.txt, split_report.json

Usage:
    python DatasetConstruct/data_prep/prepare_scienceqa.py --src data/raw/ScienceQA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

_DC = Path(__file__).resolve().parents[1]
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from common import resolve, write_json, write_jsonl  # noqa: E402

from evidencetree.data.prep import save_image  # noqa: E402
from evidencetree.data.splits import SplitSpec, select_splits, write_ids  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("dataset.prepare_scienceqa")

SPLIT_PREFIXES = {"train": "train", "validation": "validation", "test": "test"}
SEED_SPLITS = ("train", "validation")


def iter_split_rows(src: Path, split: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted(p for p in src.rglob("*.parquet") if p.name.startswith(SPLIT_PREFIXES[split]))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet under {src}")
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            yield from batch.to_pylist()


def format_question(row: dict[str, Any]) -> str:
    choices = list(row.get("choices") or [])
    return row["question"].strip() + "\nChoices: " + "; ".join(
        f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
    )


def _image_bytes(row: dict[str, Any]) -> bytes | None:
    image = row.get("image")
    if isinstance(image, dict):
        return image.get("bytes")
    return image if isinstance(image, (bytes, bytearray)) else None


def run(src: Path, out: Path, splits_dir: Path, eval_n: int = 2000, seed_n: int = 20000,
        image_side: int = 768) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        for i, row in enumerate(iter_split_rows(src, split)):
            row["_split"] = split
            rows[f"scienceqa-{split}-{i:05d}"] = row
    log.info("ScienceQA rows: %s", dict(Counter(r["_split"] for r in rows.values())))

    records = [
        {"query_id": qid, "pool": "seed" if r["_split"] in SEED_SPLITS else "eval"}
        for qid, r in rows.items()
    ]
    result = select_splits(records, SplitSpec(eval_n=eval_n, seed_n=seed_n,
                                               salt="etbench-open-v1:scienceqa"))

    def query_row(qid: str) -> dict[str, Any]:
        r = rows[qid]
        choices = [str(c) for c in (r.get("choices") or [])]
        answer_index = int(r["answer"])
        letter = chr(65 + answer_index)
        image_path = None
        data = _image_bytes(r)
        if data:
            rel = f"images/{r['_split']}/{qid}.jpg"
            if not (out / rel).exists():
                save_image(data, out / rel, max_side=image_side)
            image_path = rel
        return {
            "query_id": qid,
            "question": format_question(r),
            "gold_answers": [choices[answer_index], letter, f"{letter}. {choices[answer_index]}"],
            "image_path": image_path,
            "metadata": {
                "split": r["_split"],
                "choices": choices,
                "answer_index": answer_index,
                "has_image": bool(data),
                "subject": r.get("subject"),
                "topic": r.get("topic"),
                "category": r.get("category"),
                "skill": r.get("skill"),
                "grade": r.get("grade"),
            },
        }

    n_seed = write_jsonl(out / "queries.jsonl", (query_row(q) for q in result.seed_ids))
    n_eval = write_jsonl(out / "eval_queries.jsonl", (query_row(q) for q in result.eval_ids))

    corpus: list[dict[str, Any]] = []
    seen_lectures: set[str] = set()
    kinds: Counter[str] = Counter()
    for qid, r in rows.items():  # every split: the corpus is the benchmark's own text
        lecture = (r.get("lecture") or "").strip()
        hint = (r.get("hint") or "").strip()
        if lecture:
            h = hashlib.sha1(lecture.encode("utf-8")).hexdigest()[:12]
            if h not in seen_lectures:
                seen_lectures.add(h)
                topic = r.get("skill") or r.get("topic") or "science"
                corpus.append({"doc_id": f"sqa-lecture-{h}", "title": f"Lecture: {topic}",
                               "text": lecture, "entity_id": f"sqa-lecture-{h}"})
                kinds["lecture"] += 1
        if hint:
            corpus.append({"doc_id": f"sqa-hint-{qid}", "title": f"Context: {r['question'].strip()[:120]}",
                           "text": hint, "entity_id": f"sqa-hint-{qid}"})
            kinds["hint"] += 1
        if not lecture and not hint:
            corpus.append({"doc_id": f"sqa-question-{qid}", "title": "Question",
                           "text": r["question"].strip(), "entity_id": f"sqa-question-{qid}"})
            kinds["question_fallback"] += 1
    n_docs = write_jsonl(out / "corpus.jsonl", corpus)

    write_ids(splits_dir / "seed_ids.txt", result.seed_ids)
    write_ids(splits_dir / "eval_ids.txt", result.eval_ids)
    write_json(splits_dir / "split_report.json", result.report)
    stats = {
        "rows": dict(Counter(r["_split"] for r in rows.values())),
        "seed_queries": n_seed,
        "eval_queries": n_eval,
        "seed_with_image": sum(1 for q in result.seed_ids if _image_bytes(rows[q])),
        "corpus_docs": n_docs,
        "corpus_kinds": dict(kinds),
        "solution_indexed": False,
    }
    write_json(out / "stats.json", stats)
    log.info("ScienceQA prepared: %s", json.dumps(stats))
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepare ScienceQA.")
    p.add_argument("--src", type=Path, required=True, help="directory holding the parquet files")
    p.add_argument("--out", default="data/corpus/scienceqa")
    p.add_argument("--splits-dir", default="DatasetConstruct/splits/scienceqa")
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--seed-n", type=int, default=20000)
    p.add_argument("--image-side", type=int, default=768)
    args = p.parse_args(argv)
    run(args.src, resolve(args.out), resolve(args.splits_dir), args.eval_n, args.seed_n,
        args.image_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
