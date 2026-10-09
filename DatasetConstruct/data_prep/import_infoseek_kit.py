"""Import an InfoSeek fetch kit (DatasetConstruct/fetch_kit/fetch_infoseek.py)
into the shared benchmark layout.

    data/corpus/infoseek/
    ├── queries.jsonl        construction seeds (kit splits/seed_ids.txt)
    ├── eval_queries.jsonl   held-out evaluation questions (never used to build)
    ├── corpus.jsonl         Wiki6M pages chunked to ~1000 characters, plus one
    │                        image document per entity that has a picture
    ├── images/query/        OVEN query images
    ├── images/corpus/       Wikipedia entity images
    └── stats.json
    DatasetConstruct/splits/infoseek/{seed_ids,eval_ids}.txt, split_report.json

Every image path written is relative to the output directory. The kit's
manifest checksums are verified first.

Usage:
    python DatasetConstruct/data_prep/import_infoseek_kit.py --kit /path/to/infoseek_kit
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_DC = Path(__file__).resolve().parents[1]   # DatasetConstruct/ (common.py puts src/ on the path)
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from common import read_jsonl, resolve, write_json, write_jsonl  # noqa: E402

from evidencetree.data.prep import chunk_text, sha256_file  # noqa: E402
from evidencetree.data.splits import read_ids, write_ids  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("dataset.import_infoseek")


def verify_manifest(kit: Path) -> dict[str, Any]:
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    bad = [rel for rel, digest in manifest["sha256"].items()
           if not (kit / rel).exists() or sha256_file(kit / rel) != digest]
    if bad:
        raise RuntimeError(f"kit files missing or corrupt: {bad[:10]}")
    return manifest


def gold_list(row: dict[str, Any]) -> list[str]:
    gold = row.get("answer_eval") or row.get("answer") or []
    if isinstance(gold, (dict, str)):
        gold = [gold]
    return [str(g) for g in gold]


def _copy_images(src: Path, dst: Path, move: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.glob("*.jpg"):
        target = dst / p.name
        if not target.exists():
            (shutil.move if move else shutil.copy2)(str(p), str(target))
        n += 1
    return n


def run(kit: Path, out: Path, splits_dir: Path, chunk_chars: int = 1000,
        max_chunks: int = 40, move: bool = False, verify: bool = True) -> dict[str, Any]:
    manifest = verify_manifest(kit) if verify else {}
    ann = kit / "annotations"
    kb = {r["data_id"]: r for r in read_jsonl(ann / "infoseek_val_withkb.jsonl")}
    rows = {r["data_id"]: r for r in read_jsonl(ann / "infoseek_val.jsonl")}
    seeds = read_ids(kit / "splits" / "seed_ids.txt")
    eval_file = kit / "splits" / "eval_ids_final.txt"
    evals = read_ids(eval_file if eval_file.exists() else kit / "splits" / "eval_ids.txt")
    overlap = set(seeds) & set(evals)
    if overlap:
        raise RuntimeError(f"{len(overlap)} ids are both seed and eval — corrupt kit")

    n_query_images = _copy_images(kit / "query_images", out / "images" / "query", move)
    n_corpus_images = _copy_images(kit / "corpus_images", out / "images" / "corpus", move)
    have_query = {p.stem for p in (out / "images" / "query").glob("*.jpg")}
    have_corpus = {p.stem for p in (out / "images" / "corpus").glob("*.jpg")}

    def query_row(qid: str) -> dict[str, Any]:
        r, k = rows[qid], kb.get(qid, {})
        image_id = r.get("image_id", "")
        return {
            "query_id": qid,
            "question": r["question"],
            "gold_answers": gold_list(r),
            "image_path": f"images/query/{image_id}.jpg" if image_id in have_query else None,
            "metadata": {
                "image_id": image_id,
                "data_split": r.get("data_split", ""),
                "entity_id": k.get("entity_id"),
                "entity_text": k.get("entity_text"),
            },
        }

    n_seed = write_jsonl(out / "queries.jsonl", (query_row(q) for q in seeds if q in rows))
    n_eval = write_jsonl(out / "eval_queries.jsonl", (query_row(q) for q in evals if q in rows))

    roles: Counter[str] = Counter()
    chunks_per_entity: list[int] = []

    def corpus_rows():
        for ent in read_jsonl(kit / "wiki" / "entities.jsonl.gz"):
            eid, title = ent["entity_id"], ent.get("title", "")
            chunks = chunk_text(ent.get("content") or "", chunk_chars)[:max_chunks]
            roles[ent.get("role", "")] += 1
            chunks_per_entity.append(len(chunks))
            for i, text in enumerate(chunks):
                yield {"doc_id": f"{eid}_p{i}", "title": title, "text": text, "entity_id": eid}
            if eid in have_corpus:
                yield {"doc_id": f"{eid}_img", "title": title, "text": "",
                       "image_path": f"images/corpus/{eid}.jpg", "entity_id": eid}

    n_docs = write_jsonl(out / "corpus.jsonl", corpus_rows())
    splits_dir.mkdir(parents=True, exist_ok=True)
    write_ids(splits_dir / "seed_ids.txt", seeds)
    write_ids(splits_dir / "eval_ids.txt", evals)
    report_src = kit / "splits" / "split_report.json"
    if report_src.exists():
        shutil.copy2(report_src, splits_dir / "split_report.json")
    for stale in ("unified_clip_index", "bm25_index"):  # corpus changed -> rebuild indexes
        if (out / stale).exists():
            shutil.rmtree(out / stale)

    stats = {
        "seed_queries": n_seed,
        "eval_queries": n_eval,
        "corpus_docs": n_docs,
        "entities": sum(roles.values()),
        "entity_roles": dict(roles),
        "chunks_per_entity_median": statistics.median(chunks_per_entity) if chunks_per_entity else 0,
        "query_images": n_query_images,
        "corpus_images": n_corpus_images,
        "chunk_chars": chunk_chars,
        "max_chunks_per_entity": max_chunks,
        "kit_counts": manifest.get("counts"),
    }
    write_json(out / "stats.json", stats)
    log.info("InfoSeek imported: %s", json.dumps(stats))
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Import an InfoSeek fetch kit.")
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--out", default="data/corpus/infoseek")
    p.add_argument("--splits-dir", default="DatasetConstruct/splits/infoseek")
    p.add_argument("--chunk-chars", type=int, default=1000)
    p.add_argument("--max-chunks", type=int, default=40, help="per entity page")
    p.add_argument("--move", action="store_true", help="move images instead of copying")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args(argv)
    run(args.kit, resolve(args.out), resolve(args.splits_dir), args.chunk_chars,
        args.max_chunks, args.move, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
