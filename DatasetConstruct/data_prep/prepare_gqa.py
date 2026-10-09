"""Prepare GQA for ETBench-Open construction and evaluation.

Inputs:
    --src           the ``lmms-lab/GQA`` parquet release (ModelScope mirror):
                    train_balanced_{instructions,images}/ and
                    testdev_balanced_{instructions,images}/
    --scene-graphs  the official ``sceneGraphs.zip`` (train_sceneGraphs.json);
                    fetch it on a machine with internet access with
                    ``fetch_kit/fetch_infoseek.py gqa-scenegraphs``

Splits (paper Appendix E):
    seeds       train-balanced questions, at most one per image (diversity)
    evaluation  a fixed sample of --eval-n test-dev balanced questions; test-dev
                images are never put in the corpus

Corpus: one document per train-balanced image that has a scene graph — the
image plus its scene graph rendered as text (objects, attributes,
relations). A train question can therefore reach its own image's scene graph
(image_search, then a text lookup): the oracle-perception probe the paper
uses GQA for. ``--corpus-scope seeds`` keeps only the seed images plus
``--corpus-extra`` random others (smaller disk / faster pilots).

Outputs (image paths relative to --out):
    data/corpus/gqa/{queries,eval_queries,corpus}.jsonl, images/, stats.json
    DatasetConstruct/splits/gqa/{seed_ids,eval_ids}.txt, split_report.json

Usage:
    python DatasetConstruct/data_prep/prepare_gqa.py --src data/raw/GQA \
        --scene-graphs data/raw/GQA/sceneGraphs.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

_DC = Path(__file__).resolve().parents[1]
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from common import resolve, write_json, write_jsonl  # noqa: E402

from evidencetree.data.prep import save_image  # noqa: E402
from evidencetree.data.splits import SplitSpec, select_splits, stable_rank, write_ids  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("dataset.prepare_gqa")

_INSTRUCTION_COLUMNS = ["id", "imageId", "question", "answer", "types"]


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_instructions(src: Path, split: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted((src / f"{split}_instructions").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {src / f'{split}_instructions'}")
    rows: list[dict[str, Any]] = []
    for path in files:
        cols = [c for c in _INSTRUCTION_COLUMNS if c in pq.ParquetFile(path).schema.names]
        rows.extend(pq.read_table(path, columns=cols).to_pylist())
    return rows


def load_scene_graphs(zip_path: Path, member: str = "train_sceneGraphs.json") -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(member)]
        if not names:
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        with zf.open(names[0]) as f:
            return json.loads(f.read().decode("utf-8"))


def scene_graph_text(graph: dict[str, Any], max_objects: int = 80) -> str:
    """Objects with attributes, then subject-relation-object triples."""
    objects = graph.get("objects") or {}
    names: dict[str, str] = {}
    lines: list[str] = []
    for oid, obj in list(objects.items())[:max_objects]:
        name = _clean(obj.get("name", "object"))
        names[str(oid)] = name
        attrs = [_clean(a) for a in (obj.get("attributes") or []) if _clean(a)]
        lines.append(f"{name} ({', '.join(attrs[:8])})." if attrs else f"{name}.")
    for oid, obj in list(objects.items())[:max_objects]:
        subj = names.get(str(oid), _clean(obj.get("name", "object")))
        for rel in (obj.get("relations") or [])[:8]:
            rel_name = _clean(rel.get("name", ""))
            target = names.get(str(rel.get("object", "")))
            if rel_name and target:
                lines.append(f"{subj} {rel_name} {target}.")
    return " ".join(lines)


def save_split_images(src: Path, split: str, wanted: set[str], out: Path, side: int) -> dict[str, str]:
    """Write the wanted images of one split; returns imageId -> relative path."""
    import pyarrow.parquet as pq

    subdir = "train" if split.startswith("train") else "testdev"
    found: dict[str, str] = {}
    for path in sorted((src / f"{split}_images").glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["id", "image"]):
            for row in batch.to_pylist():
                image_id = str(row["id"])
                if image_id not in wanted or image_id in found:
                    continue
                data = (row.get("image") or {}).get("bytes")
                if not data:
                    continue
                rel = f"images/{subdir}/{image_id}.jpg"
                if not (out / rel).exists():
                    save_image(data, out / rel, max_side=side)
                found[image_id] = rel
        if len(found) >= len(wanted):
            break
    return found


def run(src: Path, scene_graphs: Path, out: Path, splits_dir: Path, eval_n: int = 2000,
        seed_n: int = 20000, corpus_scope: str = "all", corpus_extra: int = 20000,
        image_side: int = 640) -> dict[str, Any]:
    train = read_instructions(src, "train_balanced")
    testdev = read_instructions(src, "testdev_balanced")
    graphs = load_scene_graphs(scene_graphs)
    log.info("GQA: %d train-balanced, %d test-dev questions, %d scene graphs",
             len(train), len(testdev), len(graphs))

    by_id: dict[str, dict[str, Any]] = {}
    records = []
    for pool, rows in (("seed", train), ("eval", testdev)):
        for r in rows:
            qid = f"gqa-{r['id']}"
            by_id[qid] = r
            image_id = str(r["imageId"])
            # Seeds need their own scene graph in the corpus (oracle property).
            if pool == "seed" and image_id not in graphs:
                continue
            records.append({"query_id": qid, "pool": pool, "image_id": image_id})
    result = select_splits(records, SplitSpec(
        eval_n=eval_n, seed_n=seed_n, group_by="image_id", max_seeds_per_group=1,
        salt="etbench-open-v1:gqa",
    ))

    seed_images = {str(by_id[q]["imageId"]) for q in result.seed_ids}
    train_images = {str(r["imageId"]) for r in train if str(r["imageId"]) in graphs}
    if corpus_scope == "all":
        corpus_images = train_images
    else:
        others = sorted(train_images - seed_images, key=lambda i: stable_rank(i, "gqa-corpus"))
        corpus_images = seed_images | set(others[:corpus_extra])
    eval_images = {str(by_id[q]["imageId"]) for q in result.eval_ids}
    assert not (eval_images & corpus_images), "test-dev images must stay out of the corpus"

    saved_train = save_split_images(src, "train_balanced", corpus_images, out, image_side)
    saved_eval = save_split_images(src, "testdev_balanced", eval_images, out, image_side)

    def query_row(qid: str, saved: dict[str, str], split: str) -> dict[str, Any]:
        r = by_id[qid]
        image_id = str(r["imageId"])
        return {
            "query_id": qid,
            "question": _clean(r["question"]),
            "gold_answers": [_clean(r["answer"])],
            "image_path": saved.get(image_id),
            "metadata": {"image_id": image_id, "split": split, "types": r.get("types")},
        }

    n_seed = write_jsonl(out / "queries.jsonl",
                         (query_row(q, saved_train, "train_balanced") for q in result.seed_ids))
    n_eval = write_jsonl(out / "eval_queries.jsonl",
                         (query_row(q, saved_eval, "testdev_balanced") for q in result.eval_ids))
    n_docs = write_jsonl(out / "corpus.jsonl", (
        {"doc_id": f"gqa-img-{i}", "title": f"GQA image {i}",
         "text": scene_graph_text(graphs[i]), "image_path": saved_train[i],
         "entity_id": f"gqa-img-{i}"}
        for i in sorted(corpus_images) if i in saved_train
    ))

    write_ids(splits_dir / "seed_ids.txt", result.seed_ids)
    write_ids(splits_dir / "eval_ids.txt", result.eval_ids)
    write_json(splits_dir / "split_report.json", result.report)
    stats = {
        "seed_queries": n_seed,
        "eval_queries": n_eval,
        "corpus_docs": n_docs,
        "corpus_scope": corpus_scope,
        "missing_train_images": len(corpus_images) - len(saved_train),
        "missing_eval_images": len(eval_images) - len(saved_eval),
        "seed_answer_top20": Counter(_clean(by_id[q]["answer"]) for q in result.seed_ids).most_common(20),
    }
    write_json(out / "stats.json", stats)
    log.info("GQA prepared: %s", json.dumps({k: v for k, v in stats.items() if k != "seed_answer_top20"}))
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prepare GQA.")
    p.add_argument("--src", type=Path, required=True, help="lmms-lab/GQA parquet root")
    p.add_argument("--scene-graphs", type=Path, required=True, help="sceneGraphs.zip")
    p.add_argument("--out", default="data/corpus/gqa")
    p.add_argument("--splits-dir", default="DatasetConstruct/splits/gqa")
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--seed-n", type=int, default=20000)
    p.add_argument("--corpus-scope", choices=("all", "seeds"), default="all")
    p.add_argument("--corpus-extra", type=int, default=20000,
                   help="with --corpus-scope seeds: extra random train images")
    p.add_argument("--image-side", type=int, default=640)
    args = p.parse_args(argv)
    run(args.src, args.scene_graphs, resolve(args.out), resolve(args.splits_dir),
        args.eval_n, args.seed_n, args.corpus_scope, args.corpus_extra, args.image_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
