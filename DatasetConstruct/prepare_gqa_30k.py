"""Prepare a 30k GQA subset for EvidenceTree.

Inputs are the Hugging Face parquet files from ``lmms-lab/GQA``:

  GQA/hf/train_balanced_instructions/train-00000-of-00001.parquet
  GQA/hf/train_balanced_images/train-*.parquet

Optional:
  GQA/raw/sceneGraphs.zip from the official GQA download page. When present,
  scene graph objects/attributes/relations are used as corpus text. Otherwise
  the script falls back to semantic programs from the sampled questions.

Outputs:
  data/corpus/gqa/queries.jsonl
  data/corpus/gqa/corpus.jsonl
  data/corpus/gqa/stats.json
  /media/.../GQA/images_30k/*.jpg
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _semantic_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    semantic = row.get("semantic") or []
    for item in semantic:
        op = _clean(item.get("operation", ""))
        arg = _clean(item.get("argument", ""))
        if op and arg:
            parts.append(f"{op}: {arg}")
        elif op:
            parts.append(op)
        elif arg:
            parts.append(arg)
    if not parts and row.get("semanticStr"):
        parts.append(_clean(row["semanticStr"]))
    types = row.get("types") or {}
    type_text = ", ".join(
        f"{key}={value}" for key, value in types.items() if value is not None
    )
    if type_text:
        parts.append(f"question_type: {type_text}")
    return ". ".join(parts)


def _read_instruction_rows(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    for row in rows:
        if not row.get("id") or not row.get("imageId"):
            continue
        qid = str(row["id"])
        if qid in seen_qids:
            continue
        seen_qids.add(qid)
        selected.append(row)
        if len(selected) >= n:
            break
    if len(selected) < n:
        raise RuntimeError(f"Only selected {len(selected)} rows, wanted {n}.")
    return selected


def _save_images(
    parquet_dir: Path,
    wanted_ids: set[str],
    image_out: Path,
) -> dict[str, str]:
    image_out.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=512, columns=["id", "image"]):
            for row in batch.to_pylist():
                image_id = str(row["id"])
                if image_id not in wanted_ids or image_id in found:
                    continue
                image_data = row.get("image") or {}
                raw = image_data.get("bytes")
                if not raw:
                    continue
                out_path = image_out / f"{image_id}.jpg"
                if not out_path.exists():
                    with Image.open(BytesIO(raw)) as im:
                        im.convert("RGB").save(out_path, "JPEG", quality=92)
                found[image_id] = str(out_path.resolve())
        if len(found) >= len(wanted_ids):
            break
    missing = wanted_ids - set(found)
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} images, examples: {sorted(missing)[:10]}"
        )
    return found


def _load_scene_graphs(zip_path: Path, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        return {}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            json_names = [name for name in names if name.endswith(".json")]
            graphs: dict[str, dict[str, Any]] = {}
            for name in json_names:
                with zf.open(name) as f:
                    obj = json.loads(f.read().decode("utf-8"))
                if isinstance(obj, dict):
                    for image_id, graph in obj.items():
                        if str(image_id) in wanted_ids:
                            graphs[str(image_id)] = graph
                if len(graphs) >= len(wanted_ids):
                    break
            return graphs
    except zipfile.BadZipFile:
        return {}


def _scene_graph_text(graph: dict[str, Any], max_objects: int = 80) -> str:
    objects = graph.get("objects") or {}
    lines: list[str] = []
    object_names: dict[str, str] = {}
    for object_id, obj in list(objects.items())[:max_objects]:
        name = _clean(obj.get("name", "object"))
        object_names[str(object_id)] = name
        attrs = [_clean(a) for a in (obj.get("attributes") or []) if _clean(a)]
        attr_text = f" with attributes {', '.join(attrs[:8])}" if attrs else ""
        lines.append(f"{name}{attr_text}.")
    for object_id, obj in list(objects.items())[:max_objects]:
        subj = object_names.get(str(object_id), _clean(obj.get("name", "object")))
        for rel in (obj.get("relations") or [])[:8]:
            rel_name = _clean(rel.get("name", ""))
            target_id = str(rel.get("object", ""))
            obj_name = object_names.get(target_id, target_id)
            if rel_name and obj_name:
                lines.append(f"{subj} {rel_name} {obj_name}.")
    return " ".join(lines)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/GQA"),
        help="Directory holding the downloaded GQA release.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/corpus/gqa"),
        help="Output directory for the converted corpus.",
    )
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw = args.raw_root
    hf = raw / "hf"
    instruction_path = hf / "train_balanced_instructions" / "train-00000-of-00001.parquet"
    image_parquet_dir = hf / "train_balanced_images"
    image_out = raw / "images_30k"
    scene_zip = raw / "raw" / "sceneGraphs.zip"

    if not instruction_path.exists():
        raise FileNotFoundError(instruction_path)
    if not image_parquet_dir.exists():
        raise FileNotFoundError(image_parquet_dir)

    rows = _read_instruction_rows(instruction_path, n=args.n, seed=args.seed)
    wanted_image_ids = {str(row["imageId"]) for row in rows}
    image_paths = _save_images(image_parquet_dir, wanted_image_ids, image_out)
    scene_graphs = _load_scene_graphs(scene_zip, wanted_image_ids)

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["imageId"])].append(row)

    queries: list[dict[str, Any]] = []
    corpus: list[dict[str, Any]] = []
    answer_counter = Counter()
    type_counter = Counter()

    for row in rows:
        image_id = str(row["imageId"])
        answer = str(row.get("answer", ""))
        full_answer = str(row.get("fullAnswer", "")) if row.get("fullAnswer") else answer
        types = row.get("types") or {}
        if types.get("semantic"):
            type_counter[str(types["semantic"])] += 1
        answer_counter[answer] += 1
        queries.append(
            {
                "query_id": f"gqa-{row['id']}",
                "question": row["question"],
                "gold_answers": [answer, full_answer] if full_answer != answer else [answer],
                "image_path": image_paths[image_id],
                "metadata": {
                    "source": "lmms-lab/GQA/train_balanced",
                    "question_id": row["id"],
                    "image_id": image_id,
                    "full_answer": row.get("fullAnswer"),
                    "types": types,
                    "groups": row.get("groups"),
                    "semantic": row.get("semantic"),
                    "semanticStr": row.get("semanticStr"),
                },
            }
        )

    for image_id, image_rows in by_image.items():
        graph = scene_graphs.get(image_id)
        if graph:
            text = _scene_graph_text(graph)
            source = "gqa_scene_graph"
        else:
            semantic_parts = [_semantic_text(row) for row in image_rows]
            # Fallback avoids fullAnswer/answer text to reduce direct leakage.
            text = " ".join(part for part in semantic_parts if part)
            source = "gqa_semantic_program_fallback"
        if not text:
            text = "Visual scene from GQA/Visual Genome."
        corpus.append(
            {
                "doc_id": f"gqa-img-{image_id}",
                "title": f"GQA image {image_id}",
                "text": text,
                "image_path": image_paths[image_id],
                "metadata": {
                    "source": source,
                    "image_id": image_id,
                    "num_sampled_questions": len(image_rows),
                    "has_scene_graph": bool(graph),
                },
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "queries.jsonl", queries)
    _write_jsonl(args.out_dir / "corpus.jsonl", corpus)
    stats = {
        "queries": len(queries),
        "corpus_docs": len(corpus),
        "unique_images": len(wanted_image_ids),
        "images_saved": len(image_paths),
        "scene_graph_docs": sum(1 for doc in corpus if doc["metadata"]["has_scene_graph"]),
        "semantic_fallback_docs": sum(
            1 for doc in corpus if not doc["metadata"]["has_scene_graph"]
        ),
        "raw_root": str(raw),
        "image_dir": str(image_out),
        "top_answer_counts": answer_counter.most_common(20),
        "semantic_type_counts": type_counter.most_common(),
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "README.md").write_text(
        "# GQA 30k subset\n\n"
        "Generated from `lmms-lab/GQA` train balanced instructions/images.\n\n"
        f"- queries: {len(queries)}\n"
        f"- corpus docs: {len(corpus)}\n"
        f"- unique images: {len(wanted_image_ids)}\n"
        f"- scene graph docs: {stats['scene_graph_docs']}\n"
        f"- semantic fallback docs: {stats['semantic_fallback_docs']}\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
