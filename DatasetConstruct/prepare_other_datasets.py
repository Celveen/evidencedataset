"""Convert leakage-safe samples from locally downloaded VQA datasets.

The generated directories use EvidenceTree's shared format:

    data/corpus/<dataset>/queries.jsonl
    data/corpus/<dataset>/corpus.jsonl
    data/corpus/<dataset>/assets/*

Gold evidence identifiers are retained only in query metadata for evaluation.
They are never used as query images or exposed as specially labelled retrieval
candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from urllib.parse import unquote
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image

from common import resolve, write_jsonl

from evidencetree.utils import get_logger

log = get_logger("dataset.prepare_other")

DEFAULT_SOURCE_ROOT = Path(
    "/media/wenke/BBC23084DC1B0A00/datasetForAiii"
)
DATASETS = ("scienceqa", "mrag_bench", "kvqa")
_MRAG_NOISE_TOKENS = {
    "archive",
    "by",
    "car",
    "class",
    "data",
    "gt",
    "image",
    "imagenet",
    "input",
    "test",
    "train",
    "val",
}


def _iter_parquet_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64):
        yield from batch.to_pylist()


def _save_image(image: dict[str, Any], stem: Path) -> Path:
    data = image.get("bytes") if image else None
    if not data:
        raise ValueError(f"No embedded image bytes available for {stem.name}.")
    with Image.open(io.BytesIO(data)) as loaded:
        suffix = {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
        }.get(loaded.format or "", ".png")
    path = stem.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def _write_dataset(
    name: str,
    output_root: Path,
    queries: Iterable[dict[str, Any]],
    corpus: Iterable[dict[str, Any]],
) -> None:
    out = output_root / name
    n_queries = write_jsonl(out / "queries.jsonl", queries)
    n_docs = write_jsonl(out / "corpus.jsonl", corpus)
    log.info(
        "Prepared %s: %d queries, %d documents -> %s",
        name,
        n_queries,
        n_docs,
        out,
    )


def _mrag_weak_label_from_filename(path: Path) -> str:
    stem = unquote(path.stem)
    if "_gt_" in stem:
        stem = stem.split("_gt_", 1)[1]
    elif "_input" in stem:
        stem = stem.split("_input", 1)[0]
    stem = re.sub(r"ILSVRC\d*", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d{2,}x\d{2,}\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d{5,}\b", " ", stem)
    stem = re.sub(r"[_+%./\\-]+", " ", stem)
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    tokens = []
    for token in re.findall(r"[A-Za-z0-9]+", stem):
        low = token.lower()
        if low in _MRAG_NOISE_TOKENS:
            continue
        if low.isdigit() and len(low) > 4:
            continue
        if len(low) == 1 and not low.isdigit():
            continue
        tokens.append(token)
    label = " ".join(tokens)
    label = re.sub(r"\s+", " ", label).strip()
    return label or path.stem


def _mrag_doc_text(path: Path) -> str:
    label = _mrag_weak_label_from_filename(path)
    return (
        f"Image filename weak label: {label}. "
        f"Original filename: {path.name}."
    )


SCIENCEQA_SPLITS = {
    "train": "train-00000-of-00001-1028f23e353fbe3e.parquet",
    "validation": "validation-00000-of-00001-6c7328ff6c84284c.parquet",
    "test": "test-00000-of-00001-f0e719df791966ff.parquet",
}


def prepare_scienceqa(source_root: Path, output_root: Path, n: int = 1) -> None:
    data_dir = source_root / "ScienceQA/dataset_hf/data"
    limit = None if n == 0 else n
    rows = []
    for split, filename in SCIENCEQA_SPLITS.items():
        path = data_dir / filename
        for candidate in _iter_parquet_rows(path):
            if (candidate.get("image") or {}).get("bytes"):
                candidate["_et_split"] = split
                rows.append(candidate)
                if limit is not None and len(rows) >= limit:
                    break
        if limit is not None and len(rows) >= limit:
            break
    if limit is not None and len(rows) < limit:
        raise RuntimeError(f"ScienceQA contains only {len(rows)} image samples.")

    queries = []
    corpus = []
    for index, row in enumerate(rows):
        split = row["_et_split"]
        image_path = _save_image(
            row["image"], output_root / f"scienceqa/assets/{split}/query_{index:06d}"
        )
        choices = row["choices"]
        answer_index = int(row["answer"])
        answer = choices[answer_index]
        answer_letter = chr(65 + answer_index)
        question = row["question"] + "\nChoices: " + "; ".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        )
        corpus.extend(
            [
                {
                    "doc_id": f"scienceqa-{index}-lecture",
                    "title": f"Science lesson {index + 1}",
                    "text": row.get("lecture") or row.get("hint") or question,
                },
                {
                    "doc_id": f"scienceqa-{index}-hint",
                    "title": f"Question context {index + 1}",
                    "text": row.get("hint") or "Use the pictured scientific evidence.",
                },
            ]
        )
        queries.append(
            {
                "query_id": f"scienceqa-smoke-{index}",
                "question": question,
                "gold_answers": [
                    answer,
                    answer_letter,
                    f"{answer_letter}. {answer}",
                ],
                "image_path": str(image_path),
                "metadata": {
                    "subject": row.get("subject"),
                    "topic": row.get("topic"),
                    "split": split,
                    "source_pid": row.get("pid"),
                    "adapter_mode": "native_text_and_image",
                },
            }
        )
    _write_dataset("scienceqa", output_root, queries, corpus)


def prepare_mrag_bench(source_root: Path, output_root: Path, n: int = 1) -> None:
    path = source_root / "MRAG-Bench/data/test-00000-of-000028.parquet"
    rows = list(_iter_parquet_rows(path))[:n]
    if len(rows) < n:
        raise RuntimeError(f"MRAG-Bench contains only {len(rows)} rows.")

    queries = []
    query_hashes: set[tuple[int, bytes]] = set()
    for index, row in enumerate(rows):
        image_path = _save_image(
            row["image"], output_root / f"mrag_bench/assets/query_{index:03d}"
        )
        query_hashes.add(
            (image_path.stat().st_size, hashlib.sha256(image_path.read_bytes()).digest())
        )
        choices = [row[key] for key in ("A", "B", "C", "D")]
        question = row["question"] + "\nChoices: " + "; ".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)
        )
        queries.append(
            {
                "query_id": f"mrag-{row['id']}",
                "question": question,
                "gold_answers": [
                    row["answer"],
                    row["answer_choice"],
                    f"{row['answer_choice']}. {row['answer']}",
                ],
                "image_path": str(image_path),
                "metadata": {
                    "scenario": row["scenario"],
                    "aspect": row["aspect"],
                    "adapter_mode": "full_unlabelled_visual_corpus",
                },
            }
        )

    corpus_root = source_root / "MRAG-Bench/image_corpus"
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def is_query_duplicate(candidate: Path) -> bool:
        size = candidate.stat().st_size
        matching = {digest for query_size, digest in query_hashes if query_size == size}
        return bool(matching) and hashlib.sha256(candidate.read_bytes()).digest() in matching

    candidate_paths = [
        candidate.resolve()
        for candidate in sorted(corpus_root.iterdir())
        if candidate.is_file()
        and candidate.suffix.lower() in image_suffixes
        and "_input." not in candidate.name.lower()
        and not is_query_duplicate(candidate)
    ]
    if not candidate_paths:
        raise FileNotFoundError(f"MRAG image corpus is empty: {corpus_root}")
    corpus = [
        {
            "doc_id": f"mrag-corpus-{index:06d}",
            "title": f"MRAG image: {_mrag_weak_label_from_filename(candidate)}",
            "text": _mrag_doc_text(candidate),
            "image_path": str(candidate),
        }
        for index, candidate in enumerate(candidate_paths)
    ]
    for query in queries:
        query["metadata"]["corpus_size"] = len(corpus)
    _write_dataset("mrag_bench", output_root, queries, corpus)


def _kvqa_raw_dir(source_root: Path) -> Path:
    candidates = [
        source_root / "KVQA/OpenDataLab___KVQA/raw",
        source_root / "KVQA/official",
        source_root / "KVQA",
    ]
    for candidate in candidates:
        if (candidate / "dataset.json").exists():
            return candidate
    raise FileNotFoundError(
        "KVQA dataset.json not found. Expected one of:\n"
        + "\n".join(f"  {path / 'dataset.json'}" for path in candidates)
    )


def _kvqa_image_path(source_root: Path, raw_img_path: str) -> Path:
    """Return the expected query-image path after KVQAimgs.tar.gz extraction."""
    target_root = source_root / "KVQA/OpenDataLab___KVQA/raw"
    return (target_root / raw_img_path).resolve()


def _kvqa_text_doc(qid: str, entity: str, captions: list[str]) -> str:
    unique_caps = []
    for caption in captions:
        caption = re.sub(r"\s+", " ", str(caption or "")).strip()
        if caption and caption not in unique_caps:
            unique_caps.append(caption)
    pieces = []
    if entity:
        pieces.append(f"Entity name: {entity}.")
    if unique_caps:
        pieces.append("Wiki caption: " + " ".join(unique_caps))
    if qid:
        pieces.append(f"Wikidata QID: {qid}.")
    return " ".join(pieces).strip()


def prepare_kvqa(source_root: Path, output_root: Path, n: int = 1) -> None:
    """Prepare KVQA using weak text docs from dataset.json only.

    This does not fetch Wikidata/Wikipedia. The text corpus is intentionally
    weak: one doc per QID/entity with entity name, wiki caption(s), and QID.
    """
    raw_dir = _kvqa_raw_dir(source_root)
    data = json.loads((raw_dir / "dataset.json").read_text(encoding="utf-8"))
    limit = None if n == 0 else n

    queries: list[dict[str, Any]] = []
    docs_by_qid: dict[str, dict[str, Any]] = {}
    captions_by_qid: dict[str, list[str]] = {}

    for image_id, row in data.items():
        entities = [str(x) for x in row.get("NamedEntities", [])]
        qids = [str(x) for x in row.get("Qids", [])]
        if not qids:
            qids = [f"kvqa-image-{image_id}"]
        questions = row.get("Questions", []) or []
        answers = row.get("Answers", []) or []
        types = row.get("Type of Question", []) or []
        splits = row.get("split", []) or []
        raw_img = str(row.get("imgPath", ""))
        image_path = str(_kvqa_image_path(source_root, raw_img)) if raw_img else None
        wiki_cap = str(row.get("wikiCap", "") or "")
        primary_entity = entities[0] if entities else ""
        primary_qid = qids[0]

        for qid, entity in zip(qids, entities or [primary_entity] * len(qids)):
            captions_by_qid.setdefault(qid, []).append(wiki_cap)
            docs_by_qid.setdefault(
                qid,
                {
                    "doc_id": qid,
                    "title": entity or qid,
                    "text": "",
                    "metadata": {
                        "source": "kvqa_dataset_json_weak_text",
                        "named_entity": entity,
                        "qid": qid,
                    },
                },
            )

        for index, (question, answer) in enumerate(zip(questions, answers)):
            query_id = f"kvqa-{image_id}-{index:02d}"
            queries.append(
                {
                    "query_id": query_id,
                    "question": str(question),
                    "gold_answers": [str(answer)],
                    "image_path": image_path,
                    "metadata": {
                        "source_image_id": image_id,
                        "imgPath": raw_img,
                        "named_entities": entities,
                        "qids": qids,
                        "primary_qid": primary_qid,
                        "primary_entity": primary_entity,
                        "wikiCap": wiki_cap,
                        "split": splits[index] if index < len(splits) else None,
                        "type_of_question": types[index] if index < len(types) else None,
                        "adapter_mode": "weak_text_from_dataset_json",
                    },
                }
            )
            if limit is not None and len(queries) >= limit:
                break
        if limit is not None and len(queries) >= limit:
            break

    for qid, doc in docs_by_qid.items():
        doc["text"] = _kvqa_text_doc(
            qid=qid,
            entity=str(doc.get("title", "")),
            captions=captions_by_qid.get(qid, []),
        )
    corpus = sorted(docs_by_qid.values(), key=lambda doc: doc["doc_id"])
    _write_dataset("kvqa", output_root, queries, corpus)


PREPARERS = {
    "scienceqa": prepare_scienceqa,
    "mrag_bench": prepare_mrag_bench,
    "kvqa": prepare_kvqa,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe local samples from downloaded VQA datasets."
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default="data/corpus")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Queries per dataset. Use 0 to convert all available samples.",
    )
    args = parser.parse_args(argv)

    if args.n < 0:
        parser.error("--n must be non-negative")
    output_root = resolve(args.output_root)
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in selected:
        PREPARERS[name](args.source_root, output_root, n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
