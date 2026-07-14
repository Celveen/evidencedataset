"""Prepare a size-limited Encyclopedic-VQA subset.

This script builds the EvidenceTree shared format:

    data/corpus/evqa/queries.jsonl
    data/corpus/evqa/corpus.jsonl

The subset is intentionally conservative:

* E-VQA rows are taken from the HuggingFace metadata mirror.
* Only iNaturalist rows whose image ids exist in iNat2021 train_mini are kept.
* A fixed number of QA rows is selected.
* Detailed text is streamed out of the official E-VQA Wikipedia KB zip.
* The first Wikipedia image for each entity is downloaded as corpus-side image
  evidence when available.

Query images are expected to be extracted later from iNat train_mini.tar.gz
using ``extract_evqa_query_images.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from common import write_jsonl
from evidencetree.utils import get_logger

log = get_logger("dataset.prepare_evqa")

DEFAULT_SOURCE_ROOT = Path("/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA")
DEFAULT_OUTPUT_ROOT = Path("data/corpus")


def _safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:120] or "doc"


def _load_train_mini_map(path: Path) -> dict[str, str]:
    with tarfile.open(path) as tf:
        member = tf.getmembers()[0]
        data = json.load(tf.extractfile(member))
    return {str(image["id"]): image["file_name"] for image in data["images"]}


def _select_rows(
    n: int,
    mini_id_to_file: dict[str, str],
    split: str,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("reonokiy/vsp-encyclopedic-vqa", split=split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if row.get("dataset_name") != "inaturalist":
            continue
        ids = [item for item in row["dataset_image_ids"].split("|") if item]
        matched = [item for item in ids if item in mini_id_to_file]
        if not matched:
            continue
        item = dict(row)
        item["selected_image_id"] = matched[0]
        item["selected_image_file_name"] = mini_id_to_file[matched[0]]
        rows.append(item)
        if len(rows) >= n:
            break
    if len(rows) < n:
        log.warning("Selected only %d rows; requested %d.", len(rows), n)
    return rows


def _stream_kb_subset(
    *,
    kb_zip: Path,
    urls: set[str],
) -> dict[str, dict[str, Any]]:
    import ijson

    found: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(kb_zip) as zf:
        with zf.open("encyclopedic_kb_wiki.json") as raw:
            for url, value in ijson.kvitems(raw, ""):
                if url in urls:
                    found[url] = value
                    if len(found) % 100 == 0:
                        log.info("  KB matched %d/%d urls", len(found), len(urls))
                    if len(found) == len(urls):
                        break
    missing = urls - set(found)
    if missing:
        log.warning("Missing %d KB urls.", len(missing))
    return found


def _image_suffix(url: str, content_type: str | None) -> str:
    path = url.split("?", 1)[0].lower()
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(suffix):
            return ".jpg" if suffix == ".jpeg" else suffix
    if content_type:
        if "png" in content_type:
            return ".png"
        if "webp" in content_type:
            return ".webp"
    return ".jpg"


def _download_wiki_images(
    *,
    kb_by_url: dict[str, dict[str, Any]],
    url_to_title: dict[str, str],
    image_dir: Path,
    timeout: int = 20,
) -> dict[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "EvidenceTree dataset preparation "
                "(research use; contact: local)"
            )
        }
    )
    for index, (url, kb) in enumerate(kb_by_url.items(), 1):
        urls = kb.get("image_urls") or []
        image_url = next((item for item in urls if item), None)
        if not image_url:
            continue
        stem = _safe_id(url_to_title.get(url) or url)
        existing = next(image_dir.glob(stem + ".*"), None)
        if existing and existing.stat().st_size > 0:
            out[url] = str(existing.resolve())
            continue
        try:
            response = session.get(image_url, timeout=timeout, stream=True)
            response.raise_for_status()
            suffix = _image_suffix(image_url, response.headers.get("content-type"))
            path = image_dir / f"{stem}{suffix}"
            with path.open("wb") as handle:
                for chunk in response.iter_content(1 << 16):
                    if chunk:
                        handle.write(chunk)
            if path.stat().st_size > 0:
                out[url] = str(path.resolve())
        except Exception as exc:  # noqa: BLE001 - keep preparing the subset.
            log.warning("Failed to download wiki image for %s: %s", url, exc)
        if index % 100 == 0:
            log.info("  wiki images processed %d/%d", index, len(kb_by_url))
        time.sleep(0.02)
    return out


def _answers(answer: str) -> list[str]:
    seen = []
    for item in answer.split("|"):
        item = item.strip()
        if item and item not in seen:
            seen.append(item)
    return seen or [answer]


def prepare(args: argparse.Namespace) -> None:
    raw = args.source_root / "raw"
    output_dir = args.output_root / "evqa"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "assets"
    wiki_image_dir = assets_dir / "wiki_images"

    selected_path = raw / f"selected_{args.n}_{args.split}.jsonl"
    if selected_path.exists():
        rows = [
            json.loads(line)
            for line in selected_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        log.info("Loaded selected rows: %s", selected_path)
    else:
        mini_id_to_file = _load_train_mini_map(raw / "train_mini.json.tar.gz")
        log.info("Loaded %d iNat train_mini image ids.", len(mini_id_to_file))
        rows = _select_rows(args.n, mini_id_to_file, args.split)
        write_jsonl(selected_path, rows)
        log.info("Selected rows written: %s", selected_path)

    urls = {row["wikipedia_url"] for row in rows}
    url_to_title = {row["wikipedia_url"]: row["wikipedia_title"] for row in rows}
    kb_subset_path = raw / f"kb_subset_{args.n}_{args.split}.json"
    if kb_subset_path.exists():
        kb_by_url = json.loads(kb_subset_path.read_text(encoding="utf-8"))
        log.info("Loaded KB subset: %s", kb_subset_path)
    else:
        kb_by_url = _stream_kb_subset(
            kb_zip=raw / "encyclopedic_kb_wiki.zip",
            urls=urls,
        )
        kb_subset_path.write_text(
            json.dumps(kb_by_url, ensure_ascii=False),
            encoding="utf-8",
        )
    wiki_images = {}
    if not args.skip_wiki_images:
        wiki_images = _download_wiki_images(
            kb_by_url=kb_by_url,
            url_to_title=url_to_title,
            image_dir=wiki_image_dir,
        )

    corpus = []
    for url, kb in kb_by_url.items():
        title = url_to_title.get(url, url)
        section_texts = kb.get("section_texts") or []
        section_titles = kb.get("section_titles") or []
        image_path = wiki_images.get(url)
        for section_index, text in enumerate(section_texts):
            text = (text or "").strip()
            if not text:
                continue
            section_title = ""
            if section_index < len(section_titles):
                section_title = section_titles[section_index] or ""
            corpus.append(
                {
                    "doc_id": f"evqa-{_safe_id(title)}-s{section_index}",
                    "title": title if not section_title else f"{title} - {section_title}",
                    "text": text,
                    "image_path": image_path,
                    "metadata": {
                        "wikipedia_url": url,
                        "source": "encyclopedic_kb_wiki",
                        "section_index": section_index,
                    },
                }
            )

    image_root = args.source_root / "images"
    queries = []
    for index, row in enumerate(rows):
        image_rel = row.get("selected_image_file_name")
        image_path = str((image_root / image_rel).resolve()) if image_rel else None
        queries.append(
            {
                "query_id": f"evqa-{args.split}-{index:06d}",
                "question": row["question"],
                "gold_answers": _answers(row["answer"]),
                "image_path": image_path,
                "metadata": {
                    "wikipedia_title": row["wikipedia_title"],
                    "wikipedia_url": row["wikipedia_url"],
                    "question_type": row["question_type"],
                    "dataset_name": row["dataset_name"],
                    "dataset_category_id": row["dataset_category_id"],
                    "selected_image_id": row["selected_image_id"],
                    "selected_image_file_name": image_rel,
                    "evidence": row.get("evidence", ""),
                    "evidence_section_id": row.get("evidence_section_id", ""),
                    "evidence_section_title": row.get("evidence_section_title", ""),
                },
            }
        )

    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    stats = {
        "queries": len(queries),
        "corpus_docs": len(corpus),
        "unique_wikipedia_urls": len(urls),
        "kb_matched": len(kb_by_url),
        "wiki_images": len(wiki_images),
        "query_images_pending_extraction": sum(1 for q in queries if q["image_path"]),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Prepared EVQA subset: %s", stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--skip-wiki-images",
        action="store_true",
        help="Do not download corpus-side Wikipedia images.",
    )
    args = parser.parse_args(argv)
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
