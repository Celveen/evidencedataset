"""Convert the official InfoSeek annotations into this project's queries JSONL format.

Place the raw files (official GCS links, see DatasetConstruct/README.md) in
``data/corpus/infoseek/raw/``：

    infoseek_val.jsonl         # 73,620 rows with answer/answer_eval - our test set
    infoseek_test.jsonl        # 347,980 rows, no answers (leaderboard only) - skipped
    infoseek_val_withkb.jsonl  # data_id -> Wikidata entity map (used to build the corpus)
    infoseek_human.jsonl       # human split, no public answers - skipped

Writes ``data/corpus/infoseek/infoseek_queries.jsonl``, one row per query:
    {"query_id", "question", "gold_answers": [...], "image_path",
     "metadata": {"image_id", "data_split", "entity_id", "entity_text"}}

Images: InfoSeek uses OVEN images (image_id such as "oven_04990048"). If the
OVEN images are available locally, pass --images-dir (files are looked up as
<image_id>.jpg); otherwise image_path is set to null and the text-only pipeline
still runs.

Usage:
    python DatasetConstruct/prepare_infoseek.py
    python DatasetConstruct/prepare_infoseek.py --images-dir /path/to/oven_images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]   # DatasetConstruct/ (common.py, which
if str(_DC) not in sys.path:                # also puts src/ on the path)
    sys.path.insert(0, str(_DC))

from common import read_jsonl, resolve, write_jsonl

from evidencetree.utils import get_logger

log = get_logger("dataset.prepare")


def convert(raw_dir: Path, out_path: Path, images_dir: Path | None) -> int:
    val_path = raw_dir / "infoseek_val.jsonl"
    kb_path = raw_dir / "infoseek_val_withkb.jsonl"
    if not val_path.exists():
        raise FileNotFoundError(
            f"{val_path} not found — download it first (see DatasetConstruct/README.md)."
        )

    entity_by_id: dict[str, dict] = {}
    if kb_path.exists():
        entity_by_id = {row["data_id"]: row for row in read_jsonl(kb_path)}
    else:
        log.warning("KB mapping %s missing — entity metadata will be empty.", kb_path)

    # Pre-scan the image dir once: map image_id (file stem) -> path. Handles any
    # extension and nested subdirectories, so it works whatever layout the OVEN
    # download used on the server (oven_xxx.jpg / .jpeg / .png, sharded dirs...).
    image_index: dict[str, str] = {}
    if images_dir is not None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        for p in images_dir.rglob("*"):
            if p.suffix.lower() in exts:
                image_index.setdefault(p.stem, str(p))
        log.info("Indexed %d image files under %s", len(image_index), images_dir)

    rows = []
    n_with_image = 0
    for obj in read_jsonl(val_path):
        image_id = obj.get("image_id", "")
        image_path = image_index.get(image_id)
        if image_path:
            n_with_image += 1
        kb = entity_by_id.get(obj["data_id"], {})
        gold = obj.get("answer_eval") or obj.get("answer") or []
        rows.append(
            {
                "query_id": obj["data_id"],
                "question": obj["question"],
                "gold_answers": [str(a) for a in gold],
                "image_path": image_path,
                "metadata": {
                    "image_id": image_id,
                    "data_split": obj.get("data_split", ""),
                    "entity_id": kb.get("entity_id"),
                    "entity_text": kb.get("entity_text"),
                },
            }
        )

    n = write_jsonl(out_path, rows)
    log.info(
        "Converted %d queries (%d with local images) -> %s", n, n_with_image, out_path
    )
    return n


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert raw InfoSeek to pipeline format.")
    p.add_argument("--raw-dir", default="data/corpus/infoseek/raw")
    p.add_argument("--out", default="data/corpus/infoseek/infoseek_queries.jsonl")
    p.add_argument("--images-dir", default=None, help="Local OVEN image directory (optional).")
    args = p.parse_args(argv)
    convert(
        raw_dir=resolve(args.raw_dir),
        out_path=resolve(args.out),
        images_dir=resolve(args.images_dir) if args.images_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
