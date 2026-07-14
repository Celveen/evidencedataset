"""Extract selected E-VQA query images from iNat2021 train_mini.tar.gz."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from evidencetree.utils import get_logger

log = get_logger("dataset.extract_evqa_images")

DEFAULT_SOURCE_ROOT = Path("/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA")
DEFAULT_DATA_DIR = Path("data/corpus/evqa")


def _needed_members(queries_path: Path) -> set[str]:
    needed = set()
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = (row.get("metadata") or {}).get("selected_image_file_name")
        if rel:
            needed.add(rel)
    return needed


def extract(args: argparse.Namespace) -> None:
    archive = args.source_root / "raw" / "train_mini.tar.gz"
    output_root = args.source_root / "images"
    needed = _needed_members(args.data_dir / "queries.jsonl")
    if not needed:
        raise RuntimeError("No selected_image_file_name values found in queries.jsonl")

    existing = sum(1 for rel in needed if (output_root / rel).exists())
    log.info("Need %d images; already extracted %d.", len(needed), existing)
    if existing == len(needed):
        return

    extracted = 0
    with tarfile.open(archive) as tf:
        for member in tf:
            if not member.isfile() or member.name not in needed:
                continue
            target = output_root / member.name
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                continue
            with target.open("wb") as handle:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
            if extracted % 100 == 0:
                log.info("  extracted %d/%d selected images", extracted, len(needed))
            if extracted + existing == len(needed):
                break

    final_existing = sum(1 for rel in needed if (output_root / rel).exists())
    log.info("Extracted selected images: %d/%d present.", final_existing, len(needed))
    if final_existing < len(needed):
        raise RuntimeError(
            f"Only {final_existing}/{len(needed)} selected images were found in archive."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)
    extract(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
