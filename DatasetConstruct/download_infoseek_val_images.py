"""Download only the OVEN images referenced by InfoSeek validation data.

The OVEN Hugging Face snapshot is about 294 GB. This script processes the
relevant tar shards one at a time, extracts only requested images, and removes
each completed tar file immediately to keep peak disk usage manageable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download


REPO_ID = "ychenNLP/oven"
SHARDS = ("shard04.tar", "shard05.tar", "shard08.tar")
AIRCRAFT_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
    "fgvc-aircraft-2013b.tar.gz"
)


def load_targets(ids_path: Path, mapping_path: Path) -> dict[str, str]:
    wanted = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
    targets: dict[str, str] = {}
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        for image_id, source_path in csv.reader(handle):
            if image_id in wanted:
                targets[image_id] = source_path
    missing = wanted - targets.keys()
    if missing:
        raise RuntimeError(f"{len(missing)} image IDs are absent from the OVEN mapping")
    return targets


def existing_ids(images_dir: Path) -> set[str]:
    return {path.stem for path in images_dir.glob("oven_*.jpg") if path.stat().st_size}


def extract_targets(archive: Path, images_dir: Path, wanted: set[str]) -> int:
    found = 0
    with archive.open("rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                image_id = Path(member.name).stem
                if image_id not in wanted:
                    continue
                destination = images_dir / f"{image_id}.jpg"
                if destination.exists() and destination.stat().st_size:
                    wanted.discard(image_id)
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    continue
                temporary = destination.with_suffix(".jpg.part")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temporary.replace(destination)
                wanted.discard(image_id)
                found += 1
                if found % 500 == 0:
                    print(f"  extracted {found} images; {len(wanted)} still pending")
    return found


def process_shard(
    root: Path, shard: str, targets: set[str], images_dir: Path
) -> None:
    marker = root / "state" / f"{shard}.complete"
    if marker.exists():
        print(f"{shard}: already complete")
        return

    pending = targets - existing_ids(images_dir)
    if not pending:
        marker.touch()
        print(f"{shard}: no pending images")
        return

    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    print(f"{shard}: downloading for {len(pending)} possible targets")
    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=shard,
            repo_type="dataset",
            local_dir=downloads,
        )
    )
    print(f"{shard}: scanning {archive}")
    extracted = extract_targets(archive, images_dir, pending)
    archive.unlink()
    marker.touch()
    print(f"{shard}: extracted {extracted}; temporary archive removed")


def download_http(url: str, destination: Path) -> Path:
    partial = destination.with_suffix(destination.suffix + ".part")
    total = None
    attempts = 0
    while total is None or partial.stat().st_size < total:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        mode = "ab" if offset else "wb"
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                    mode = "wb"
                content_range = response.headers.get("content-range", "")
                if "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[1])
                else:
                    total = offset + int(response.headers["content-length"])
                downloaded = offset
                next_report = downloaded + 256 * 1024 * 1024
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            print(
                                f"  downloaded {downloaded / 1e9:.2f}/"
                                f"{total / 1e9:.2f} GB"
                            )
                            next_report += 256 * 1024 * 1024
            attempts = 0
        except requests.RequestException as error:
            attempts += 1
            if attempts > 10:
                raise
            print(f"  connection interrupted; retrying from {offset} bytes: {error}")
            time.sleep(min(30, attempts * 3))
    partial.replace(destination)
    return destination


def process_aircraft(
    root: Path, targets: dict[str, str], images_dir: Path
) -> None:
    marker = root / "state" / "fgvc-aircraft.complete"
    if marker.exists():
        print("fgvc-aircraft: already complete")
        return

    source_to_id = {
        source_path.removeprefix("aircraft/"): image_id
        for image_id, source_path in targets.items()
        if source_path.startswith("aircraft/")
        and not (images_dir / f"{image_id}.jpg").exists()
    }
    if not source_to_id:
        marker.touch()
        print("fgvc-aircraft: no pending images")
        return

    archive = root / "downloads" / "fgvc-aircraft-2013b.tar.gz"
    print(f"fgvc-aircraft: downloading for {len(source_to_id)} targets")
    download_http(AIRCRAFT_URL, archive)

    found = 0
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            image_id = source_to_id.get(member.name)
            if image_id is None or not member.isfile():
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            destination = images_dir / f"{image_id}.jpg"
            temporary = destination.with_suffix(".jpg.part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(destination)
            found += 1
    archive.unlink()
    marker.touch()
    print(f"fgvc-aircraft: extracted {found}; temporary archive removed")


def write_manifest(root: Path, targets: dict[str, str], images_dir: Path) -> int:
    manifest_path = root / "annotations" / "val_image_manifest.jsonl"
    missing_path = root / "annotations" / "missing_val_image_ids.txt"
    rows = []
    missing = []
    for image_id, source_path in sorted(targets.items()):
        local_path = images_dir / f"{image_id}.jpg"
        if local_path.exists() and local_path.stat().st_size:
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(local_path),
                    "source_path": source_path,
                    "size": local_path.stat().st_size,
                }
            )
        else:
            missing.append(image_id)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    missing_path.write_text(
        "".join(f"{image_id}\n" for image_id in missing), encoding="utf-8"
    )
    print(f"manifest: {len(rows)} images; missing: {len(missing)}")
    return len(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("INFOSEEK_RAW_ROOT", "data/raw/infoseek")),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    annotations = root / "annotations"
    images_dir = root / "images"
    state_dir = root / "state"
    images_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(
        annotations / "val_image_ids.txt",
        root / "oven_snapshot" / "ovenid2impath.csv",
    )
    shard_targets = {
        "shard04.tar": {key for key in targets if key.startswith("oven_04")},
        "shard05.tar": {key for key in targets if key.startswith("oven_05")},
        "shard08.tar": {key for key in targets if key.startswith("oven_05")},
    }
    for shard in SHARDS:
        process_shard(root, shard, shard_targets[shard], images_dir)

    # The remaining oven_00 validation images all originate from FGVC Aircraft.
    # Its source archive is much smaller than OVEN shard06 and shard07.
    process_aircraft(root, targets, images_dir)
    missing = write_manifest(root, targets, images_dir)
    return 0 if missing == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
