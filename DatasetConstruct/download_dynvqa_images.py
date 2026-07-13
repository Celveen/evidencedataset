"""Download all images referenced by the official Dyn-VQA snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


def image_name(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return f"{digest}{suffix}"


def collect_rows(dataset_dir: Path) -> tuple[list[Path], dict[str, str]]:
    files = sorted(dataset_dir.glob("DynVQA_*/*.jsonl"))
    urls: dict[str, str] = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                url = json.loads(line)["image_url"]
                urls[url] = image_name(url)
    return files, urls


def valid_image(path: Path) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def download_one(url: str, destination: Path, attempts: int = 8) -> dict:
    if valid_image(destination):
        return {"url": url, "path": str(destination), "status": "existing"}

    temporary = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "Dyn-VQA academic dataset downloader/1.0"}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 120)
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "image" not in content_type.lower():
                    raise RuntimeError(f"unexpected content type: {content_type}")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response.raw, output, length=1024 * 1024)
            if not valid_image(temporary):
                raise RuntimeError("downloaded file is not a valid image")
            temporary.replace(destination)
            return {
                "url": url,
                "path": str(destination),
                "status": "downloaded",
                "size": destination.stat().st_size,
            }
        except Exception as error:  # noqa: BLE001 - retry all network/image errors
            last_error = f"{type(error).__name__}: {error}"
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(30, attempt * 2))
    return {"url": url, "path": str(destination), "status": "failed", "error": last_error}


def write_local_annotations(
    files: list[Path],
    urls: dict[str, str],
    images_dir: Path,
    available_urls: set[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in files:
        language = source.parent.name.removeprefix("DynVQA_")
        target = output_dir / language / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open(encoding="utf-8") as fin, target.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["image_url"] in available_urls:
                    row["image_path"] = str(images_dir / urls[row["image_url"]])
                    row["image_download_status"] = "available"
                else:
                    row["image_path"] = None
                    row["image_download_status"] = "source_url_unavailable"
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/wenke/BBC23084DC1B0A00/datasetForAiii/Dyn-VQA"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()

    root = args.root.resolve()
    dataset_dir = root / "OmniSearch" / "dataset"
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    files, urls = collect_rows(dataset_dir)
    print(f"annotation files: {len(files)}; unique image URLs: {len(urls)}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, url, images_dir / filename, args.attempts
            ): url
            for url, filename in urls.items()
        }
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 50 == 0 or index == len(futures):
                failed = sum(row["status"] == "failed" for row in results)
                print(f"processed {index}/{len(futures)}; failed {failed}")

    results.sort(key=lambda row: row["url"])
    manifest = root / "image_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = [row for row in results if row["status"] == "failed"]
    (root / "failed_images.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    write_local_annotations(
        files,
        urls,
        images_dir,
        {row["url"] for row in results if row["status"] != "failed"},
        root / "annotations_with_local_images",
    )
    print(f"complete: {len(results) - len(failures)}; failed: {len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
