"""Download corpus-side Wikipedia images for the local E-VQA subset.

The official E-VQA KB contains ``image_urls`` for most Wikipedia entities. This
script downloads one representative image per entity, rewrites
``data/corpus/evqa/corpus.jsonl`` to attach the downloaded image path to every
section of that entity, and updates ``stats.json``.

Images are stored under the external dataset disk by default so the root
filesystem does not fill up.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
from PIL import Image


DEFAULT_SOURCE_ROOT = Path("/media/wenke/BBC23084DC1B0A00/datasetForAiii/EVQA")
DEFAULT_DATA_DIR = Path("data/corpus/evqa")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:120] or "doc"


def image_suffix(url: str, content_type: str | None) -> str:
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


def thumbnail_redirect_url(url: str, width: int) -> str:
    filename = unquote(url.split("?", 1)[0].rsplit("/", 1)[-1])
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(filename)}?width={width}"


def expanded_image_urls(urls: list[str], *, thumbnail_width: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if thumbnail_width > 0:
            thumb = thumbnail_redirect_url(url, thumbnail_width)
            if thumb not in seen:
                seen.add(thumb)
                out.append(thumb)
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_title_map(selected_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with selected_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[row["wikipedia_url"]] = row["wikipedia_title"]
    return mapping


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def existing_for_stem(image_dir: Path, stem: str) -> Path | None:
    for path in image_dir.glob(stem + ".*"):
        if valid_image(path):
            return path
        path.unlink(missing_ok=True)
    return None


def migrate_existing_images(data_dir: Path, image_dir: Path) -> None:
    old_dir = data_dir / "assets" / "wiki_images"
    if not old_dir.exists() or old_dir.resolve() == image_dir.resolve():
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    for source in old_dir.glob("*"):
        if not source.is_file():
            continue
        target = image_dir / source.name
        if not target.exists() and valid_image(source):
            shutil.copy2(source, target)


def download_one(
    *,
    session: requests.Session,
    urls: list[str],
    path_stem: Path,
    min_sleep: float,
    max_sleep: float,
    max_attempts: int,
    thumbnail_width: int,
    max_urls_per_entity: int,
) -> Path | None:
    expanded_urls = expanded_image_urls(urls, thumbnail_width=thumbnail_width)
    if max_urls_per_entity > 0:
        expanded_urls = expanded_urls[:max_urls_per_entity]
    for url in expanded_urls:
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(url, timeout=(20, 120), stream=True)
                if response.status_code == 429:
                    wait = min(20, 3 * attempt)
                    print(
                        f"429 rate limited; sleeping {wait}s then trying next URL: {url}",
                        flush=True,
                    )
                    time.sleep(wait)
                    break
                if response.status_code == 404:
                    print(f"image URL not found, trying next URL: {url}", flush=True)
                    break
                response.raise_for_status()
                suffix = image_suffix(url, response.headers.get("content-type"))
                tmp = path_stem.with_suffix(suffix + ".part")
                final = path_stem.with_suffix(suffix)
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                if valid_image(tmp):
                    tmp.replace(final)
                    time.sleep(random.uniform(min_sleep, max_sleep))
                    return final
                tmp.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001 - try next URL/attempt.
                print(f"download failed attempt {attempt}: {url}: {exc}", flush=True)
                time.sleep(min(120, 5 * attempt))
    return None


def prepare_candidate(
    *,
    url: str,
    kb: dict[str, Any],
    url_to_title: dict[str, str],
    image_dir: Path,
) -> tuple[str, str, list[str], str | None]:
    title = url_to_title.get(url, url)
    stem = safe_id(title)
    existing = existing_for_stem(image_dir, stem)
    urls = [item for item in (kb.get("image_urls") or []) if item]
    return url, stem, urls, str(existing.resolve()) if existing else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="For debugging; 0 means all.")
    parser.add_argument("--min-sleep", type=float, default=0.6)
    parser.add_argument("--max-sleep", type=float, default=1.4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-urls-per-entity",
        type=int,
        default=0,
        help="Try at most this many expanded URLs per entity; 0 means all.",
    )
    parser.add_argument("--shuffle-pending", action="store_true")
    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=512,
        help="Prefer Commons thumbnail redirects at this width; 0 disables.",
    )
    parser.add_argument(
        "--proxy",
        default="",
        help="Optional HTTP proxy, for example http://127.0.0.1:7890.",
    )
    args = parser.parse_args(argv)

    raw = args.source_root / "raw"
    kb_path = raw / f"kb_subset_{args.n}_{args.split}.json"
    selected_path = raw / f"selected_{args.n}_{args.split}.jsonl"
    image_dir = args.source_root / "wiki_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    migrate_existing_images(args.data_dir, image_dir)

    kb_by_url = json.loads(kb_path.read_text(encoding="utf-8"))
    url_to_title = load_title_map(selected_path)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "EvidenceTree local dataset preparation/1.0 "
                "(research use; respectful rate-limited image download)"
            )
        }
    )
    if args.proxy:
        session.trust_env = False
        session.proxies.update({"http": args.proxy, "https": args.proxy})

    downloaded: dict[str, str] = {}
    candidates = [(url, kb) for url, kb in kb_by_url.items() if kb.get("image_urls")]
    if args.limit:
        candidates = candidates[: args.limit]

    prepared = [
        prepare_candidate(url=url, kb=kb, url_to_title=url_to_title, image_dir=image_dir)
        for url, kb in candidates
    ]

    for url, _stem, _urls, existing in prepared:
        if existing:
            downloaded[url] = existing

    pending = [(url, stem, urls) for url, stem, urls, existing in prepared if not existing]
    if args.shuffle_pending:
        random.shuffle(pending)

    if args.workers <= 1:
        processed = 0
        for url, stem, urls, existing in prepared:
            processed += 1
            if existing:
                pass
            else:
                path = download_one(
                    session=session,
                    urls=urls,
                    path_stem=image_dir / stem,
                    min_sleep=args.min_sleep,
                    max_sleep=args.max_sleep,
                    max_attempts=args.max_attempts,
                    thumbnail_width=args.thumbnail_width,
                    max_urls_per_entity=args.max_urls_per_entity,
                )
                if path:
                    downloaded[url] = str(path.resolve())
            if processed % 50 == 0:
                print(
                    f"processed {processed}/{len(candidates)}; "
                    f"downloaded_or_existing={len(downloaded)}",
                    flush=True,
                )
    else:
        done_count = len(downloaded)
        print(
            f"existing={done_count}; pending={len(pending)}; workers={args.workers}",
            flush=True,
        )

        def worker(item: tuple[str, str, list[str]]) -> tuple[str, str | None]:
            url, stem, urls = item
            local_session = requests.Session()
            local_session.headers.update(session.headers)
            local_session.trust_env = session.trust_env
            local_session.proxies.update(session.proxies)
            path = download_one(
                session=local_session,
                urls=urls,
                path_stem=image_dir / stem,
                min_sleep=args.min_sleep,
                max_sleep=args.max_sleep,
                max_attempts=args.max_attempts,
                thumbnail_width=args.thumbnail_width,
                max_urls_per_entity=args.max_urls_per_entity,
            )
            return url, str(path.resolve()) if path else None

        completed = 0
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {executor.submit(worker, item): item for item in pending}
            for future in futures.as_completed(future_to_item):
                url, path = future.result()
                completed += 1
                if path:
                    downloaded[url] = path
                total_processed = done_count + completed
                if total_processed % 50 == 0 or completed == len(pending):
                    print(
                        f"processed {total_processed}/{len(candidates)}; "
                        f"downloaded_or_existing={len(downloaded)}",
                        flush=True,
                    )

    corpus_path = args.data_dir / "corpus.jsonl"
    corpus = load_jsonl(corpus_path)
    docs_with_images = 0
    entities_with_docs = set()
    for row in corpus:
        url = (row.get("metadata") or {}).get("wikipedia_url")
        image_path = downloaded.get(url)
        row["image_path"] = image_path
        if image_path:
            docs_with_images += 1
            entities_with_docs.add(url)
    write_jsonl(corpus_path, corpus)

    stats_path = args.data_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats["wiki_images"] = len(downloaded)
    stats["corpus_docs_with_images"] = docs_with_images
    stats["corpus_entities_with_images"] = len(entities_with_docs)
    stats["corpus_image_dir"] = str(image_dir.resolve())
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
