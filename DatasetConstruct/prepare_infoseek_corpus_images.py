"""Add corpus-side Wikipedia images to the local InfoSeek corpus.

Official InfoSeek releases the Wikipedia KB as text records with an optional
``wikipedia_image_url`` field. Images are not bundled; the official instruction
is to download them from that URL. This script follows that flow for the local
subset only:

1. read entity ids used by ``infoseek_corpus.jsonl``;
2. scan the official ``Wiki6M_ver_1_0.jsonl.gz`` for ``wikipedia_image_url``;
3. download one representative image per entity;
4. attach ``image_path`` to every corpus chunk of that entity;
5. rebuild CLIP indexes so image_search can target corpus-side images.

The full Wiki6M gzip is large, so keep it on the external dataset disk and run
this script in tmux. The script is idempotent and safe to resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
from PIL import Image


DEFAULT_DATA_DIR = Path("data/corpus/infoseek")
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get("INFOSEEK_RAW_ROOT", "data/raw/infoseek")
)
DEFAULT_WIKI6M_URL = (
    "https://storage.googleapis.com/gresearch/open-vision-language/"
    "Wiki6M_ver_1_0.jsonl.gz"
)


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:120] or "entity"


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


def commons_filename(url: str) -> str:
    return unquote(url.split("?", 1)[0].rsplit("/", 1)[-1])


def commons_api_image_urls(
    *,
    session: requests.Session,
    image_url: str,
    thumbnail_width: int,
    max_attempts: int,
) -> list[str]:
    if thumbnail_width <= 0:
        return []
    filename = commons_filename(image_url)
    if not filename:
        return []
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": str(thumbnail_width),
        "titles": "File:" + filename,
    }
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(endpoint, params=params, timeout=(20, 120))
            if response.status_code == 429:
                print(f"429 rate limited; skip for now: Commons API {filename}", flush=True)
                return []
            response.raise_for_status()
            data = response.json()
            urls: list[str] = []
            for page in data.get("query", {}).get("pages", {}).values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                thumb = info.get("thumburl")
                if thumb:
                    urls.append(str(thumb))
            return urls
        except Exception as exc:  # noqa: BLE001 - fall back to redirect URL.
            print(
                f"Commons API failed attempt={attempt}/{max_attempts}: "
                f"{filename}: {exc}",
                flush=True,
            )
            time.sleep(min(30, 3 * attempt))
    return []


def expanded_image_urls(url: str, *, thumbnail_width: int) -> list[str]:
    if thumbnail_width > 0:
        return [thumbnail_redirect_url(url, thumbnail_width)]
    return [url]


def entity_from_doc_id(doc_id: str) -> str:
    return doc_id.split("_p", 1)[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_needed_entities(corpus_path: Path) -> set[str]:
    needed: set[str] = set()
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            needed.add(entity_from_doc_id(str(row["doc_id"])))
    return needed


def load_entity_titles(corpus_path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            entity_id = entity_from_doc_id(str(row["doc_id"]))
            title = str(row.get("title") or "").strip()
            if title and entity_id not in titles:
                titles[entity_id] = title
    return titles


def remote_content_length(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=(30, 120))
    response.raise_for_status()
    return int(response.headers["content-length"])


def curl_range(url: str, target: Path, start: int, end: int, retries: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return
    tmp = target.with_suffix(target.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        tmp.unlink(missing_ok=True)
        cmd = [
            "curl",
            "--http1.1",
            "-L",
            "--connect-timeout",
            "30",
            "--retry",
            "3",
            "--retry-delay",
            "3",
            "-r",
            f"{start}-{end}",
            "-o",
            str(tmp),
            url,
        ]
        try:
            subprocess.run(cmd, check=True)
            if tmp.stat().st_size != expected:
                raise RuntimeError(
                    f"range {start}-{end} expected {expected} bytes, "
                    f"got {tmp.stat().st_size}"
                )
            tmp.replace(target)
            return
        except Exception as exc:  # noqa: BLE001 - retry this chunk.
            print(
                f"range download failed attempt={attempt}/{retries} "
                f"bytes={start}-{end}: {exc}",
                flush=True,
            )
            time.sleep(min(120, 5 * attempt))
    raise RuntimeError(f"failed to download range {start}-{end}")


def download_by_ranges(
    *,
    url: str,
    path: Path,
    chunk_mb: int,
    workers: int,
    retries: int,
    size: int | None = None,
) -> None:
    """Download a large file as byte ranges and concatenate verified chunks."""
    expected_size = size or remote_content_length(url)
    if path.exists():
        if path.stat().st_size == expected_size:
            print(f"Wiki6M already downloaded: {path}", flush=True)
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, chunk_mb) * 1024 * 1024
    parts_dir = path.with_suffix(path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = [
        (start, min(start + chunk_size - 1, size - 1))
        for start in range(0, expected_size, chunk_size)
    ]
    print(
        f"Range downloading {url} -> {path} "
        f"size={expected_size:,} chunks={len(ranges)} workers={workers}",
        flush=True,
    )

    def part_path(start: int, end: int) -> Path:
        return parts_dir / f"{start:012d}-{end:012d}.part"

    def worker(item: tuple[int, int]) -> tuple[int, int]:
        start, end = item
        curl_range(url, part_path(start, end), start, end, retries)
        return start, end

    done = 0
    if workers <= 1:
        for item in ranges:
            worker(item)
            done += 1
            if done % 10 == 0 or done == len(ranges):
                print(f"downloaded chunks {done}/{len(ranges)}", flush=True)
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_range = {executor.submit(worker, item): item for item in ranges}
            for future in futures.as_completed(future_to_range):
                future.result()
                done += 1
                if done % 10 == 0 or done == len(ranges):
                    print(f"downloaded chunks {done}/{len(ranges)}", flush=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as out:
        for start, end in ranges:
            part = part_path(start, end)
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out)
    if tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"assembled file has wrong size: {tmp.stat().st_size} != {expected_size}"
        )
    tmp.replace(path)
    print(f"Range download complete: {path}", flush=True)


def load_url_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["entity_id"])] = row
    return cache


def save_url_cache(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows)]
    write_jsonl(path, ordered)


def extract_image_urls(
    *,
    wiki6m_gz: Path,
    needed_entities: set[str],
    cache_path: Path,
    progress_every: int,
) -> dict[str, dict[str, Any]]:
    cache = load_url_cache(cache_path)
    missing = needed_entities - set(cache)
    if not missing:
        return cache

    print(
        f"Scanning Wiki6M for {len(missing)} missing entity URL records "
        f"({len(cache)} cached).",
        flush=True,
    )
    scanned = 0
    matched = 0
    with gzip.open(wiki6m_gz, "rt", encoding="utf-8") as handle:
        for line in handle:
            scanned += 1
            if not line.strip():
                continue
            row = json.loads(line)
            entity_id = str(row.get("wikidata_id") or "")
            if entity_id not in missing:
                if progress_every and scanned % progress_every == 0:
                    print(
                        f"scan rows={scanned:,}; matched_new={matched}; "
                        f"remaining={len(missing)}",
                        flush=True,
                    )
                continue
            cache[entity_id] = {
                "entity_id": entity_id,
                "title": row.get("wikipedia_title") or "",
                "image_url": row.get("wikipedia_image_url"),
                "summary": row.get("wikipedia_summary") or "",
            }
            missing.remove(entity_id)
            matched += 1
            if matched % 50 == 0:
                save_url_cache(cache_path, cache)
                print(
                    f"matched_new={matched}; total_cached={len(cache)}; "
                    f"remaining={len(missing)}",
                    flush=True,
                )
            if not missing:
                break
    save_url_cache(cache_path, cache)
    print(
        f"Wiki6M scan done: scanned={scanned:,}; matched_new={matched}; "
        f"cached={len(cache)}.",
        flush=True,
    )
    return cache


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def commons_redirect_url(filename: str) -> str:
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(filename)}"


def make_session(proxy: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "EvidenceTree InfoSeek corpus image preparation/1.0 "
                "(research use; Wikidata/Wikipedia fallback)"
            )
        }
    )
    if proxy:
        session.trust_env = False
        session.proxies.update({"http": proxy, "https": proxy})
    return session


def wikipedia_pageimages(
    *,
    session: requests.Session,
    entity_to_title: dict[str, str],
    max_attempts: int,
) -> dict[str, str]:
    title_to_entity = {
        title: entity_id
        for entity_id, title in entity_to_title.items()
        if title
    }
    found: dict[str, str] = {}
    titles = sorted(title_to_entity)
    endpoint = "https://en.wikipedia.org/w/api.php"
    for idx, group in enumerate(batched(titles, 50), 1):
        params = {
            "action": "query",
            "format": "json",
            "prop": "pageimages",
            "piprop": "original|thumbnail",
            "pithumbsize": "1024",
            "redirects": "1",
            "titles": "|".join(group),
        }
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(endpoint, params=params, timeout=(20, 120))
                if response.status_code == 429:
                    time.sleep(min(120, 10 * attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                normalized = {
                    row.get("to"): row.get("from")
                    for row in data.get("query", {}).get("normalized", [])
                    if row.get("from") and row.get("to")
                }
                redirects = {
                    row.get("to"): row.get("from")
                    for row in data.get("query", {}).get("redirects", [])
                    if row.get("from") and row.get("to")
                }
                for page in data.get("query", {}).get("pages", {}).values():
                    url = (
                        page.get("original", {}).get("source")
                        or page.get("thumbnail", {}).get("source")
                    )
                    if not url:
                        continue
                    resolved = page.get("title") or ""
                    candidates = [
                        resolved,
                        normalized.get(resolved, ""),
                        redirects.get(resolved, ""),
                    ]
                    entity_id = next(
                        (title_to_entity[c] for c in candidates if c in title_to_entity),
                        None,
                    )
                    if entity_id:
                        found[entity_id] = url
                break
            except Exception as exc:  # noqa: BLE001 - retry API batch.
                print(
                    f"Wikipedia pageimages failed batch={idx} "
                    f"attempt={attempt}/{max_attempts}: {exc}",
                    flush=True,
                )
                time.sleep(min(120, 5 * attempt))
    return found


def extract_image_urls_from_wikidata(
    *,
    corpus_path: Path,
    needed_entities: set[str],
    cache_path: Path,
    proxy: str,
    max_attempts: int,
) -> dict[str, dict[str, Any]]:
    """Fetch corpus entity image URLs without the full Wiki6M gzip.

    This is a practical fallback for the same local entity subset. It first asks
    Wikidata for P18 image claims and English Wikipedia sitelinks, then fills
    remaining entities from the Wikipedia pageimages API.
    """
    cache = load_url_cache(cache_path)
    missing = sorted(needed_entities - set(cache))
    if not missing:
        return cache

    corpus_titles = load_entity_titles(corpus_path)
    session = make_session(proxy)
    endpoint = "https://www.wikidata.org/w/api.php"
    pending_pageimage_titles: dict[str, str] = {}
    matched = 0
    print(
        f"Fetching Wikidata image records for {len(missing)} missing entities "
        f"({len(cache)} cached).",
        flush=True,
    )
    for batch_idx, group in enumerate(batched(missing, 50), 1):
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(group),
            "props": "claims|sitelinks|labels",
            "languages": "en",
            "sitefilter": "enwiki",
        }
        data: dict[str, Any] | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(endpoint, params=params, timeout=(20, 120))
                if response.status_code == 429:
                    time.sleep(min(120, 10 * attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:  # noqa: BLE001 - retry API batch.
                print(
                    f"Wikidata failed batch={batch_idx} "
                    f"attempt={attempt}/{max_attempts}: {exc}",
                    flush=True,
                )
                time.sleep(min(120, 5 * attempt))
        if not data:
            continue

        for entity_id, entity in data.get("entities", {}).items():
            title = (
                entity.get("sitelinks", {}).get("enwiki", {}).get("title")
                or corpus_titles.get(entity_id)
                or entity.get("labels", {}).get("en", {}).get("value")
                or ""
            )
            image_url = None
            p18_claims = entity.get("claims", {}).get("P18", [])
            for claim in p18_claims:
                value = (
                    claim.get("mainsnak", {})
                    .get("datavalue", {})
                    .get("value")
                )
                if isinstance(value, str) and value.strip():
                    image_url = commons_redirect_url(value.strip())
                    break
            cache[entity_id] = {
                "entity_id": entity_id,
                "title": title,
                "image_url": image_url,
                "summary": "",
                "source": "wikidata_p18" if image_url else "wikidata_no_p18",
            }
            if image_url:
                matched += 1
            elif title:
                pending_pageimage_titles[entity_id] = title

        if batch_idx % 5 == 0 or batch_idx == len(batched(missing, 50)):
            save_url_cache(cache_path, cache)
            print(
                f"wikidata batches={batch_idx}; cached={len(cache)}; "
                f"p18_urls={sum(1 for r in cache.values() if r.get('image_url'))}",
                flush=True,
            )

    still_without = {
        entity_id: title
        for entity_id, title in pending_pageimage_titles.items()
        if not cache.get(entity_id, {}).get("image_url")
    }
    if still_without:
        print(
            f"Fetching Wikipedia pageimages for {len(still_without)} entities "
            "without P18.",
            flush=True,
        )
        page_urls = wikipedia_pageimages(
            session=session,
            entity_to_title=still_without,
            max_attempts=max_attempts,
        )
        for entity_id, url in page_urls.items():
            cache.setdefault(entity_id, {"entity_id": entity_id})
            cache[entity_id]["image_url"] = url
            cache[entity_id]["source"] = "wikipedia_pageimages"
            matched += 1

    save_url_cache(cache_path, cache)
    print(
        f"Wikidata/Wikipedia image URL fetch done: cached={len(cache)}; "
        f"records_with_image_url={sum(1 for r in cache.values() if r.get('image_url'))}; "
        f"new_matches={matched}.",
        flush=True,
    )
    return cache


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


def download_one(
    *,
    session: requests.Session,
    image_url: str,
    path_stem: Path,
    min_sleep: float,
    max_sleep: float,
    max_attempts: int,
    thumbnail_width: int,
) -> Path | None:
    urls = commons_api_image_urls(
        session=session,
        image_url=image_url,
        thumbnail_width=thumbnail_width,
        max_attempts=max(1, max_attempts),
    )
    if not urls:
        if thumbnail_width > 0:
            return None
        urls = expanded_image_urls(image_url, thumbnail_width=thumbnail_width)
    for url in urls:
        for attempt in range(1, max_attempts + 1):
            try:
                response = session.get(url, timeout=(20, 120), stream=True)
                if response.status_code == 429 and session.proxies:
                    try:
                        direct = requests.get(
                            url,
                            headers=session.headers,
                            timeout=(20, 120),
                            stream=True,
                        )
                        if direct.ok:
                            response.close()
                            response = direct
                        else:
                            direct.close()
                    except Exception as exc:  # noqa: BLE001 - keep proxy 429 path.
                        print(f"direct fallback failed: {url}: {exc}", flush=True)
                if response.status_code == 429:
                    print(f"429 rate limited; skip for now: {url}", flush=True)
                    break
                if response.status_code == 404:
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
            except Exception as exc:  # noqa: BLE001 - try next attempt/url.
                print(f"download failed attempt={attempt}: {url}: {exc}", flush=True)
                time.sleep(min(120, 5 * attempt))
    return None


def download_images(
    *,
    url_rows: dict[str, dict[str, Any]],
    image_dir: Path,
    manifest_path: Path,
    workers: int,
    min_sleep: float,
    max_sleep: float,
    max_attempts: int,
    thumbnail_width: int,
    proxy: str,
    limit: int,
    shuffle_pending: bool,
) -> dict[str, str]:
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_url_cache(manifest_path)
    downloaded: dict[str, str] = {}
    for entity_id, row in manifest.items():
        path = Path(str(row.get("image_path") or ""))
        if path.exists() and valid_image(path):
            downloaded[entity_id] = str(path.resolve())

    candidates: list[tuple[str, str, str]] = []
    for entity_id, row in sorted(url_rows.items()):
        image_url = row.get("image_url")
        if not image_url:
            continue
        stem = safe_id(f"{entity_id}_{row.get('title') or entity_id}")
        existing = existing_for_stem(image_dir, stem)
        if existing:
            downloaded[entity_id] = str(existing.resolve())
            continue
        if entity_id in downloaded:
            continue
        candidates.append((entity_id, stem, image_url))

    if limit:
        candidates = candidates[:limit]
    if shuffle_pending:
        random.shuffle(candidates)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "EvidenceTree InfoSeek corpus image preparation/1.0 "
                "(research use; rate-limited Wikimedia download)"
            )
        }
    )
    if proxy:
        session.trust_env = False
        session.proxies.update({"http": proxy, "https": proxy})

    print(
        f"image candidates={len(candidates)}; existing={len(downloaded)}; "
        f"workers={workers}",
        flush=True,
    )

    def record(entity_id: str, path: str) -> None:
        downloaded[entity_id] = path
        manifest[entity_id] = {
            "entity_id": entity_id,
            "image_path": path,
            "image_url": url_rows.get(entity_id, {}).get("image_url"),
            "title": url_rows.get(entity_id, {}).get("title", ""),
        }

    if workers <= 1:
        for idx, (entity_id, stem, image_url) in enumerate(candidates, 1):
            path = download_one(
                session=session,
                image_url=image_url,
                path_stem=image_dir / stem,
                min_sleep=min_sleep,
                max_sleep=max_sleep,
                max_attempts=max_attempts,
                thumbnail_width=thumbnail_width,
            )
            if path:
                record(entity_id, str(path.resolve()))
            if idx % 50 == 0 or idx == len(candidates):
                save_url_cache(manifest_path, manifest)
                print(
                    f"processed={idx}/{len(candidates)}; "
                    f"downloaded_or_existing={len(downloaded)}",
                    flush=True,
                )
    else:
        def worker(item: tuple[str, str, str]) -> tuple[str, str | None]:
            entity_id, stem, image_url = item
            local = requests.Session()
            local.headers.update(session.headers)
            local.trust_env = session.trust_env
            local.proxies.update(session.proxies)
            path = download_one(
                session=local,
                image_url=image_url,
                path_stem=image_dir / stem,
                min_sleep=min_sleep,
                max_sleep=max_sleep,
                max_attempts=max_attempts,
                thumbnail_width=thumbnail_width,
            )
            return entity_id, str(path.resolve()) if path else None

        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(worker, item): item for item in candidates}
            for idx, future in enumerate(futures.as_completed(future_to_item), 1):
                entity_id, path = future.result()
                if path:
                    record(entity_id, path)
                if idx % 50 == 0 or idx == len(candidates):
                    save_url_cache(manifest_path, manifest)
                    print(
                        f"processed={idx}/{len(candidates)}; "
                        f"downloaded_or_existing={len(downloaded)}",
                        flush=True,
                    )
    save_url_cache(manifest_path, manifest)
    return downloaded


def attach_images_to_corpus(
    *,
    corpus_path: Path,
    downloaded: dict[str, str],
) -> tuple[int, int, int]:
    rows = load_jsonl(corpus_path)
    docs_with_images = 0
    entities_with_images: set[str] = set()
    for row in rows:
        entity_id = entity_from_doc_id(str(row["doc_id"]))
        image_path = downloaded.get(entity_id)
        row["image_path"] = image_path
        if image_path:
            row.setdefault("metadata", {})["corpus_image_entity_id"] = entity_id
            docs_with_images += 1
            entities_with_images.add(entity_id)
    write_jsonl(corpus_path, rows)
    return len(rows), docs_with_images, len(entities_with_images)


def rebuild_clip_indexes(
    *,
    data_dir: Path,
    model_name: str,
    device: str,
    batch_size: int,
) -> dict[str, int]:
    from evidencetree.actions.retrievers import ClipImageRetriever, CrossModalCLIPRetriever
    from evidencetree.eval.benchmarks import Document

    corpus_path = data_dir / "infoseek_corpus.jsonl"
    docs = [
        Document(
            doc_id=str(row["doc_id"]),
            text=row["text"],
            title=row.get("title", ""),
            image_path=row.get("image_path"),
        )
        for row in load_jsonl(corpus_path)
    ]

    for dirname in ("clip_index", "cross_modal_clip_index"):
        shutil.rmtree(data_dir / dirname, ignore_errors=True)

    ClipImageRetriever(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    ).build(docs).save(data_dir / "clip_index")
    cross = CrossModalCLIPRetriever(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    ).build(docs)
    cross.save(data_dir / "cross_modal_clip_index")
    return {
        "docs": len(docs),
        "image_docs": sum(1 for doc in docs if doc.image_path),
        "text_docs": sum(1 for doc in docs if doc.text),
        "cross_image_docs": cross.image_doc_count,
        "cross_text_docs": cross.text_doc_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--wiki6m-gz", type=Path, default=None)
    parser.add_argument("--wiki6m-url", default=DEFAULT_WIKI6M_URL)
    parser.add_argument(
        "--wiki6m-size",
        type=int,
        default=7408031599,
        help="Official Wiki6M gzip size in bytes; avoids flaky HEAD requests.",
    )
    parser.add_argument("--download-wiki6m", action="store_true")
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--download-chunk-mb", type=int, default=64)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument(
        "--url-source",
        choices=("auto", "wiki6m", "wikidata"),
        default="auto",
        help=(
            "Where to obtain corpus image URLs. 'wiki6m' scans the official "
            "gzip; 'wikidata' fetches P18/pageimages for the local entity "
            "subset; 'auto' uses Wiki6M if present, otherwise Wikidata."
        ),
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-sleep", type=float, default=0.4)
    parser.add_argument("--max-sleep", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--thumbnail-width", type=int, default=512)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--shuffle-pending", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500000)
    parser.add_argument("--model-name", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    source_root = args.source_root
    source_root.mkdir(parents=True, exist_ok=True)
    wiki6m_gz = args.wiki6m_gz or source_root / "Wiki6M_ver_1_0.jsonl.gz"
    image_dir = args.image_dir or source_root / "wiki_images"
    cache_path = data_dir / "raw" / "wiki6m_image_urls_for_corpus.jsonl"
    manifest_path = data_dir / "raw" / "wiki_image_manifest.jsonl"
    corpus_path = data_dir / "infoseek_corpus.jsonl"

    if args.download_wiki6m and not wiki6m_gz.exists():
        download_by_ranges(
            url=args.wiki6m_url,
            path=wiki6m_gz,
            chunk_mb=args.download_chunk_mb,
            workers=args.download_workers,
            retries=args.download_retries,
            size=args.wiki6m_size if args.wiki6m_size > 0 else None,
        )

    if args.url_source == "auto":
        url_source = "wiki6m" if wiki6m_gz.exists() else "wikidata"
    else:
        url_source = args.url_source

    if url_source == "wiki6m" and not wiki6m_gz.exists():
        raise FileNotFoundError(
            f"Wiki6M gzip not found: {wiki6m_gz}. "
            "Pass --download-wiki6m or download it first."
        )

    needed = load_needed_entities(corpus_path)
    print(f"needed_entities={len(needed)} from {corpus_path}", flush=True)
    print(f"url_source={url_source}", flush=True)
    if url_source == "wiki6m":
        url_rows = extract_image_urls(
            wiki6m_gz=wiki6m_gz,
            needed_entities=needed,
            cache_path=cache_path,
            progress_every=args.progress_every,
        )
    else:
        cache_path = data_dir / "raw" / "wikidata_image_urls_for_corpus.jsonl"
        url_rows = extract_image_urls_from_wikidata(
            corpus_path=corpus_path,
            needed_entities=needed,
            cache_path=cache_path,
            proxy=args.proxy,
            max_attempts=args.max_attempts,
        )
    with_urls = sum(1 for row in url_rows.values() if row.get("image_url"))
    print(f"url_records={len(url_rows)}; records_with_image_url={with_urls}", flush=True)

    downloaded = download_images(
        url_rows=url_rows,
        image_dir=image_dir,
        manifest_path=manifest_path,
        workers=args.workers,
        min_sleep=args.min_sleep,
        max_sleep=args.max_sleep,
        max_attempts=args.max_attempts,
        thumbnail_width=args.thumbnail_width,
        proxy=args.proxy,
        limit=args.limit_images,
        shuffle_pending=args.shuffle_pending,
    )
    total_docs, docs_with_images, entities_with_images = attach_images_to_corpus(
        corpus_path=corpus_path,
        downloaded=downloaded,
    )

    stats_path = data_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats.update(
        {
            "infoseek_wiki6m_gz": str(wiki6m_gz.resolve()),
            "corpus_image_dir": str(image_dir.resolve()),
            "corpus_entities": len(needed),
            "corpus_entities_with_image_url": with_urls,
            "corpus_entities_with_downloaded_images": entities_with_images,
            "corpus_docs": total_docs,
            "corpus_docs_with_images": docs_with_images,
        }
    )
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    if not args.skip_index:
        index_stats = rebuild_clip_indexes(
            data_dir=data_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(json.dumps({"index": index_stats}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
