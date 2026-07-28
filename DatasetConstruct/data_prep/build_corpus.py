"""Build the local retrieval corpus for InfoSeek (one-time offline data prep).

The validation split covers ~1.8K unique Wikidata entities
(raw/infoseek_val_withkb.jsonl). Steps:
    1. Wikidata API, batched (50 per request): entity -> English Wikipedia title
       (sitelinks/enwiki)
    2. Wikipedia API, page by page: full plaintext, resumable via
       raw/wiki_pages.jsonl
    3. split into ~1200-character paragraph chunks ->
       data/corpus/infoseek/infoseek_corpus.jsonl, one
       {"doc_id": "Q123_p0", "title", "text"} per line

This is dataset preparation: the corpus is fetched once, then frozen and
releasable alongside the dataset. It does not violate the offline-retrieval
constraint, since retrieval at run time only ever hits the local index.

Usage:
    python DatasetConstruct/build_corpus.py            # all entities
    python DatasetConstruct/build_corpus.py --limit 50 # trial run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

import sys
from pathlib import Path

_DC = Path(__file__).resolve().parents[1]   # DatasetConstruct/ (common.py, which
if str(_DC) not in sys.path:                # also puts src/ on the path)
    sys.path.insert(0, str(_DC))

from common import append_jsonl, read_jsonl, resolve, write_jsonl

from evidencetree.utils import get_logger

log = get_logger("dataset.corpus")

_HEADERS = {"User-Agent": "EvidenceTree-research/0.1 (academic project; offline corpus prep)"}
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
CHUNK_CHARS = 1200


def unique_entities(withkb_path: Path) -> dict[str, str]:
    """entity_id -> entity_text (fallback title)."""
    out: dict[str, str] = {}
    for row in read_jsonl(withkb_path):
        if row.get("entity_id"):
            out.setdefault(row["entity_id"], row.get("entity_text", ""))
    return out


def fetch_enwiki_titles(
    session, entity_ids: list[str], cache_path: Path | None = None
) -> dict[str, str]:
    """Batch-resolve Wikidata ids to English Wikipedia titles (50 per call).

    Results are cached to ``cache_path`` so a rerun never refetches; 429s get
    long backoff (Wikimedia throttles aggressively).
    """
    titles: dict[str, str] = {}
    if cache_path and cache_path.exists():
        titles = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [q for q in entity_ids if q not in titles]
    for i in range(0, len(todo), 50):
        batch = todo[i: i + 50]
        for attempt in range(5):
            try:
                resp = session.get(
                    _WIKIDATA_API,
                    params={
                        "action": "wbgetentities", "ids": "|".join(batch),
                        "props": "sitelinks", "sitefilter": "enwiki", "format": "json",
                    },
                    headers=_HEADERS, timeout=30,
                )
                resp.raise_for_status()
                for qid, ent in resp.json().get("entities", {}).items():
                    title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")
                    if title:
                        titles[qid] = title
                break
            except Exception as e:  # noqa: BLE001 - includes 429; back off hard
                wait = 15.0 * (attempt + 1)
                log.warning("  wikidata batch %d attempt %d: %s — backoff %.0fs",
                            i // 50, attempt + 1, e, wait)
                time.sleep(wait)
        if cache_path:
            cache_path.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
        log.info("  wikidata titles: %d/%d", min(i + 50, len(todo)), len(todo))
        time.sleep(1.0)
    return titles


def fetch_page_text(session, title: str) -> str:
    resp = session.get(
        _WIKIPEDIA_API,
        params={
            "action": "query", "prop": "extracts", "explaintext": 1,
            "redirects": 1, "format": "json", "titles": title,
        },
        headers=_HEADERS, timeout=30,
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "") or ""
    return ""


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Greedy paragraph packing into ~max_chars chunks."""
    chunks, current = [], ""
    for para in (p.strip() for p in text.split("\n") if p.strip()):
        if len(current) + len(para) + 1 > max_chars and current:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


def run(raw_dir: Path, out_path: Path, limit: int | None = None) -> Path:
    entities = unique_entities(raw_dir / "infoseek_val_withkb.jsonl")
    qids = sorted(entities)
    if limit:
        qids = qids[:limit]
    log.info("Entities to cover: %d", len(qids))

    pages_path = raw_dir / "wiki_pages.jsonl"  # fetch cache (resumable)
    # Only pages with non-empty text count as done; empty pages (usually 429
    # rate limiting) are re-fetched on the next run.
    done = (
        {row["entity_id"] for row in read_jsonl(pages_path) if row.get("text")}
        if pages_path.exists()
        else set()
    )
    todo = [q for q in qids if q not in done]
    log.info("Already fetched: %d | to fetch: %d", len(done), len(todo))

    session = requests.Session()
    if todo:
        titles = fetch_enwiki_titles(session, todo, cache_path=raw_dir / "wiki_titles.json")
        for n, qid in enumerate(todo, 1):
            title = titles.get(qid) or entities[qid]
            text = ""
            for attempt in range(4):
                try:
                    text = fetch_page_text(session, title)
                    break
                except requests.HTTPError as e:  # 429 -> long backoff
                    wait = 10.0 * (attempt + 1) if e.response is not None and \
                        e.response.status_code == 429 else 2.0 * (attempt + 1)
                    log.warning("  %s (%s) attempt %d: %s — backoff %.0fs",
                                qid, title, attempt + 1, e, wait)
                    time.sleep(wait)
                except Exception as e:  # noqa: BLE001 - retry then record empty
                    log.warning("  %s (%s) attempt %d failed: %s", qid, title, attempt + 1, e)
                    time.sleep(2.0 * (attempt + 1))
            append_jsonl(pages_path, [{"entity_id": qid, "title": title, "text": text}])
            if n % 100 == 0:
                log.info("  wikipedia pages: %d/%d", n, len(todo))
            time.sleep(0.8)  # be gentle: steady and slow beats triggering 429 backoff

    # pages -> chunked corpus
    rows = []
    n_empty = 0
    for page in read_jsonl(pages_path):
        if page["entity_id"] not in set(qids):
            continue
        if not page["text"]:
            n_empty += 1
            continue
        for i, chunk in enumerate(chunk_text(page["text"])):
            rows.append(
                {"doc_id": f"{page['entity_id']}_p{i}", "title": page["title"], "text": chunk}
            )
    write_jsonl(out_path, rows)
    log.info(
        "Corpus written: %d chunks from %d entities (%d empty pages) -> %s",
        len(rows), len(qids) - n_empty, n_empty, out_path,
    )
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build the InfoSeek Wikipedia corpus.")
    p.add_argument("--raw-dir", default="data/corpus/infoseek/raw")
    p.add_argument("--out", default="data/corpus/infoseek/infoseek_corpus.jsonl")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N entities (trial run).")
    args = p.parse_args(argv)
    run(resolve(args.raw_dir), resolve(args.out), limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
