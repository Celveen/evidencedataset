"""Fetch a few entity images (Wikidata P18) for LOCAL testing without OVEN.

For local smoke tests when the OVEN images live only on the server: pulls each
query's Wikidata entity representative image (P18) from Wikimedia Commons and
writes them next to a mini queries JSONL with image_path filled in. Same entity
as the InfoSeek query image, so it exercises "VLM sees the image -> identifies
the entity -> no hallucination". NOT for the real dataset (use real OVEN images
via prepare_infoseek.py --images-dir on the server).

Usage:
    python DatasetConstruct/fetch_test_images.py --n 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests

from common import read_jsonl, resolve, write_jsonl

from evidencetree.utils import get_logger

log = get_logger("dataset.fetch_test_images")
_HEADERS = {"User-Agent": "EvidenceTree-research/0.1 (academic; local test image fetch)"}


def p18_filename(session, entity_id: str) -> str | None:
    r = session.get(
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbgetclaims", "entity": entity_id, "property": "P18",
                "format": "json"},
        headers=_HEADERS, timeout=30,
    )
    r.raise_for_status()
    claims = r.json().get("claims", {}).get("P18", [])
    if not claims:
        return None
    return claims[0]["mainsnak"]["datavalue"]["value"]


def download_commons(session, filename: str, dest: Path) -> bool:
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + filename.replace(" ", "_")
    r = session.get(url, headers=_HEADERS, timeout=60)
    if r.status_code != 200 or not r.content:
        return False
    dest.write_bytes(r.content)
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fetch P18 images for local testing.")
    p.add_argument("--queries", default="data/corpus/infoseek/infoseek_queries.jsonl")
    p.add_argument("--out-dir", default="data/corpus/infoseek_localtest")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args(argv)

    out_dir = resolve(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    rows, n = [], 0
    for o in read_jsonl(resolve(args.queries)):
        if n >= args.n:
            break
        eid = o["metadata"].get("entity_id")
        if not eid:
            continue
        try:
            fname = p18_filename(session, eid)
            if not fname:
                log.info("  %s (%s): no P18 image, skip", eid, o["metadata"].get("entity_text"))
                continue
            dest = img_dir / f"{o['metadata']['image_id']}{Path(fname).suffix.lower()}"
            if download_commons(session, fname, dest):
                o = dict(o); o["image_path"] = str(dest)
                rows.append(o)
                n += 1
                log.info("  %s -> %s", o["metadata"]["entity_text"], dest.name)
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            log.warning("  %s failed: %s", eid, e)

    out_q = out_dir / "infoseek_queries.jsonl"
    write_jsonl(out_q, rows)
    log.info("Wrote %d queries with images -> %s", len(rows), out_q)
    log.info("Test it: point a run at data_dir=%s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
