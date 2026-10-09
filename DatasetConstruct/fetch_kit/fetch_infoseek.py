"""InfoSeek fetch kit — run on a machine WITH internet access, then copy the kit
directory to the construction server.

The construction server cannot reach Google Cloud Storage, HuggingFace or
Wikimedia, and could not hold the full sources anyway (OVEN alone is ~294 GB).
This kit selects the questions FIRST and then downloads only what those
questions need:

    annotations  InfoSeek val annotations + entity map          (GCS, ~100 MB)
    splits       2K evaluation + 20K seed questions (+ backups)  (no download)
    plan         which OVEN images those questions need: the list,
                 its breakdown, and the shard sizes on HuggingFace (tiny)
    oven         their OVEN query images, one shard at a time,
                 each shard deleted after extraction; stops as soon
                 as every wanted image is found                   (HF)
    wiki6m       the 60K-entity corpus from the official Wiki6M KB:
                 gold entities + same-category hard negatives +
                 random fill; keeps only those pages              (GCS, 7.4 GB)
    wiki-images  one thumbnail per corpus entity                  (Wikimedia)
    finalize     drop seeds whose image is missing, fill from the
                 backups, write the manifest
    all          every step above in order

Every step is resumable. The resulting kit is ~5-6 GB:

    <kit>/manifest.json               counts + sha256 of every non-image file
    <kit>/annotations/*.jsonl         raw InfoSeek val files
    <kit>/splits/{eval,seed}_ids.txt  final splits (+ split_report.json)
    <kit>/query_images/<image_id>.jpg
    <kit>/wiki/entities.jsonl.gz      selected Wiki6M pages
    <kit>/corpus_images/<entity>.jpg

Then:  tar -cf infoseek_kit.tar <kit>;  scp it to the server;  on the server
    python DatasetConstruct/data_prep/import_infoseek_kit.py --kit <kit>

Requirements on the local machine: Python >= 3.10, ``pip install requests
pillow huggingface_hub``, and a checkout of this repository (the kit imports
the split logic from ``src/`` so the selection is identical everywhere).

Usage:
    python DatasetConstruct/fetch_kit/fetch_infoseek.py all --kit ~/infoseek_kit
    python DatasetConstruct/fetch_kit/fetch_infoseek.py splits --kit ~/infoseek_kit --dry-run
    python DatasetConstruct/fetch_kit/fetch_infoseek.py plan --kit ~/infoseek_kit
    python DatasetConstruct/fetch_kit/fetch_infoseek.py gqa-scenegraphs --out ~/sceneGraphs.zip
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, unquote

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.data.prep import save_image, sha256_file  # noqa: E402
from evidencetree.data.splits import SplitSpec, read_ids, select_splits, stable_rank, write_ids  # noqa: E402

KIT_VERSION = 1
GCS = "https://storage.googleapis.com/gresearch/open-vision-language"
ANNOTATION_URLS = {
    "infoseek_val.jsonl": f"{GCS}/infoseek/infoseek_val.jsonl",
    "infoseek_val_withkb.jsonl": f"{GCS}/infoseek/infoseek_val_withkb.jsonl",
}
WIKI6M_URL = f"{GCS}/Wiki6M_ver_1_0.jsonl.gz"
OVEN_REPO = "ychenNLP/oven"
OVEN_MAPPING = "ovenid2impath.csv"
# InfoSeek val images were found in these shards on the previous build; the
# `oven` step reports anything it could not find so the list can be extended.
DEFAULT_OVEN_SHARDS = ("shard04.tar", "shard05.tar", "shard08.tar")
AIRCRAFT_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
)
GQA_SCENEGRAPHS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip"
USER_AGENT = "ETBench-Open-fetch-kit/1.0 (academic dataset construction; offline corpus prep)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Network primitives (module-level so tests can replace them)
# --------------------------------------------------------------------------- #
def http_download(url: str, dest: Path, retries: int = 8, timeout: int = 120) -> Path:
    """Resumable streaming download (HTTP Range)."""
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if offset and r.status_code == 206 else "wb"
                with part.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=4 << 20):
                        if chunk:
                            f.write(chunk)
            break
        except Exception as exc:  # noqa: BLE001 - retry with backoff
            if attempt == retries:
                raise
            log(f"  download interrupted ({exc}); retry {attempt}/{retries}")
            time.sleep(min(60, 5 * attempt))
    part.replace(dest)
    return dest


def http_get_bytes(url: str, timeout: int = 60) -> bytes | None:
    """GET a small resource; None on 404, raises on other errors."""
    import requests

    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def hf_download(repo_id: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=dest_dir,
    ))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_records(kit: Path) -> list[dict[str, Any]]:
    """InfoSeek val questions joined with their entity ids (split candidates)."""
    ann = kit / "annotations"
    kb = {r["data_id"]: r for r in read_jsonl(ann / "infoseek_val_withkb.jsonl")}
    records = []
    for r in read_jsonl(ann / "infoseek_val.jsonl"):
        k = kb.get(r["data_id"], {})
        records.append({
            "query_id": r["data_id"],
            "pool": "both",                      # InfoSeek val serves eval and seeds
            "image_id": r.get("image_id", ""),
            "data_split": r.get("data_split", ""),
            "entity_id": k.get("entity_id"),
            "entity_text": k.get("entity_text", ""),
        })
    return records


def oven_mapping(kit: Path) -> dict[str, str]:
    path = kit / "oven" / OVEN_MAPPING
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {row[0]: row[1] for row in csv.reader(f) if len(row) >= 2}


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def step_annotations(kit: Path) -> None:
    for name, url in ANNOTATION_URLS.items():
        dest = kit / "annotations" / name
        if dest.exists():
            log(f"annotations: {name} present")
            continue
        log(f"annotations: downloading {name}")
        http_download(url, dest)


def step_splits(
    kit: Path, eval_n: int, seed_n: int, backup: int,
    image_prefixes: list[str] | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    records = load_records(kit)
    mapping = oven_mapping(kit)
    if mapping:  # questions whose image OVEN does not have can never be used
        before = len(records)
        records = [r for r in records if r["image_id"] in mapping]
        log(f"splits: {before - len(records)} questions dropped (image not in OVEN mapping)")
    if image_prefixes:
        records = [r for r in records if any(r["image_id"].startswith(p) for p in image_prefixes)]
        log(f"splits: {len(records)} questions after image-prefix filter {image_prefixes}")
    spec = SplitSpec(
        eval_n=eval_n, seed_n=seed_n, eval_strata="data_split", group_by="image_id",
        entity_field="entity_id", entity_disjoint_strata=("val_unseen_entity",),
        seed_backup=backup,
    )
    result = select_splits(records, spec)
    rep = result.report
    log(
        f"splits: eval {rep['eval']['n']} {rep['eval']['per_stratum']} | seeds "
        f"{rep['seeds']} (+{rep['backup_seeds']} backup) | seed pool "
        f"{rep['seed_pool_after_exclusion']} | dropped {rep['seed_dropped']}"
    )
    if rep["seed_shortfall"]:
        log(f"splits: WARNING seed shortfall of {rep['seed_shortfall']}")
    if not dry_run:
        out = kit / "splits"
        write_ids(out / "eval_ids.txt", result.eval_ids)
        write_ids(out / "seed_ids_ranked.txt", result.seed_ids + result.backup_seed_ids)
        (out / "split_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def _wanted_images(kit: Path) -> dict[str, str]:
    """image_id -> query_id for eval + ranked seeds (backups included)."""
    records = {r["query_id"]: r for r in load_records(kit)}
    ids = read_ids(kit / "splits" / "eval_ids.txt") + read_ids(kit / "splits" / "seed_ids_ranked.txt")
    return {records[q]["image_id"]: q for q in ids if q in records and records[q]["image_id"]}


def _ensure_mapping(kit: Path) -> dict[str, str]:
    oven_dir = kit / "oven"
    if not (oven_dir / OVEN_MAPPING).exists():
        log("oven: downloading the OVEN id -> source mapping")
        hf_download(OVEN_REPO, OVEN_MAPPING, oven_dir)
    return oven_mapping(kit)


def _shard_indexes(kit: Path) -> dict[str, set[str]]:
    """Image ids found in each OVEN shard already scanned (oven/shard_index/)."""
    index_dir = kit / "oven" / "shard_index"
    return {p.name[: -len(".txt")]: set(read_ids(p)) for p in sorted(index_dir.glob("*.txt"))}


def hf_list_files(repo_id: str) -> list[tuple[str, int | None]]:
    """(path, size in bytes) of every file in a HuggingFace dataset repo."""
    from huggingface_hub import HfApi

    return [
        (item.path, getattr(item, "size", None))
        for item in HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        if hasattr(item, "size")
    ]


def step_plan(kit: Path, shards: list[str]) -> dict[str, Any]:
    """Which query images the selected questions need — before any big download.

    Every InfoSeek question names one OVEN image (``image_id``); the images to
    fetch are the distinct image ids of the evaluation, seed and backup
    questions chosen by ``splits``. OVEN publishes images only inside large tar
    shards, without a per-image index, so this also reports how the wanted ids
    break down by id prefix and by source dataset (from OVEN's id -> source
    mapping), which already-scanned shards contain them, and the size of every
    shard on HuggingFace. Writes oven/wanted_images.tsv and oven/plan.json.
    """
    mapping = _ensure_mapping(kit)
    wanted = _wanted_images(kit)
    have = {p.stem for p in (kit / "query_images").glob("*.jpg")}
    located: dict[str, str] = {}
    for shard, members in _shard_indexes(kit).items():
        for image_id in members & set(wanted):
            located.setdefault(image_id, shard)
    ids = sorted(wanted)
    by_prefix = Counter(i[:7] for i in ids)
    by_source = Counter((mapping.get(i) or "not-in-mapping/").split("/", 1)[0] for i in ids)

    out = kit / "oven"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "wanted_images.tsv").open("w", encoding="utf-8") as f:
        f.write("image_id\tsource_path\tquery_id\tshard\tpresent\n")
        for i in ids:
            f.write(f"{i}\t{mapping.get(i, '')}\t{wanted[i]}\t{located.get(i, '')}\t{int(i in have)}\n")

    try:
        tars = sorted((p, s) for p, s in hf_list_files(OVEN_REPO) if p.endswith(".tar"))
    except Exception as exc:  # noqa: BLE001 - the plan is still useful offline
        log(f"plan: could not list {OVEN_REPO} on HuggingFace ({type(exc).__name__})")
        tars = []
    n_eval = len(read_ids(kit / "splits" / "eval_ids.txt"))
    n_ranked = len(read_ids(kit / "splits" / "seed_ids_ranked.txt"))
    aircraft = by_source.get("aircraft", 0)
    planned_bytes = sum(s for p, s in tars if s and p.rsplit("/", 1)[-1] in shards)
    plan = {
        "questions": {"eval": n_eval, "seeds_with_backups": n_ranked},
        "distinct_images": len(ids),
        "already_present": len(have & set(ids)),
        "by_id_prefix": dict(sorted(by_prefix.items())),
        "by_source_dataset": dict(by_source.most_common()),
        "located_in_scanned_shards": dict(Counter(located.values())),
        "hf_shards": [{"file": p, "gb": round(s / 1e9, 1) if s else None,
                       "planned": p.rsplit("/", 1)[-1] in shards} for p, s in tars],
        "planned_shards": shards,
        "planned_download_gb": round(planned_bytes / 1e9, 1) if planned_bytes else None,
        "aircraft_archive_needed": aircraft > 0,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    log(f"plan: {n_eval} eval + {n_ranked} seed/backup questions -> {len(ids)} distinct "
        f"images ({plan['already_present']} already present)")
    log(f"plan: by id prefix {plan['by_id_prefix']}")
    log(f"plan: by source dataset {plan['by_source_dataset']}")
    if by_source.get("not-in-mapping"):
        log(f"plan: WARNING {by_source['not-in-mapping']} ids are not in {OVEN_MAPPING}")
    if located:
        log(f"plan: found in scanned shards {plan['located_in_scanned_shards']}")
    for row in plan["hf_shards"]:
        gb = row["gb"] if row["gb"] is not None else "?"
        log(f"plan:   {row['file']:<22} {gb:>6} GB" + ("   <- planned" if row["planned"] else ""))
    if aircraft:
        log(f"plan: {aircraft} images come from FGVC-Aircraft (separate ~2.8 GB archive)")
    log(f"plan: list written to {out / 'wanted_images.tsv'}")
    return plan


def step_oven(kit: Path, shards: list[str], max_side: int = 768) -> dict[str, int]:
    """Extract the wanted query images, one OVEN shard at a time.

    Each scanned shard's full member list is saved to oven/shard_index/, so a
    later run skips shards known to hold none of the pending images, fetches
    an already processed shard again only when the selection grew into it,
    and tries shards known to hold pending images first. Stops once nothing
    is pending, without downloading the remaining shards.
    """
    out_dir = kit / "query_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    oven_dir = kit / "oven"
    mapping = _ensure_mapping(kit)
    wanted = set(_wanted_images(kit))
    have = {p.stem for p in out_dir.glob("*.jpg")}
    pending = wanted - have
    log(f"oven: {len(wanted)} query images wanted, {len(pending)} pending")

    aircraft = {i: mapping[i] for i in pending if mapping.get(i, "").startswith("aircraft/")}
    if aircraft:
        _extract_aircraft(kit, aircraft, out_dir, max_side)
        pending -= {p.stem for p in out_dir.glob("*.jpg")}

    state = oven_dir / "state"
    state.mkdir(parents=True, exist_ok=True)
    index_dir = oven_dir / "shard_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    indexes = _shard_indexes(kit)
    # Shards known to hold pending images first, then unscanned ones, in the given order.
    order = sorted(shards, key=lambda s: (0 if pending & indexes.get(s, set())
                                          else 1 if s not in indexes else 2, shards.index(s)))
    for shard in order:
        if not pending:
            break
        if shard in indexes and not pending & indexes[shard]:
            continue  # scanned before; holds none of the pending images
        if (state / f"{shard}.complete").exists() and shard not in indexes:
            continue  # processed by an older kit version that kept no index
        log(f"oven: {shard} — downloading ({len(pending)} images still pending)")
        archive = hf_download(OVEN_REPO, shard, oven_dir / "downloads")
        found, members = 0, []
        with tarfile.open(archive, mode="r|*") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                image_id = Path(member.name).stem
                members.append(image_id)
                if image_id not in pending:
                    continue
                src = bundle.extractfile(member)
                if src is None:
                    continue
                try:
                    save_image(src.read(), out_dir / f"{image_id}.jpg", max_side=max_side)
                except Exception as exc:  # noqa: BLE001 - corrupt member
                    log(f"  unreadable image {image_id}: {exc}")
                    continue
                pending.discard(image_id)
                found += 1
        Path(archive).unlink(missing_ok=True)
        write_ids(index_dir / f"{shard}.txt", members)
        (state / f"{shard}.complete").touch()
        log(f"oven: {shard} done — {len(members)} images inside, extracted {found}, "
            f"archive removed, {len(pending)} still pending")
    missing = sorted(pending)
    (oven_dir / "missing_image_ids.txt").write_text("".join(f"{m}\n" for m in missing))
    log(f"oven: {len(wanted) - len(missing)}/{len(wanted)} query images present; "
        f"{len(missing)} missing (see oven/missing_image_ids.txt)")
    if missing:
        log(f"oven: missing by id prefix {dict(Counter(m[:7] for m in missing))} — `plan` lists "
            "the shards; add some with --oven-shards, or let `finalize` replace those seeds")
    return {"wanted": len(wanted), "missing": len(missing)}


def _extract_aircraft(kit: Path, targets: dict[str, str], out_dir: Path, max_side: int) -> None:
    archive = kit / "oven" / "downloads" / "fgvc-aircraft-2013b.tar.gz"
    log(f"oven: {len(targets)} images come from FGVC-Aircraft — fetching its archive")
    http_download(AIRCRAFT_URL, archive)
    by_source = {src.removeprefix("aircraft/"): i for i, src in targets.items()}
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            image_id = by_source.get(member.name)
            if image_id is None or not member.isfile():
                continue
            src = bundle.extractfile(member)
            if src is not None:
                save_image(src.read(), out_dir / f"{image_id}.jpg", max_side=max_side)
    archive.unlink(missing_ok=True)


# ---- Wiki6M corpus selection ------------------------------------------------ #
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TITLE_STOP = {
    "the", "of", "and", "a", "an", "in", "on", "de", "la", "le", "for", "to",
    "at", "by", "st", "saint", "new", "house", "river", "lake",
}
_COPULA_RE = re.compile(r"\b(?:is|was|are|were)\s+(?:a|an|the|one of the)\s+([^.;:()]+)", re.I)
_HEAD_STOP = {"in", "of", "that", "which", "located", "from", "founded", "built",
              "based", "native", "found", "known", "with", "on", "by", "for", "at"}
_SPECIES_RE = re.compile(
    r"\b(?:species|genus|family|type|kind|breed|variety|cultivar)\s+of\s+([^.;:()]+)", re.I
)


def title_tokens(title: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(title.lower()) if t not in _TITLE_STOP and len(t) > 2}


def _head_noun(phrase: str) -> str | None:
    """Last word of a noun phrase, read up to the first preposition/relative."""
    words = []
    for w in _TOKEN_RE.findall(phrase.lower()):
        if w in _HEAD_STOP:
            break
        words.append(w)
    return words[-1] if words else None


def category_head(text: str) -> str | None:
    """Coarse category of an entity from its first sentence: 'X is a species of
    flowering plant in ...' -> 'plant'; 'X is a museum in Canton' -> 'museum'."""
    first = str(text)[:400]
    m = _SPECIES_RE.search(first)
    if m:
        return _head_noun(m.group(1))
    m = _COPULA_RE.search(first)
    return _head_noun(m.group(1)) if m else None


def _content(row: dict[str, Any]) -> str:
    for key in ("wikipedia_content", "content", "text", "wikipedia_summary"):
        if row.get(key):
            return str(row[key])
    return ""


def step_wiki6m(
    kit: Path, corpus_entities: int = 60000, hard_per_gold: int = 2, seed: int = 0,
    wiki6m_path: Path | None = None,
) -> dict[str, Any]:
    src = wiki6m_path or kit / "raw" / "Wiki6M_ver_1_0.jsonl.gz"
    if not src.exists():
        log("wiki6m: downloading Wiki6M_ver_1_0.jsonl.gz (7.4 GB, resumable)")
        http_download(WIKI6M_URL, src)
    records = {r["query_id"]: r for r in load_records(kit)}
    ids = read_ids(kit / "splits" / "eval_ids.txt") + read_ids(kit / "splits" / "seed_ids_ranked.txt")
    gold_ids = {records[q]["entity_id"] for q in ids if q in records and records[q]["entity_id"]}
    gold_tokens = set()
    for q in ids:
        if q in records:
            gold_tokens |= title_tokens(records[q].get("entity_text") or "")
    log(f"wiki6m: {len(gold_ids)} gold entities; pass 1 (candidates)")

    rng = random.Random(seed)
    k_tok, k_head = 20, 15
    tok_res: dict[str, list[str]] = defaultdict(list)
    tok_seen: Counter[str] = Counter()
    head_res: dict[str, list[str]] = defaultdict(list)
    head_seen: Counter[str] = Counter()
    random_res: list[str] = []
    gold_rows: dict[str, dict[str, Any]] = {}
    n = 0

    def reservoir(res: list[str], seen: int, item: str, k: int) -> None:
        if len(res) < k:
            res.append(item)
        else:
            j = rng.randrange(seen)
            if j < k:
                res[j] = item

    for row in read_jsonl(src):
        eid = str(row.get("wikidata_id") or "")
        if not eid:
            continue
        n += 1
        if n == 1:
            log(f"wiki6m: record fields = {sorted(row)}")
        if eid in gold_ids:
            gold_rows[eid] = row
            continue
        title = str(row.get("wikipedia_title") or "")
        for tok in title_tokens(title) & gold_tokens:
            tok_seen[tok] += 1
            reservoir(tok_res[tok], tok_seen[tok], eid, k_tok)
        head = category_head(row.get("wikipedia_summary") or _content(row))
        if head:
            head_seen[head] += 1
            reservoir(head_res[head], head_seen[head], eid, k_head)
        reservoir(random_res, n, eid, corpus_entities)
        if n % 500_000 == 0:
            log(f"  pass 1: {n:,} records scanned")
    missing_gold = gold_ids - set(gold_rows)
    log(f"wiki6m: pass 1 done — {n:,} records, {len(gold_rows)} gold found, "
        f"{len(missing_gold)} gold missing from Wiki6M")

    chosen: dict[str, str] = {}  # entity id -> role
    for eid in sorted(gold_rows, key=lambda e: stable_rank(e, f"hard:{seed}")):
        row = gold_rows[eid]
        cands: list[str] = []
        for tok in sorted(title_tokens(str(row.get("wikipedia_title") or ""))):
            cands += tok_res.get(tok, [])
        head = category_head(row.get("wikipedia_summary") or _content(row))
        if head:
            cands += head_res.get(head, [])
        picked = 0
        for c in sorted(set(cands), key=lambda e: stable_rank(e, f"{eid}:{seed}")):
            if picked >= hard_per_gold:
                break
            if c in gold_rows or c in chosen:
                continue
            chosen[c] = "hard_negative"
            picked += 1
    budget = max(0, corpus_entities - len(gold_rows) - len(chosen))
    for eid in sorted(random_res, key=lambda e: stable_rank(e, f"fill:{seed}")):
        if budget <= 0:
            break
        if eid not in chosen and eid not in gold_rows:
            chosen[eid] = "random"
            budget -= 1

    log(f"wiki6m: pass 2 (collect {len(chosen)} non-gold pages)")
    out = kit / "wiki" / "entities.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    roles = Counter()
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        def emit(row: dict[str, Any], role: str) -> None:
            roles[role] += 1
            f.write(json.dumps({
                "entity_id": str(row["wikidata_id"]),
                "title": row.get("wikipedia_title") or "",
                "content": _content(row),
                "image_url": row.get("wikipedia_image_url"),
                "role": role,
                "category": category_head(row.get("wikipedia_summary") or _content(row)),
            }, ensure_ascii=False) + "\n")

        for eid in sorted(gold_rows):
            emit(gold_rows[eid], "gold")
        for row in read_jsonl(src):
            eid = str(row.get("wikidata_id") or "")
            if eid in chosen:
                emit(row, chosen[eid])
    tmp.replace(out)
    report = {"scanned": n, "gold_found": len(gold_rows), "gold_missing": len(missing_gold),
              "roles": dict(roles)}
    (kit / "wiki" / "selection_report.json").write_text(json.dumps(report, indent=2))
    log(f"wiki6m: wrote {sum(roles.values())} entities {dict(roles)} -> {out}")
    return report


def commons_thumbnail_url(image_url: str, width: int) -> str:
    filename = unquote(image_url.split("?", 1)[0].rsplit("/", 1)[-1])
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(filename)}?width={width}"


def step_wiki_images(
    kit: Path, width: int = 512, workers: int = 2, pause: float = 0.5,
) -> dict[str, int]:
    from concurrent.futures import ThreadPoolExecutor

    out_dir = kit / "corpus_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in out_dir.glob("*.jpg")}
    todo = [
        (e["entity_id"], e["image_url"])
        for e in read_jsonl(kit / "wiki" / "entities.jsonl.gz")
        if e.get("image_url") and e["entity_id"] not in have
    ]
    log(f"wiki-images: {len(have)} present, {len(todo)} to download")
    failures: list[dict[str, str]] = []

    def fetch(item: tuple[str, str]) -> None:
        eid, url = item
        for attempt in range(3):
            try:
                data = http_get_bytes(commons_thumbnail_url(url, width)) or http_get_bytes(url)
                if data:
                    save_image(data, out_dir / f"{eid}.jpg", max_side=width)
                else:
                    failures.append({"entity_id": eid, "error": "404"})
                break
            except Exception as exc:  # noqa: BLE001 - back off on throttling, else record
                if "429" in str(exc) and attempt < 2:
                    time.sleep(30 * (attempt + 1))
                    continue
                failures.append({"entity_id": eid, "error": str(exc)[:200]})
                break
        time.sleep(pause)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, _ in enumerate(pool.map(fetch, todo), 1):
            if i % 2000 == 0:
                log(f"  wiki-images: {i}/{len(todo)} ({len(failures)} failed)")
    logs = kit / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "wiki_image_failures.jsonl").open("a", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row) + "\n")
    n_ok = len(list(out_dir.glob("*.jpg")))
    log(f"wiki-images: {n_ok} corpus images present, {len(failures)} failed this run")
    return {"present": n_ok, "failed": len(failures)}


def step_finalize(kit: Path, seed_n: int) -> dict[str, Any]:
    records = {r["query_id"]: r for r in load_records(kit)}
    have = {p.stem for p in (kit / "query_images").glob("*.jpg")}

    def has_image(q: str) -> bool:
        return q in records and records[q]["image_id"] in have

    ranked = read_ids(kit / "splits" / "seed_ids_ranked.txt")
    seeds = [q for q in ranked if has_image(q)][:seed_n]
    evals = read_ids(kit / "splits" / "eval_ids.txt")
    evals_ok = [q for q in evals if has_image(q)]
    write_ids(kit / "splits" / "seed_ids.txt", seeds)
    write_ids(kit / "splits" / "eval_ids_final.txt", evals_ok)

    files = {}
    for p in sorted(kit.rglob("*")):
        rel = p.relative_to(kit).as_posix()
        if not p.is_file() or rel.startswith(("raw/", "oven/", "logs/")) or rel.endswith(".jpg") \
                or rel == "manifest.json":
            continue
        files[rel] = sha256_file(p)
    images = {
        d: sorted(f"{p.name}\t{p.stat().st_size}" for p in (kit / d).glob("*.jpg"))
        for d in ("query_images", "corpus_images")
    }
    for d, lines in images.items():
        (kit / f"{d}.tsv").write_text("".join(f"{line}\n" for line in lines))
        files[f"{d}.tsv"] = sha256_file(kit / f"{d}.tsv")
    manifest = {
        "kit_version": KIT_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {
            "seeds": len(seeds), "seed_target": seed_n, "eval": len(evals_ok),
            "eval_without_image": len(evals) - len(evals_ok),
            "query_images": len(images["query_images"]),
            "corpus_images": len(images["corpus_images"]),
        },
        "sha256": files,
    }
    (kit / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"finalize: seeds {len(seeds)}/{seed_n}, eval {len(evals_ok)}/{len(evals)}; manifest written.")
    log(f"Next: tar -cf infoseek_kit.tar --exclude={kit.name}/raw --exclude={kit.name}/oven "
        f"-C {kit.parent} {kit.name}   (raw/ and oven/ are local caches, not needed)")
    if len(seeds) < seed_n:
        log("finalize: WARNING — not enough seeds with images; raise --backup or add OVEN shards")
    return manifest


def step_gqa_scenegraphs(out: Path) -> None:
    log(f"gqa-scenegraphs: downloading {GQA_SCENEGRAPHS_URL}")
    http_download(GQA_SCENEGRAPHS_URL, out)
    log(f"gqa-scenegraphs: {out} ({out.stat().st_size / 1e6:.0f} MB) — copy it to the server")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["annotations", "splits", "plan", "oven", "wiki6m",
                                    "wiki-images", "finalize", "all", "gqa-scenegraphs"])
    p.add_argument("--kit", type=Path, default=Path("infoseek_kit"))
    p.add_argument("--eval-n", type=int, default=2000)
    p.add_argument("--seed-n", type=int, default=20000)
    p.add_argument("--backup", type=int, default=2000, help="extra ranked seeds replacing image failures")
    p.add_argument("--image-prefixes", default="", help="comma list, e.g. oven_04,oven_05 (fewer shards)")
    p.add_argument("--oven-shards", default=",".join(DEFAULT_OVEN_SHARDS))
    p.add_argument("--query-image-side", type=int, default=768)
    p.add_argument("--corpus-entities", type=int, default=60000)
    p.add_argument("--hard-per-gold", type=int, default=2)
    p.add_argument("--wiki6m", type=Path, default=None, help="already-downloaded Wiki6M gzip")
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--workers", type=int, default=2, help="wiki-images parallel downloads (be polite)")
    p.add_argument("--dry-run", action="store_true", help="splits: report only")
    p.add_argument("--out", type=Path, default=Path("sceneGraphs.zip"))
    args = p.parse_args(argv)

    kit = args.kit.expanduser().resolve()
    if args.step == "gqa-scenegraphs":
        step_gqa_scenegraphs(args.out.expanduser())
        return 0
    kit.mkdir(parents=True, exist_ok=True)
    prefixes = [x for x in args.image_prefixes.split(",") if x]
    steps = ["annotations", "splits", "plan", "oven", "wiki6m", "wiki-images", "finalize"] \
        if args.step == "all" else [args.step]
    for step in steps:
        if step == "annotations":
            step_annotations(kit)
        elif step == "splits":
            step_splits(kit, args.eval_n, args.seed_n, args.backup, prefixes, args.dry_run)
        elif step == "plan":
            step_plan(kit, [s for s in args.oven_shards.split(",") if s])
        elif step == "oven":
            step_oven(kit, [s for s in args.oven_shards.split(",") if s], args.query_image_side)
        elif step == "wiki6m":
            step_wiki6m(kit, args.corpus_entities, args.hard_per_gold, wiki6m_path=args.wiki6m)
        elif step == "wiki-images":
            step_wiki_images(kit, args.image_width, args.workers)
        elif step == "finalize":
            step_finalize(kit, args.seed_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
