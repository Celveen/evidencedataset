"""Data preparation: InfoSeek fetch kit + import, ScienceQA, GQA — all offline,
with the network calls of the fetch kit replaced by local fakes."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_DC = ROOT / "DatasetConstruct"
for p in (_DC, _DC / "data_prep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evidencetree.data.prep import chunk_text  # noqa: E402
from evidencetree.eval.benchmarks import load_benchmark  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(color=(200, 30, 30), size=(40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_chunk_text_packs_paragraphs_and_splits_long_ones():
    text = "Short para.\n" + "Sentence one is here. " * 80 + "\nTail."
    chunks = chunk_text(text, max_chars=300)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("Short para.") and chunks[-1].endswith("Tail.")
    assert "".join(chunks).count("Sentence one") == 80


# --------------------------------------------------------------------------- #
# InfoSeek fetch kit -> import
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_sources(tmp_path):
    src = tmp_path / "sources"
    val, kb = [], []
    for i in range(30):                       # 30 images x 2 questions
        for j in range(2):
            qid = f"infoseek_val_{i:03d}_{j}"
            val.append({
                "data_id": qid, "image_id": f"oven_04{i:06d}",
                "question": f"What is the weight of this bird {j}?",
                "answer_eval": [{"wikidata": 100.0 + i, "range": [90.0 + i, 110.0 + i]}] if j
                else [f"Answer {i}"],
                "data_split": "val_unseen_entity" if i % 2 else "val_unseen_question",
            })
            kb.append({"data_id": qid, "entity_id": f"Q{i % 10}", "entity_text": f"Dove {i % 10}"})
    _jsonl(src / "infoseek_val.jsonl", val)
    _jsonl(src / "infoseek_val_withkb.jsonl", kb)
    wiki = []
    for e in range(10):
        wiki.append({"wikidata_id": f"Q{e}", "wikipedia_title": f"Dove {e}",
                     "wikipedia_summary": f"Dove {e} is a species of bird in the family Columbidae.",
                     "wikipedia_content": f"Dove {e} is a species of bird.\n" + "It eats seeds. " * 150,
                     "wikipedia_image_url": f"https://upload.wikimedia.org/x/Dove_{e}.jpg"})
    for e in range(10, 60):
        kind = "bird" if e < 30 else "museum"
        wiki.append({"wikidata_id": f"Q{e}", "wikipedia_title": f"{'Dove' if e < 15 else 'Thing'} {e}",
                     "wikipedia_summary": f"Thing {e} is a {kind} in Europe.",
                     "wikipedia_content": f"Thing {e} is a {kind} in Europe.",
                     "wikipedia_image_url": None if e % 3 else f"https://upload.wikimedia.org/x/T{e}.jpg"})
    _jsonl(src / "Wiki6M_ver_1_0.jsonl.gz", wiki)
    mapping = src / "ovenid2impath.csv"
    mapping.write_text("".join(f"oven_04{i:06d},inat/img{i}.jpg\n" for i in range(30)))
    shard = src / "shard04.tar"
    with tarfile.open(shard, "w") as tar:
        for i in range(29):                   # image 29 is missing from every shard
            data = _png((i * 8, 100, 50))
            info = tarfile.TarInfo(f"images/oven_04{i:06d}.jpg")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return src


@pytest.fixture
def kit_module(fake_sources, monkeypatch):
    kit_mod = _load("fetch_infoseek_t", _DC / "fetch_kit" / "fetch_infoseek.py")

    def fake_http_download(url, dest, **_):
        name = url.rsplit("/", 1)[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((fake_sources / name).read_bytes())
        return dest

    def fake_hf_download(repo, filename, dest_dir):
        dest = Path(dest_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((fake_sources / filename).read_bytes())
        return dest

    monkeypatch.setattr(kit_mod, "http_download", fake_http_download)
    monkeypatch.setattr(kit_mod, "hf_download", fake_hf_download)
    monkeypatch.setattr(kit_mod, "http_get_bytes", lambda url, **_: _png())
    monkeypatch.setattr(kit_mod.time, "sleep", lambda *_: None)
    return kit_mod


def test_fetch_kit_and_import_end_to_end(kit_module, tmp_path):
    kit = tmp_path / "kit"
    args = ["--kit", str(kit), "--eval-n", "10", "--seed-n", "30", "--backup", "10",
            "--corpus-entities", "25", "--hard-per-gold", "1"]
    kit_module.main(["annotations", *args])
    kit_module.main(["splits", *args])
    report = json.loads((kit / "splits" / "split_report.json").read_text())
    assert report["eval"]["n"] >= 10 and report["excluded_entities"] > 0

    kit_module.main(["oven", *args])
    wanted = set(kit_module._wanted_images(kit))
    present = {p.stem for p in (kit / "query_images").glob("*.jpg")}
    missing = set((kit / "oven" / "missing_image_ids.txt").read_text().split())
    assert present | missing == wanted and not present & missing  # only what the splits need
    assert present and missing <= {"oven_04000029"}
    assert not (kit / "oven" / "downloads" / "shard04.tar").exists()  # deleted after use
    with Image.open(next((kit / "query_images").glob("*.jpg"))) as im:
        assert im.format == "JPEG"

    kit_module.main(["wiki6m", *args])
    ents = [json.loads(l) for l in gzip.open(kit / "wiki" / "entities.jsonl.gz", "rt")]
    roles = {e["role"] for e in ents}
    assert {"gold", "hard_negative", "random"} <= roles and len(ents) == 25
    assert {e["entity_id"] for e in ents if e["role"] == "gold"} <= {f"Q{i}" for i in range(10)}
    assert kit_module.category_head("Dove 1 is a species of bird in the family X.") == "bird"
    assert kit_module.category_head("The Louvre is a museum in Paris.") == "museum"

    kit_module.main(["wiki-images", *args])
    kit_module.main(["finalize", *args])
    manifest = json.loads((kit / "manifest.json").read_text())
    seeds = (kit / "splits" / "seed_ids.txt").read_text().split()
    evals = (kit / "splits" / "eval_ids_final.txt").read_text().split()
    assert len(seeds) == manifest["counts"]["seeds"] <= 30
    assert not set(seeds) & set(evals)
    assert "wiki/entities.jsonl.gz" in manifest["sha256"]

    importer = _load("import_infoseek_kit_t", _DC / "data_prep" / "import_infoseek_kit.py")
    out, splits_dir = tmp_path / "corpus" / "infoseek", tmp_path / "splits"
    stats = importer.run(kit, out, splits_dir, chunk_chars=300, max_chunks=5)
    assert stats["seed_queries"] == len(seeds) and stats["eval_queries"] == len(evals)
    rows = [json.loads(l) for l in (out / "queries.jsonl").open()]
    assert all(r["image_path"] and not r["image_path"].startswith("/") for r in rows)
    value_gold = [r for r in rows if "range" in r["gold_answers"][0]]
    assert value_gold  # numeric answers survive as InfoSeek range reprs
    docs = [json.loads(l) for l in (out / "corpus.jsonl").open()]
    assert any(d["doc_id"].endswith("_img") and d["text"] == "" for d in docs)
    assert max(sum(1 for d in docs if d["entity_id"] == e and d["text"]) for e in ("Q0", "Q1")) <= 5

    queries, corpus = load_benchmark(
        "infoseek", n=1000, data_dir=out,
        include_ids=(splits_dir / "seed_ids.txt").read_text().split(),
        exclude_ids=(splits_dir / "eval_ids.txt").read_text().split(),
    )
    assert len(queries) == len(seeds) and Path(queries[0].image_path).exists()
    with pytest.raises(ValueError, match="held-out"):
        load_benchmark("infoseek", data_dir=out, queries_file="eval_queries.jsonl",
                       exclude_ids=evals)

    (kit / "wiki" / "entities.jsonl.gz").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="corrupt"):
        importer.run(kit, tmp_path / "again", tmp_path / "splits2")


# --------------------------------------------------------------------------- #
# ScienceQA / GQA from synthetic parquet
# --------------------------------------------------------------------------- #
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _sqa_rows(n, split):
    rows = []
    for i in range(n):
        rows.append({
            "image": {"bytes": _png(), "path": None} if i % 2 == 0 else None,
            "question": f"{split} question {i}: which animal is a mammal?",
            "choices": ["frog", "dog", "eagle"], "answer": 1,
            "hint": f"Context for {split} {i}." if i % 3 else "",
            "lecture": "Mammals have fur." if i % 4 else "",
            "solution": "A dog is a mammal. SECRET_SOLUTION",
            "subject": "natural science", "topic": "biology", "category": "c",
            "skill": "classify animals", "grade": "grade3",
        })
    return rows


def test_prepare_scienceqa(tmp_path):
    src = tmp_path / "ScienceQA" / "data"
    src.mkdir(parents=True)
    for split, n in (("train", 12), ("validation", 4), ("test", 8)):
        pq.write_table(pa.Table.from_pylist(_sqa_rows(n, split)),
                       src / f"{split}-00000-of-00001-abc.parquet")
    mod = _load("prepare_scienceqa_t", _DC / "data_prep" / "prepare_scienceqa.py")
    out, splits = tmp_path / "out", tmp_path / "splits"
    stats = mod.run(tmp_path / "ScienceQA", out, splits, eval_n=5, seed_n=100)
    assert stats["seed_queries"] == 16 and stats["eval_queries"] == 5
    seeds = [json.loads(l) for l in (out / "queries.jsonl").open()]
    assert all(q["metadata"]["split"] in ("train", "validation") for q in seeds)
    assert "Context for" not in seeds[0]["question"]          # hints must be retrieved
    assert seeds[0]["metadata"]["choices"] == ["frog", "dog", "eagle"]
    corpus_text = (out / "corpus.jsonl").read_text()
    assert "SECRET_SOLUTION" not in corpus_text                # solution never indexed
    kinds = stats["corpus_kinds"]
    assert kinds == {"lecture": 1, "hint": kinds["hint"]} and kinds["hint"] > 0
    docs = [json.loads(l) for l in corpus_text.splitlines()]
    assert all(d["doc_id"].startswith(("sqa-lecture-", "sqa-hint-")) for d in docs)
    q, _ = load_benchmark("scienceqa", n=100, data_dir=out)
    assert any(x.image_path and Path(x.image_path).exists() for x in q)


def test_prepare_gqa(tmp_path):
    src = tmp_path / "GQA"
    train = [{"id": f"t{i}", "imageId": f"{i // 3}", "question": f"What color is object {i}?",
              "answer": "red", "types": {"semantic": "attr"}} for i in range(30)]
    dev = [{"id": f"d{i}", "imageId": f"n{i}", "question": f"Is there a cat {i}?",
            "answer": "no", "types": {"semantic": "exist"}} for i in range(6)]
    for split, rows in (("train_balanced", train), ("testdev_balanced", dev)):
        (src / f"{split}_instructions").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), src / f"{split}_instructions" / "x.parquet")
        ids = sorted({r["imageId"] for r in rows})
        (src / f"{split}_images").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"id": i, "image": {"bytes": _png(), "path": None}}
                                             for i in ids]),
                       src / f"{split}_images" / "x.parquet")
    graphs = {str(i): {"objects": {"o1": {"name": "ball", "attributes": ["red"],
                                          "relations": [{"name": "left of", "object": "o2"}]},
                                   "o2": {"name": "box", "attributes": [], "relations": []}}}
              for i in range(9)}                       # image 9 has no scene graph
    zpath = tmp_path / "sceneGraphs.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("train_sceneGraphs.json", json.dumps(graphs))
    mod = _load("prepare_gqa_t", _DC / "data_prep" / "prepare_gqa.py")
    out = tmp_path / "out"
    stats = mod.run(src, zpath, out, tmp_path / "splits", eval_n=4, seed_n=100)
    seeds = [json.loads(l) for l in (out / "queries.jsonl").open()]
    assert stats["seed_queries"] == 9                   # one per image with a graph
    assert len({s["metadata"]["image_id"] for s in seeds}) == 9
    assert stats["eval_queries"] == 4
    docs = [json.loads(l) for l in (out / "corpus.jsonl").open()]
    assert len(docs) == 9 and "ball (red). box. ball left of box." == docs[0]["text"]
    assert not any(d["doc_id"].startswith("gqa-img-n") for d in docs)   # no test-dev images
    evals = [json.loads(l) for l in (out / "eval_queries.jsonl").open()]
    assert all((out / e["image_path"]).exists() for e in evals)


def test_prepare_gqa_without_scene_graphs_uses_program_fallback(tmp_path):
    src = tmp_path / "GQA"
    train = [{"id": f"t{i}", "imageId": f"{i // 3}", "question": f"What is on the wall {i}?",
              "answer": "SECRETANSWER", "fullAnswer": "The SECRETANSWER is on the wall.",
              "types": {"semantic": "rel"},
              "semanticStr": f"select: wall (1{i})->relate: _,on,s (2{i}) [0]->query: name [1]"}
             for i in range(12)]
    dev = [{"id": f"d{i}", "imageId": f"n{i}", "question": "Is there a cat?", "answer": "no",
            "types": {"semantic": "exist"}, "semanticStr": "select: cat (9)->exist [0]"}
           for i in range(4)]
    for split, rows in (("train_balanced", train), ("testdev_balanced", dev)):
        (src / f"{split}_instructions").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows), src / f"{split}_instructions" / "x.parquet")
        (src / f"{split}_images").mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{"id": i, "image": {"bytes": _png(), "path": None}}
                                             for i in sorted({r["imageId"] for r in rows})]),
                       src / f"{split}_images" / "x.parquet")
    mod = _load("prepare_gqa_fallback_t", _DC / "data_prep" / "prepare_gqa.py")
    out = tmp_path / "out"
    stats = mod.run(src, None, out, tmp_path / "splits", eval_n=2, seed_n=100)
    assert stats["seed_queries"] == 4 and stats["corpus_docs"] == 4
    assert stats["corpus_docs_program_fallback"] == 4 and stats["corpus_docs_scene_graph"] == 0
    docs = [json.loads(l) for l in (out / "corpus.jsonl").open()]
    assert docs[0]["text"].startswith("Questions asked about this image")
    assert "select: wall (10) -> relate: _,on,s (20) -> query: name" in docs[0]["text"]
    assert "SECRETANSWER" not in (out / "corpus.jsonl").read_text()   # answers never indexed
    assert not any(d["doc_id"].startswith("gqa-img-n") for d in docs)  # no test-dev images


@pytest.mark.parametrize("text,head", [
    ("Dove 1 is a species of bird in the family X.", "bird"),
    ("Rosa x is a species of flowering plant in the rose family.", "plant"),
    ("The Louvre is a museum in Paris.", "museum"),
    ("Pro Football Hall of Fame is the hall of fame for professional football.", "hall"),
    ("It was one of the largest ocean liners built in Belfast.", "liners"),
    ("No copula here.", None),
])
def test_category_head(text, head):
    kit_mod = _load("fetch_infoseek_head", _DC / "fetch_kit" / "fetch_infoseek.py")
    assert kit_mod.category_head(text) == head


def test_plan_lists_needed_images_and_shard_index_avoids_refetch(kit_module, tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    args = ["--kit", str(kit), "--eval-n", "10", "--seed-n", "30", "--backup", "10"]
    kit_module.main(["annotations", *args])
    kit_module.main(["splits", *args])
    monkeypatch.setattr(kit_module, "hf_list_files", lambda repo: [
        ("shard04.tar", 40_000_000_000), ("shard05.tar", 30_000_000_000), ("ovenid2impath.csv", 10),
    ])
    plan = kit_module.step_plan(kit, ["shard04.tar"])
    wanted = kit_module._wanted_images(kit)
    rows = (kit / "oven" / "wanted_images.tsv").read_text().splitlines()[1:]
    assert len(rows) == plan["distinct_images"] == len(wanted)
    assert {r.split("\t")[0] for r in rows} == set(wanted)          # exactly the selected images
    assert plan["by_id_prefix"] == {"oven_04": len(wanted)}
    assert plan["by_source_dataset"] == {"inat": len(wanted)}
    assert plan["planned_download_gb"] == 40.0
    assert [s["file"] for s in plan["hf_shards"] if s["planned"]] == ["shard04.tar"]

    fetched = []
    fake = kit_module.hf_download
    monkeypatch.setattr(kit_module, "hf_download",
                        lambda repo, name, dest: fetched.append(name) or fake(repo, name, dest))
    kit_module.step_oven(kit, ["shard04.tar", "shard05.tar"])
    index = kit / "oven" / "shard_index" / "shard04.tar.txt"
    assert len(index.read_text().split()) == 29                     # every image in the shard
    assert fetched.count("shard04.tar") == 1
    # Nothing new is pending (image 29 is in no shard): a rerun downloads nothing,
    # and shard05 is never fetched once everything findable has been found.
    kit_module.step_oven(kit, ["shard04.tar"])
    assert fetched.count("shard04.tar") == 1
    plan2 = kit_module.step_plan(kit, ["shard04.tar"])
    present = sum(1 for i in wanted if (kit / "query_images" / f"{i}.jpg").exists())
    assert plan2["already_present"] == present
    assert plan2["located_in_scanned_shards"] == {"shard04.tar": present}


def test_plan_survives_offline_hf_listing(kit_module, tmp_path, monkeypatch):
    kit = tmp_path / "kit"
    args = ["--kit", str(kit), "--eval-n", "4", "--seed-n", "8", "--backup", "0"]
    kit_module.main(["annotations", *args])
    kit_module.main(["splits", *args])

    def offline(repo):
        raise ConnectionError("no network")

    monkeypatch.setattr(kit_module, "hf_list_files", offline)
    plan = kit_module.step_plan(kit, ["shard04.tar"])
    assert plan["hf_shards"] == [] and plan["distinct_images"] > 0
