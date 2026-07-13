"""Run standalone image_search diagnostics and write a visual report.

This isolates the image retriever from MCTS/Qwen answer generation. It uses the
same retriever wiring as DatasetConstruct step 1:

* Local image corpora: CLIP FAISS index.

The report shows the query image ("before") and Top-K retrieved images/items
("after") so retrieval failures are visible without reading JSONL by hand.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

from common import load_env

from evidencetree.actions import ClipImageRetriever
from evidencetree.eval import benchmarks
from evidencetree.utils import config as cfgutil


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _query_ids_from_trajectories(path: Path) -> list[str]:
    ids: list[str] = []
    seen = set()
    for row in _read_jsonl(path):
        qid = str(row["query_id"])
        if qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


def _load_queries(dataset: str, data_dir: Path, n: int, query_ids: list[str]):
    queries, corpus = benchmarks.load_benchmark(
        name=dataset,
        n=1000000 if query_ids else n,
        mock=False,
        data_dir=data_dir,
        seed=0,
    )
    if query_ids:
        wanted = set(query_ids)
        by_id = {query.query_id: query for query in queries}
        queries = [by_id[qid] for qid in query_ids if qid in by_id]
    return queries[:n], corpus


def _build_retriever(dataset: str, cfg: dict[str, Any], data_dir: Path, corpus):
    retriever_cfg = cfg.get("retriever", {})
    image_cfg = retriever_cfg.get("image", {})
    index_value = image_cfg.get("index_dir")
    index_dir = Path(index_value) if index_value else data_dir / "clip_index"
    model_name = str(image_cfg.get("model_name", "clip-ViT-B-32"))
    device = image_cfg.get("device", "cpu")
    batch_size = int(image_cfg.get("batch_size", 32))
    if (index_dir / "index.faiss").exists() and (index_dir / "docs.jsonl").exists():
        return ClipImageRetriever.load(index_dir, model_name=model_name, device=device)
    retriever = ClipImageRetriever(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    ).build(corpus)
    retriever.save(index_dir)
    return retriever


def _image_src(path_or_url: str | None) -> str:
    if not path_or_url:
        return ""
    if path_or_url.startswith(("http://", "https://", "file://")):
        return path_or_url
    return "file://" + str(Path(path_or_url).resolve())


def _source_url(text: str) -> str:
    match = re.search(r"Image source:\s*(https?://\S+)", text)
    return match.group(1) if match else ""


def _entities(text: str) -> str:
    match = re.search(r"\[(.*?)\]", text, flags=re.S)
    if not match:
        return ""
    raw = match.group(1).replace('"', "").replace("\n", " ")
    return raw[:180]


def _short(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<style>",
        "body{font-family:Arial,sans-serif;line-height:1.35;margin:24px}",
        ".query{border:1px solid #ddd;border-radius:10px;padding:16px;margin:18px 0}",
        ".grid{display:grid;grid-template-columns:220px 1fr;gap:18px}",
        ".hits{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
        ".card{border:1px solid #ddd;border-radius:8px;padding:10px;background:#fafafa}",
        "img{max-width:100%;max-height:190px;object-fit:contain;background:white}",
        ".score{font-weight:bold;color:#245}",
        "code{white-space:pre-wrap}",
        "</style>",
        "<h1>Image Search Diagnostic Report</h1>",
        "<p>Standalone image_search results. Left: query image. Right: Top-K retrieval results.</p>",
    ]
    for row in rows:
        parts.append("<section class='query'>")
        parts.append(f"<h2>{html.escape(row['dataset'])} / {html.escape(row['query_id'])}</h2>")
        parts.append(f"<p><b>Question:</b> {html.escape(row['question'])}</p>")
        parts.append(f"<p><b>Gold:</b> {html.escape(' / '.join(row['gold_answers']))}</p>")
        parts.append("<div class='grid'>")
        parts.append(
            "<div><h3>Before: query image</h3>"
            f"<img src='{html.escape(_image_src(row['query_image']))}'>"
            f"<p><code>{html.escape(row['query_image'])}</code></p></div>"
        )
        parts.append("<div><h3>After: retrieved Top-K</h3><div class='hits'>")
        for index, hit in enumerate(row["hits"], 1):
            visual = hit.get("image_path") or hit.get("source_url")
            parts.append("<div class='card'>")
            parts.append(f"<h4>#{index} <span class='score'>score={hit['score']:.4f}</span></h4>")
            if visual:
                parts.append(f"<img src='{html.escape(_image_src(visual))}'>")
            else:
                parts.append("<p><i>No retrievable image path/url in hit metadata.</i></p>")
            parts.append(f"<p><b>doc:</b> {html.escape(hit['doc_id'])}</p>")
            parts.append(f"<p><b>title/entity:</b> {html.escape(hit['title_or_entity'])}</p>")
            parts.append(f"<p>{html.escape(hit['snippet'])}</p>")
            parts.append("</div>")
        parts.append("</div></div></div></section>")
    report_path.write_text("\n".join(parts), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="DatasetConstruct/config.qwen25vl.yaml")
    parser.add_argument("--datasets", nargs="+", default=["mrag_bench"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pipeline-root", type=Path, default=Path("data/pipeline_5q"))
    parser.add_argument("--output", type=Path, default=Path("data/image_search_debug/report.html"))
    args = parser.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config)
    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        data_dir = Path("data/corpus") / dataset
        query_ids = _query_ids_from_trajectories(
            args.pipeline_root / dataset / "trajectories.jsonl"
        )
        queries, corpus = _load_queries(dataset, data_dir, args.n, query_ids)
        retriever = _build_retriever(dataset, cfg, data_dir, corpus)
        for query in queries:
            if not query.image_path:
                continue
            hits = retriever.search_image(query.image_path, top_k=args.top_k)
            all_rows.append(
                {
                    "dataset": dataset,
                    "query_id": query.query_id,
                    "question": query.question,
                    "gold_answers": query.gold_answers,
                    "query_image": query.image_path,
                    "hits": [
                        {
                            "doc_id": hit.doc_id,
                            "score": float(hit.score),
                            "image_path": hit.image_path,
                            "source_url": _source_url(hit.text),
                            "title_or_entity": _entities(hit.text) or hit.title,
                            "snippet": _short(hit.text),
                        }
                        for hit in hits
                    ],
                }
            )
    _write_report(args.output, all_rows)
    print(f"wrote {args.output}")
    for row in all_rows:
        print(f"\n[{row['dataset']}] {row['query_id']}: {row['question']}")
        for index, hit in enumerate(row["hits"], 1):
            print(
                f"  #{index} score={hit['score']:.4f} "
                f"{hit['title_or_entity']} | {hit['doc_id']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
