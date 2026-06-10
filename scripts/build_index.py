"""Build a dense FAISS index over a benchmark's own corpus (offline, one-time).

Reads a corpus JSONL (one ``{"doc_id", "text", "title"?, "image_path"?}`` per
line), encodes it, and writes ``index.faiss`` + ``docs.jsonl`` to the output
directory. BM25 needs no prebuilt index (it builds in-memory at load time).

Usage:
    python scripts/build_index.py \
        --corpus data/corpus/infoseek/infoseek_corpus.jsonl \
        --out data/corpus/infoseek/dense_index
    # CLIP image-text index instead of a text encoder:
    python scripts/build_index.py --corpus ... --out ... --clip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.actions.retrievers import ClipImageRetriever, DenseRetriever  # noqa: E402
from evidencetree.eval.benchmarks import Document  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("build_index")


def load_corpus(path: Path) -> list[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(obj["doc_id"]),
                    text=obj["text"],
                    title=obj.get("title", ""),
                    image_path=obj.get("image_path"),
                )
            )
    return docs


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a dense FAISS corpus index.")
    p.add_argument("--corpus", required=True, help="Corpus JSONL path.")
    p.add_argument("--out", required=True, help="Output index directory.")
    p.add_argument(
        "--model", default=None,
        help="Encoder model (default: MiniLM for text, clip-ViT-B-32 with --clip).",
    )
    p.add_argument("--clip", action="store_true", help="Build a CLIP image-text index.")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    docs = load_corpus(Path(args.corpus))
    log.info("Loaded %d docs from %s", len(docs), args.corpus)

    if args.clip:
        retriever = ClipImageRetriever(
            model_name=args.model or "clip-ViT-B-32", device=args.device
        )
    else:
        retriever = DenseRetriever(
            model_name=args.model or "sentence-transformers/all-MiniLM-L6-v2",
            device=args.device,
        )
    retriever.build(docs)
    out = retriever.save(args.out)
    log.info("Index written: %s (%d docs)", out, len(docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
