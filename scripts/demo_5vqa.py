"""5-sample VQA end-to-end demo of the unified-retrieval MCTS — fully local.

Builds 5 tiny VQA samples (PIL colored-shape query images), a ~15-unit corpus
(per entity: one text chunk holding the answer fact + one image), and runs the
REAL MCTSSearcher through the SAME shared assembly used by DatasetConstruct
step1 and scripts/run_inference.py (``build_search_stack``).

Retrieval mechanics are REAL (unified CLIP-style space, FAISS/numpy index,
entity dedupe, mixed-modality top-k) — only the encoders are injectable
deterministic stand-ins so nothing is downloaded:

* image encoder — 8-bin color histogram (2-level RGB quantization);
* text encoder — color words map onto the SAME 8 color bins + a 24-dim
  hashed bag-of-words block; both towers project to one 32-dim space and are
  L2-normalized by the retriever.

Two policy modes (--proposer):

* heuristic (default) — HeuristicProposer + mock generator (assembled by
  build_search_stack in mock mode). No VLM: fixed template proposals.
* llm — the REAL LLMProposer code path (prompt build -> JSON parse -> dedupe
  -> unconditional image/text branch guarantees) driven by a scripted VLM
  stand-in that reproduces the collapse bias observed on the server (root:
  image_search only; after evidence: answer only). The demo then shows the
  guarantees turning that single-candidate policy into a branching tree.
  On the server the same LLMProposer is driven by Qwen2.5-VL instead.

Scorer (--scorer): overlap (default) or visualprm — the latter runs
VisualPRMScorer in mock mode to validate the PRM wiring end to end (on the
server the same flag loads the real / fine-tuned VisualPRM-8B).

For every query the script prints the full tree trajectories: each rollout's
action sequence with per-step retrieved units (modality tags), node Q values,
and the best path; output is saved to data/demo_5vqa_trajectories[_llm].txt.

Usage:
    python scripts/demo_5vqa.py
    python scripts/demo_5vqa.py --proposer llm --scorer visualprm
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.eval.benchmarks import Document, Query  # noqa: E402
from evidencetree.generation.base import GenerationConfig, Generator  # noqa: E402
from evidencetree.mcts import LLMProposer  # noqa: E402
from evidencetree.pipeline import build_search_stack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "demo_5vqa_trajectories.txt"

DIM = 32           # 8 color bins + 24 hashed bag-of-words dims
_COLOR_BINS = 8    # 2-level quantization per RGB channel -> 2^3 bins
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# color word -> solid RGB (shared vocabulary between the two towers)
_COLOR_RGB = {
    "red": (220, 30, 30),
    "green": (30, 180, 30),
    "blue": (30, 30, 220),
    "yellow": (230, 210, 30),
    "purple": (150, 30, 200),
}


# --------------------------------------------------------------------------- #
# Deterministic encoders (one shared 32-dim space, no model downloads)
# --------------------------------------------------------------------------- #
def _color_bin(rgb) -> int:
    r, g, b = rgb[:3]
    return (int(r >= 128) << 2) | (int(g >= 128) << 1) | int(b >= 128)


def image_encoder(images):
    """8-bin color histogram in dims [0..7]; dims [8..31] stay zero."""
    vecs = np.zeros((len(images), DIM), dtype="float32")
    for i, image in enumerate(images):
        pixels = np.asarray(image.convert("RGB").resize((16, 16)), dtype="float32")
        for px in pixels.reshape(-1, 3):
            vecs[i, _color_bin(px)] += 1.0
        # de-emphasize the white background bin so shape color dominates
        vecs[i, _color_bin((255, 255, 255))] *= 0.15
    return vecs


def text_encoder(texts):
    """Color words hit the SAME color bins; other tokens hash into [8..31]."""
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for tok in _TOKEN_RE.findall(str(text).lower()):
            if tok in _COLOR_RGB:
                vecs[i, _color_bin(_COLOR_RGB[tok])] += 2.0
            vecs[i, 8 + zlib.crc32(tok.encode()) % (DIM - 8)] += 1.0
    return vecs


# --------------------------------------------------------------------------- #
# Tiny VQA world: 5 entities, 5 queries, ~15-unit corpus
# --------------------------------------------------------------------------- #
_ENTITIES = [
    ("red", "circle", "Paris", "The red circle emblem is kept in Paris."),
    ("blue", "square", "Tokyo", "The blue square banner was made in Tokyo."),
    ("green", "triangle", "Cairo", "The green triangle flag flies over Cairo."),
    ("yellow", "star", "Rome", "The yellow star medal is awarded in Rome."),
    ("purple", "hexagon", "Berlin", "The purple hexagon crest belongs to Berlin."),
]

_DISTRACTORS = [
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The stock market reflects investor sentiment about future earnings.",
    "Tectonic plates move slowly over the mantle of the Earth.",
    "Classical music flourished in Europe for two centuries.",
    "The history of cartography spans thousands of years.",
]


def _draw_shape(path: Path, color: str, shape: str, size: int) -> None:
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    rgb = _COLOR_RGB[color]
    lo, hi = size // 6, size - size // 6
    mid = size // 2
    if shape == "circle":
        d.ellipse([lo, lo, hi, hi], fill=rgb)
    elif shape == "square":
        d.rectangle([lo, lo, hi, hi], fill=rgb)
    elif shape == "triangle":
        d.polygon([(mid, lo), (hi, hi), (lo, hi)], fill=rgb)
    elif shape == "star":
        d.polygon([(mid, lo), (hi, mid), (mid, hi), (lo, mid)], fill=rgb)
    else:  # hexagon
        q = size // 4
        d.polygon([(q, lo), (hi - q, lo), (hi, mid), (hi - q, hi), (q, hi), (lo, mid)],
                  fill=rgb)
    img.save(path)


def build_world(tmp: Path):
    queries, corpus = [], []
    for i, (color, shape, city, fact) in enumerate(_ENTITIES):
        q_img = tmp / f"query_{color}_{shape}.png"
        c_img = tmp / f"corpus_{color}_{shape}.png"
        _draw_shape(q_img, color, shape, size=24)   # query rendering
        _draw_shape(c_img, color, shape, size=32)   # corpus rendering
        entity = f"{color} {shape}"
        queries.append(Query(
            query_id=f"vqa_{i}",
            question=f"Which city is linked to the {entity} in this image?",
            gold_answers=[city],
            image_path=str(q_img),
        ))
        # ONE document with text AND image -> TWO independent retrieval units.
        corpus.append(Document(
            doc_id=f"doc_{i}", title=entity, text=fact,
            image_path=str(c_img), entity_id=entity,
        ))
    for j, text in enumerate(_DISTRACTORS):
        corpus.append(Document(doc_id=f"distractor_{j}", text=text))
    return queries, corpus


# --------------------------------------------------------------------------- #
# Scripted VLM (llm mode): drive the REAL LLMProposer path offline
# --------------------------------------------------------------------------- #
_CITIES = [e[2] for e in _ENTITIES]


class ScriptedVLM(Generator):
    """Deterministic stand-in for the policy VLM (Qwen2.5-VL on the server).

    Reproduces the single-candidate collapse bias measured in the 40q/200q
    server ablations so the demo exercises LLMProposer's structural
    guarantees: with no evidence it proposes ONLY image_search; once any
    evidence exists it proposes ONLY answer. Answer text is read from the
    evidence (never from gold), like a faithful evidence-grounded policy.
    """

    def __init__(self) -> None:
        super().__init__(GenerationConfig(backend="mock", model="scripted-vlm"))

    def generate(self, question, context_docs, **kwargs) -> str:
        if "Propose exactly" in question:  # LLMProposer's action-proposal prompt
            if "Evidence collected so far:\n(none)" in question:
                return '{"type": "image_search"}'
            return f'{{"type": "answer", "text": "{self._answer(question)}"}}'
        # answer drafting (LLMProposer.propose_answer / simulation)
        return self._answer("\n".join(context_docs))

    @staticmethod
    def _answer(visible_text: str) -> str:
        # Read the answer out of the highest-ranked evidence line that names a
        # city (evidence lines arrive in retrieval-rank order).
        for line in visible_text.splitlines():
            for city in _CITIES:
                if city in line:
                    return city
        return "unknown"


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def _fmt_evidence(state, step_index: int) -> list[str]:
    lines = []
    for e in state.evidence:
        if e.step_index != step_index:
            continue
        snippet = (e.text or e.title or "<no text>")[:48]
        lines.append(
            f"        [{e.result_modality or '?'}] {e.doc_id:<14} "
            f"score={e.score:+.3f}  {snippet}"
        )
    return lines


def _walk_tree(node, out: list[str], depth: int = 0) -> None:
    for child in node.children:
        out.append(
            f"    {'  ' * depth}- {child.action.describe()[:58]:<58} "
            f"N={child.visits:<2d} Q={child.q_value:.3f}"
        )
        _walk_tree(child, out, depth + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposer", choices=("heuristic", "llm"), default="heuristic",
                    help="llm = real LLMProposer path driven by a scripted VLM")
    ap.add_argument("--scorer", choices=("overlap", "visualprm"), default="overlap",
                    help="visualprm = VisualPRMScorer (mock) to validate PRM wiring")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="demo5vqa_"))
    queries, corpus = build_world(tmp)

    cfg = {
        "benchmark": "demo5vqa",
        "retriever": {"backend": "unified", "dedupe": "entity", "top_k": 3,
                      "image_search": True, "image": {"top_k": 3}},
        "policy": {"backend": "mock", "mock_accuracy": 1.0, "mock_seed": 0},
        "mcts": {"rollouts": 6, "max_depth": 3, "top_k_children": 3,
                 "early_stop_q": 2.0, "seed": 0},
        "prm": {"scorer": args.scorer},
    }
    searcher = build_search_stack(
        cfg, queries, corpus, mock=True, benchmark="demo5vqa",
        text_encoder=text_encoder, image_encoder=image_encoder,
    )
    if args.proposer == "llm":
        # Same LLMProposer class + code path as real inference/step1; only the
        # generator is the scripted VLM instead of Qwen2.5-VL.
        searcher.proposer = LLMProposer(ScriptedVLM(), enable_image_search=True)

    out_path = OUT_PATH if args.proposer == "heuristic" else (
        OUT_PATH.with_name(OUT_PATH.stem + "_llm" + OUT_PATH.suffix)
    )

    lines: list[str] = []
    seen_modalities: set[str] = set()
    trees_with_both_branches = 0
    n_correct = 0

    lines.append("=" * 78)
    lines.append("5-sample VQA demo — unified CLIP space, REAL MCTSSearcher "
                 "(rollouts=6, depth=3)")
    lines.append(f"proposer={args.proposer} (llm = real LLMProposer + scripted "
                 f"VLM) | scorer={args.scorer}")
    lines.append(f"corpus units: {len(searcher.executor.text_retriever)} "
                 f"({searcher.executor.text_retriever.text_unit_count} text + "
                 f"{searcher.executor.text_retriever.image_unit_count} image), "
                 f"dedupe=entity")
    lines.append("=" * 78)

    for query in queries:
        result = searcher.search(query.question, image_path=query.image_path)
        lines.append("")
        lines.append(f"QUERY {query.query_id}: {query.question}")
        lines.append(f"  gold={query.gold_answers}  image={Path(query.image_path).name}")

        for record in result.rollout_log:
            state = record.state
            lines.append(f"  rollout t={record.t}  reward={record.reward:.3f}  "
                         f"answer={state.final_answer!r}")
            for i, action in enumerate(state.actions_taken):
                lines.append(f"      step {i}: {action.describe()[:64]}")
                lines.extend(_fmt_evidence(state, i))
            seen_modalities |= {
                e.result_modality for e in state.evidence if e.result_modality
            }

        lines.append("  TREE (visits, mean Q):")
        _walk_tree(result.root, lines)
        root_types = {c.action.action_type for c in result.root.children}
        if {"text_search", "image_search"} <= root_types:
            trees_with_both_branches += 1
        best = " -> ".join(a.describe()[:44] for a in result.best_state.actions_taken)
        lines.append(f"  BEST PATH (reward={result.best_reward:.3f}): {best}")
        correct = result.answer.strip().lower() == query.gold_answers[0].lower()
        n_correct += int(correct)
        lines.append(f"  FINAL answer={result.answer!r}  correct={correct}")

    lines.append("")
    lines.append("=" * 78)
    lines.append(f"SUMMARY: {n_correct}/{len(queries)} correct | "
                 f"modalities retrieved: {sorted(seen_modalities)} | "
                 f"trees with BOTH search branches: {trees_with_both_branches}/{len(queries)}")
    lines.append("=" * 78)

    report = "\n".join(lines)
    print(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Health assertions (the point of the demo)
    assert seen_modalities == {"text", "image"}, (
        f"expected BOTH result modalities across the run, got {seen_modalities}"
    )
    assert trees_with_both_branches >= 1, (
        "no query tree contains both a text_search and an image_search branch"
    )
    print("HEALTH: mixed modalities retrieved, mixed search branches present, "
          "no crash. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
