"""Trace ONE MCTS trajectory end-to-end on an image+text scene (no downloads).

Simulates the real dataset-construction setting — a query that carries an image,
a corpus mixing entity text and entity images — and prints every search
decision: rollout lambda, SELECTION (UCT per candidate), EXPANSION (proposer
candidates + PRM prior, top-k kept), SIMULATION (greedy roll to answer + reward),
BACKUP (reward propagated to the root). Uses fake CLIP so it is deterministic.

    python DatasetConstruct/trace_trajectory.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evidencetree.actions import (
    ActionExecutor, BM25Retriever, ClipImageRetriever, SearchState,
)
from evidencetree.eval.benchmarks import Document
from evidencetree.generation import build_generator
from evidencetree.mcts import HeuristicProposer, MCTSSearcher, SearchConfig
from evidencetree.prm.model import HeuristicOverlapScorer


# Fake CLIP: "owl" axis vs "fish" axis, shared by both towers (red img -> owl).
def _txt_enc(texts):
    return np.array([[1.0, 0.0] if "owl" in str(t).lower() else [0.0, 1.0]
                     for t in texts], dtype="float32")


def _img_enc(images):
    return np.array([[1.0, 0.0] if im.getpixel((0, 0))[0] > im.getpixel((0, 0))[2]
                     else [0.0, 1.0] for im in images], dtype="float32")


def make_tracer():
    def tracer(ev):
        e = ev["event"]
        if e == "rollout_start":
            print(f"\n{'='*78}\n● ROLLOUT t={ev['t']}")
        elif e == "select":
            print("  SELECT (UCT = Q + c·√(lnN/n)):")
            for c in ev["candidates"]:
                print(f"     {c['action'][:46]:46}  UCT={c['uct']:.3f} "
                      f"(Q={c['q']:.2f} N={c['visits']} prior={c['prior']:.2f})")
            print(f"     -> chosen: {ev['chosen']}")
        elif e == "expand":
            print("  EXPAND (proposer candidates, PRM prior; keep top-k):")
            for c in ev["proposed"]:
                print(f"     prior={c['prior']:.3f}  ev={c['n_evidence']}  {c['action'][:54]}")
            print(f"     -> kept as children: {ev['kept']}")
        elif e == "simulate":
            print(f"  SIMULATE roll-out: {ev['rollout_actions']}")
            print(f"     answer={ev['answer']!r}  PRM reward={ev['reward']:.3f}")
        elif e == "backup":
            chain = " -> ".join(f"{a[:22]}(N={n},Q={q:.2f})" for a, n, q in ev["path"])
            print(f"  BACKUP reward={ev['reward']:.3f} up: {chain}")
    return tracer


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    owl_q = tmp / "owl_query.png"; Image.new("RGB", (8, 8), (255, 0, 0)).save(owl_q)
    owl_c = tmp / "owl_corpus.png"; Image.new("RGB", (8, 8), (255, 0, 0)).save(owl_c)
    fish_c = tmp / "fish_corpus.png"; Image.new("RGB", (8, 8), (0, 0, 255)).save(fish_c)

    corpus = [
        Document(doc_id="txt_owl", title="Snowy owl",
                 text="The snowy owl has a wingspan of about 1.5 meters and is a large white owl."),
        Document(doc_id="img_owl", title="owl photo", text="", image_path=str(owl_c)),
        Document(doc_id="txt_fish", title="Blue tang",
                 text="The blue tang is a small tropical reef fish."),
        Document(doc_id="img_fish", title="fish photo", text="", image_path=str(fish_c)),
    ]
    text_ret = BM25Retriever().build(corpus)
    img_ret = ClipImageRetriever(encoder=_txt_enc, image_encoder=_img_enc).build(corpus)
    executor = ActionExecutor(text_retriever=text_ret, image_retriever=img_ret, top_k=2)

    # Deterministic policy: mock generator answers "1.5 meters" when evidence
    # contains it (the overlap scorer then rewards evidence-backed answers).
    gen = build_generator({"backend": "mock", "mock_accuracy": 1.0, "mock_seed": 0})
    proposer = HeuristicProposer(
        gen, answer_kwargs_fn=lambda s: {"reference": ["1.5 meters"]})
    scorer = HeuristicOverlapScorer()

    cfg = SearchConfig(rollouts=3, max_depth=3, top_k_children=2,
                       c_uct=1.0, early_stop_q=2.0, seed=0)
    searcher = MCTSSearcher(executor, proposer, scorer, cfg, tracer=make_tracer())

    print("QUERY: 'What is the wingspan of this owl?'  (+ owl query image)")
    print("CORPUS: txt_owl(answer), img_owl, txt_fish, img_fish")
    result = searcher.search("What is the wingspan of this owl?", image_path=str(owl_q))

    print(f"\n{'='*78}\nRESULT answer={result.answer!r}  best_reward={result.best_reward:.3f} "
          f"rollouts={result.rollouts_run}")
    print("FINAL TREE (root children, visit counts, mean Q):")
    for ch in result.root.children:
        print(f"   {ch.action.describe()[:50]:50} N={ch.visits} Q={ch.q_value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
