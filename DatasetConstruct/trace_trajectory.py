"""Trace one MCTS trajectory end-to-end on an image+text scene.

This is a deterministic, no-download debug script. It simulates the current
dataset-construction setting: a query with an image, a corpus mixing text and
image units, and ONE unified CLIP index (text and image units independently
indexed; both search actions return mixed-modality hits). It prints selection
UCT scores, expansion priors, simulation actions, and backup values.

Usage:
    python DatasetConstruct/trace_trajectory.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from evidencetree.actions import ActionExecutor, UnifiedClipRetriever
from evidencetree.eval.benchmarks import Document
from evidencetree.generation import build_generator
from evidencetree.mcts import HeuristicProposer, MCTSSearcher, SearchConfig
from evidencetree.prm.model import HeuristicOverlapScorer


def _txt_enc(texts):
    return np.array(
        [[1.0, 0.0] if "owl" in str(t).lower() else [0.0, 1.0] for t in texts],
        dtype="float32",
    )


def _img_enc(images):
    return np.array(
        [
            [1.0, 0.0]
            if image.getpixel((0, 0))[0] > image.getpixel((0, 0))[2]
            else [0.0, 1.0]
            for image in images
        ],
        dtype="float32",
    )


def make_tracer():
    def tracer(ev):
        event = ev["event"]
        if event == "rollout_start":
            print(f"\n{'=' * 78}\nROLLOUT t={ev['t']}")
        elif event == "select":
            print("  SELECT (UCT = Q + c·sqrt(lnN/n)):")
            for cand in ev["candidates"]:
                print(
                    f"     {cand['action'][:46]:46}  UCT={cand['uct']:.3f} "
                    f"(Q={cand['q']:.2f} N={cand['visits']} "
                )
            print(f"     -> chosen: {ev['chosen']}")
        elif event == "expand":
            print("  EXPAND (proposer candidates, PRM prior; keep top-k):")
            for cand in ev["proposed"]:
                print(
                    f"     prior={cand['prior']:.3f}  ev={cand['n_evidence']}  "
                    f"{cand['action'][:54]}"
                )
            print(f"     -> kept as children: {ev['kept']}")
        elif event == "simulate":
            print(f"  SIMULATE roll-out: {ev['rollout_actions']}")
            print(f"     answer={ev['answer']!r}  PRM reward={ev['reward']:.3f}")
        elif event == "backup":
            chain = " -> ".join(
                f"{action[:22]}(N={visits},Q={q:.2f})"
                for action, visits, q in ev["path"]
            )
            print(f"  BACKUP reward={ev['reward']:.3f} up: {chain}")

    return tracer


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    owl_q = tmp / "owl_query.png"
    owl_c = tmp / "owl_corpus.png"
    fish_c = tmp / "fish_corpus.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(owl_q)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(owl_c)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(fish_c)

    corpus = [
        Document(
            doc_id="txt_owl",
            title="Snowy owl",
            text="The snowy owl has a wingspan of about 1.5 meters.",
        ),
        Document(doc_id="img_owl", title="owl photo", text="", image_path=str(owl_c)),
        Document(doc_id="txt_fish", title="Blue tang", text="The blue tang is a fish."),
        Document(doc_id="img_fish", title="fish photo", text="", image_path=str(fish_c)),
    ]
    unified = UnifiedClipRetriever(
        text_encoder=_txt_enc,
        image_encoder=_img_enc,
    ).build(corpus)
    executor = ActionExecutor(text_retriever=unified, image_retriever=unified, top_k=2)

    gen = build_generator({"backend": "mock", "mock_accuracy": 1.0, "mock_seed": 0})
    proposer = HeuristicProposer(
        gen,
        answer_kwargs_fn=lambda _state: {"reference": ["1.5 meters"]},
    )
    scorer = HeuristicOverlapScorer()

    cfg = SearchConfig(
        rollouts=3,
        max_depth=3,
        top_k_children=2,
        c_uct=1.0,
        early_stop_q=2.0,
        seed=0,
    )
    searcher = MCTSSearcher(executor, proposer, scorer, cfg, tracer=make_tracer())

    print("QUERY: 'What is the wingspan of this owl?' (+ owl query image)")
    print("CORPUS: txt_owl(answer), img_owl, txt_fish, img_fish")
    result = searcher.search("What is the wingspan of this owl?", image_path=str(owl_q))

    print(
        f"\n{'=' * 78}\nRESULT answer={result.answer!r} "
        f"best_reward={result.best_reward:.3f} rollouts={result.rollouts_run}"
    )
    print("FINAL TREE (root children, visit counts, mean Q):")
    for child in result.root.children:
        print(f"   {child.action.describe()[:50]:50} N={child.visits} Q={child.q_value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
