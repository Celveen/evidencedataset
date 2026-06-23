"""Self-contained demo of the 4 retrieval actions on a tiny image+text corpus.

Uses a deterministic fake CLIP (no model download) so you can SEE every action
type fire end-to-end and how the inspector renders them, before real OVEN
images are available. Run:

    python DatasetConstruct/demo_cross_modal.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from common import write_jsonl  # noqa: F401  (kept for parity; demo prints inline)

from evidencetree.actions import (
    ActionExecutor, BM25Retriever, CrossModalCLIPRetriever,
    AnswerAction, ImageSearchAction, ImageToTextAction, SearchState,
    TextSearchAction, TextToImageAction,
)
from evidencetree.eval.benchmarks import Document
from evidencetree.prm.data_gen import label_steps
from evidencetree.prm.rationale_gen import RationaleGenerator
from evidencetree.prm.verifiers import ClipGroundingScorer, GroundingVerifier
import inspect_trajectories as inspect


# Fake CLIP: "owl" axis vs "fish" axis, shared by text and image towers.
def _txt_enc(texts):
    return np.array([[1.0, 0.0] if "owl" in str(t).lower() else [0.0, 1.0] for t in texts],
                    dtype="float32")


def _img_enc(images):
    # red-ish image -> owl axis, blue-ish -> fish axis
    return np.array([[1.0, 0.0] if im.getpixel((0, 0))[0] > im.getpixel((0, 0))[2]
                     else [0.0, 1.0] for im in images], dtype="float32")


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    owl_img = tmp / "owl.png"; Image.new("RGB", (8, 8), (255, 0, 0)).save(owl_img)
    fish_img = tmp / "fish.png"; Image.new("RGB", (8, 8), (0, 0, 255)).save(fish_img)

    corpus = [
        Document(doc_id="img_owl", text="", title="owl photo", image_path=str(owl_img)),
        Document(doc_id="img_fish", text="", title="fish photo", image_path=str(fish_img)),
        Document(doc_id="txt_owl", text="The snowy owl is a large white owl of the tundra.",
                 title="Snowy owl"),
        Document(doc_id="txt_fish", text="The blue tang is a tropical reef fish.",
                 title="Blue tang"),
    ]
    text_ret = BM25Retriever().build(corpus)
    cross_ret = CrossModalCLIPRetriever(text_encoder=_txt_enc, image_encoder=_img_enc).build(corpus)
    ex = ActionExecutor(text_retriever=text_ret, image_retriever=cross_ret, top_k=2)

    # One trajectory exercising every action type (the query carries an owl image).
    s = SearchState(question="What is the wingspan of this owl?", image_path=str(owl_img))
    s = ex.execute(s, TextSearchAction(query="owl wingspan"))     # text -> text
    s = ex.execute(s, ImageToTextAction())                         # image -> text
    s = ex.execute(s, TextToImageAction(query="owl"))              # text -> image
    s = ex.execute(s, ImageSearchAction())                         # image -> image
    s = ex.execute(s, AnswerAction(text="around 1.5 meters"))

    # Serialize to the trajectory schema (mirrors step1.trajectory_dict).
    steps = []
    for i, a in enumerate(s.actions_taken):
        a_in = getattr(a, "query", None) or getattr(a, "text", "") or a.describe()
        step = {"step_index": i, "action_type": a.action_type, "action_input": str(a_in),
                "evidence": [{"evidence_id": e.evidence_id, "doc_id": e.doc_id,
                              "title": e.title, "text": e.text,
                              "image_path": e.image_path, "score": e.score}
                             for e in s.evidence if e.step_index == i]}
        if a.action_type in ("image_search", "image_to_text"):
            step["region"] = list(a.region) if a.region else None
            step["image_path"] = a.image_path or s.image_path
        steps.append(step)
    traj = {"traj_id": "demo#t0", "query_id": "demo", "question": s.question,
            "image_path": str(owl_img), "gold_answers": ["1.5 m"], "lambda": 0.3,
            "outcome_em": 0.0, "steps": steps}

    print("\n########## TRAJECTORY (inspector view) ##########\n")
    print(inspect.render_trajectory(traj, width=90))

    # Score (RESULT vs question) in ONE unified CLIP space (fake encoders here).
    verifier = GroundingVerifier(
        clip_scorer=ClipGroundingScorer(
            text_encoder=_txt_enc, image_encoder=_img_enc,
            text_band=(0.0, 1.0), image_band=(0.0, 1.0),  # fake cos in {0,1}
        )
    )
    samples = label_steps([traj], verifier=verifier, alpha=0.5)
    rgen = RationaleGenerator(backend="mock")
    print("\n########## STEP SAMPLES (scores + rationale) ##########\n")
    for smp in samples:
        smp.update(rgen.generate_for(smp))
        print(inspect.render_sample(smp, width=90))
        print()
    print("action-type usage:",
          {a: sum(1 for s in samples if s["action"]["type"] == a)
           for a in sorted({s["action"]["type"] for s in samples})})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
