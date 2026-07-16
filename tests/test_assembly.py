"""build_search_stack: one shared assembly for construction AND inference."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from evidencetree.actions import BM25Retriever, HybridTextRetriever, UnifiedClipRetriever
from evidencetree.eval.benchmarks import Document, Query
from evidencetree.mcts import HeuristicProposer, MCTSSearcher
from evidencetree.mcts.proposer import GatedProposer
from evidencetree.pipeline import assembly, build_search_stack
from evidencetree.prm.model import HeuristicOverlapScorer

ROOT = Path(__file__).resolve().parents[1]


def _txt_enc(texts):
    return np.array(
        [[1.0, 0.0] if "red" in str(t).lower() else [0.0, 1.0] for t in texts],
        dtype="float32",
    )


def _img_enc(images):
    out = []
    for im in images:
        r, _, b = im.getpixel((0, 0))
        out.append([1.0, 0.0] if r > b else [0.0, 1.0])
    return np.array(out, dtype="float32")


@pytest.fixture
def data(tmp_path):
    from PIL import Image

    red = tmp_path / "red.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    queries = [
        Query(query_id="q0", question="which flower is red?",
              gold_answers=["rose"], image_path=str(red)),
    ]
    corpus = [
        Document(doc_id="rose", title="red rose",
                 text="a red rose is a red flower", image_path=str(red)),
        Document(doc_id="sea", title="blue sea", text="the blue sea"),
    ]
    return queries, corpus


def _cfg(**retriever):
    return {
        "benchmark": "infoseek",
        "retriever": {"backend": "unified", "dedupe": "entity", "top_k": 3,
                      "image_search": True, **retriever},
        "policy": {"backend": "mock", "mock_accuracy": 1.0, "mock_seed": 0},
        "mcts": {"rollouts": 2, "max_depth": 2, "top_k_children": 2,
                 "early_stop_q": 2.0, "seed": 0},
        "prm": {"scorer": "overlap"},
    }


def test_unified_backend_wires_unified_for_text_and_image(data):
    queries, corpus = data
    searcher = build_search_stack(
        _cfg(), queries, corpus, mock=True,
        text_encoder=_txt_enc, image_encoder=_img_enc,
    )
    assert isinstance(searcher, MCTSSearcher)
    ex = searcher.executor
    assert isinstance(ex.text_retriever, UnifiedClipRetriever)
    assert ex.image_retriever is ex.text_retriever  # ONE shared index
    assert ex.text_retriever.dedupe_key == "entity"
    assert isinstance(searcher.proposer, HeuristicProposer)
    assert searcher.proposer.enable_image_actions is True
    assert isinstance(searcher.scorer, HeuristicOverlapScorer)
    assert searcher.cfg.rollouts == 2 and searcher.cfg.max_depth == 2


def test_bm25_backend_still_available(data):
    queries, corpus = data
    cfg = _cfg(backend="bm25", image_search=False)
    searcher = build_search_stack(cfg, queries, corpus, mock=True,
                                  text_encoder=_txt_enc, image_encoder=_img_enc)
    assert isinstance(searcher.executor.text_retriever, BM25Retriever)
    assert searcher.executor.image_retriever is None


def test_hybrid_backend(data):
    queries, corpus = data
    searcher = build_search_stack(
        _cfg(backend="hybrid"), queries, corpus, mock=True,
        text_encoder=_txt_enc, image_encoder=_img_enc,
    )
    tr = searcher.executor.text_retriever
    assert isinstance(tr, HybridTextRetriever)
    hits = tr.search("red flower", top_k=3)
    assert hits and len({(h.doc_id, h.result_modality) for h in hits}) == len(hits)


def test_mock_without_encoders_falls_back_to_bm25(data):
    """Smoke runs must never download CLIP: unified -> bm25, image off."""
    queries, corpus = data
    searcher = build_search_stack(_cfg(), queries, corpus, mock=True)
    assert isinstance(searcher.executor.text_retriever, BM25Retriever)
    assert searcher.executor.image_retriever is None
    assert searcher.proposer.enable_image_actions is False


def test_gate_availability_includes_image_search(data):
    queries, corpus = data
    cfg = _cfg()
    cfg["action_gate"] = {"enabled": True, "mode": "dataset", "max_actions": 3}
    searcher = build_search_stack(cfg, queries, corpus, mock=True,
                                  text_encoder=_txt_enc, image_encoder=_img_enc)
    gate = searcher.proposer
    assert isinstance(gate, GatedProposer)
    assert gate.available_actions == {"text_search", "image_search", "answer"}


def test_invalid_backend_and_dedupe_raise(data):
    queries, corpus = data
    with pytest.raises(ValueError):
        build_search_stack(_cfg(backend="dense"), queries, corpus, mock=True)
    with pytest.raises(ValueError):
        build_search_stack(_cfg(dedupe="banana"), queries, corpus, mock=True,
                           text_encoder=_txt_enc, image_encoder=_img_enc)


def _load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step1_and_inference_share_the_same_assembly():
    """Both entry points must call the ONE shared build_search_stack."""
    step1 = _load_module(
        "step1_gen_trajectories_asm",
        ROOT / "DatasetConstruct" / "step1_gen_trajectories.py",
    )
    run_inf = _load_module(
        "run_inference_asm", ROOT / "scripts" / "run_inference.py"
    )
    assert step1.build_search_stack is assembly.build_search_stack
    assert run_inf.build_search_stack is assembly.build_search_stack
    # Neither entry point keeps a private assembly path around.
    for mod in (step1, run_inf):
        assert not hasattr(mod, "build_searcher")
        leftovers = [n for n in vars(mod)
                     if n.startswith(("_build_local", "_clip_index",
                                      "_available_actions"))]
        assert not leftovers


def test_same_cfg_builds_identical_stacks(data):
    """Given the same cfg, two builds wire identically-typed components."""
    queries, corpus = data

    def fingerprint(searcher):
        ex = searcher.executor
        return (
            type(ex.text_retriever).__name__,
            type(ex.image_retriever).__name__,
            ex.top_k, ex.image_top_k,
            type(searcher.proposer).__name__,
            type(searcher.scorer).__name__,
            searcher.cfg,
        )

    a = build_search_stack(_cfg(), queries, corpus, mock=True,
                           text_encoder=_txt_enc, image_encoder=_img_enc)
    b = build_search_stack(_cfg(), queries, corpus, mock=True,
                           text_encoder=_txt_enc, image_encoder=_img_enc)
    assert fingerprint(a) == fingerprint(b)
