"""Regression: dataset construction records the FULL action space.

The mock benchmark carries no images, so the pipeline smoke test can never
exercise image_search — the exact blind spot behind the "constructed dataset
is 100% text_search" report. This test builds a tiny image-bearing world with
injected deterministic encoders (same device as scripts/demo_5vqa.py), runs
the search stack exactly as DatasetConstruct step1 does, and asserts that the
serialized trajectory rows contain image_search steps and evidence tagged with
BOTH result modalities.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

_DC = Path(__file__).resolve().parents[1] / "DatasetConstruct"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

import step1_gen_trajectories  # noqa: E402

from evidencetree.eval.benchmarks import Document, Query  # noqa: E402
from evidencetree.pipeline import build_search_stack  # noqa: E402

DIM = 8
_COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}


def _text_encoder(texts):
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for word, rgb in _COLORS.items():
            if word in str(text).lower():
                vecs[i, :3] += rgb
        vecs[i, 3 + (len(str(text)) % (DIM - 3))] += 0.3
    return vecs


def _image_encoder(images):
    vecs = np.zeros((len(images), DIM), dtype="float32")
    for i, image in enumerate(images):
        mean = np.asarray(image.convert("RGB"), dtype="float32").mean((0, 1)) / 255.0
        vecs[i, :3] = mean
    return vecs


@pytest.fixture()
def world(tmp_path):
    queries, corpus = [], []
    for i, (color, city) in enumerate(
        [("red", "Paris"), ("green", "Cairo"), ("blue", "Tokyo")]
    ):
        img = tmp_path / f"{color}.png"
        Image.new("RGB", (8, 8), tuple(int(255 * c) for c in _COLORS[color])).save(img)
        queries.append(Query(
            query_id=f"q{i}",
            question=f"Which city is linked to the {color} flag?",
            gold_answers=[city],
            image_path=str(img),
        ))
        corpus.append(Document(
            doc_id=f"doc_{i}", title=f"{color} flag",
            text=f"The {color} flag is flown in {city}.",
            image_path=str(img), entity_id=f"{color} flag",
        ))
    corpus.append(Document(doc_id="distractor", text="Tectonic plates move slowly."))
    return queries, corpus


def test_step1_trajectories_record_mixed_modalities(world):
    queries, corpus = world
    cfg = {
        "benchmark": "unittest",
        "retriever": {"backend": "unified", "dedupe": "entity", "top_k": 3,
                      "image_search": True, "image": {"top_k": 2}},
        "policy": {"backend": "mock", "mock_accuracy": 1.0, "mock_seed": 0},
        "mcts": {"rollouts": 5, "max_depth": 3, "top_k_children": 3,
                 "early_stop_q": 2.0, "seed": 0},
        "prm": {"scorer": "overlap"},
    }
    searcher = build_search_stack(
        cfg, queries, corpus, mock=True, benchmark="unittest",
        text_encoder=_text_encoder, image_encoder=_image_encoder,
    )

    rows = []
    for query in queries:
        result = searcher.search(query.question, image_path=query.image_path)
        for record in result.rollout_log:
            if record.state is not None:
                rows.append(
                    step1_gen_trajectories.trajectory_dict(query, record, len(rows))
                )

    assert rows, "no trajectories generated"
    json.dumps(rows)  # every row must be JSONL-serializable

    action_types = {s["action_type"] for r in rows for s in r["steps"]}
    assert "image_search" in action_types, (
        f"construction never records image_search; got {action_types}"
    )
    assert "text_search" in action_types

    modalities = {
        e["result_modality"]
        for r in rows for s in r["steps"] for e in s["evidence"]
        if e["result_modality"]
    }
    assert modalities == {"text", "image"}, (
        f"expected both evidence modalities in the dataset rows, got {modalities}"
    )

    image_steps = [
        s for r in rows for s in r["steps"] if s["action_type"] == "image_search"
    ]
    assert all(s.get("image_path") for s in image_steps), (
        "image_search steps must record the query image_path"
    )
