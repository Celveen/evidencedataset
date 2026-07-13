"""Benchmark loaders.

Benchmarks are normalized to a shared pair of JSONL files:

* ``queries.jsonl`` with question, answer, and optional image fields.
* ``corpus.jsonl`` with retrievable text documents.

InfoSeek keeps its historical ``infoseek_queries.jsonl`` /
``infoseek_corpus.jsonl`` filenames. Two modes:

* ``mock=True``  -> a tiny synthetic, self-contained dataset. No download, no
  GPU. The corpus is constructed so BM25 can actually retrieve the supporting
  document, which lets the smoke test exercise the full pipeline with signal.
* ``mock=False`` -> load real InfoSeek from a local path / HuggingFace
  ``datasets``. Raises a clear error if the data is not available so the failure
  is obvious on the GPU server.

Later stages (E-VQA / MRAG-Bench / OK-VQA) will add loaders here (Stage 7).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A retrievable corpus item (text, optionally with an associated image)."""

    doc_id: str
    text: str
    title: str = ""
    image_path: str | None = None


@dataclass
class Query:
    """A benchmark question with gold answer(s)."""

    query_id: str
    question: str
    gold_answers: list[str]
    image_path: str | None = None
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_infoseek(
    n: int = 1000,
    mock: bool = False,
    data_dir: str | Path | None = None,
    seed: int = 0,
) -> tuple[list[Query], list[Document]]:
    """Load (queries, corpus) for InfoSeek.

    Args:
        n: number of queries to load.
        mock: if True, return a synthetic self-contained dataset.
        data_dir: directory holding ``infoseek_queries.jsonl`` and
            ``infoseek_corpus.jsonl`` (real mode). Defaults to
            ``data/corpus/infoseek``.
        seed: RNG seed for mock data / subsampling.

    Returns:
        ``(queries, corpus)``.
    """
    if mock:
        return _mock_infoseek(n=n, seed=seed)
    return _load_infoseek_real(n=n, data_dir=data_dir, seed=seed)


def load_benchmark(
    name: str,
    n: int = 1000,
    mock: bool = False,
    data_dir: str | Path | None = None,
    seed: int = 0,
) -> tuple[list[Query], list[Document]]:
    """Load any benchmark converted to the shared JSONL format."""
    normalized = name.lower().replace("-", "").replace("_", "")
    if mock or normalized == "infoseek":
        return load_infoseek(n=n, mock=mock, data_dir=data_dir, seed=seed)
    if data_dir is None:
        raise ValueError(f"data.data_dir is required for benchmark {name!r}.")
    return _load_jsonl_benchmark(n=n, data_dir=data_dir, seed=seed)


# --------------------------------------------------------------------------- #
# Real loader
# --------------------------------------------------------------------------- #
def _load_infoseek_real(
    n: int, data_dir: str | Path | None, seed: int
) -> tuple[list[Query], list[Document]]:
    data_dir = Path(data_dir) if data_dir else Path("data/corpus/infoseek")
    queries_path = data_dir / "infoseek_queries.jsonl"
    corpus_path = data_dir / "infoseek_corpus.jsonl"

    if not queries_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            "Real InfoSeek data not found.\n"
            f"  Expected: {queries_path}\n"
            f"            {corpus_path}\n"
            "On the GPU server, download InfoSeek + its Wikipedia corpus and write\n"
            "them as JSONL (one object per line):\n"
            '  queries: {"query_id","question","gold_answers":[...],"image_path"?}\n'
            '  corpus:  {"doc_id","text","title"?,"image_path"?}\n'
            "Or run with --mock for a local smoke test."
        )

    corpus = [
        Document(
            doc_id=str(obj["doc_id"]),
            text=obj["text"],
            title=obj.get("title", ""),
            image_path=obj.get("image_path"),
        )
        for obj in _read_jsonl(corpus_path)
    ]

    queries = [
        Query(
            query_id=str(obj["query_id"]),
            question=obj["question"],
            gold_answers=_as_str_list(obj["gold_answers"]),
            image_path=obj.get("image_path"),
            metadata=obj.get("metadata", {}),
        )
        for obj in _read_jsonl(queries_path)
    ]

    if n < len(queries):
        rng = random.Random(seed)
        queries = rng.sample(queries, n)
    return queries, corpus


def _load_jsonl_benchmark(
    n: int, data_dir: str | Path, seed: int
) -> tuple[list[Query], list[Document]]:
    data_dir = Path(data_dir)
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"
    if not queries_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"Converted benchmark data not found in {data_dir}.\n"
            f"  Expected: {queries_path}\n"
            f"            {corpus_path}"
        )

    corpus = [
        Document(
            doc_id=str(obj["doc_id"]),
            text=obj["text"],
            title=obj.get("title", ""),
            image_path=obj.get("image_path"),
        )
        for obj in _read_jsonl(corpus_path)
    ]
    queries = [
        Query(
            query_id=str(obj["query_id"]),
            question=obj["question"],
            gold_answers=_as_str_list(obj["gold_answers"]),
            image_path=obj.get("image_path"),
            metadata=obj.get("metadata", {}),
        )
        for obj in _read_jsonl(queries_path)
    ]
    if n < len(queries):
        queries = random.Random(seed).sample(queries, n)
    return queries, corpus


# --------------------------------------------------------------------------- #
# Mock loader (synthetic, self-contained)
# --------------------------------------------------------------------------- #
# A small factual knowledge base. Each fact yields one query whose supporting
# document is retrievable from the corpus via BM25 (shared salient tokens).
_MOCK_FACTS: list[dict[str, str]] = [
    {"entity": "Eiffel Tower", "attr": "city", "value": "Paris",
     "fact": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France."},
    {"entity": "Mount Fuji", "attr": "country", "value": "Japan",
     "fact": "Mount Fuji is the tallest mountain in Japan and an active stratovolcano."},
    {"entity": "Great Barrier Reef", "attr": "country", "value": "Australia",
     "fact": "The Great Barrier Reef is the world's largest coral reef system off the coast of Australia."},
    {"entity": "Statue of Liberty", "attr": "city", "value": "New York",
     "fact": "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York."},
    {"entity": "Colosseum", "attr": "city", "value": "Rome",
     "fact": "The Colosseum is an ancient amphitheatre in the centre of Rome, Italy."},
    {"entity": "Taj Mahal", "attr": "country", "value": "India",
     "fact": "The Taj Mahal is an ivory-white marble mausoleum located in Agra, India."},
    {"entity": "Sydney Opera House", "attr": "country", "value": "Australia",
     "fact": "The Sydney Opera House is a multi-venue performing arts centre in Australia."},
    {"entity": "Christ the Redeemer", "attr": "city", "value": "Rio de Janeiro",
     "fact": "Christ the Redeemer is an Art Deco statue overlooking Rio de Janeiro, Brazil."},
    {"entity": "Big Ben", "attr": "city", "value": "London",
     "fact": "Big Ben is the nickname for the Great Bell of the clock tower in London."},
    {"entity": "Machu Picchu", "attr": "country", "value": "Peru",
     "fact": "Machu Picchu is a 15th-century Inca citadel located in southern Peru."},
    {"entity": "Sagrada Familia", "attr": "city", "value": "Barcelona",
     "fact": "The Sagrada Familia is a large unfinished basilica in Barcelona, designed by Gaudi."},
    {"entity": "Golden Gate Bridge", "attr": "city", "value": "San Francisco",
     "fact": "The Golden Gate Bridge is a suspension bridge spanning the strait at San Francisco."},
    {"entity": "Petra", "attr": "country", "value": "Jordan",
     "fact": "Petra is a famous archaeological site in southern Jordan, carved into rose-red cliffs."},
    {"entity": "Acropolis", "attr": "city", "value": "Athens",
     "fact": "The Acropolis is an ancient citadel on a rocky outcrop above the city of Athens."},
    {"entity": "Brandenburg Gate", "attr": "city", "value": "Berlin",
     "fact": "The Brandenburg Gate is an 18th-century neoclassical monument in Berlin."},
    {"entity": "Niagara Falls", "attr": "country", "value": "Canada",
     "fact": "Niagara Falls straddles the border between Ontario, Canada and New York."},
    {"entity": "Stonehenge", "attr": "country", "value": "England",
     "fact": "Stonehenge is a prehistoric monument of standing stones in Wiltshire, England."},
    {"entity": "Burj Khalifa", "attr": "city", "value": "Dubai",
     "fact": "The Burj Khalifa is the tallest building in the world, located in Dubai."},
    {"entity": "Angkor Wat", "attr": "country", "value": "Cambodia",
     "fact": "Angkor Wat is a vast Hindu-Buddhist temple complex in Cambodia."},
    {"entity": "Pyramids of Giza", "attr": "country", "value": "Egypt",
     "fact": "The Pyramids of Giza are ancient monumental tombs on the Giza plateau in Egypt."},
]

# Several paraphrase templates per attribute so the mock set can grow to
# len(facts) * len(templates) queries — enough samples (~100) for the smoke
# run's correlation estimate to sit near its configured target.
_ATTR_TO_QUESTIONS = {
    "city": [
        "In which city is the {entity} located?",
        "Which city is home to the {entity}?",
        "The {entity} can be found in which city?",
        "Name the city where the {entity} stands.",
        "What city hosts the {entity}?",
    ],
    "country": [
        "In which country is the {entity} located?",
        "Which country is home to the {entity}?",
        "The {entity} can be found in which country?",
        "Name the country where the {entity} is found.",
        "What country hosts the {entity}?",
    ],
}

# Filler documents (distractors) to make retrieval non-trivial.
_MOCK_DISTRACTORS = [
    "The history of cartography spans thousands of years across many cultures.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The stock market reflects investor sentiment about future earnings.",
    "Classical music flourished in Europe during the 18th and 19th centuries.",
    "Tectonic plates move slowly over the Earth's semi-fluid mantle.",
]


def _mock_infoseek(n: int, seed: int) -> tuple[list[Query], list[Document]]:
    rng = random.Random(seed)
    corpus: list[Document] = []
    queries: list[Query] = []

    n_templates = min(len(v) for v in _ATTR_TO_QUESTIONS.values())
    for i, fact in enumerate(_MOCK_FACTS):
        corpus.append(
            Document(
                doc_id=f"doc_{i}",
                title=fact["entity"],
                text=f"{fact['entity']}. {fact['fact']}",
            )
        )
        for v in range(n_templates):
            question = _ATTR_TO_QUESTIONS[fact["attr"]][v].format(entity=fact["entity"])
            queries.append(
                Query(
                    query_id=f"q_{i}_{v}",
                    question=question,
                    gold_answers=[fact["value"]],
                    metadata={"supporting_doc": f"doc_{i}", "attr": fact["attr"]},
                )
            )

    for j, text in enumerate(_MOCK_DISTRACTORS):
        corpus.append(Document(doc_id=f"distractor_{j}", text=text))

    rng.shuffle(queries)
    if n < len(queries):
        queries = queries[:n]
    return queries, corpus


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_str_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]
