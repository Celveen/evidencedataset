# DatasetConstruct — ETBench-Open construction pipeline

ETBench-Open is the step-level dataset used to train ET-PRM. This directory turns a
benchmark's own corpus into labeled `(state, action, score, rationale)` records: a policy
VLM runs MCTS over the typed action space `{text_search, image_search, answer}`, and every
step of every rollout is scored and explained. Retrieval never leaves the local index.

```text
queries.jsonl + corpus.jsonl
  Step 1  policy VLM drives MCTS; every rollout is kept, not just the best one
     → <name>_trajectories.jsonl
  Step 2  local grounding (unified CLIP) + tree-level outcome credit + answer-support gate
     → <name>_scored.jsonl        (one step-level sample per step)
  Step 3  strong LLM writes a rationale for the score; 5-rule QC with regeneration
     → <name>_rationales.jsonl
  Step 4  quality filter + query-disjoint train/val split
     → etbench_open/{train,val}.jsonl + stats.json
```

The algorithms live in the main package (`src/evidencetree/prm/{verifiers,data_gen,rationale_gen}.py`,
`src/evidencetree/mcts/`); this directory only orchestrates. Step 1 assembles its search
stack with [`build_search_stack`](../src/evidencetree/pipeline/assembly.py) — the same
function `scripts/run_inference.py` calls — so construction and inference trees cannot
drift apart. Record formats are specified in [SCHEMA.md](SCHEMA.md).

## Quickstart

```bash
# Smoke test: synthetic data, template rationales, zero API calls, no downloads
python DatasetConstruct/run_pipeline.py --mock

# Real run — needs prepared data and API keys (below). Always pilot a small run first.
python DatasetConstruct/run_pipeline.py --n 100
python DatasetConstruct/run_pipeline.py --config DatasetConstruct/config.qwen25vl.yaml --n 100

python DatasetConstruct/run_pipeline.py --steps 3,4             # a subset of steps
python DatasetConstruct/run_pipeline.py --force                 # ignore existing output
python DatasetConstruct/run_pipeline.py --set mcts.rollouts=4   # dotted-key overrides
python DatasetConstruct/run_pipeline.py --seed 1                # replicate seed (paper: 0/1/2)
```

Mock mode loads a synthetic InfoSeek-shaped benchmark, forces the policy and rationale
backends to `mock`, falls back to BM25 retrieval (no CLIP download), and uses the offline
lexical verifiers. Its outputs take a `mock_` filename prefix and Step 4 writes to
`<dataset_dir>/mock/`, so a smoke run never touches real data.

Each step is also a standalone script (`step1_gen_trajectories.py` … `step4_quality_filter.py`,
same flags), and all steps resume from partial output: Step 1 skips `query_id`s already
written, Steps 2-4 skip `sample_id`s already written. Rerunning an interrupted command
continues where it stopped.

## Data preparation

| Script | Benchmark | Produces |
| --- | --- | --- |
| `data_prep/prepare_infoseek.py` | InfoSeek | `infoseek_queries.jsonl` from the official annotations |
| `data_prep/build_corpus.py` | InfoSeek | `infoseek_corpus.jsonl` — Wikidata entity → Wikipedia plaintext → ~1200-character chunks |
| `data_prep/download_infoseek_val_images.py` | InfoSeek | OVEN query images for the validation split (shard by shard, low peak disk) |
| `data_prep/prepare_infoseek_corpus_images.py` | InfoSeek | corpus-side Wikipedia images; attaches `image_path` to chunks and invalidates the cached index |
| `data_prep/prepare_gqa_30k.py` | GQA | 30k balanced-train subset: `queries.jsonl`, `corpus.jsonl` (scene-graph text), `stats.json` |
| `data_prep/download_gqa_assets.sh` | GQA | official questions / scene graphs / images archives |
| `data_prep/prepare_other_datasets.py` | ScienceQA, MRAG-Bench, KVQA | `queries.jsonl`, `corpus.jsonl`, `assets/` |

Every benchmark is normalized to the same gitignored layout. InfoSeek keeps the historical
filenames `infoseek_queries.jsonl` / `infoseek_corpus.jsonl`.

```text
data/corpus/<benchmark>/
├── queries.jsonl        # query_id, question, gold_answers, image_path, metadata
├── corpus.jsonl         # doc_id, title, text, optional image_path / entity_id
├── assets/              # extracted query and page images
└── unified_clip_index/  # cached CLIP index, built on the first real run
```

InfoSeek raw annotations go in `data/corpus/infoseek/raw/`, from
`http://storage.googleapis.com/gresearch/open-vision-language/infoseek/<filename>`:
`infoseek_val.jsonl` (validation questions with `answer` / `answer_eval`),
`infoseek_val_withkb.jsonl` (query → Wikidata entity mapping, which drives corpus
building), and `infoseek_test.jsonl` (answers withheld, leaderboard only).

```bash
python DatasetConstruct/data_prep/prepare_infoseek.py --raw-dir data/corpus/infoseek/raw
python DatasetConstruct/data_prep/prepare_infoseek.py --images-dir /path/to/oven_images   # optional
python DatasetConstruct/data_prep/build_corpus.py --limit 50   # trial run; omit --limit for all entities
```

`data_prep/build_corpus.py` resolves each Wikidata id to its English Wikipedia title, fetches page
plaintext with backoff, caches pages in `raw/wiki_pages.jsonl` (resumable), and chunks them.
Corpus construction is a one-time offline step; the pipeline itself only queries the local
index. Queries whose image is unavailable get `image_path=null` and still run the text-only
path. Scripts that read from an external dataset disk take the source location as a flag —
pass your own paths rather than relying on the built-in defaults:

```bash
python DatasetConstruct/data_prep/prepare_gqa_30k.py --raw-root /path/to/GQA --out-dir data/corpus/gqa --n 30000
python DatasetConstruct/data_prep/prepare_other_datasets.py --dataset scienceqa \
    --source-root /path/to/datasets --output-root data/corpus --n 0   # --n 0 = all samples
```

To build data for another benchmark, override `benchmark`, `data.data_dir` and the four
`output.*` paths with `--set`.

## Policy VLM and API keys

`config.yaml` points `policy` at a hosted OpenAI-compatible endpoint;
`config.qwen25vl.yaml` points it at a local Qwen2.5-VL-7B-Instruct server, which is how the
released data is generated.

```bash
python DatasetConstruct/serve_qwen25vl.py --model /path/to/Qwen2.5-VL-7B-Instruct --port 8000
```

`--model` defaults to `models/Qwen2.5-VL-7B-Instruct` relative to the repository root, and
the server exposes `/v1/chat/completions`, `/v1/models` and `/health`.
`config.qwen25vl.yaml` keeps the Step 2 CLIP verifier and the retriever's image encoder on
CPU (`verifier.device`, `retriever.image.device`) so they do not compete with the served
model for GPU memory.

Copy `.env.example` to `DatasetConstruct/.env` and fill it in. `.env` is gitignored and
never committed; already-exported environment variables take precedence over the file. Only
variable *names* appear anywhere in the repository.

| Variable | Used by | Notes |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | Step 1 policy VLM | the hosted endpoint configured in `config.yaml` |
| `LOCAL_QWEN_API_KEY` | Step 1 policy VLM | the local server ignores the key; any non-empty placeholder works |
| `DEEPSEEK_API_KEY` | Step 2 answer-support judge, Step 3 rationale | |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | optional | only if you switch provider |

Models and endpoints are changed in the `policy`, `verifier.support.generation` and
`rationale.generation` blocks: `provider` (`openai` covers any OpenAI-compatible endpoint,
`anthropic` for Claude), `model`, `base_url`, and `api_key_env` (which variable to read).
Images attached to multimodal queries are uploaded as base64 automatically.

## Configuration

Meaningful knobs in [`config.yaml`](config.yaml); `--set key.subkey=value` overrides any of
them on the command line.

| Key | Default | Meaning |
| --- | --- | --- |
| `data.data_dir` / `n_queries` / `seed` | `data/corpus/infoseek` / 100 / 0 | input directory, query budget, sampling seed (`--seed` sets query *and* policy sampling seeds together) |
| `mcts.rollouts` | 10 | rollout budget *P* per query; upper bound on trajectories per query |
| `mcts.max_depth` / `top_k_children` | 3 / 3 | maximum actions per trajectory; candidates kept per expansion |
| `mcts.early_stop_q` | 2.0 | > 1 disables early stopping, so rollouts stay diverse |
| `retriever.backend` | `unified` | `unified` (one CLIP space for text and image units) \| `bm25` \| `hybrid` |
| `retriever.dedupe` | `entity` | `entity` \| `doc` \| `none` — collapse duplicate units before top-*k* truncation |
| `retriever.top_k` / `clip_model` | 5 / `clip-ViT-B-32` | results per search action; CLIP model backing the unified index |
| `retriever.image_search` | `true` | enable `image_search`; auto-disabled when queries carry no images, or in mock runs without injected encoders |
| `verifier.backend` | `clip` | `clip` (real grounding) \| `lexical` (offline fallback, forced in mock) |
| `verifier.{text,image}_cos_lo` / `_cos_hi` | 0.5 / 0.9, 0.15 / 0.32 | cosine bands mapping text- and image-side similarity onto `[0,1]` |
| `verifier.alpha` | 0.5 | fusion weight: `score = alpha*local + (1-alpha)*outcome` |
| `verifier.support.backend` | `api` | `api` \| `lexical` \| `off` — answer-support judge |
| `verifier.support.floor` / `unsupported_threshold` | 0.3 / 0.3 | answer-step floor `score = outcome*(floor + (1-floor)*support)`; `outcome >= 0.5` with `support <=` threshold marks `unsupported_correct` |
| `rationale.max_attempts` / `concurrency` | 3 / 8 | regenerations allowed on QC failure; parallel rationale requests |
| `quality.min_steps` / `max_steps` | 2 / 8 | trajectories outside this length are dropped whole |
| `quality.max_grounding_outcome_gap` | 0.7 | drop samples where \|local − outcome\| exceeds this (noisy label) |
| `quality.dedupe_within_query` / `val_fraction` | `true` / 0.05 | drop identical action sequences within a query; validation share, split by `query_id` |
| `prm.scorer` / `action_gate.enabled` | `overlap` / `false` | Step 1 rollout guidance signal (not a dataset label); optional per-benchmark action-type restriction |

## Outputs and inspection

Step outputs go to the paths in the `output` block: `<name>_trajectories.jsonl` (one
trajectory per line), `<name>_scored.jsonl` and `<name>_rationales.jsonl` (one step-level
sample per line), and `data/etbench_open/{train,val}.jsonl` plus `stats.json` with
kept/dropped counts, action-type distribution and mean scores.

```bash
# Render any of the four files as a compact, truncated summary
python DatasetConstruct/inspect_trajectories.py data/etbench_open/train.jsonl --n 5
python DatasetConstruct/inspect_trajectories.py <file> --query <query_id> --action image_search --full

# Deterministic, download-free trace of full trajectories on a mixed text/image scene
python scripts/demo_5vqa.py

# Check released files against SCHEMA.md (exit 0 = conformant)
python DatasetConstruct/validate_dataset.py data/etbench_open/train.jsonl data/etbench_open/val.jsonl
python DatasetConstruct/validate_dataset.py data/etbench_open/train.jsonl --require-provenance
```

`validate_dataset.py` takes one or more JSONL paths and checks field presence, score ranges,
the score-fusion identity, rationale QC, re-distributable image paths, and — given more than
one file — that no `query_id` appears in two splits. `--min-steps`, `--max-steps` and
`--max-gap` mirror the `quality` block.

Scale up in stages: `--mock`, then `--n 100`, then the full run. Step 1 issues roughly
`n_queries × rollouts` policy calls, Step 2 one judge call per answer step when
`verifier.support.backend: api`, and Step 3 about one strong-LLM call per sample plus
regenerations, so `stats.json` is worth reading before every increase. A high
`rationale_qc` drop rate points at the rationale model; a high `grounding_outcome_gap` drop
rate points at the verifier bands or the outcome metric.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Step N output not found` | Run the previous step, and keep `--mock` consistent across steps — mock outputs are separate `mock_`-prefixed files |
| `Real InfoSeek data not found` | `data.data_dir` must contain `infoseek_queries.jsonl` and `infoseek_corpus.jsonl`; run `prepare_infoseek.py` and `build_corpus.py`, or use `--mock` |
| `image_search` never fires | The log line `image_search disabled for ...` reports why: queries must carry `image_path`, and a unified index must exist |
| Empty rationales, budget consumed | Reasoning models can spend the whole token budget on thinking tokens; the generation blocks pass `extra_body.thinking: {type: disabled}` for this reason |
| Step 3 exits on "transient connection-error rationales" | Those rows are dropped automatically; rerun Step 3 and they are regenerated |
| `Ignoring stale unified CLIP index` | The cached index does not match the corpus size — delete `<data_dir>/unified_clip_index/` and let it rebuild |
