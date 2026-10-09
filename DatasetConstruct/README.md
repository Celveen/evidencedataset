# DatasetConstruct — ETBench-Open construction pipeline

ETBench-Open is the step-level dataset used to train ET-PRM. This directory turns
each benchmark's own corpus (InfoSeek, ScienceQA, GQA) into labeled
`(state, action, score, rationale)` records: a policy VLM runs MCTS over the typed
action space `{text_search, image_search, answer}`, and the tree nodes are
labeled, sampled, judged and explained. Retrieval never leaves the local index.

The design and its rationale are in [DESIGN.md](DESIGN.md); record formats in
[SCHEMA.md](SCHEMA.md); progress in [ROADMAP.md](ROADMAP.md).

```text
queries.jsonl (seeds) + corpus.jsonl            eval_queries.jsonl is never read here
  Step 1  MCTS rollouts, every rollout kept          -> trajectories.jsonl   (no API)
  Step 2  CLIP grounding + tree-level credit          -> labeled.jsonl        (no API)
  Step 3  length/gap filters, node collapse,
          budgeted sibling-group sampling             -> sampled.jsonl        (no API)
  Step 4  answer-support judge, sampled answers only  -> scored.jsonl         (API)
  Step 5  rationale + 5-rule QC                       -> rationales.jsonl     (API)
  Step 6  splits, provenance, pairs, stats, validation
                    -> data/etbench_open/<benchmark>/{train,val}.jsonl, pairs_*.jsonl, stats.json
```

Intermediate files live in `data/construction/<benchmark>/`. The algorithms live
in the main package (`src/evidencetree/prm/`, `mcts/`, `data/`); this directory
only orchestrates. Step 1 assembles its search stack with
[`build_search_stack`](../src/evidencetree/pipeline/assembly.py), the same function
`scripts/run_inference.py` calls, so construction and inference cannot drift apart.

## Quickstart

```bash
# Smoke test: synthetic image-bearing world, stand-in encoders, template
# rationales — zero API calls, no downloads
python DatasetConstruct/run_pipeline.py --mock

# Real runs (after data preparation, below)
python DatasetConstruct/preflight.py --benchmark infoseek        # checks, generates nothing
python DatasetConstruct/run_pipeline.py --benchmark infoseek --n 100 \
    --set output.dir=data/pilot/construction --set output.dataset_dir=data/pilot/etbench_open
python DatasetConstruct/audit_report.py --file data/pilot/construction/infoseek/rationales.jsonl
python DatasetConstruct/run_pipeline.py --benchmark infoseek     # full build

python DatasetConstruct/run_pipeline.py --benchmark gqa --steps 1,2,3   # free steps only
python DatasetConstruct/run_pipeline.py --config DatasetConstruct/configs/scienceqa.yaml
python DatasetConstruct/run_pipeline.py --set sampling.budget_per_query=2   # dotted overrides
```

Each step is also a standalone script (`step1_gen_trajectories.py` …
`step6_finalize.py`, same flags). Steps 1, 2, 4 and 5 resume from partial output;
step 3 keeps an existing sample unless `--force`, so the paid steps never see
their input change. Mock outputs go to `data/construction/mock/` and
`data/etbench_open/mock/`.

## Data preparation

Every benchmark is normalized to the same layout; image paths inside the files are
relative to the benchmark directory.

```text
data/corpus/<benchmark>/
├── queries.jsonl        # construction seeds: query_id, question, gold_answers, image_path, metadata
├── eval_queries.jsonl   # held-out evaluation questions (never used to build data)
├── corpus.jsonl         # doc_id, title, text, optional image_path, entity_id
├── images/
├── unified_clip_index/  # cached CLIP index, built on the first run
└── bm25_index/          # cached BM25 index, built on the first run
DatasetConstruct/splits/<benchmark>/{seed_ids,eval_ids}.txt, split_report.json   # committed
```

| Benchmark | Source | Command |
| --- | --- | --- |
| models, GQA, ScienceQA | ModelScope (reachable from the server) | `bash DatasetConstruct/data_prep/download_assets.sh` |
| InfoSeek | GCS / HuggingFace / Wikimedia (not reachable) | on a machine with internet: `python DatasetConstruct/fetch_kit/fetch_infoseek.py all --kit ~/infoseek_kit`, copy the kit, then `python DatasetConstruct/data_prep/import_infoseek_kit.py --kit <kit>` — see [fetch_kit/README.md](fetch_kit/README.md) |
| ScienceQA | — | `python DatasetConstruct/data_prep/prepare_scienceqa.py --src data/raw/ScienceQA` |
| GQA | + `sceneGraphs.zip` from `fetch_infoseek.py gqa-scenegraphs` | `python DatasetConstruct/data_prep/prepare_gqa.py --src data/raw/GQA --scene-graphs <zip>` |

Each preparation script selects the ~2K held-out evaluation questions and the
~20K construction seeds with `evidencetree.data.splits` (deterministic; the
leakage rules are in DESIGN.md §4) and writes the id lists under `splits/`.

## Policy VLM, judge, rationale model, API keys

```bash
cp DatasetConstruct/site.env.example DatasetConstruct/site.env   # weights dir, image, GPUs
bash DatasetConstruct/serve_policy_vllm.sh      # step 1: one policy server per GPU, ports 8100+
bash DatasetConstruct/serve_rationale_vllm.sh   # steps 4-5: local judge/rationale model, port 8020
```

List every server under `policy.base_url` in `configs/base.yaml`. If vLLM cannot be
installed, `serve_qwen25vl.py` is a (much slower) transformers-based fallback server. Models and
endpoints are changed only in the config: `policy`, `verifier.support.generation`
and `rationale.generation` each take `provider` (`openai` covers any
OpenAI-compatible endpoint, `anthropic` for Claude), `model`, `base_url` and
`api_key_env` (the name of the variable holding the key).

Ready endpoint blocks live in `configs/providers/` (`kimi_code`,
`kimi_code_anthropic`, `moonshot`, `deepseek`, `local_qwen3_coder`); select one with
`--provider` (steps 4/5, `try_rationale_model.py`) or `--rationale-provider` /
`--judge-provider` (`run_pipeline.py`, `preflight.py`) instead of editing `base.yaml`.

Steps 4 and 5 need only their input file and an API key, so they can run on a
machine that reaches the provider when the server cannot (Moonshot/Kimi are not
reachable from the construction server): `--input` / `--output` take explicit
paths, and step 6 records the models from the rows. Commented Kimi settings (both
the OpenAI-protocol Moonshot platform and an Anthropic-protocol Kimi Code key) are
in `configs/base.yaml`; try a model with `try_rationale_model.py --n 3` first.
See DESIGN.md §9.

Copy `.env.example` to `DatasetConstruct/.env` and fill it in. `.env` is
gitignored; already-exported variables take precedence over the file.

| Variable | Used by |
| --- | --- |
| `LOCAL_QWEN_API_KEY` | step 1 policy (vLLM `--api-key`; any value, same on both sides) |
| `DEEPSEEK_API_KEY` | step 4 support judge, step 5 rationales (default provider) |
| `MOONSHOT_API_KEY` / `KIMI_CODE_API_KEY` | optional: Kimi for step 4 / 5 |

## Configuration

`configs/base.yaml` holds the shared settings; `configs/{infoseek,scienceqa,gqa}.yaml`
inherit it (`_base_`) and add their data paths. The knobs that matter:

| Key | Default | Meaning |
| --- | --- | --- |
| `data.n_queries` / `seed_ids` / `eval_ids` | 20000 / splits files | seeds per benchmark; the id lists enforce the holdout |
| `mcts.rollouts` / `max_depth` / `top_k_children` | 10 / 3 / 3 | P, d_max, k (paper §5.1) |
| `mcts.expansion` | `diverse` | keep one child per proposed action type first |
| `step1.concurrency` | 32 | queries searched in parallel |
| `retriever.backend` / `top_k` / `image.top_k` | `hybrid` / 3 / 1 | BM25 + CLIP; text top-3, image top-1 |
| `verifier.window_words` | 48 | grounding scores chunks as the max over windows |
| `verifier.{text,image}_cos_lo/hi` | 0.5/0.9, 0.15/0.32 | calibration bands — refit from the pilot |
| `verifier.alpha`, `support.floor` | 0.5, 0.3 | Eq. 5 and Eq. 6 |
| `quality.min_steps` / `max_steps` / `max_grounding_outcome_gap` | 2 / 8 / 0.7 | drop rules |
| `sampling.budget_per_query` / `max_per_query` | 3 / 9 | sampled nodes = 3 × seeds; the main cost knob |
| `rationale.evidence_chars` | 300 / 1500 / 1500 | prompt truncation (earlier evidence, answer steps, observation) |
| `rationale.batch` / `max_group` | `group` / 6 | siblings (same state) share one request; `node` = one request per sample |
| `pairs.margin` | 0.3 | minimum score gap of a preference pair |
| `quality.val_fraction` / `test_fraction` | 0.05 / 0 | query-disjoint splits |

## Outputs and inspection

```bash
python DatasetConstruct/inspect_trajectories.py data/construction/infoseek/scored.jsonl --n 5
python DatasetConstruct/inspect_trajectories.py <file> --query <query_id> --action image_search --full
python DatasetConstruct/audit_report.py --benchmark infoseek --n 100          # Markdown audit sheet
python DatasetConstruct/validate_dataset.py data/etbench_open/infoseek/train.jsonl \
    data/etbench_open/infoseek/val.jsonl --require-provenance
```

`stats.json` reports per-split counts, action types, mean trajectory length, score /
grounding / credit histograms, outcome positive rate per answer type, support labels,
rationale QC, drops per rule, pair counts, the validation result and the provenance.
Read it before every scale-up (DESIGN.md §6 lists what to look for).

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Step N output not found` | run the previous step; keep `--mock` consistent across steps |
| `Prepared benchmark data not found` | run the benchmark's data preparation, or use `--mock` |
| `refusing to load them for construction` | the seeds file contains evaluation ids — rerun data preparation |
| `data.seed_ids and data.eval_ids are required` | the split id files are missing (`require_splits`) |
| `image_search disabled for ...` | queries lack `image_path` or no unified index exists |
| empty rationales, budget consumed | reasoning models spend the budget on thinking tokens; keep `extra_body.thinking: {type: disabled}` |
| step 5 exits on transient-failure rationales | rerun step 5; those rows are regenerated |
| `Ignoring stale unified CLIP index` | the corpus changed; the index is rebuilt automatically |
