# ETBench-Open construction — remaining work

Working branch for finishing the dataset-construction side of the repository. This file
records where the pipeline stands and what is still missing before ETBench-Open can be
released as specified in [SCHEMA.md](SCHEMA.md). Tick items off here as they land.

## Current state (baseline: `b4d979a`)

What works today:

- **Steps 1–4 run end to end.** `python DatasetConstruct/run_pipeline.py --mock --force`
  produces 449 trajectories → 1147 step samples → 1132 kept (train 1089 / val 43) and
  `validate_dataset.py` reports the output CONFORMANT.
- **Test suite is green** (`pytest`, 96 tests, base requirements only).
- **Data preparation** exists for InfoSeek (queries, Wikipedia corpus, query and corpus
  images), GQA (30k balanced subset), and ScienceQA / MRAG-Bench / KVQA.
- **Validator** enforces the SCHEMA §2 invariants, including split purity and the
  `--require-provenance` release mode.

Gaps found while auditing against SCHEMA.md and the top-level README:

| # | Gap | Where | Spec reference |
| - | --- | --- | --- |
| G1 | `gen_provenance` is never written; release validation (`--require-provenance`) cannot pass | step 4 | SCHEMA §1.1, §1.4, §2.11 |
| G2 | Only `train` / `val` are produced; no `test` split | step 4 | SCHEMA §1.1, §3 |
| G3 | `stats.json` lacks the published statistics: histograms of `score` / `local_grounding` / `outcome_credit`, mean trajectory length, rationale QC pass rate, outcome positive rate **per answer type** (string vs. value) | step 4 | SCHEMA §4 |
| G4 | No release bundle: `corpus.jsonl` export, `image_urls.jsonl` + download script, `manifest.json` (line counts + sha256 + version), `DATASET_CARD.md`, `load_dataset.py` | new `export_release.py` | SCHEMA §3, §6 |
| G5 | No DPO preference-pair mining, although the README says the repository provides "the preference pairs" (action-typed, same-state contrastive pairs from the search trees; `unsupported_correct` answers as negatives) | new step / exporter | README §3, SCHEMA §1.1 |
| G6 | Step 4's trajectory-length rule counts samples that *survived step 3*, not the complete trajectory; a step-3 drop can shorten a trajectory below `min_steps` and wrongly discard it | step 4 | SCHEMA §2.10 |
| G7 | Mock runs never exercise `image_search` (no injected encoders), so the mixed-modality path of steps 2–4 has no end-to-end smoke coverage | mock benchmark / tests | README "Quickstart" |
| G8 | Non-InfoSeek benchmarks (GQA, ScienceQA, …) have data preparation but no documented pilot run or per-benchmark config; ScienceQA's multiple-choice gold answers need an outcome rule that accepts the option letter or the option text | configs, `eval/metrics.py` | README "Reproducing" |

## Plan

Ordered so each stage leaves the pipeline runnable and the validator green.

### Stage 1 — Close the step-4 spec gaps
- [ ] G6: take trajectory length from the step-1 trajectories file, not from step-3 survivors.
- [ ] G2: deterministic query-level three-way split (`quality.val_fraction`, new
      `quality.test_fraction`), still hashed on `query_id`.
- [ ] G3: extend `stats.json` to the SCHEMA §4 list; per-answer-type positive rate uses
      the same string/value distinction as `metrics.exact_match`.
- [ ] G1: attach `gen_provenance` in step 4 from the resolved config (policy model and
      deployment, retriever, rationale model, MCTS block, outcome metric, seed,
      `git rev-parse HEAD`, `dataset_version`).
- [ ] Tests covering each of the above on the mock pipeline.

### Stage 2 — Preference pairs for ET-PRM DPO
- [ ] G5: mine same-state contrastive pairs from step-2/3 samples — siblings sharing
      `state.actions_before` with a score margin above a configurable threshold, grouped by
      action type; `unsupported_correct` answer steps paired against supported ones.
- [ ] Write `pairs_{train,val,test}.jsonl` inside the existing query-disjoint splits and
      document the record format in SCHEMA.md.

### Stage 3 — Release bundle
- [ ] G4: `DatasetConstruct/export_release.py` assembling the SCHEMA §3 layout, with
      `manifest.json` checksums and a `DATASET_CARD.md` pre-filled from `stats.json` and
      `gen_provenance`.
- [ ] Image redistribution: emit `image_urls.jsonl` plus a download script instead of image
      bytes for Wikipedia / OVEN sources.
- [ ] `load_dataset.py` (HF `datasets` loader whose `Features` match SCHEMA §1.1).
- [ ] `validate_dataset.py --require-provenance` passes on the exported bundle.

### Stage 4 — Coverage and real runs
- [ ] G7: mock encoders so the mock pipeline emits `image_search` steps with image-side
      evidence.
- [ ] G8: per-benchmark configs and a documented `--n 100` pilot for GQA and ScienceQA;
      multiple-choice outcome rule.
- [ ] Real InfoSeek pilot (`--n 100`), read `stats.json`, tune verifier bands, then scale up.

## Working conventions

- Every change keeps `pytest` and `run_pipeline.py --mock --force` +
  `validate_dataset.py` green.
- Algorithms go in `src/evidencetree/`; `DatasetConstruct/` only orchestrates.
- Generated data stays under `data/` (gitignored); never commit outputs or `.env`.
