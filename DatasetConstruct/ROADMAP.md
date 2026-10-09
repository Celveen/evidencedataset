# ETBench-Open construction — status and remaining work

Tracks the build of ETBench-Open on the three benchmarks of the paper. The design
is in [DESIGN.md](DESIGN.md); tick items off here as they land.

## Current state (pipeline v2)

The six-step pipeline runs end to end offline: `python DatasetConstruct/run_pipeline.py
--mock` builds a CONFORMANT mock split from a synthetic image-bearing world (140 queries
→ 1,227 trajectories → 4,036 step records → 2,747 unique nodes → 420 sampled, with
text_search, image_search and answer nodes, support judging, rationales, pairs and
stats). The unit tests cover every component, including the data preparation
scripts and the InfoSeek fetch kit with the network replaced by local fakes. No real
data has been generated yet: the construction server holds no data or models.

Closed from the v1 audit:

| # | Gap | Resolution |
| - | --- | --- |
| G1 | `gen_provenance` never written | step 6 attaches it; the validator runs in release mode |
| G2 | no `test` split | decided: the benchmark evaluation sets are the held-out test sets; `quality.test_fraction` exists if one is wanted |
| G3 | `stats.json` incomplete | step 6 writes the SCHEMA §4 report |
| G5 | no DPO pairs | step 6 mines same-state pairs per split (6 types) |
| G6 | length rule counted post-filter survivors | step 3 filters on the full trajectory length (`traj_len`) |
| G7 | mock never exercised image_search | mock world with images + deterministic stand-in CLIP encoders |
| G8 | no per-benchmark configs / MC outcome rule | `configs/{infoseek,scienceqa,gqa}.yaml`; `metrics.outcome_match` |

Also new in v2: evaluation holdout with committed id lists and a loader guard; 60K-entity
InfoSeek corpus from Wiki6M with hard negatives; hybrid retrieval with a fast sparse BM25;
windowed CLIP grounding with raw cosines stored; node collapse; budgeted sibling-group
sampling; support judge on sampled correct answers only, with titles and image evidence;
marginal-gain field; preflight and audit tools; vLLM multi-server policy serving.

## Remaining work

### Stage A — Data and environment
- [ ] Server environment: venv, `pip install -r requirements/server.txt`, `pip install -e .`
- [ ] `data_prep/download_assets.sh` — Qwen2.5-VL-7B, CLIP ViT-B/32, GQA, ScienceQA
- [ ] InfoSeek fetch kit on a machine with internet → copy → `import_infoseek_kit.py`
- [ ] GQA `sceneGraphs.zip` via the kit → `prepare_gqa.py`
- [ ] `prepare_scienceqa.py`
- [ ] Commit `DatasetConstruct/splits/<benchmark>/` (the evaluation holdouts)
- [ ] New judge / rationale models and API keys in `configs/base.yaml` and `.env`

### Stage B — Pilot and calibration (per benchmark, 100 seeds)
- [ ] `preflight.py --check-apis` passes
- [ ] Pilot run in a separate output tree; read `stats.json`
- [ ] Refit grounding bands from `grounding_detail.raw_cos` (DESIGN §7); rerun steps 2–6
- [ ] Audit sheet of 100 queries read and signed off
- [ ] Measure step-1 throughput and API tokens; confirm the sampling budget

### Stage C — Full builds
- [ ] InfoSeek (20K seeds)
- [ ] ScienceQA (~17K seeds)
- [ ] GQA (20K seeds)
- [ ] Audit of 100 queries per full build

### Stage D — Release bundle (G4)
- [ ] `export_release.py`: merge benchmarks, `corpus.jsonl` export, `manifest.json`
      (line counts + sha256 + version), `DATASET_CARD.md` from `stats.json` + provenance
- [ ] Image redistribution: `image_urls.jsonl` + download script for Wikipedia / OVEN images
- [ ] `load_dataset.py` (HF `datasets` loader whose `Features` match SCHEMA §1.1)

## Working conventions

- Every change keeps `pytest` and `run_pipeline.py --mock` green (the mock build must
  validate CONFORMANT).
- Algorithms go in `src/evidencetree/`; `DatasetConstruct/` only orchestrates.
- Generated data stays under `data/` (gitignored); never commit outputs or `.env`.
  The split id lists under `DatasetConstruct/splits/` are committed.
