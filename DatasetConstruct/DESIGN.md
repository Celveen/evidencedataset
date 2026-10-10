# ETBench-Open construction design (v2)

This document is the design of record for building ETBench-Open, the step-level
dataset that trains ET-PRM. It replaces the first pipeline (four steps, one
benchmark at a time, every step record paid for) with a six-step pipeline built
around two goals: **label quality** and **cost that scales with what is kept,
not with what is generated**. Where this document and the older READMEs
disagree, this document and the paper win.

Record formats: [SCHEMA.md](SCHEMA.md). Commands: [README.md](README.md).
Progress: [ROADMAP.md](ROADMAP.md).

## 1. What changed and why

| Area | v1 | v2 | Why |
| --- | --- | --- | --- |
| Scope | InfoSeek pilot only | InfoSeek, ScienceQA, GQA, ~20K seeds each (paper §4) | the paper's three benchmarks |
| Evaluation holdout | none — construction sampled from the evaluation pool | fixed ~2K eval set per benchmark, committed id lists, loader refuses eval ids | leakage |
| InfoSeek corpus | ~1.8K gold entities fetched page by page from the Wikipedia API | 60K entities from the official Wiki6M KB: gold + same-category hard negatives + random fill | a corpus of only gold pages makes retrieval trivial and labels optimistic |
| Retrieval | CLIP-only unified index, top-5 | hybrid (BM25 over text + one CLIP index), text top-3, image top-1 (paper §3.2, §5.1) | CLIP's text tower reads 77 tokens, so ~200-word chunks were matched by their first sentence |
| Grounding (Eq. 3) | CLIP cosine of the chunk (truncated) | max over ~48-word windows, page title prefixed; raw cosine + modality stored | same truncation problem; stored raw cosines let the bands be refit without re-encoding |
| Step-1 guidance | overlap scorer gave every unanswered child prior 0 → expansion pruned search children in favour of answers | neutral prior for partial states; type-diverse expansion keeps one child per proposed action type first | tree diversity is the product |
| Answers | free-form, often sentences | short-answer prompt + normalization; per-benchmark official outcome rules, incl. ScienceQA multiple choice | exact-match false negatives poison outcome credit |
| Duplicates | one record per trajectory step — the same node appeared once per trajectory through it | one record per unique tree node | duplicate labels over-weight frequent nodes and multiply cost |
| Support judge | every answer step; evidence text only (no titles, image hits invisible) | only sampled answers that are correct; evidence shown with page titles and `[image] <title>` lines | cost; a chunk without its title often cannot be tied to its entity |
| Rationales | every step record | only sampled nodes; prompt with fixed prefix first, earlier evidence truncated for search steps | cost; provider prefix caching |
| Sampling | none | budgeted, query-balanced sampling of sibling groups (paper Table A4: 462K records → 60K) | the paper samples; siblings kept together for DPO |
| Release items | no provenance, no stats, no pairs | `gen_provenance`, SCHEMA §4 stats, same-state DPO pairs, validator run on every build | ROADMAP G1, G3, G5 |

## 2. Targets

Per benchmark (all numbers are config values, `DatasetConstruct/configs/`):

| | InfoSeek | ScienceQA | GQA |
| --- | --- | --- | --- |
| Seed pool | val (73,620) minus eval groups | train + validation (~17K) | train-balanced (~943K q / 72K images) |
| Seeds | 20,000 | all ~17K (the dataset is smaller than 20K) | 20,000, at most one per image |
| Held-out evaluation | 2,000 from val (1,000 unseen-question + 1,000 unseen-entity) | 2,000 from test | 2,000 from test-dev balanced |
| Corpus | 60K Wiki6M entities, ~1,000-char chunks + one image doc per entity | lectures (deduplicated) + hints | one doc per train-balanced image: image + scene-graph text |
| Rollouts | P = 10, d_max = 3, k = 3, no early stop | same | same |
| Sampled nodes | 3 × seeds ≈ 60K | ≈ 51K | ≈ 60K |

The paper's InfoSeek build (Table A4b) went 20K queries → 153,291 trajectories →
462,409 step records → 60,000 sampled → 58,994 released. v2 reproduces that
funnel per benchmark; `sampling.budget_per_query` is the main cost knob.

## 3. Data acquisition

The construction server reaches ModelScope, the Aliyun PyPI mirror and
DeepSeek directly; it cannot reach GitHub, HuggingFace, Google Cloud Storage,
Wikimedia or Moonshot/Kimi, and a proxy is not an option.

Machine-specific settings — where the weights live, which vLLM image to run,
which GPUs each server may use and which are reserved for other jobs — live in
`DatasetConstruct/site.env` (gitignored; see `site.env.example`), never in the
repository. The models are served locally with vLLM in containers:
`serve_policy_vllm.sh` (the step-1 policy, one server per GPU, ports 8100+) and
`serve_rationale_vllm.sh` (the step-4/5 judge and rationale model,
tensor-parallel, port 8020). When both need the same GPUs they take turns:
step 1 for every benchmark first, then steps 4–5.

| Source | Where | How |
| --- | --- | --- |
| Qwen2.5-VL-7B-Instruct, CLIP ViT-B/32 | ModelScope | `data_prep/download_assets.sh models` |
| GQA questions + images (lmms-lab/GQA) | ModelScope | `data_prep/download_assets.sh gqa` (~10.5 GB) |
| GQA `sceneGraphs.zip` | Stanford | fetch kit `gqa-scenegraphs` on a machine with internet, then copy |
| ScienceQA | ModelScope | `data_prep/download_assets.sh scienceqa` (~0.6 GB) |
| InfoSeek annotations, Wiki6M (7.4 GB), OVEN images (~294 GB snapshot), Wikipedia images | GCS / HF / Wikimedia | **fetch kit** (below) |

**The InfoSeek fetch kit** (`fetch_kit/fetch_infoseek.py`) runs on a machine
with internet access and inverts the usual order: it selects the questions
first and downloads only what they need.

1. `annotations` — InfoSeek val + entity map.
2. `splits` — the eval holdout and ranked seeds (+2,000 backups), with the same
   `evidencetree.data.splits` code the server uses. `--dry-run` reports only.
   `plan` then lists the distinct `image_id`s of those questions — exactly the
   images needed — with their source datasets and the OVEN shard sizes.
3. `oven` — the needed OVEN query images, one shard at a time; each shard is
   deleted after extraction (peak disk = one shard) and its member list kept as
   an index, so no shard is downloaded twice and none is downloaded once every
   wanted image is found. FGVC-Aircraft images come
   from their own small archive. `--image-prefixes` restricts selection to
   fewer shards if bandwidth is tight.
4. `wiki6m` — two streaming passes over Wiki6M: collect gold pages and
   candidates, then keep gold + up to 2 hard negatives per gold entity + random
   fill to 60K entities.
5. `wiki-images` — one 512-px thumbnail per corpus entity (polite rate, 429
   back-off, resumable).
6. `finalize` — drop seeds whose image is missing, fill from backups, write a
   manifest with sha256 checksums.

The kit is ~5–6 GB. The server-side `data_prep/import_infoseek_kit.py` verifies
the checksums, chunks the pages, and writes the shared layout.

## 4. Leakage control

Leakage is prevented structurally, not only checked afterwards.

* **Fixed evaluation sets.** Each benchmark's ~2K evaluation questions are
  selected once (deterministic, salted hashing, no RNG state) and written to
  `DatasetConstruct/splits/<benchmark>/eval_ids.txt`, committed to the repo.
  Construction loads only `seed_ids.txt`; the loader raises if any loaded
  question is an evaluation id, and `data.require_splits: true` refuses to run
  without both files.
* **InfoSeek** — construction and evaluation share the val split, so:
  (i) questions sharing an image land on one side only; (ii) every entity of an
  unseen-entity evaluation question is removed from the seed pool, so "unseen
  entity" stays unseen by the PRM too.
* **ScienceQA** — seeds from train/validation, evaluation from test. The
  corpus indexes `lecture` and `hint`, never `solution`; questions are posed
  without their hint, so the context must be retrieved.
* **GQA** — seeds from train-balanced, evaluation from test-dev; test-dev
  images are never put in the corpus (asserted by the prep script).
* **Train/val** of ETBench-Open are split by query hash; no query spans two
  splits (checked by the validator).
* **Rationales** may not use evaluation phrasing or name a gold answer that the
  visible evidence does not contain (QC rule d).

## 5. Pipeline

```text
seed queries + corpus
 Step 1  MCTS rollouts (policy VLM)            -> trajectories.jsonl   GPU, no API
 Step 2  free labels: g, R_out, marginal gain  -> labeled.jsonl        GPU, no API
 Step 3  length/gap filters, node collapse,
         budgeted sibling-group sampling        -> sampled.jsonl       CPU
 Step 4  answer-support judge (sampled, correct
         answers only)                          -> scored.jsonl        API
 Step 5  rationale + 5-rule QC                  -> rationales.jsonl    API
 Step 6  QC filter, splits, provenance, pairs,
         stats, validation                      -> etbench_open/<benchmark>/
```

Everything before step 4 is free, so the paid steps only ever see what will be
kept. Every step resumes from partial output; step 3 keeps an existing sample
unless `--force`, so the paid steps never see their input change.

### 5.1 Step 1 — rollouts

* Policy: Qwen2.5-VL-7B-Instruct, training-free, temperature 0.7, served by
  vLLM in containers (one server per GPU, `serve_policy_vllm.sh`);
  `policy.base_url` lists every server and requests rotate over them. `step1.concurrency` queries are
  searched in parallel so the servers batch.
* Search: UCB1, P = 10, d_max = 3, k = 3, early stop disabled.
  `mcts.expansion: diverse` keeps the best child of each proposed action type
  before a second child of any type — without a trained PRM the prior cannot
  rank types, and same-state contrasts across types are exactly what the
  dataset is for. The step-1 guidance scorer gives partial states a neutral
  prior (0.5) and scores finished rollouts by answer coverage in the evidence;
  it is never a label.
* Retrieval: hybrid — BM25 over the text side (a sparse-matrix implementation
  scoring like rank-bm25, cached on disk) interleaved with one CLIP index over
  text and image units; entity dedupe; text top-3, image top-1; an LRU cache
  memoizes repeated queries.
* Answers: the proposal prompt and a dedicated answer prompt ask for the short
  answer only; drafts are normalized (first line, no "Answer:", no quotes).
* Outcome: `metrics.outcome_match` — InfoSeek relaxed accuracy (numeric range
  for value questions), ScienceQA multiple choice (letter, option text, or
  both; contradictions rejected), GQA normalized exact match.
* Records carry portable image keys (relative to the benchmark data dir).

### 5.2 Step 2 — free labels

* **Local grounding** g (Eq. 3): cosine of question vs. result in one CLIP
  space, text results as the max over 48-word windows (title-prefixed), images
  with the image tower; per-modality calibration bands map cosines to [0, 1].
  The raw cosine and modality of the winning result are stored.
* **Outcome credit** R_out (Eq. 4): Monte Carlo success rate of all rollouts
  through the node, with the number of trajectories.
* **Answer steps** get a provisional score (= outcome) and label `pending`.
* **Marginal gain** (paper App. H, recorded not used): redundancy = max
  similarity of the best result to same-modality evidence already held (a
  repeated document counts 1.0); `marginal_gain = g × (1 − redundancy)`;
  `n_repeat_docs` counts re-retrieved documents.

### 5.3 Step 3 — filters, node collapse, sampling

1. Drop trajectories whose full length is outside [2, 8].
2. Collapse step records to unique nodes `(query, action prefix)`; rollouts
   share prefixes, so v1 wrote the same state/action/label several times.
3. Drop non-answer nodes with |g − R_out| > 0.7 (self-contradictory labels).
4. Group nodes by parent state (siblings = same state, different actions) and
   sample groups round-robin over queries — every query contributes before any
   contributes twice — up to `budget_per_query × #queries` nodes, at most
   `max_per_query` per query. Within a query, groups are taken most
   informative first: has a contrast (≥ 2 children), larger label spread, more
   reliable credit (more trajectories through the least-visited child).
5. Rotate depths. Deep groups rest on one trajectory each, so their credit is
   0/1 and their spread wide; ranked by spread alone, the first pilot drew 83%
   of its sample from depth 1 and 4% from root decisions. The first group of
   query *i* (in the seeded query order) is therefore its best contrast group
   at depth *i* mod 3 (`sampling.depth_cycle`, when the query has one), so the
   sample covers every decision depth; later groups follow the order above.

### 5.4 Step 4 — answer-support gate (Eq. 6)

`score = R_out × (0.3 + 0.7 × supp)`, with SUPPORTED / PARTIAL / UNSUPPORTED →
1 / 0.5 / 0 and NOT_REQUIRED → no gating. Only sampled answers are judged, and
wrong answers are skipped (`skipped_incorrect`, score 0 either way). The judge
sees the evidence with page titles, and image hits as `[image] <title>`, and is
told it cannot see the question image. Correct-but-unsupported answers are
kept and flagged as DPO rejected sides.

The judge writes a one- or two-sentence reason, then the verdict alone on the
last line; the verdict is read from that line and the reason is stored as
`answer_support_reason` for audits. In the first pilot a one-word verdict from
the local Qwen3-Coder never used NOT_REQUIRED (0 of 89 judged answers) and
called evidence that had nothing to do with the answer SUPPORTED (about 9 of
56). With the reason first and NOT_REQUIRED spelled out — answerable from the
question, its options and its image by grammar, logic, arithmetic, reading a
given passage or diagram, comparing the options or everyday commonsense; never
for entity facts — reasoning questions get NOT_REQUIRED as the paper intends
(App. H, Case 3) and SUPPORTED becomes stricter. This departs from the
one-word prompt printed in the paper's Appendix D; the four verdicts and their
mapping are unchanged.

### 5.5 Step 5 — rationales

The five QC rules are unchanged: cite a visible evidence id; name the action
type; 30–150 tokens; no gold leakage or evaluation phrasing; a `VERDICT` line
consistent with the score. Up to 3 attempts, with the failure reasons fed
back. Transient API failures are not QC failures: they are removed and retried.

Every sampled node still gets its own rationale — each is the training target
of one (state, action) sample, and the PRM scores one candidate at a time. But
the requests are merged: **siblings (candidates from the same state) share one
request** (`rationale.batch: group`, at most `max_group: 6` per request). The
question and shared evidence are sent once; the model writes one labeled block
per candidate; each block passes the same QC on its own plus a stand-alone
rule (it may not name evidence that only another candidate retrieved, since
the PRM never sees the siblings); a missing or failing block is regenerated
alone. Merging along a trajectory instead would show the model the later steps
and the final answer, inviting hindsight rationales (the paper requires
ex-ante ones), and would rarely help anyway because sampling keeps sibling
groups, not paths. On the mock build, 420 nodes need 181 requests (2.3× fewer)
with no extra QC failures. Requests get longer, so wall-clock time under the
same concurrency changes little; what drops is the request count (decisive for
request-quota plans) and repeated input tokens.

Cost: fixed instructions come first (prefix-cacheable); earlier evidence is
truncated to 300 characters for search steps, while answer steps see it in
full (their support lives there).

### 5.6 Step 6 — finalize

Drop QC failures; split train/val by query hash (`test_fraction` defaults to 0:
the benchmark evaluation sets are the test sets); attach `gen_provenance`
(models, deployments, retriever, grounding bands, MCTS, sampling, outcome rule,
seed, commit and dirty flag, version); mine same-state pairs per split;
write `stats.json`; run the SCHEMA §2 validator in release mode.

**Pair types** (margin ≥ 0.3 unless noted): `action_type` (text vs. image
search), `query_choice`, `search_vs_answer`, `answer_support` (supported vs.
unsupported-correct), `answer_outcome`, and — tagged, within the margin —
`answer_now_vs_search_more` (a supported correct answer preferred over a
successful further search from the same state).

## 6. Quality gates

1. **Preflight** (`preflight.py`): data files, split disjointness, image
   paths, local models, packages, API keys, vLLM endpoints, GPU memory, disk.
2. **Pilot** of 100 seeds per benchmark through all six steps.
3. **Audit** (`audit_report.py`): a Markdown sheet of 100 sampled queries with
   every sampled node, its evidence, labels, support verdict and rationale. A
   human reads it; construction scales up only after it passes (paper §4).
4. **Calibration** from the pilot (next section).
5. **Stats review** before and after each scale-up — the quantities to watch:

| Signal | Where | Healthy | If not |
| --- | --- | --- | --- |
| trajectories per query | stats.construction | ≥ 5 | proposal diversity (prompt, temperature) |
| image_search share (InfoSeek) | construction.step_action_types | clearly > 0 | image path / index / prompt |
| outcome positive rate by answer type | construction / splits | not ~0 for value questions | answer format, metric |
| gap drops | sampling.dropped | ≲ 15% of nodes | grounding bands |
| support labels | splits.answer_support_labels | few `unparseable` | judge prompt / model |
| rationale QC pass rate | rationales.qc_pass_rate | ≥ 90% first attempt | rationale model / prompt |
| unsupported_correct | splits | present, not dominant | judge strictness |

## 7. Calibration after the pilot

* **Grounding bands.** `grounding_detail.raw_cos` holds every winning cosine
  by modality. Set `text_cos_lo/hi` and `image_cos_lo/hi` to roughly the 5th
  and 95th percentiles of the pilot distribution (per modality), rerun step 2
  with `--force` (free), then steps 3+. Bands differ by corpus, so they live in
  the benchmark config: ScienceQA uses text [0.67, 0.945] (pilot, 2026-10-09;
  the shared [0.5, 0.9] put the median at 0.79 and clipped the top 15%).
* **Gap threshold.** Check what 0.7 removes; the rule should catch
  contradictions, not a whole action type.
* **Support judge.** Read 50 judged answers from the audit sheet; confirm
  NOT_REQUIRED is not used for entity facts and UNSUPPORTED catches lucky
  guesses.

## 8. Cost model (estimates — replace with pilot measurements)

Per benchmark, 20K seeds, ~30 policy calls per query:

| Step | Unit | v1 flow | v2 flow |
| --- | --- | --- | --- |
| 1 policy (local vLLM) | calls | ~600K | ~600K (same; free) |
| 4 support judge | calls | every answer step, ~150K | sampled correct answers, ~10K |
| 5 rationales | calls | every step record, ~460K × 1.2 | ~25K group requests for ~60K sampled nodes |
| 5 rationale input | tokens | ~1.4B | ~0.1B (plus prefix-cache hits) |

Step 1 on 4× RTX 3090 should take on the order of half a day to a day per
benchmark (to be measured in the pilot); step 2 is under an hour of GPU time.
Disk: intermediate JSONL ≈ 10 GB per benchmark (evidence snapshots per step),
indexes ≈ 4 GB for InfoSeek, raw GQA ≈ 10.5 GB.

## 9. Run book

```bash
# 0. environment (server): python -m venv .venv && source .venv/bin/activate
pip install -r requirements/server.txt && pip install -e .
bash DatasetConstruct/data_prep/download_assets.sh            # models, GQA, ScienceQA

# 1. data
python DatasetConstruct/fetch_kit/fetch_infoseek.py all --kit ~/infoseek_kit   # LOCAL machine
python DatasetConstruct/fetch_kit/fetch_infoseek.py gqa-scenegraphs --out ~/sceneGraphs.zip  # LOCAL
python DatasetConstruct/data_prep/import_infoseek_kit.py --kit /path/to/infoseek_kit
python DatasetConstruct/data_prep/prepare_scienceqa.py --src data/raw/ScienceQA
python DatasetConstruct/data_prep/prepare_gqa.py --src data/raw/GQA --scene-graphs data/raw/GQA/sceneGraphs.zip

# 2. serve the policy, check everything
cp DatasetConstruct/site.env.example DatasetConstruct/site.env   # then adapt
bash DatasetConstruct/serve_policy_vllm.sh                        # step 1 policy servers
python DatasetConstruct/preflight.py --benchmark infoseek --check-apis

# 3. pilot + audit in a separate output tree, then the full build (per benchmark)
PILOT="--set output.dir=data/pilot/construction --set output.dataset_dir=data/pilot/etbench_open"
python DatasetConstruct/run_pipeline.py --benchmark infoseek --n 100 $PILOT
python DatasetConstruct/audit_report.py --benchmark infoseek --n 100 \
    --file data/pilot/construction/infoseek/rationales.jsonl
python DatasetConstruct/run_pipeline.py --benchmark infoseek          # full build
```

Keep the pilot out of the full build's output tree: step 3 deliberately keeps
an existing sample, and `--force` would also discard step 1's rollouts.

**Endpoint presets.** `configs/providers/*.yaml` hold ready endpoint blocks
(`kimi_code`, `kimi_code_anthropic`, `moonshot`, `deepseek`,
`local_qwen3_coder`). `--provider NAME` (steps 4/5, `try_rationale_model.py`) or
`--rationale-provider` / `--judge-provider` (`run_pipeline.py`, `preflight.py`)
swap only the endpoint keys (provider, model, base_url, api_key_env,
extra_body) and keep the prompt and sampling settings. For Kimi Code: put
`KIMI_CODE_API_KEY=...` in `DatasetConstruct/.env`, check the base URL and model
id in `configs/providers/kimi_code.yaml` against the Kimi Code docs, then
`python DatasetConstruct/try_rationale_model.py --provider kimi_code`.

Models and API keys are switched in the config only: `policy`,
`verifier.support.generation` and `rationale.generation` each take `provider`,
`model`, `base_url` and `api_key_env` (the variable name; the key itself lives
in `DatasetConstruct/.env`). Try a new rationale model on a few samples first:
`python DatasetConstruct/try_rationale_model.py --n 3`.

**Rationales with Kimi.** The construction server cannot reach Moonshot/Kimi
(api.moonshot.cn, api.kimi.com time out; DeepSeek, ModelScope and the Aliyun
mirror are reachable). Either have the domains allow-listed, or run step 5 on
a machine that reaches them — it needs only `scored.jsonl`, the repository,
`pip install pyyaml openai` (or `anthropic` for an Anthropic-protocol key) and
the key in `.env`:

```bash
# on the server, after step 4
scp data/construction/infoseek/scored.jsonl <laptop>:~/
# on the laptop (edit rationale.generation in configs/base.yaml as shown there first)
python DatasetConstruct/try_rationale_model.py --n 3
python DatasetConstruct/step5_gen_rationales.py --benchmark infoseek \
    --input ~/scored.jsonl --output ~/rationales.jsonl
# back on the server
scp <laptop>:~/rationales.jsonl data/construction/infoseek/rationales.jsonl
python DatasetConstruct/run_pipeline.py --benchmark infoseek --steps 6
```

Each row records `rationale_model` (and judged answers `support_judge_model`);
step 6 takes the provenance from the rows, so it stays right when steps 4–5
ran elsewhere. A coding-plan key (Kimi Code) is meant for coding tools and is
quota-limited; check its terms before sending ~60K requests per benchmark
through it — a pay-as-you-go key from the Moonshot platform is the safer fit.

## 10. Decisions and open points

Decided here (change in config if needed):

* Sampled nodes = 3 × seeds per benchmark (paper ratio); ≈ 170K rationales for
  all three benchmarks. This is the dominant API cost.
* No ETBench-Open test split; the benchmark evaluation sets are held out.
* Marginal gain = g × (1 − redundancy of the best result), recorded only.
* InfoSeek chunks: ~1,000 characters, at most 40 per entity page.
* ScienceQA corpus: lectures and hints only, no question-text documents for
  questions that have neither. The paper's leakage audit (App. E) says the
  corpus falls back to the question text; the pilot showed such copies score
  near 1.0 grounding while carrying no information, so they were removed —
  the paper's sentence needs the same update.
* Support judge: reason first, verdict on the last line (§5.4).
* Sampling: each query's first group rotates over depths 0–2 (§5.3).
* GQA seeds: one question per image; corpus = all train-balanced images with a
  scene graph.

To confirm with the pilot: grounding bands; gap threshold; judge and rationale
models after the planned model/API switch; whether the ScienceQA text-only
questions (~half) should stay in the seed pool (they exercise text retrieval
and answering without an image).
