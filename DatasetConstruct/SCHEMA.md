# ETBench-Open — dataset specification

This file is the release specification for ETBench-Open. Everything produced by
`run_pipeline.py` must pass `validate_dataset.py`, and every section of the dataset card in
§6 must be filled in before publication. The goal is a dataset that is reproducible,
auditable and re-distributable.

## 1. Record schema

The dataset is **step-level**: one record is one retrieval or answer step of one
trajectory, and one training sample for the PRM. JSONL, one UTF-8 JSON object per line.

### 1.1 Training sample records (`train.jsonl` / `val.jsonl` / `test.jsonl`)

| Field | Type | Nullable | Constraints | Meaning |
| --- | --- | --- | --- | --- |
| `sample_id` | str | no | globally unique, `<traj_id>#s<step_index>` | primary key |
| `traj_id` | str | no | `<query_id>#t<k>` | owning trajectory |
| `query_id` | str | no | unique within the benchmark | owning query; **the unit of the train/val/test split** |
| `question` | str | no | non-empty | the original question |
| `image_path` | str \| null | yes | **relative key or image id** (machine-absolute paths rejected) | query image; null for text-only queries |
| `state.actions_before` | list[str] | no | may be empty | actions already taken, as `type(input)` strings |
| `state.evidence_before` | list[obj] | no | see §1.3 | evidence accumulated before this step |
| `action.type` | str | no | `text_search` \| `image_search` \| `answer` | action type |
| `action.input` | str | no | non-empty | search query, image-region description, or answer text |
| `observation_evidence` | list[obj] | no | see §1.3; empty on answer steps | evidence this step retrieved |
| `local_grounding` | float \| null | yes | `[0,1]`; null **iff** `action.type == "answer"` | relevance of the retrieved results to the question, scored in one CLIP space; the best modality present wins |
| `outcome_credit` | float | no | `[0,1]` | tree-level credit: Monte Carlo success rate of all trajectories through this node |
| `n_traj_through` | int | no | `>= 1` | number of trajectories through the node (credit sample size) |
| `alpha` | float | no | `[0,1]` | dual-source fusion weight |
| `answer_support` | float \| null | yes | `[0,1]`; numeric only on answer steps | whether the answer is backed by accumulated evidence (null = not judged) |
| `answer_support_label` | str \| null | yes | `supported` \| `partial` \| `unsupported` \| `not_required` \| `lexical` \| `no_evidence` \| `unverifiable` \| `unparseable` | raw judge label, kept for diagnostics |
| `support_floor` | float | no | `[0,1]`, default 0.3 | answer-step fusion floor (see `score`) |
| `unsupported_correct` | bool | no | — | answer step with `outcome >= 0.5` and `support <= unsupported_threshold`: a parametric lucky guess. **Not dropped** — retained as a DPO negative |
| `gold_answers` | list | no | see §1.2 | reference answers, for leakage checks and analysis; not part of the PRM input |
| `score` | float | no | `[0,1]` | training target. Non-answer: `alpha*local + (1-alpha)*outcome`. Answer: `outcome*(floor + (1-floor)*support)`, falling back to `outcome` when `support` is null |
| `score_source` | str | no | `local_plus_outcome` \| `answer_support` \| `outcome_only` | which fusion rule produced `score` |
| `rationale` | str | no | non-empty, 30–150 whitespace tokens | explanation of the score, with the `VERDICT` line stripped |
| `rationale_verdict` | str \| null | yes | `good` \| `mixed` \| `poor` | verdict direction, checked against the score |
| `rationale_backend` | str | no | `api` \| `mock` | rationale source |
| `rationale_attempts` | int | no | `>= 1` | generation attempts including QC retries |
| `rationale_qc_pass` | bool | no | **must be true** in a released split | whether QC passed |
| `rationale_qc_reasons` | list[str] | no | empty when `rationale_qc_pass` | failure reasons |
| `gen_provenance`* | obj | no | see §1.4 | **required for release**: model versions and seeds |

\* `gen_provenance` is not written by the pipeline; attach it before publishing (§1.4), at
which point `validate_dataset.py --require-provenance` will check for it.

### 1.2 Trajectory records (intermediate `*_trajectories.jsonl`, optionally released for reproduction)

| Field | Type | Constraints | Meaning |
| --- | --- | --- | --- |
| `traj_id` / `query_id` / `question` / `image_path` | — | as in §1.1 | — |
| `gold_answers` | list | non-empty | string-type answers are a list of acceptable strings; value-type answers are `{wikidata, range: [lo, hi]}` (see §5) |
| `rollout_t` | int | `>= 0` | index of the rollout that produced this trajectory |
| `gen_reward` | float | — | generation-time guidance score; **recorded only, never a label** |
| `final_answer` | str | — | the trajectory's answer |
| `outcome_em` | float | `{0,1}` | answer correctness (§5) |
| `steps` | list[obj] | length 2–8 after filtering | `{step_index, action_type, action_input, evidence[]}`; `image_search` steps additionally carry `region` (normalized `[x1,y1,x2,y2]` or null) and the `image_path` actually queried |

### 1.3 Evidence objects

| Field | Type | Meaning |
| --- | --- | --- |
| `evidence_id` | str | unique within a trajectory: `e0`, `e1`, … — this is what a rationale cites |
| `doc_id` | str | corpus document key (present in trajectory records; omitted from the sample-level snapshots) |
| `title` | str | document title; may be an empty string |
| `text` | str | document or chunk body |
| `image_path` | str \| null | relative key of an image-side result, used for image grounding |
| `score` | float | retriever similarity |
| `result_modality` | str \| null | modality of the retrieved **unit**: `text` \| `image`. In the unified CLIP index, corpus text chunks and images are indexed as independent units, so a single search action can return both. `null` means the record came from a legacy document-level retriever |

Both `text_search` and `image_search` query the same unified index, so either action may
return either modality; `result_modality` is what distinguishes them downstream.

### 1.4 `gen_provenance` (required for release)

Records the policy that was actually used. `policy_deployment` separates `local` from
`api` so a development-time API model is never mistaken for the released policy.

```json
{"policy_model": "Qwen2.5-VL-7B-Instruct", "policy_deployment": "local",
 "policy_snapshot": "2026-06", "retriever": "unified-clip-ViT-B-32",
 "rationale_model": "deepseek-v4-pro", "rationale_deployment": "api",
 "mcts": {"rollouts": 10, "max_depth": 3, "c_uct": 1.0},
 "outcome_metric": "infoseek_relaxed", "seed": 0,
 "pipeline_commit": "<git-sha>", "dataset_version": "v0.1"}
```

## 2. Invariants enforced by `validate_dataset.py`

1. Required fields present; `sample_id` unique within a file; the `#s<n>` suffix is present
   and consistent with the trajectory.
2. `action.type` is one of the three typed actions; `action.input` is non-empty.
3. Score domains: `local_grounding`, `outcome_credit`, `score`, `alpha`, `answer_support`
   all lie in `[0,1]`; `local_grounding is null ⇔ action.type == "answer"`;
   `answer_support` is numeric only on answer steps.
4. Score fusion identity holds to 1e-4, using the answer-step formula when
   `action.type == "answer"` (with `support_floor` defaulting to 0.3) and the non-answer
   formula otherwise.
5. `n_traj_through >= 1`.
6. `|local − outcome| <= max_grounding_outcome_gap` (default 0.7); a released split should
   contain no sample outside the band.
7. `rationale_qc_pass == true`, and the rationale is 30–150 whitespace tokens, mentions
   `action.type`, and cites at least one `evidence_id` visible in this sample. If
   `rationale_verdict` is present it must not hard-contradict the score: no `poor` when
   `score >= 0.6`, no `good` when `score <= 0.4`.
8. Split purity: no `query_id` appears in more than one of the files passed together.
9. Re-distributable paths: `image_path` must not be machine-absolute — a relative key, an
   image id, or null.
10. Trajectory length is a **trajectory-level pre-filter** applied by step 4 to the complete
    trajectory. Per-sample gap filtering legitimately leaves some trajectories with a single
    surviving sample; that sample is still a valid independent `(state, action, label)` node,
    so a short post-filter count is reported as info, not a violation. More samples than
    `max_steps` is a violation, since it would mean the step-4 gate failed.
11. Release mode (`--require-provenance`): every record carries `gen_provenance`.

Gold-answer leakage is enforced at generation time rather than by the validator. The
rationale QC in `src/evidencetree/prm/rationale_gen.py` rejects evaluation phrasing
("ground truth", "gold answer", "correct answer") and any gold-answer string that does not
also appear in the visible evidence, question, or action input — quoting an answer found in
the evidence is legitimate, knowing it otherwise is leakage.

## 3. Release bundle

```text
etbench_open/
├── train.jsonl  val.jsonl  test.jsonl     # samples (§1.1)
├── corpus.jsonl                           # doc_id -> {text, title, image_id}: evidence stays traceable
├── images/ or image_urls.jsonl            # bundled images where redistribution is allowed, otherwise URLs + a download script (§6)
├── stats.json                             # distribution report (§5)
├── manifest.json                          # per-split line counts + sha256 + dataset_version
├── SCHEMA.md  DATASET_CARD.md  LICENSE
└── load_dataset.py                        # HF datasets loader whose Features match §1.1
```

## 4. Provenance and statistics to publish

`stats.json` in a release should report: per-split line counts, action-type distribution,
mean trajectory length, histograms of `score` / `local_grounding` / `outcome_credit`, the
rationale QC pass rate, drop counts per rule, and the **outcome positive rate reported
separately per answer type** (§5).

## 5. Outcome labels

`outcome_em` comes from `evidencetree.eval.metrics.exact_match`, which is deterministic and
rule-based. No LLM judge is involved. InfoSeek answers are of two kinds and each uses its
own official tolerance:

- **String questions.** `gold_answers` is a list of acceptable strings; the prediction is
  normalized SQuAD-style (lower-cased, punctuation and articles removed) and must equal any
  one of them.
- **Value questions.** The gold entry encodes `{"wikidata": v, "range": [lo, hi]}`, as a
  dict or its string repr. Any number parsed out of the prediction that falls inside
  `[lo, hi]` counts as correct. A bare string comparison would score every value question 0,
  so this rule is applied inside `exact_match` itself; `metrics.infoseek_accuracy` exposes
  the same relaxed rule for evaluation.

## 6. Dataset card template (`DATASET_CARD.md`, complete every section before release)

```markdown
# ETBench-Open

## Motivation
Why the dataset exists: retrieval-aware PRMs have no step-level supervision.

## Composition
Sample and trajectory counts, fields (point at SCHEMA.md), answer-type distribution,
multimodal share.

## Collection
Sources (InfoSeek / OVEN / Wikipedia), policy and rationale models with versions and
dates, MCTS configuration, seeds.

## Preprocessing
Tree-level credit, grounding scores, QC rules, split strategy.

## Uses
Training and evaluating retrieval-action PRMs. Not suitable as a general-purpose QA
training set.

## Distribution
License (below), images distributed by URL or id, version number and changelog.

## Maintenance
Maintainer, contact, update plan, errata channel.

## Ethics and license
Source licenses passed through (Wikipedia images are typically CC-BY-SA), derived-dataset
license, PII statement.
```

**License constraint.** Wikipedia and OVEN images generally cannot be redistributed
directly. Publish `image_url` / `image_id` plus a download script rather than image bytes,
and pass the source licenses through explicitly in the card.
