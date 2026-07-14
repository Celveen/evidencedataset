# Action Gate Comparison, N=30, Rollout=5

Run setting:
- Date: 2026-06-21
- Root: `/home/wenke/SQJ/code/EvidenceTree`
- Requested samples: 30 per dataset
- Actual samples: InfoSeek 30, ScienceQA 25, MRAG-Bench 25, CRAG-MM 25
- Rollouts: 5
- OCR: off
- Compared modes: old action space, full new action space, dataset gate, heuristic gate, strict gate
- Gate runs executed only step1/step2, so `kept`, `mean_score`, and `rationale_qc` are not measured for gate modes.

## Aggregate Results

Supported datasets here means InfoSeek, ScienceQA, and MRAG-Bench. These are the datasets where the new cross-modal actions can actually be used. CRAG-MM currently ignores `cross_modal=true` because it uses separate official web/image indexes.

| mode | supported EM | supported soft | supported mean F1 | supported steps | step cost vs old | all EM | all soft | all mean F1 | all steps | step cost vs old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old actions | 0.512 | 0.550 | 0.512 | 334 | 1.00x | 0.457 | 0.514 | 0.470 | 502 | 1.00x |
| full new actions | 0.650 | 0.662 | 0.535 | 1282 | 3.84x | 0.543 | 0.590 | 0.471 | 1410 | 2.81x |
| dataset gate | 0.613 | 0.688 | 0.546 | 892 | 2.67x | 0.524 | 0.610 | 0.495 | 1038 | 2.07x |
| heuristic gate | 0.600 | 0.650 | 0.528 | 807 | 2.42x | 0.514 | 0.600 | 0.471 | 987 | 1.97x |
| strict gate | 0.575 | 0.625 | 0.543 | 801 | 2.40x | 0.505 | 0.581 | 0.493 | 982 | 1.96x |

## Per-Dataset Results

| mode | dataset | query EM | query soft | mean F1 | trajectories | steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old actions | infoseek | 7/30 | 10/30 | 0.264 | 62 | 139 |
| old actions | scienceqa | 21/25 | 21/25 | 0.812 | 48 | 102 |
| old actions | mrag_bench | 13/25 | 13/25 | 0.509 | 43 | 93 |
| old actions | crag_mm | 7/25 | 10/25 | 0.337 | 69 | 168 |
| full new actions | infoseek | 13/30 | 14/30 | 0.276 | 143 | 482 |
| full new actions | scienceqa | 23/25 | 23/25 | 0.825 | 122 | 409 |
| full new actions | mrag_bench | 16/25 | 16/25 | 0.555 | 124 | 391 |
| full new actions | crag_mm | 5/25 | 9/25 | 0.269 | 57 | 128 |
| dataset gate | infoseek | 8/30 | 14/30 | 0.277 | 142 | 422 |
| dataset gate | scienceqa | 22/25 | 22/25 | 0.834 | 43 | 79 |
| dataset gate | mrag_bench | 19/25 | 19/25 | 0.581 | 124 | 391 |
| dataset gate | crag_mm | 6/25 | 9/25 | 0.332 | 62 | 146 |
| heuristic gate | infoseek | 9/30 | 13/30 | 0.314 | 121 | 321 |
| heuristic gate | scienceqa | 21/25 | 21/25 | 0.762 | 51 | 92 |
| heuristic gate | mrag_bench | 18/25 | 18/25 | 0.551 | 124 | 394 |
| heuristic gate | crag_mm | 6/25 | 11/25 | 0.290 | 74 | 180 |
| strict gate | infoseek | 7/30 | 11/30 | 0.287 | 115 | 323 |
| strict gate | scienceqa | 21/25 | 21/25 | 0.824 | 51 | 89 |
| strict gate | mrag_bench | 18/25 | 18/25 | 0.570 | 125 | 389 |
| strict gate | crag_mm | 7/25 | 11/25 | 0.332 | 74 | 181 |

## Interpretation

Best overall gate in this run: **dataset gate**.

Why:
- It keeps most of the gain from the full new action space.
- It has the best supported soft accuracy and supported mean F1.
- It cuts supported step cost from 1282 to 892, about 30.4% fewer steps than full new actions.
- It is much cheaper than full new actions while still clearly better than the old action space on MRAG-Bench.

Heuristic gate is the cheapest useful option:
- It reduces supported steps to 807, slightly cheaper than dataset gate.
- It performs well on InfoSeek F1 and MRAG-Bench, but ScienceQA drops from 22/25 soft under dataset gate to 21/25.
- It is acceptable when collection cost matters more than squeezing out the last few correct queries.

Strict gate is too narrow as the default:
- It saves almost the same number of steps as heuristic gate.
- It loses more query-level accuracy on InfoSeek.
- Its main use should be a very cheap smoke-test mode, not production data construction.

Full new action space is still the upper-bound exploration mode:
- It gives the highest supported EM.
- But it costs 3.84x the old step count on supported datasets.
- It also previously showed lower rationale QC because more weak/detour steps are produced.

## Recommendation

Use this operating policy:

| scenario | recommended mode |
| --- | --- |
| cheap smoke tests | old actions or strict gate |
| normal data construction | dataset gate |
| cost-sensitive larger runs | heuristic gate |
| diagnosis / maximum exploration | full new actions |

For the next 60k-style collection, the practical default should be:

```bash
--set retriever.cross_modal=true \
--set retriever.ocr.enabled=false \
--set action_gate.enabled=true \
--set action_gate.mode=dataset \
--set action_gate.max_actions=5
```

If runtime becomes too expensive, switch only the gate mode:

```bash
--set action_gate.mode=heuristic \
--set action_gate.max_actions=4
```

