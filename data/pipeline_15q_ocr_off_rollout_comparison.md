# OCR Off Rollout Comparison: 15 queries each

Run settings:
- Datasets: infoseek, scienceqa, mrag_bench, crag_mm
- N=15 per dataset
- retriever.ocr.enabled=false
- retriever.cross_modal=false
- Compared mcts.rollouts=3, 5, 10

## Query-Level Metrics

| dataset | r3_em | r5_em | r10_em | r3_soft | r5_soft | r10_soft | r3_f1 | r5_f1 | r10_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 5/15 | 5/15 | 6/15 | 6/15 | 6/15 | 7/15 | 0.284 | 0.268 | 0.281 |
| scienceqa | 13/15 | 13/15 | 13/15 | 13/15 | 13/15 | 13/15 | 0.802 | 0.821 | 0.750 |
| mrag_bench | 11/15 | 9/15 | 7/15 | 11/15 | 9/15 | 7/15 | 0.620 | 0.607 | 0.586 |
| crag_mm | 4/15 | 4/15 | 4/15 | 6/15 | 6/15 | 6/15 | 0.278 | 0.303 | 0.258 |

## Data Volume / Exploration

| dataset | r3_traj | r5_traj | r10_traj | r3_steps | r5_steps | r10_steps | r3_kept | r5_kept | r10_kept |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 31 | 32 | 32 | 64 | 70 | 75 | 42/64 | 48/70 | 47/75 |
| scienceqa | 26 | 28 | 28 | 52 | 57 | 58 | 50/52 | 52/57 | 48/58 |
| mrag_bench | 26 | 34 | 27 | 54 | 80 | 59 | 45/54 | 66/80 | 44/59 |
| crag_mm | 29 | 43 | 35 | 65 | 108 | 82 | 59/65 | 93/108 | 72/82 |

## Action Distribution

| dataset | r3_actions | r5_actions | r10_actions |
| --- | --- | --- | --- |
| infoseek | answer:31, text_search:33 | answer:32, text_search:38 | answer:32, text_search:43 |
| scienceqa | answer:26, text_search:26 | answer:28, text_search:29 | answer:28, text_search:30 |
| mrag_bench | answer:26, image_search:26, text_search:2 | answer:34, image_search:30, text_search:16 | answer:27, image_search:24, text_search:8 |
| crag_mm | answer:29, image_search:27, text_search:9 | answer:43, image_search:36, text_search:29 | answer:35, image_search:31, text_search:16 |

## Macro Average

| rollouts | avg_query_em | avg_query_soft | avg_mean_f1 | total_trajectories | total_steps | total_kept |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.550 | 0.600 | 0.496 | 112 | 235 | 196/235 |
| 5 | 0.517 | 0.567 | 0.500 | 137 | 315 | 259/315 |
| 10 | 0.500 | 0.550 | 0.469 | 122 | 274 | 211/274 |

## Interpretation

- In this OCR-off base action space, increasing rollouts from 3 to 5/10 does not monotonically improve accuracy on the 15-query sample.
- InfoSeek and ScienceQA are mostly saturated by rollout=3: trajectory counts barely increase, and query-level metrics change little.
- MRAG-Bench and CRAG-MM are more sensitive because image_search competes with text_search/answer, but higher rollout still does not guarantee better final query hits in this small sample.
- The main effect of larger rollout is more step-level data and more diverse actions, not consistently better answer accuracy.
- If we enlarge the action space again, for example enabling OCR or cross-modal text_to_image/image_to_text, more rollout is generally needed for coverage. But this run suggests simply raising rollout is inefficient; action gating/prior routing is more important.
