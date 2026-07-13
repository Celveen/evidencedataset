# Action Space Comparison: baseline vs cross-modal actions

Run settings:
- Datasets: infoseek, scienceqa, mrag_bench, crag_mm
- N=15 per dataset
- mcts.rollouts=5
- retriever.ocr.enabled=false
- Baseline: retriever.cross_modal=false
- New action space: retriever.cross_modal=true, enabling text_to_image and image_to_text where supported
- CRAG-MM warning: cross_modal=true is ignored for CRAG-MM because its current retrievers use separate official web/image indexes.

## Success Metrics

| dataset | base_em | new_em | delta_em | base_soft | new_soft | delta_soft | base_f1 | new_f1 | delta_f1 | base_score | new_score | delta_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 5/15 | 6/15 | +0.067 | 6/15 | 9/15 | +0.200 | 0.268 | 0.397 | +0.129 | 0.257 | 0.310 | +0.053 |
| scienceqa | 13/15 | 14/15 | +0.067 | 13/15 | 14/15 | +0.067 | 0.821 | 0.806 | -0.015 | 0.854 | 0.812 | -0.042 |
| mrag_bench | 9/15 | 12/15 | +0.200 | 9/15 | 12/15 | +0.200 | 0.607 | 0.573 | -0.034 | 0.486 | 0.582 | +0.096 |
| crag_mm | 4/15 | 4/15 | +0.000 | 6/15 | 6/15 | +0.000 | 0.303 | 0.340 | +0.037 | 0.265 | 0.320 | +0.055 |

## Cost / Data Volume

| dataset | base_traj | new_traj | traj_x | base_steps | new_steps | steps_x | base_kept | new_kept | base_qc | new_qc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 32 | 74 | 2.31x | 70 | 248 | 3.54x | 48/70 | 149/248 | 70/70 | 171/248 |
| scienceqa | 28 | 73 | 2.61x | 57 | 244 | 4.28x | 52/57 | 155/244 | 57/57 | 168/244 |
| mrag_bench | 34 | 74 | 2.18x | 80 | 230 | 2.88x | 66/80 | 144/230 | 80/80 | 204/230 |
| crag_mm | 43 | 44 | 1.02x | 108 | 110 | 1.02x | 93/108 | 103/110 | 108/108 | 110/110 |

## Action Distribution

| dataset | baseline_actions | new_actions |
| --- | --- | --- |
| infoseek | answer:32, text_search:38 | answer:74, image_search:59, image_to_text:42, text_search:38, text_to_image:35 |
| scienceqa | answer:28, text_search:29 | answer:73, image_search:61, image_to_text:35, text_search:36, text_to_image:39 |
| mrag_bench | answer:34, image_search:30, text_search:16 | answer:74, image_search:67, image_to_text:29, text_search:21, text_to_image:39 |
| crag_mm | answer:43, image_search:36, text_search:29 | answer:44, image_search:37, text_search:29 |

## Macro Average All Four

| setting | avg_em | avg_soft | avg_f1 | avg_score | total_traj | total_steps | qc_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.517 | 0.567 | 0.500 | 0.466 | 137 | 315 | 315/315 (1.000) |
| new | 0.600 | 0.683 | 0.529 | 0.506 | 265 | 832 | 653/832 (0.785) |

## Macro Average Supported Cross-Modal Only

| setting | avg_em | avg_soft | avg_f1 | avg_score | total_traj | total_steps | qc_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.600 | 0.622 | 0.565 | 0.532 | 94 | 207 | 207/207 (1.000) |
| new | 0.711 | 0.778 | 0.592 | 0.568 | 221 | 722 | 543/722 (0.752) |

## Interpretation

- The new cross-modal actions improve query-level hit rate on the supported datasets in this 15-query smoke test: InfoSeek improves from 5/15 to 6/15 EM and 6/15 to 9/15 soft; ScienceQA improves from 13/15 to 14/15; MRAG-Bench improves from 9/15 to 12/15.
- The improvement is not free: supported datasets jump from 94 trajectories / 207 steps to 221 trajectories / 722 steps, about 3.5x more step-level data.
- Rationale QC drops sharply on InfoSeek and ScienceQA after adding cross-modal actions. This suggests many new action steps are harder to justify or produce weaker evidence, even when query-level success improves.
- CRAG-MM should not be used to judge the new cross-modal action space in this run, because cross_modal=true is explicitly ignored for its official local retrievers.
- Overall: the new action space helps success rate on this small test, especially MRAG-Bench and InfoSeek soft matching, but it increases collection time/cost and lowers label cleanliness. It should be used with gating/prior routing, not enabled blindly for every sample.
