# OCR Ablation: 15 queries, rollouts=10

Run settings:
- Datasets: infoseek, scienceqa, mrag_bench, crag_mm
- N=15 per dataset
- mcts.rollouts=10
- retriever.cross_modal=false for both runs
- Compared retriever.ocr.enabled=true vs false

## Comparison Table

| dataset | query_em_on | query_em_off | delta_query_em | query_soft_on | query_soft_off | delta_query_soft | mean_f1_on | mean_f1_off | delta_f1 | traj_em_on | traj_em_off | delta_traj_em | steps_on | steps_off | kept_on | kept_off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 6/15 | 6/15 | +0.000 | 7/15 | 7/15 | +0.000 | 0.326 | 0.292 | +0.034 | 16/83 | 6/41 | +0.046 | 259 | 99 | 223/259 | 63/99 |
| scienceqa | 13/15 | 13/15 | +0.000 | 13/15 | 13/15 | +0.000 | 0.797 | 0.806 | -0.009 | 35/46 | 25/31 | -0.046 | 112 | 67 | 103/112 | 61/67 |
| mrag_bench | 13/15 | 9/15 | +0.267 | 13/15 | 9/15 | +0.267 | 0.601 | 0.625 | -0.024 | 36/76 | 11/23 | -0.005 | 210 | 47 | 183/210 | 38/47 |
| crag_mm | 5/15 | 4/15 | +0.067 | 8/15 | 6/15 | +0.133 | 0.283 | 0.375 | -0.092 | 24/121 | 11/45 | -0.046 | 387 | 108 | 354/387 | 96/108 |

## Full Summaries

### OCR on

| dataset | queries | query_hit_em | query_hit_soft | traj_em | traj_soft | mean_f1 | trajectories | steps | kept | train | val | mean_score | rationale_qc | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 15 | 6/15 | 7/15 | 16/83 | 24/83 | 0.326 | 83 | 259 | 223/259 | 210 | 13 | 0.228 | 259/259 | answer:83, ocr:69, text_search:107 |
| scienceqa | 15 | 13/15 | 13/15 | 35/46 | 37/46 | 0.797 | 46 | 112 | 103/112 | 101 | 2 | 0.753 | 112/112 | answer:46, ocr:36, text_search:30 |
| mrag_bench | 15 | 13/15 | 13/15 | 36/76 | 36/76 | 0.601 | 76 | 210 | 183/210 | 175 | 8 | 0.538 | 210/210 | answer:76, image_search:65, ocr:63, text_search:6 |
| crag_mm | 15 | 5/15 | 8/15 | 24/121 | 35/121 | 0.283 | 121 | 387 | 354/387 | 354 | 0 | 0.256 | 387/387 | answer:121, image_search:103, ocr:98, text_search:65 |

### OCR off

| dataset | queries | query_hit_em | query_hit_soft | traj_em | traj_soft | mean_f1 | trajectories | steps | kept | train | val | mean_score | rationale_qc | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 15 | 6/15 | 7/15 | 6/41 | 11/41 | 0.292 | 41 | 99 | 63/99 | 58 | 5 | 0.224 | 99/99 | answer:41, text_search:58 |
| scienceqa | 15 | 13/15 | 13/15 | 25/31 | 25/31 | 0.806 | 31 | 67 | 61/67 | 58 | 3 | 0.872 | 67/67 | answer:31, text_search:36 |
| mrag_bench | 15 | 9/15 | 9/15 | 11/23 | 11/23 | 0.625 | 23 | 47 | 38/47 | 34 | 4 | 0.575 | 47/47 | answer:23, image_search:23, text_search:1 |
| crag_mm | 15 | 4/15 | 6/15 | 11/45 | 14/45 | 0.375 | 45 | 108 | 96/108 | 96 | 0 | 0.321 | 108/108 | answer:45, image_search:37, text_search:26 |

## OCR Evidence Stats

| setting | dataset | trajectories | mean_traj_len | ocr_empty | ocr_avg_chars | actions |
| --- | --- | --- | --- | --- | --- | --- |
| OCR on | infoseek | 83 | 3.12 | 65/69 | 1.7 | answer:83, ocr:69, text_search:107 |
| OCR on | scienceqa | 46 | 2.43 | 26/36 | 24.5 | answer:46, ocr:36, text_search:30 |
| OCR on | mrag_bench | 76 | 2.76 | 59/63 | 1.6 | answer:76, image_search:65, ocr:63, text_search:6 |
| OCR on | crag_mm | 121 | 3.20 | 68/98 | 14.4 | answer:121, image_search:103, ocr:98, text_search:65 |
| OCR off | infoseek | 41 | 2.41 | - | - | answer:41, text_search:58 |
| OCR off | scienceqa | 31 | 2.16 | - | - | answer:31, text_search:36 |
| OCR off | mrag_bench | 23 | 2.04 | - | - | answer:23, image_search:23, text_search:1 |
| OCR off | crag_mm | 45 | 2.40 | - | - | answer:45, image_search:37, text_search:26 |

## Short Analysis

- Rollout=10 increases exploration, but also magnifies the cost of every enabled action. OCR-on creates far more trajectories and step-level samples than OCR-off.
- InfoSeek: query-level metrics are identical with OCR on/off, while OCR text is mostly empty. OCR should stay gated or disabled for InfoSeek-like encyclopedia image QA.
- ScienceQA: query-level metrics are identical. OCR-off has slightly higher mean_f1 and much higher mean_score, so OCR is not necessary here.
- MRAG-Bench: OCR-on improves query EM/soft from 9/15 to 13/15, but OCR text is mostly empty. The gain is probably from added MCTS branching/exploration, not reliable OCR evidence.
- CRAG-MM: OCR-on improves query soft hit from 6/15 to 8/15, but OCR-off has higher mean_f1 and mean_score. CRAG-MM is still limited by retrieval/entity/long-answer matching.
- Because cross_modal=false, text_to_image and image_to_text were not included in this ablation. Treat this report as OCR-only ablation under the current base action space.
