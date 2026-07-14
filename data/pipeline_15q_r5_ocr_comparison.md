# OCR Ablation: 15 queries, rollouts=5

Run settings:
- Datasets: infoseek, scienceqa, mrag_bench, crag_mm
- N=15 per dataset
- mcts.rollouts=5
- retriever.cross_modal=false for both runs
- Compared retriever.ocr.enabled=true vs false

## Comparison Table

| dataset | query_em_on | query_em_off | delta_query_em | query_soft_on | query_soft_off | delta_query_soft | mean_f1_on | mean_f1_off | delta_f1 | traj_em_on | traj_em_off | delta_traj_em | steps_on | steps_off | kept_on | kept_off |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 5/15 | 5/15 | +0.000 | 6/15 | 7/15 | -0.067 | 0.276 | 0.254 | +0.023 | 11/59 | 5/34 | +0.039 | 168 | 81 | 141/168 | 52/81 |
| scienceqa | 13/15 | 13/15 | +0.000 | 13/15 | 13/15 | +0.000 | 0.897 | 0.862 | +0.035 | 36/42 | 25/29 | -0.005 | 100 | 59 | 94/100 | 55/59 |
| mrag_bench | 12/15 | 8/15 | +0.267 | 12/15 | 8/15 | +0.267 | 0.709 | 0.536 | +0.173 | 36/58 | 8/21 | +0.240 | 160 | 43 | 148/160 | 32/43 |
| crag_mm | 5/15 | 4/15 | +0.067 | 8/15 | 5/15 | +0.200 | 0.327 | 0.247 | +0.080 | 16/73 | 4/34 | +0.102 | 210 | 80 | 192/210 | 76/80 |

## Full Summaries

### OCR on

| dataset | queries | query_hit_em | query_hit_soft | traj_em | traj_soft | mean_f1 | trajectories | steps | kept | train | val | mean_score | rationale_qc | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 15 | 5/15 | 6/15 | 11/59 | 15/59 | 0.276 | 59 | 168 | 141/168 | 134 | 7 | 0.195 | 168/168 | answer:59, ocr:47, text_search:62 |
| scienceqa | 15 | 13/15 | 13/15 | 36/42 | 38/42 | 0.897 | 42 | 100 | 94/100 | 89 | 5 | 0.860 | 100/100 | answer:42, ocr:32, text_search:26 |
| mrag_bench | 15 | 12/15 | 12/15 | 36/58 | 36/58 | 0.709 | 58 | 160 | 148/160 | 133 | 15 | 0.679 | 160/160 | answer:58, image_search:53, ocr:48, text_search:1 |
| crag_mm | 15 | 5/15 | 8/15 | 16/73 | 24/73 | 0.327 | 73 | 210 | 192/210 | 192 | 0 | 0.288 | 210/210 | answer:73, image_search:58, ocr:39, text_search:40 |

### OCR off

| dataset | queries | query_hit_em | query_hit_soft | traj_em | traj_soft | mean_f1 | trajectories | steps | kept | train | val | mean_score | rationale_qc | actions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 15 | 5/15 | 7/15 | 5/34 | 11/34 | 0.254 | 34 | 81 | 52/81 | 49 | 3 | 0.169 | 81/81 | answer:34, text_search:47 |
| scienceqa | 15 | 13/15 | 13/15 | 25/29 | 25/29 | 0.862 | 29 | 59 | 55/59 | 53 | 2 | 0.874 | 59/59 | answer:29, text_search:30 |
| mrag_bench | 15 | 8/15 | 8/15 | 8/21 | 8/21 | 0.536 | 21 | 43 | 32/43 | 30 | 2 | 0.498 | 43/43 | answer:21, image_search:21, text_search:1 |
| crag_mm | 15 | 4/15 | 5/15 | 4/34 | 5/34 | 0.247 | 34 | 80 | 76/80 | 76 | 0 | 0.207 | 80/80 | answer:34, image_search:30, text_search:16 |

## OCR Evidence Stats

| setting | dataset | trajectories | mean_traj_len | ocr_empty | ocr_avg_chars | actions |
| --- | --- | --- | --- | --- | --- | --- |
| OCR on | infoseek | 59 | 2.85 | 44/47 | 1.9 | answer:59, ocr:47, text_search:62 |
| OCR on | scienceqa | 42 | 2.38 | 22/32 | 22.6 | answer:42, ocr:32, text_search:26 |
| OCR on | mrag_bench | 58 | 2.76 | 46/48 | 1.6 | answer:58, image_search:53, ocr:48, text_search:1 |
| OCR on | crag_mm | 73 | 2.88 | 27/39 | 10.4 | answer:73, image_search:58, ocr:39, text_search:40 |
| OCR off | infoseek | 34 | 2.38 | - | - | answer:34, text_search:47 |
| OCR off | scienceqa | 29 | 2.03 | - | - | answer:29, text_search:30 |
| OCR off | mrag_bench | 21 | 2.05 | - | - | answer:21, image_search:21, text_search:1 |
| OCR off | crag_mm | 34 | 2.35 | - | - | answer:34, image_search:30, text_search:16 |

## Short Analysis

- OCR increases trajectory/step count substantially, because it adds another action branch.
- InfoSeek: OCR output is almost always empty (44/47). Query EM is unchanged, soft hit is worse with OCR, but mean F1 is slightly higher. Treat OCR as mostly noise for InfoSeek.
- ScienceQA: OCR does not change query-level hits, but slightly improves mean F1 while mean_score is slightly lower. Effect is small.
- MRAG-Bench: OCR-on is much better in this run, but OCR text is almost always empty (46/48). The improvement is likely from extra exploration/branching rather than useful OCR content.
- CRAG-MM: OCR-on improves EM/soft/F1 in this run, but many OCR outputs are empty/noisy (27/39 empty). It likely helps by increasing action diversity, not because OCR is a reliable evidence source.
- Because cross_modal=false, the newly merged text_to_image/image_to_text actions were not part of this OCR ablation.
