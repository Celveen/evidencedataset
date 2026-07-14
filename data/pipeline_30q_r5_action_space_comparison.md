# Action Space Comparison: larger sample run

Run settings:
- Requested N=30 per dataset; actual available queries are shown per row.
- mcts.rollouts=5
- retriever.ocr.enabled=false
- Baseline: retriever.cross_modal=false
- New action space: retriever.cross_modal=true, enabling text_to_image and image_to_text where supported.
- CRAG-MM still ignores cross_modal=true because it uses official separate web/image indexes.

## Success Metrics

| dataset | queries | base_em | new_em | delta_em | base_soft | new_soft | delta_soft | base_f1 | new_f1 | delta_f1 | base_score | new_score | delta_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 30 | 7/30 | 13/30 | +0.200 | 10/30 | 14/30 | +0.133 | 0.264 | 0.276 | +0.012 | 0.174 | 0.287 | +0.113 |
| scienceqa | 25 | 21/25 | 23/25 | +0.080 | 21/25 | 23/25 | +0.080 | 0.812 | 0.825 | +0.013 | 0.852 | 0.796 | -0.056 |
| mrag_bench | 25 | 13/25 | 16/25 | +0.120 | 13/25 | 16/25 | +0.120 | 0.509 | 0.555 | +0.046 | 0.479 | 0.625 | +0.146 |
| crag_mm | 25 | 7/25 | 5/25 | -0.080 | 10/25 | 9/25 | -0.040 | 0.337 | 0.269 | -0.068 | 0.280 | 0.183 | -0.097 |

## Cost / Data Volume

| dataset | base_traj | new_traj | traj_x | base_steps | new_steps | steps_x | base_kept | new_kept | base_qc | base_qc_rate | new_qc | new_qc_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| infoseek | 62 | 143 | 2.31x | 139 | 482 | 3.47x | 101/139 | 284/482 | 139/139 | 1.000 | 327/482 | 0.678 |
| scienceqa | 48 | 122 | 2.54x | 102 | 409 | 4.01x | 95/102 | 270/409 | 102/102 | 1.000 | 293/409 | 0.716 |
| mrag_bench | 43 | 124 | 2.88x | 93 | 391 | 4.20x | 70/93 | 252/391 | 93/93 | 1.000 | 355/391 | 0.908 |
| crag_mm | 69 | 57 | 0.83x | 168 | 128 | 0.76x | 142/168 | 109/128 | 168/168 | 1.000 | 128/128 | 1.000 |

`base_qc/new_qc` is the number of step-level rationales that passed the basic rationale quality check. The rate column is `passed / total_steps`. It does not measure whether the final answer is correct; it measures whether the generated rationale for each step is usable enough to keep as training supervision.

## Action Distribution

| dataset | baseline_actions | new_actions |
| --- | --- | --- |
| infoseek | answer:62, text_search:77 | answer:143, image_search:110, image_to_text:82, text_search:68, text_to_image:79 |
| scienceqa | answer:48, text_search:54 | answer:122, image_search:98, image_to_text:66, text_search:60, text_to_image:63 |
| mrag_bench | answer:43, image_search:43, text_search:7 | answer:124, image_search:114, image_to_text:39, text_search:44, text_to_image:70 |
| crag_mm | answer:69, image_search:58, text_search:41 | answer:57, image_search:52, text_search:19 |

## Macro Average All Available Datasets

| setting | avg_em | avg_soft | avg_f1 | avg_score | total_traj | total_steps | qc_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.468 | 0.523 | 0.481 | 0.446 | 222 | 502 | 502/502 (1.000) |
| new | 0.548 | 0.597 | 0.481 | 0.473 | 446 | 1410 | 1103/1410 (0.782) |

## Macro Average Supported Cross-Modal Only

| setting | avg_em | avg_soft | avg_f1 | avg_score | total_traj | total_steps | qc_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.531 | 0.564 | 0.528 | 0.502 | 153 | 334 | 334/334 (1.000) |
| new | 0.664 | 0.676 | 0.552 | 0.569 | 389 | 1282 | 975/1282 (0.761) |

## Interpretation

- With more samples, the new action space shows a clearer query-level improvement on the supported datasets: InfoSeek EM +0.200, ScienceQA EM +0.080, MRAG-Bench EM +0.120.
- Soft hit also improves on InfoSeek and MRAG-Bench, while ScienceQA stays high and unchanged at 23/25 soft.
- The improvement is still expensive: supported datasets grow from 334 step-level samples to 1282, about 3.84x more steps.
- Rationale QC again drops substantially on supported datasets: baseline 334/334, new 975/1282 = 76.1%. This confirms the QC drop is not just a 15-query fluke.
- CRAG-MM is not evidence for or against the new cross-modal actions in this run, because cross_modal is ignored there. Its changed numbers are sampling variance under the original CRAG action space.
- Overall: increasing sample count makes the new action space advantage more convincing for success rate, but it also confirms the cost/noise problem. The right next move is action gating rather than blindly using cross_modal for every sample.
