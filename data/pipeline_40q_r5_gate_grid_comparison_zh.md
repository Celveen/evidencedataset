# EvidenceTree Gate 网格实验报告：max_actions=3/4/5，N=40，rollout=5

## 1. 实验目的

这次实验主要解决两个问题：

1. **索引重复构建问题**

   之前每次切换 gate 或 `max_actions`，MRAG-Bench 都会重新构建 cross-modal CLIP 索引，耗时很高。现在已经为 `CrossModalCLIPRetriever` 增加本地缓存：

   ```text
   data/corpus/infoseek/cross_modal_clip_index
   data/corpus/scienceqa/cross_modal_clip_index
   data/corpus/mrag_bench/cross_modal_clip_index
   ```

   后续运行会优先加载本地缓存，不再重复建索引。

2. **系统比较 gate 与 max_actions 的组合**

   这次统一测试：

   ```text
   gate ∈ {dataset, heuristic, strict}
   max_actions ∈ {3, 4, 5}
   ```

   每个组合都跑所有当前保留的数据集，观察 gate 规则和动作候选数量对 MCTS 的影响。

## 2. 实验设置

| 项目 | 设置 |
| --- | --- |
| 请求样本数 | 每个数据集 40 条 |
| 实际样本数 | InfoSeek 40，ScienceQA 25，MRAG-Bench 25，CRAG-MM 25 |
| rollout | 5 |
| OCR | 关闭 |
| cross_modal | 开启 |
| 测试步骤 | step1/step2 |
| rationale | 当前保留的 supported 数据集已全部补齐 step3 rationale，并重新统计 rationale QC |
| 测试组合 | 3 个 gate × 3 个 max_actions = 9 组 |

注意：虽然请求每个数据集 40 条，但当前本地 prepared corpus 中 ScienceQA、MRAG-Bench、CRAG-MM 只有 25 条可测样本。因此这次实际是：

```text
InfoSeek 40 + ScienceQA 25 + MRAG-Bench 25 + CRAG-MM 25
```

输出根目录：

```text
data/pipeline_40q_r5_gate_grid/
```

汇总表：

```text
data/pipeline_40q_r5_gate_grid/aggregate_tables.md
```

## 3. 索引缓存改动

新增缓存逻辑后，cross-modal CLIP 索引会保存为：

```text
image_index.faiss
text_index.faiss
image_docs.jsonl
text_docs.jsonl
meta.json
```

加载时会检查缓存中的文档总数是否和当前 corpus 一致：

```text
一致：直接加载缓存
不一致：忽略旧缓存并重建
```

本次运行中已经确认缓存生效。例如后续组合会出现：

```text
Loaded cached cross-modal CLIP index: data/corpus/mrag_bench/cross_modal_clip_index
```

CRAG-MM 仍然不使用这个 cross-modal 缓存，因为它当前走官方分开的 web/image index。

相关代码：

```text
src/evidencetree/actions/retrievers.py
DatasetConstruct/step1_gen_trajectories.py
```

验证：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-qwen25vl/bin/python -m pytest \
  tests/test_cross_modal.py tests/test_retrievers.py -q
```

结果：

```text
20 passed
```

## 4. Supported 数据集整体结果

这里的 supported 数据集指真正能使用新 cross-modal 动作的数据集：

```text
InfoSeek + ScienceQA + MRAG-Bench
```

CRAG-MM 当前会忽略 `cross_modal=true`，所以不放进 supported 主结论里。

| mode | supported queries | EM | soft | F1 | steps | cost vs old | trajectories | rationale QC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_actions | 80 | 0.512 | 0.550 | 0.512 | 334 | 1.00x | 153 | 334/334 (1.000) |
| full_new | 80 | 0.650 | 0.662 | 0.535 | 1282 | 3.84x | 389 | 975/1282 (0.761) |
| dataset_max3 | 90 | 0.544 | 0.589 | 0.510 | 1036 | 3.10x | 354 | 842/1036 (0.813) |
| dataset_max4 | 90 | 0.589 | 0.656 | 0.508 | 1028 | 3.08x | 361 | 835/1028 (0.812) |
| dataset_max5 | 90 | 0.578 | 0.633 | 0.525 | 1016 | 3.04x | 356 | 810/1016 (0.797) |
| heuristic_max3 | 90 | 0.589 | 0.622 | 0.536 | 915 | 2.74x | 330 | 758/915 (0.828) |
| heuristic_max4 | 90 | 0.544 | 0.611 | 0.524 | 897 | 2.69x | 334 | 744/897 (0.829) |
| heuristic_max5 | 90 | 0.589 | 0.644 | 0.522 | 875 | 2.62x | 324 | 721/875 (0.824) |
| strict_max3 | 90 | 0.611 | 0.667 | 0.515 | 910 | 2.72x | 338 | 757/910 (0.832) |
| strict_max4 | 90 | 0.589 | 0.644 | 0.526 | 873 | 2.61x | 327 | 713/873 (0.817) |
| strict_max5 | 90 | 0.567 | 0.622 | 0.531 | 902 | 2.70x | 332 | 732/902 (0.812) |

说明：

- `old_actions` 和 `full_new` 是之前 N=30 实验的参照值，supported queries 为 80。
- 本次 9 个 gate-grid 组合是新跑的 N=40 请求实验，supported queries 为 90。
- 因为 query 数量不同，`old_actions/full_new` 只作为参考，不作为严格同分布对比。
- `rationale QC` 表示 step-level rationale 通过基础质检的比例。当前表中的 gate-grid QC 已经全部补齐，统计范围为当前保留的 supported 数据集：InfoSeek、ScienceQA、MRAG-Bench。

## 5. 全数据集整体结果

| mode | all queries | EM | soft | F1 | steps | cost vs old | trajectories | rationale QC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old_actions | 105 | 0.457 | 0.514 | 0.470 | 502 | 1.00x | 222 | 502/502 (1.000) |
| full_new | 105 | 0.543 | 0.590 | 0.471 | 1410 | 2.81x | 446 | 1103/1410 (0.782) |
| dataset_max3 | 115 | 0.478 | 0.548 | 0.469 | 1214 | 2.42x | 427 | 842/1036 (0.813, supported only) |
| dataset_max4 | 115 | 0.513 | 0.609 | 0.456 | 1160 | 2.31x | 417 | 835/1028 (0.812, supported only) |
| dataset_max5 | 115 | 0.504 | 0.600 | 0.464 | 1150 | 2.29x | 413 | 810/1016 (0.797, supported only) |
| heuristic_max3 | 115 | 0.513 | 0.565 | 0.511 | 1072 | 2.14x | 395 | 758/915 (0.828, supported only) |
| heuristic_max4 | 115 | 0.478 | 0.548 | 0.475 | 1053 | 2.10x | 397 | 744/897 (0.829, supported only) |
| heuristic_max5 | 115 | 0.513 | 0.609 | 0.473 | 1047 | 2.09x | 394 | 721/875 (0.824, supported only) |
| strict_max3 | 115 | 0.539 | 0.609 | 0.467 | 1075 | 2.14x | 405 | 757/910 (0.832, supported only) |
| strict_max4 | 115 | 0.513 | 0.609 | 0.490 | 1041 | 2.07x | 395 | 713/873 (0.817, supported only) |
| strict_max5 | 115 | 0.496 | 0.583 | 0.481 | 1075 | 2.14x | 401 | 732/902 (0.812, supported only) |

## 6. 分数据集结果

这部分是 gate-grid 的分数据集结果。当前保留的 supported 数据集已经全部补齐 rationale，QC 如下：

| mode | infoseek QC | scienceqa QC | mrag_bench QC | status |
| --- | ---: | ---: | ---: | --- |
| dataset_max3 | 443/578 | 59/71 | 340/387 | complete |
| dataset_max4 | 426/557 | 79/91 | 330/380 | complete |
| dataset_max5 | 420/560 | 64/79 | 326/377 | complete |
| heuristic_max3 | 323/415 | 85/100 | 350/400 | complete |
| heuristic_max4 | 321/411 | 81/96 | 342/390 | complete |
| heuristic_max5 | 313/401 | 67/83 | 341/391 | complete |
| strict_max3 | 343/431 | 82/99 | 332/380 | complete |
| strict_max4 | 310/409 | 72/83 | 331/381 | complete |
| strict_max5 | 328/437 | 71/82 | 333/383 | complete |

说明：`a/b` 表示 `QC pass / scored steps`。这些 scored steps 均已生成对应 rationale。

| mode | dataset | queries | EM | soft | F1 | trajectories | steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dataset_max3 | infoseek | 40 | 12/40 | 15/40 | 0.284 | 189 | 578 |
| dataset_max3 | scienceqa | 25 | 20/25 | 21/25 | 0.821 | 40 | 71 |
| dataset_max3 | mrag_bench | 25 | 17/25 | 17/25 | 0.561 | 125 | 387 |
| dataset_max3 | crag_mm | 25 | 6/25 | 10/25 | 0.323 | 73 | 178 |
| dataset_max4 | infoseek | 40 | 13/40 | 18/40 | 0.268 | 189 | 557 |
| dataset_max4 | scienceqa | 25 | 21/25 | 22/25 | 0.837 | 50 | 91 |
| dataset_max4 | mrag_bench | 25 | 19/25 | 19/25 | 0.564 | 122 | 380 |
| dataset_max4 | crag_mm | 25 | 6/25 | 11/25 | 0.269 | 56 | 132 |
| dataset_max5 | infoseek | 40 | 12/40 | 17/40 | 0.269 | 187 | 560 |
| dataset_max5 | scienceqa | 25 | 22/25 | 22/25 | 0.869 | 47 | 79 |
| dataset_max5 | mrag_bench | 25 | 18/25 | 18/25 | 0.592 | 122 | 377 |
| dataset_max5 | crag_mm | 25 | 6/25 | 12/25 | 0.242 | 57 | 134 |
| heuristic_max3 | infoseek | 40 | 15/40 | 18/40 | 0.307 | 148 | 415 |
| heuristic_max3 | scienceqa | 25 | 21/25 | 21/25 | 0.850 | 57 | 100 |
| heuristic_max3 | mrag_bench | 25 | 17/25 | 17/25 | 0.587 | 125 | 400 |
| heuristic_max3 | crag_mm | 25 | 6/25 | 9/25 | 0.423 | 65 | 157 |
| heuristic_max4 | infoseek | 40 | 13/40 | 19/40 | 0.326 | 154 | 411 |
| heuristic_max4 | scienceqa | 25 | 20/25 | 20/25 | 0.855 | 55 | 96 |
| heuristic_max4 | mrag_bench | 25 | 16/25 | 16/25 | 0.510 | 125 | 390 |
| heuristic_max4 | crag_mm | 25 | 6/25 | 8/25 | 0.298 | 63 | 156 |
| heuristic_max5 | infoseek | 40 | 10/40 | 15/40 | 0.285 | 150 | 401 |
| heuristic_max5 | scienceqa | 25 | 22/25 | 22/25 | 0.837 | 49 | 83 |
| heuristic_max5 | mrag_bench | 25 | 21/25 | 21/25 | 0.586 | 125 | 391 |
| heuristic_max5 | crag_mm | 25 | 6/25 | 12/25 | 0.299 | 70 | 172 |
| strict_max3 | infoseek | 40 | 14/40 | 19/40 | 0.284 | 157 | 431 |
| strict_max3 | scienceqa | 25 | 21/25 | 21/25 | 0.830 | 57 | 99 |
| strict_max3 | mrag_bench | 25 | 20/25 | 20/25 | 0.571 | 124 | 380 |
| strict_max3 | crag_mm | 25 | 7/25 | 10/25 | 0.295 | 67 | 165 |
| strict_max4 | infoseek | 40 | 11/40 | 16/40 | 0.304 | 155 | 409 |
| strict_max4 | scienceqa | 25 | 22/25 | 22/25 | 0.830 | 47 | 83 |
| strict_max4 | mrag_bench | 25 | 20/25 | 20/25 | 0.579 | 125 | 381 |
| strict_max4 | crag_mm | 25 | 6/25 | 12/25 | 0.358 | 68 | 168 |
| strict_max5 | infoseek | 40 | 13/40 | 18/40 | 0.303 | 162 | 437 |
| strict_max5 | scienceqa | 25 | 21/25 | 21/25 | 0.866 | 46 | 82 |
| strict_max5 | mrag_bench | 25 | 17/25 | 17/25 | 0.561 | 124 | 383 |
| strict_max5 | crag_mm | 25 | 6/25 | 11/25 | 0.302 | 69 | 173 |

## 7. 关键观察

### 7.1 dataset gate 不再是最优默认

在 N=30 的早期实验里，dataset gate 曾经看起来较稳。但在这次更完整的 3×3 网格里，dataset gate 的成本偏高，整体收益不突出。

supported 数据集上：

| mode | EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| dataset_max3 | 0.544 | 0.589 | 0.510 | 1036 |
| dataset_max4 | 0.589 | 0.656 | 0.508 | 1028 |
| dataset_max5 | 0.578 | 0.633 | 0.525 | 1016 |

dataset gate 的问题是：它按数据集整体先验保留动作，缺少问题级过滤，因此 InfoSeek 的轨迹和 steps 明显偏多。

### 7.2 strict_max3 的 EM/soft 最强

supported 数据集上，`strict_max3` 的 EM 和 soft 最好：

```text
EM   = 0.611
soft = 0.667
```

这说明更强的问题级 gate 约束是有价值的。它把动作空间压得更窄，但没有明显牺牲命中率，反而减少了无效路径。

### 7.3 heuristic_max3 的 F1 最好，全数据集表现也最好

supported 数据集上，`heuristic_max3` 的 F1 最高：

```text
F1 = 0.536
```

全数据集上，`heuristic_max3` 的 F1 也是最高：

```text
all F1 = 0.511
```

这说明它生成的答案虽然不一定在 EM/soft 上最强，但语义相似度和部分匹配更好。

### 7.4 strict_max4 是成本最低的强候选

supported 数据集中，`strict_max4` 的 steps 最低：

```text
steps = 873
```

同时它的表现并不差：

```text
EM   = 0.589
soft = 0.644
F1   = 0.526
```

如果追求“省成本 + 不明显掉性能”，`strict_max4` 是很好的候选。

### 7.5 max_actions 不是越大越好

从结果看，`max_actions=5` 并不稳定优于 3 或 4。

例如 strict：

| mode | EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| strict_max3 | 0.611 | 0.667 | 0.515 | 910 |
| strict_max4 | 0.589 | 0.644 | 0.526 | 873 |
| strict_max5 | 0.567 | 0.622 | 0.531 | 902 |

增加候选动作会给 MCTS 更多选择，但也会引入更多噪声分支。当前小样本下，3/4 往往比 5 更合适。

## 8. 数据集维度分析

### InfoSeek

InfoSeek 最适合 heuristic/strict 的问题级 gate。

表现较好的组合：

| mode | EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| heuristic_max3 | 15/40 | 18/40 | 0.307 | 415 |
| heuristic_max4 | 13/40 | 19/40 | 0.326 | 411 |
| strict_max3 | 14/40 | 19/40 | 0.284 | 431 |

InfoSeek 里 dataset gate 的 step 明显偏多，说明只按数据集先验不够，需要看问题文本特征。

### ScienceQA

ScienceQA 整体都比较高，差距不大。它主要依赖 `text_search + answer`，跨模态动作不是主收益来源。

最高 F1 是：

```text
dataset_max5: 0.869
strict_max5:  0.866
```

但 ScienceQA 当前只有 25 条，差异不要过度解释。

### MRAG-Bench

MRAG-Bench 受 gate 影响明显。

表现较好的组合：

| mode | EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| heuristic_max5 | 21/25 | 21/25 | 0.586 | 391 |
| strict_max3 | 20/25 | 20/25 | 0.571 | 380 |
| strict_max4 | 20/25 | 20/25 | 0.579 | 381 |

如果只看 MRAG-Bench，`heuristic_max5` 的命中率最高；如果兼顾成本，`strict_max3/4` 更划算。

### CRAG-MM

CRAG-MM 当前不真正使用新增 cross-modal 动作，因此 gate 对它的意义不同。

较好的组合：

```text
strict_max3: EM 7/25
heuristic_max3: F1 0.423
strict_max4: soft 12/25, F1 0.358
```

CRAG-MM 的结果波动较大，不建议用它决定 cross-modal gate 默认策略。

## 9. 更新后的推荐

### 9.1 如果优先追求成功率

推荐：

```bash
--set action_gate.mode=strict \
--set action_gate.max_actions=3
```

理由：

- supported EM 最高；
- supported soft 最高；
- 成本比 full_new 低很多。

### 9.2 如果优先追求语义接近和答案质量

推荐：

```bash
--set action_gate.mode=heuristic \
--set action_gate.max_actions=3
```

理由：

- supported F1 最高；
- all F1 最高；
- InfoSeek 表现较好；
- 规则比 strict 稍微没那么激进。

### 9.3 如果优先追求成本控制

推荐：

```bash
--set action_gate.mode=strict \
--set action_gate.max_actions=4
```

理由：

- supported steps 最低；
- 性能仍处于第一梯队；
- 比 max5 更干净。

### 9.4 当前综合默认建议

如果现在要选一个默认方案，我建议用：

```bash
--set retriever.cross_modal=true \
--set retriever.ocr.enabled=false \
--set action_gate.enabled=true \
--set action_gate.mode=heuristic \
--set action_gate.max_actions=3
```

原因：

- F1 最稳；
- InfoSeek 和 CRAG-MM 的语义匹配表现较好；
- 比 dataset gate 成本低；
- 比 strict gate 稍微更保守，作为默认更稳。

如果后续只看 EM/soft，不太关心 F1，可以把默认切成：

```bash
--set action_gate.mode=strict \
--set action_gate.max_actions=3
```

## 10. 一句话结论

这次 3×3 网格实验说明：

```text
索引缓存已经解决重复建索引问题；
dataset gate 不适合作为当前默认；
max_actions 不是越大越好；
heuristic_max3 是当前最稳默认；
strict_max3 是最高命中率方案；
strict_max4 是最低成本强候选。
```
