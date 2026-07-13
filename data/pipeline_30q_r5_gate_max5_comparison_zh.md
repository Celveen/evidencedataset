# EvidenceTree Gate Max-Actions 控制实验报告

## 1. 实验目的

前一版 gate 对比中，不同 gate 使用了不同的 `max_actions`：

| gate | 原 max_actions |
| --- | ---: |
| dataset gate | 5 |
| heuristic gate | 4 |
| strict gate | 3 |

这会带来一个混淆因素：不同 gate 的表现差异，可能不只是来自 gate 规则本身，也可能来自每一步保留的候选动作数量不同。

因此这次实验统一设置：

```text
dataset gate   max_actions = 5
heuristic gate max_actions = 5
strict gate    max_actions = 5
```

这样可以尽量排除 `max_actions` 的影响，更直接地观察不同 gate 的过滤和排序规则本身是否有效。

## 2. 实验设置

| 项目 | 设置 |
| --- | --- |
| 请求样本数 | 每个数据集 30 条 |
| 实际样本数 | InfoSeek 30，ScienceQA 25，MRAG-Bench 25，CRAG-MM 25 |
| rollout | 5 |
| OCR | 关闭 |
| cross_modal | 开启 |
| 测试步骤 | step1/step2 |
| rationale | 未生成，因此不评估 rationale QC |
| 统一 max_actions | 5 |

输出目录：

```text
data/pipeline_30q_r5_gate_dataset_max5_ocr_off
data/pipeline_30q_r5_gate_heuristic_max5_ocr_off
data/pipeline_30q_r5_gate_strict_max5_ocr_off
```

## 3. 整体结果

这里的“支持新动作数据集”指：

```text
InfoSeek + ScienceQA + MRAG-Bench
```

CRAG-MM 当前使用官方分开的 web/image index，不真正使用 `text_to_image` / `image_to_text`，因此需要单独看。

| mode | supported EM | supported soft | supported F1 | supported steps | cost vs old | all EM | all soft | all F1 | all steps | cost vs old |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old actions | 0.512 | 0.550 | 0.512 | 334 | 1.00x | 0.457 | 0.514 | 0.470 | 502 | 1.00x |
| full new actions | 0.650 | 0.662 | 0.535 | 1282 | 3.84x | 0.543 | 0.590 | 0.471 | 1410 | 2.81x |
| dataset gate max5 | 0.600 | 0.650 | 0.537 | 911 | 2.73x | 0.514 | 0.571 | 0.484 | 1067 | 2.13x |
| heuristic gate max5 | 0.637 | 0.662 | 0.541 | 819 | 2.45x | 0.552 | 0.619 | 0.508 | 958 | 1.91x |
| strict gate max5 | 0.625 | 0.675 | 0.554 | 811 | 2.43x | 0.543 | 0.629 | 0.505 | 950 | 1.89x |

## 4. 分数据集结果

| mode | dataset | EM | soft | F1 | trajectories | steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| old actions | infoseek | 7/30 | 10/30 | 0.264 | 62 | 139 |
| old actions | scienceqa | 21/25 | 21/25 | 0.812 | 48 | 102 |
| old actions | mrag_bench | 13/25 | 13/25 | 0.509 | 43 | 93 |
| old actions | crag_mm | 7/25 | 10/25 | 0.337 | 69 | 168 |
| full new actions | infoseek | 13/30 | 14/30 | 0.276 | 143 | 482 |
| full new actions | scienceqa | 23/25 | 23/25 | 0.825 | 122 | 409 |
| full new actions | mrag_bench | 16/25 | 16/25 | 0.555 | 124 | 391 |
| full new actions | crag_mm | 5/25 | 9/25 | 0.269 | 57 | 128 |
| dataset gate max5 | infoseek | 8/30 | 12/30 | 0.251 | 141 | 421 |
| dataset gate max5 | scienceqa | 22/25 | 22/25 | 0.855 | 58 | 98 |
| dataset gate max5 | mrag_bench | 18/25 | 18/25 | 0.562 | 122 | 392 |
| dataset gate max5 | crag_mm | 6/25 | 8/25 | 0.316 | 66 | 156 |
| heuristic gate max5 | infoseek | 11/30 | 13/30 | 0.304 | 121 | 324 |
| heuristic gate max5 | scienceqa | 22/25 | 22/25 | 0.837 | 56 | 100 |
| heuristic gate max5 | mrag_bench | 18/25 | 18/25 | 0.530 | 125 | 395 |
| heuristic gate max5 | crag_mm | 7/25 | 12/25 | 0.402 | 59 | 139 |
| strict gate max5 | infoseek | 9/30 | 13/30 | 0.327 | 120 | 337 |
| strict gate max5 | scienceqa | 22/25 | 22/25 | 0.801 | 51 | 90 |
| strict gate max5 | mrag_bench | 19/25 | 19/25 | 0.580 | 124 | 384 |
| strict gate max5 | crag_mm | 7/25 | 12/25 | 0.347 | 59 | 139 |

## 5. 关键观察

### 5.1 统一 max_actions 后，gate 差异仍然存在

如果不同 gate 的差异主要来自 `max_actions`，那么统一成 5 后，`dataset / heuristic / strict` 的结果应该明显接近。

但结果并不是这样：

| gate | supported EM | supported soft | supported F1 | supported steps |
| --- | ---: | ---: | ---: | ---: |
| dataset max5 | 0.600 | 0.650 | 0.537 | 911 |
| heuristic max5 | 0.637 | 0.662 | 0.541 | 819 |
| strict max5 | 0.625 | 0.675 | 0.554 | 811 |

说明 gate 的过滤规则、动作排序规则本身确实会影响 MCTS 轨迹，而不只是 `max_actions` 在起作用。

### 5.2 dataset gate 在 max5 下不再是最优

前一次实验里，dataset gate 是综合最稳的方案。但这次统一 max=5 后：

- dataset gate 的 supported EM 为 0.600；
- heuristic gate 为 0.637；
- strict gate 为 0.625。

dataset gate 的 supported steps 也是最高的：

```text
dataset max5:   911
heuristic max5: 819
strict max5:    811
```

这说明：当 heuristic / strict 不再因为 max_actions 较小而少拿候选时，它们可以保留足够探索，同时继续利用问题级规则减少无效路径。

### 5.3 strict max5 这次表现最好，但不宜过度下结论

从 supported soft 和 supported F1 看，strict max5 这次最好：

| gate | supported soft | supported F1 |
| --- | ---: | ---: |
| dataset max5 | 0.650 | 0.537 |
| heuristic max5 | 0.662 | 0.541 |
| strict max5 | 0.675 | 0.554 |

它还保持了较低 steps：

```text
strict max5: 811
```

不过需要注意：这次样本量仍然较小，MRAG-Bench 和 ScienceQA 实际只有 25 条。strict max5 的优势还需要更大样本验证。

比较稳妥的说法是：

```text
strict gate 本身并不一定差；
前一轮 strict 表现弱，部分原因可能是 max_actions=3 太窄。
当 max_actions 提到 5 后，strict 的规则过滤反而可能变成有效约束。
```

### 5.4 MRAG-Bench 的结论更清楚了

MRAG-Bench 结果：

| mode | MRAG-Bench EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| old actions | 13/25 | 13/25 | 0.509 | 93 |
| full new actions | 16/25 | 16/25 | 0.555 | 391 |
| dataset max5 | 18/25 | 18/25 | 0.562 | 392 |
| heuristic max5 | 18/25 | 18/25 | 0.530 | 395 |
| strict max5 | 19/25 | 19/25 | 0.580 | 384 |

这里可以看出，MRAG-Bench 的提升不是简单来自 `max_actions` 数量。因为三种 gate 都是 max=5，但结果仍有区别。

更合理的解释是：

```text
MRAG-Bench 对图像/跨模态动作非常敏感；
gate 的动作排序和问题级过滤会改变实际进入 MCTS 的动作实例；
这些调度差异会影响搜索树结构和最终答案。
```

### 5.5 InfoSeek 更适合 heuristic/strict max5

InfoSeek 结果：

| mode | InfoSeek EM | soft | F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| full new actions | 13/30 | 14/30 | 0.276 | 482 |
| dataset max5 | 8/30 | 12/30 | 0.251 | 421 |
| heuristic max5 | 11/30 | 13/30 | 0.304 | 324 |
| strict max5 | 9/30 | 13/30 | 0.327 | 337 |

InfoSeek 里 heuristic/strict max5 明显比 dataset max5 更省，而且 F1 更高。

这说明 dataset gate 对 InfoSeek 可能仍然太宽，保留了较多视觉分支；heuristic/strict 使用问题特征过滤后，能减少一些无效检索。

## 6. 更新后的建议

这次控制实验后，推荐策略需要调整。

### 6.1 不建议继续用不同 max_actions 比较 gate

后续比较 gate 时，应该统一：

```bash
--set action_gate.max_actions=5
```

否则 `dataset / heuristic / strict` 的区别会混入候选数量差异，不利于判断 gate 规则本身。

### 6.2 当前更推荐 heuristic max5 或 strict max5

如果只看这次实验，推荐优先级变为：

| 场景 | 推荐 |
| --- | --- |
| 稳妥默认 | heuristic gate max5 |
| 更激进、追求更高 soft/F1 | strict gate max5 |
| 保守、不想依赖问题关键词规则 | dataset gate max5 |
| 最大探索上限诊断 | full new actions |

我更倾向于把正式默认从原来的 `dataset gate max5` 调整为：

```bash
--set retriever.cross_modal=true \
--set retriever.ocr.enabled=false \
--set action_gate.enabled=true \
--set action_gate.mode=heuristic \
--set action_gate.max_actions=5
```

原因：

- 比 dataset max5 更省；
- supported EM 和 F1 更高；
- 比 strict max5 保守一点，不至于过度依赖激进规则。

如果后续再扩大样本，比如每个数据集 100 条，strict max5 仍然稳定更好，可以再考虑把默认改成 strict max5。

## 7. 一句话结论

统一 `max_actions=5` 后，gate 之间的差异仍然明显，说明 gate 的规则过滤和动作排序本身确实影响 MCTS。

这次控制实验中：

```text
heuristic max5 和 strict max5 都优于 dataset max5；
strict max5 的 soft/F1 最好；
heuristic max5 更适合作为当前稳妥默认。
```

