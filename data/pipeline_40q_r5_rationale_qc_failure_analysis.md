# Rationale QC 失败原因分析：pipeline_40q_r5_gate_grid

统计范围：`data/pipeline_40q_r5_gate_grid` 中当前保留的 supported 数据集：InfoSeek、ScienceQA、MRAG-Bench。  
总 step-level rationale 数：8452。

## 1. 当前 QC 标准

代码位置：`src/evidencetree/prm/rationale_gen.py`

`check_rationale()` 目前只有 3 条硬规则：

| QC 条件 | 具体含义 |
| --- | --- |
| 长度合格 | rationale 的 whitespace token 数必须在 `[30, 150]` 内 |
| 提到动作类型 | rationale 文本里必须显式包含当前 action type，例如 `text_search`、`image_search`、`text_to_image` |
| 引用可见证据 | rationale 必须引用至少一个当前样本可见的 evidence id，例如 `[e0]`；可见证据来自 `state.evidence_before` 或 `observation_evidence` |

如果不通过，会最多重试 `max_attempts=3` 次；仍不通过就标记为 `rationale_qc_pass=false`，后续 step4 会过滤掉。

## 2. 总体结果

| 项目 | 数量 | 比例 |
| --- | ---: | ---: |
| 总 rationale | 8452 | 100.00% |
| QC 通过 | 6912 | 81.78% |
| QC 失败 | 1540 | 18.22% |

## 3. 失败原因比例

注意：这里的失败原因是按当前 QC 规则统计。结果非常集中：所有失败都来自同一个原因。

| 失败原因 | 数量 | 占失败样本 | 占全部样本 |
| --- | ---: | ---: | ---: |
| cites no evidence_id visible in the sample | 1540 | 100.00% | 18.22% |
| length outside [30, 150] | 0 | 0.00% | 0.00% |
| does not mention the action type | 0 | 0.00% | 0.00% |

进一步拆开看，这 1540 条失败全部属于：

| 子类型 | 数量 | 含义 |
| --- | ---: | --- |
| no_visible_evidence_available | 1540 | 当前 step 的 `state.evidence_before` 和 `observation_evidence` 都没有任何 evidence_id，因此模型无论怎么写都无法合法引用“可见证据” |

也就是说，这批失败主要不是因为 DeepSeek 忘记写动作名、写太短/太长，或者引用格式错；根本原因是 **该 step 没有可引用的证据**。

## 4. 按 gate 配置统计

| mode | total | failed | failed rate | pass rate |
| --- | ---: | ---: | ---: | ---: |
| dataset_max3 | 1036 | 194 | 18.73% | 81.27% |
| dataset_max4 | 1028 | 193 | 18.77% | 81.23% |
| dataset_max5 | 1016 | 206 | 20.28% | 79.72% |
| heuristic_max3 | 915 | 157 | 17.16% | 82.84% |
| heuristic_max4 | 897 | 153 | 17.06% | 82.94% |
| heuristic_max5 | 875 | 154 | 17.60% | 82.40% |
| strict_max3 | 910 | 153 | 16.81% | 83.19% |
| strict_max4 | 873 | 160 | 18.33% | 81.67% |
| strict_max5 | 902 | 170 | 18.85% | 81.15% |

观察：`strict_max3` 的失败率最低，`dataset_max5` 的失败率最高。整体差距不大，主要都在 17%-20% 区间。

## 5. 按数据集统计

| dataset | total | failed | failed rate | pass rate |
| --- | ---: | ---: | ---: | ---: |
| InfoSeek | 4199 | 972 | 23.15% | 76.85% |
| ScienceQA | 784 | 124 | 15.82% | 84.18% |
| MRAG-Bench | 3469 | 444 | 12.80% | 87.20% |

观察：InfoSeek 的失败率最高，说明它更容易产生“没有任何证据可引用”的 step，尤其是在跨模态动作进入动作空间之后。

## 6. 按动作类型统计

| action | total | failed | failed rate | 失败原因 |
| --- | ---: | ---: | ---: | --- |
| answer | 3056 | 197 | 6.45% | no_visible_evidence_available |
| image_search | 1511 | 389 | 25.74% | no_visible_evidence_available |
| image_to_text | 1644 | 444 | 27.01% | no_visible_evidence_available |
| text_search | 952 | 0 | 0.00% | 无 |
| text_to_image | 1289 | 510 | 39.57% | no_visible_evidence_available |

观察：

- `text_search` 没有失败，说明文本检索基本都会产生可引用 evidence。
- `text_to_image` 失败率最高，接近 40%。这说明很多文本搜图动作没有返回可转成 evidence 的结果。
- `image_to_text`、`image_search` 的失败率也明显高于 `answer`。
- `answer` 也会失败，原因通常是 answer 节点前面没有累积到任何 evidence。

## 7. 一个典型失败例子

样本：`infoseek_val_00050494#t1#s0`  
动作：`text_to_image`  
失败原因：`cites no evidence_id visible in the sample`

rationale 片段：

```text
The text_to_image action received a score of 1.00 because it directly attempts to gather visual evidence about the bird's weight in grams, aligning with the question. However, no evidence was retrieved [e0] ...
```

这里模型写了 `[e0]`，但当前 step 的 `state.evidence_before` 和 `observation_evidence` 都是空的，所以 `[e0]` 并不是可见 evidence id。按当前 QC，它必须被过滤。

## 8. 结论

当前被过滤掉的 rationale 几乎可以归结为一句话：

```text
该 step 没有任何可见 evidence，因此无法满足“必须引用可见 evidence_id”的 QC 标准。
```

这不是普通的语言生成质量问题，而是动作/检索结果和 QC 规则之间的结构性冲突。尤其是跨模态动作中，如果检索没有返回 evidence，rationale 就天然很难通过当前 QC。

## 9. 后续可选处理方式

如果我们认为“无证据动作”也应该保留为负样本，可以考虑把 QC 改成两类：

| 类型 | 建议规则 |
| --- | --- |
| 有证据 step | 必须引用可见 evidence_id |
| 无证据 step | 允许不引用 evidence_id，但必须明确写出 `no evidence was retrieved`，并解释为什么该动作没有帮助 |

这样可以避免把大量“检索失败但有训练价值”的负样本直接过滤掉。
