# EvidenceTree 动作 Gate 对比实验报告

## 1. 实验目的

新版动作空间加入了更多跨模态检索动作，例如 `text_to_image` 和 `image_to_text`。这些动作能解决老动作空间中一些“必须先看图、再找文字”或“用文本去找相关图片”的问题，但代价也很明显：

- MCTS 每一步可选动作变多；
- rollout 会探索更多分支；
- step-level 数据数量增加；
- 收集时间和后续 rationale 生成成本变高；
- 一些弱相关检索步骤会拉低 rationale QC。

因此本次实验的目标是验证：**能不能用 gate 机制实时缩小动作空间，在保留跨模态收益的同时减少搜索成本。**

本次比较的对象包括：

| 模式 | 含义 |
| --- | --- |
| `old actions` | 老动作空间，只使用原有动作，主要是 `text_search`、`image_search`、`answer` |
| `full new actions` | 新动作空间全开，不做 gate 限制 |
| `dataset gate` | 按数据集类型决定允许哪些动作 |
| `heuristic gate` | 按数据集 + 问题文本特征动态决定动作 |
| `strict gate` | 更激进的 heuristic gate，进一步压缩候选动作 |

## 2. Gate 的基本原理

Gate 不是替代 MCTS，而是放在 **动作提议器 proposer 和 MCTS expansion 之间**。

完整流程可以理解为：

```text
当前搜索状态 state
  ↓
原始 proposer 生成候选动作
  ↓
GatedProposer 过滤不合适的动作
  ↓
GatedProposer 按优先级重排动作
  ↓
最多保留 max_actions 个动作
  ↓
MCTS 用剩下的动作继续展开搜索树
```

也就是说，gate 影响的是：**MCTS 当前节点下一步“能尝试哪些动作”。**

它不会直接改 Qwen 的回答，也不会直接改 PRM 打分，而是通过减少不必要的候选动作，降低搜索树膨胀。

当前实现位置：

```text
src/evidencetree/mcts/proposer.py
```

配置入口：

```bash
--set action_gate.enabled=true
--set action_gate.mode=dataset
--set action_gate.max_actions=5
```

### 2.1 Gate 实际做了两件事：过滤 + 排序

当前 `GatedProposer` 的核心逻辑不是单纯“删掉某些动作”，而是分两步：

```text
第一步：过滤
  只保留当前 gate mode 允许的动作类型。

第二步：排序
  对保留下来的动作按数据集先验重新排序。
```

代码层面的流程可以简化成：

```text
candidates = 原始 proposer 生成的候选动作
allowed = 当前 gate 允许的动作类型
filtered = candidates 中 action_type 属于 allowed 的动作
ranked = filtered 按 _priority(action, state) 排序
return ranked[:max_actions]
```

因此，即使两个模式允许的动作类型一样，只要排序逻辑不同，MCTS 实际优先展开的节点也会不同。

这点对 MRAG-Bench 特别重要。MRAG-Bench 在 `dataset gate` 下确实保留了 `text_search`、`image_search`、`text_to_image`、`image_to_text`、`answer` 这些新动作，但 gate 会给它们一个更符合数据集特点的优先级：

```text
image_search > text_to_image > image_to_text > answer > text_search
```

也就是说，`dataset gate` 对 MRAG-Bench 的作用不是“少用某些动作”，而是：

```text
保留完整跨模态动作能力，但优先让 MCTS 尝试图像/跨模态检索路径。
```

这会改变 MCTS 的展开顺序。因为当前配置里每个节点只保留有限个 child：

```text
top_k_children = 3
```

如果候选动作排序不同，最终进入搜索树的 top children 也可能不同。于是即使动作类型集合相同，搜索轨迹和最终效果也会不同。

## 3. 几种 Gate 的设计逻辑

### 3.1 old actions：老动作空间基线

老动作空间不使用新增跨模态动作。

主要动作：

| 动作 | 作用 |
| --- | --- |
| `text_search` | 用文本问题检索文本证据 |
| `image_search` | 用 query 图片检索相似图片 |
| `answer` | 基于已有 evidence 生成最终答案 |

优点：

- 成本低；
- step 数少；
- 轨迹更干净；
- rationale QC 通常更稳定。

缺点：

- 遇到“这个建筑是谁”“这架飞机是哪国的”“图片里这个东西对应哪个文本实体”时容易卡住；
- 无法显式执行 `image_to_text` 和 `text_to_image` 这类跨模态桥接动作。

### 3.2 full new actions：新动作空间全开

新动作空间除了老动作外，还加入：

| 新动作 | 作用 |
| --- | --- |
| `text_to_image` | 用文本去检索相关图片 |
| `image_to_text` | 用图片去检索或生成相关文本描述 |
| `ocr` | 从图片中识别文字，本次实验关闭 OCR |

优点：

- 搜索能力最强；
- 对 InfoSeek、MRAG-Bench 这类图文检索任务有帮助；
- 能覆盖老动作空间解决不了的一些跨模态路径。

缺点：

- 动作空间明显变大；
- MCTS 分支变多；
- step 数大幅增加；
- 会产生更多弱相关 evidence，后续 rationale QC 可能下降。

### 3.3 dataset gate：按数据集类型裁动作

这是最宽松、也最稳定的一种 gate。

它不细看具体问题，只根据数据集特点决定动作空间，并对保留下来的动作做数据集级别的优先级排序。

当前规则大致是：

| 数据集 | 允许动作 |
| --- | --- |
| ScienceQA | `text_search`、`answer` |
| CRAG-MM | `text_search`、`image_search`、`answer` |
| InfoSeek | `text_search`、`image_search`、`text_to_image`、`image_to_text`、`answer` |
| MRAG-Bench | `text_search`、`image_search`、`text_to_image`、`image_to_text`、`answer` |

设计直觉：

- ScienceQA 当前样本主要是文本题和选择题，跨模态动作多数时候收益不高；
- InfoSeek 更像百科实体问答，经常需要先识别图像实体；
- MRAG-Bench 图文混合程度更高，应该保留跨模态动作；
- CRAG-MM 当前使用官方 web/image 分离索引，所以不真正使用 `text_to_image` 和 `image_to_text`。

需要注意：对 MRAG-Bench 来说，`dataset gate` 允许的动作类型和 `full new actions` 很接近；它真正带来的变化主要是动作排序，而不是动作种类减少。

本次配置：

```bash
--set action_gate.mode=dataset
--set action_gate.max_actions=5
```

### 3.4 heuristic gate：按问题特征动态裁动作

heuristic gate 比 dataset gate 更细。

它会看问题文本里有没有视觉指示词、选择题结构、纯文本推理词等。

例如：

| 问题特征 | 动作倾向 |
| --- | --- |
| 出现 `this building`、`this aircraft`、`this image`、`identify` 等视觉指示词 | 开启 `image_search`、`image_to_text` |
| 有 choices / 选项 | 更倾向保留 `text_to_image`，用于“选项文本找图” |
| ScienceQA 中出现 `expected ratio`、`offspring`、`punnett` | 允许少量 `image_to_text` |
| 出现 `passage`、`experiment`、`best answer` 等纯文本推理信号 | 减少不必要的视觉动作 |

本次配置：

```bash
--set action_gate.mode=heuristic
--set action_gate.max_actions=4
```

优点：

- 比 dataset gate 更省；
- 能针对明显不需要跨模态的题少走弯路。

风险：

- 规则写得不够准时，可能误杀有用动作；
- 对问题表达方式敏感。

### 3.5 strict gate：更激进地压缩动作

strict gate 基于 heuristic gate，但进一步降低一些宽泛视觉动作的优先级。

例如：

- 如果不是选择题，降低 `text_to_image` 的优先级；
- ScienceQA 中进一步压低 `image_search`；
- 每步最多保留更少动作。

本次配置：

```bash
--set action_gate.mode=strict
--set action_gate.max_actions=3
```

优点：

- 成本低；
- 适合快速 smoke test。

缺点：

- 容易剪掉真正有用的跨模态动作；
- 不适合作为默认生产配置。

## 4. 实验设置

实验日期：2026-06-21

运行设置：

| 项目 | 设置 |
| --- | --- |
| 请求样本数 | 每个数据集 30 条 |
| 实际样本数 | InfoSeek 30，ScienceQA 25，MRAG-Bench 25，CRAG-MM 25 |
| rollout | 5 |
| OCR | 关闭 |
| 测试步骤 | gate 实验只跑 step1/step2 |
| 评估重点 | query 命中率、soft match、mean F1、trajectory 数、step 数 |

注意：本次 gate 实验没有跑 rationale 生成，所以 `rationale_qc` 不作为本次判断依据。

## 5. 整体结果

这里的“支持新动作的数据集”指：

```text
InfoSeek + ScienceQA + MRAG-Bench
```

CRAG-MM 当前会忽略 `cross_modal=true`，因为它使用官方分开的 web/image 索引，不是真正共享 CLIP 跨模态索引。

| 模式 | 支持新动作数据集 EM | 支持新动作数据集 soft | 支持新动作数据集 mean F1 | 支持新动作数据集 steps | 相对老动作成本 | 全部数据集 EM | 全部数据集 soft | 全部数据集 mean F1 | 全部数据集 steps | 相对老动作成本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old actions | 0.512 | 0.550 | 0.512 | 334 | 1.00x | 0.457 | 0.514 | 0.470 | 502 | 1.00x |
| full new actions | 0.650 | 0.662 | 0.535 | 1282 | 3.84x | 0.543 | 0.590 | 0.471 | 1410 | 2.81x |
| dataset gate | 0.613 | 0.688 | 0.546 | 892 | 2.67x | 0.524 | 0.610 | 0.495 | 1038 | 2.07x |
| heuristic gate | 0.600 | 0.650 | 0.528 | 807 | 2.42x | 0.514 | 0.600 | 0.471 | 987 | 1.97x |
| strict gate | 0.575 | 0.625 | 0.543 | 801 | 2.40x | 0.505 | 0.581 | 0.493 | 982 | 1.96x |

## 6. 分数据集结果

| 模式 | 数据集 | query EM | query soft | mean F1 | trajectories | steps |
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

## 7. 结果分析

### 7.1 新动作空间确实有效，但成本很高

和老动作空间相比，`full new actions` 在支持新动作的数据集上：

- EM 从 0.512 提升到 0.650；
- soft 从 0.550 提升到 0.662；
- mean F1 从 0.512 提升到 0.535。

但代价是：

- steps 从 334 增加到 1282；
- 成本约为老动作空间的 3.84 倍。

说明新版动作空间确实补上了老动作空间的一些能力缺口，但如果全开，会显著增加数据构建成本。

### 7.2 dataset gate 是当前最适合的默认方案

`dataset gate` 的整体表现最好。

它在支持新动作的数据集上：

- EM 为 0.613，低于 full new actions 的 0.650；
- soft 为 0.688，高于 full new actions 的 0.662；
- mean F1 为 0.546，高于 full new actions 的 0.535；
- steps 为 892，比 full new actions 的 1282 少约 30.4%。

这说明 dataset gate 剪掉了一部分无效探索，但没有明显伤害跨模态能力。

尤其是 MRAG-Bench：

| 模式 | MRAG-Bench query EM | mean F1 | steps |
| --- | ---: | ---: | ---: |
| old actions | 13/25 | 0.509 | 93 |
| full new actions | 16/25 | 0.555 | 391 |
| dataset gate | 19/25 | 0.581 | 391 |

MRAG-Bench 在 dataset gate 下反而比 full new actions 更好，说明限制动作后，MCTS 可能更少被无效路径干扰。

更准确地说：MRAG-Bench 的 `dataset gate` 并不是明显减少了动作类型，而是改变了候选动作进入 MCTS 的优先级。它会更优先展开 `image_search`、`text_to_image`、`image_to_text` 这类图像/跨模态动作，使搜索更集中在 MRAG-Bench 需要的证据路径上。由于每个节点最终只保留有限个 child，排序变化本身就可能改变整棵搜索树，所以会出现 `dataset gate` 比 `full new actions` 更好的情况。

### 7.3 heuristic gate 更省，但有轻微误杀风险

`heuristic gate` 的 steps 为 807，比 dataset gate 的 892 更少。

但它的表现略低：

| 模式 | 支持新动作 EM | soft | mean F1 | steps |
| --- | ---: | ---: | ---: | ---: |
| dataset gate | 0.613 | 0.688 | 0.546 | 892 |
| heuristic gate | 0.600 | 0.650 | 0.528 | 807 |

这说明 heuristic gate 确实能省成本，但规则还不够完美，可能会误判某些问题是否需要跨模态动作。

它适合：

- 大规模低成本收集；
- 已经知道数据集中大部分问题不需要复杂跨模态检索；
- 希望牺牲一点成功率换取速度。

### 7.4 strict gate 太窄，不适合作为默认配置

`strict gate` 的 steps 为 801，和 heuristic gate 的 807 非常接近。

但它在 InfoSeek 上掉得比较明显：

| 模式 | InfoSeek query EM | query soft | steps |
| --- | ---: | ---: | ---: |
| dataset gate | 8/30 | 14/30 | 422 |
| heuristic gate | 9/30 | 13/30 | 321 |
| strict gate | 7/30 | 11/30 | 323 |

也就是说，strict gate 并没有比 heuristic gate 明显省更多成本，却剪掉了更多有用动作。

因此 strict gate 更适合：

- 快速 smoke test；
- 检查管道是否能跑通；
- 不适合正式构建高质量数据。

### 7.5 CRAG-MM 的结论要单独看

CRAG-MM 当前不真正使用新增跨模态动作。

原因是它现在走的是官方 local web/image index：

```text
text_search -> CRAG web index
image_search -> CRAG image index
```

而不是统一的 CLIP cross-modal corpus。

所以本次 gate 对 CRAG-MM 的影响主要是限制 old actions，而不是测试 `text_to_image` / `image_to_text` 的收益。

## 8. 最终建议

### 8.1 默认推荐：dataset gate

如果后续要做比较正式的数据构建，建议默认使用：

```bash
--set retriever.cross_modal=true \
--set retriever.ocr.enabled=false \
--set action_gate.enabled=true \
--set action_gate.mode=dataset \
--set action_gate.max_actions=5
```

原因：

- 比老动作空间能力强；
- 比新动作全开便宜；
- 目前综合 soft 和 mean F1 最好；
- 不容易像 heuristic / strict 那样误杀有用动作。

### 8.2 成本敏感时：heuristic gate

如果后续要跑更大规模数据，例如接近 60k 条，发现 dataset gate 成本仍然偏高，可以改成：

```bash
--set action_gate.mode=heuristic \
--set action_gate.max_actions=4
```

它会更省，但要接受轻微性能波动。

### 8.3 不建议默认：strict gate

strict gate 可以保留为快速测试模式：

```bash
--set action_gate.mode=strict \
--set action_gate.max_actions=3
```

但不建议作为正式数据构建默认配置。

### 8.4 调试上限：full new actions

如果目的是分析某个样本到底需要什么动作，或者想测试动作空间上限，可以关闭 gate，直接全开新动作：

```bash
--set retriever.cross_modal=true \
--set action_gate.enabled=false
```

但它不适合长期大规模跑，因为 step 成本太高。

## 9. 一句话结论

当前最合理的策略是：

```text
正式构建用 dataset gate；
成本太高时用 heuristic gate；
快速测试用 strict gate；
诊断上限用 full new actions。
```

本次实验中，`dataset gate` 是综合性价比最高的方案：它保留了新版跨模态动作的大部分收益，同时把 supported step 成本从 full new actions 的 1282 降到了 892。
