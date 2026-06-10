# EvidenceTree: Process-Rewarded MCTS for Open-Domain Multimodal RAG

## 项目实现报告 v1.3

> **本报告是经过多轮批判性讨论后形成的最终设计文档。**
> 它整合了对 RCTS（ICML 2025）、AR-MCTS（ACL 2025）、GroundedPRM（arXiv 2510.14942）、VisualPRM、MMSearch-R1 等近期工作的批判性分析，
> 在每一个设计决策上都附有"为什么不选另一条路"的论证。

> **v1.1 相对 v1.0 的修订**：
> 1. **移除 web API 依赖**：实验完全在 benchmark 自带语料上展开，确保可复现性与因果干净度
> 2. **新增 self-adjusting search 子组件**：在 Contribution 2 内部加入 inference-time bandit-based λ 调整，对齐 self-evolution 热点 framing
> 3. **更新数据集成本**：从 $1,000 API 费用降为 0
> 4. **更新风险评估**：移除"cached vs real retrieval 分布偏移"风险（不再相关），新增"bandit 收敛性"风险
> 5. **附录 B 增补**：明确区分本项目的 inference-time self-adjusting 与更激进的 parameter-level self-evolution（后者作为 future work）

> **v1.2 相对 v1.1 的修订**：
> 1. **明确生成式 PRM 训练范式**：训练时输出 rationale + score（密集监督、抗 reward hacking、便于诊断），推理时只取 score-only（保持速度）；3.4 节展开方法论价值
> 2. **新增 rationale label 来源说明**：3.5 节明确 rationale 由强 LLM 离线生成（约 $800），含质量过滤规则
> 3. **新增风险项**：rationale 质量决定 PRM 训练效果（5.2 节）
> 4. **更新计算预算**：增加约 $800 的 rationale 生成 API 成本（注意：此 API 仅用于生成 rationale label，不用于检索，不破坏可复现性）
> 5. **附录新增 B.6**：记录"NAS 视角"作为论文写作阶段待决定的 packaging 选项（不作为方法组件）
> 6. **写作策略备注**：对 GroundedPRM 等前作采取"方法上深度参考、写作上正常引用不反复划界"的原则，不设专门的借鉴说明小节

> **v1.3 相对 v1.2 的修订**：
> 1. **形式化 self-adjusting bandit**：3.3.1 节用规范数学符号完整表达 Thompson Sampling 机制（采样、动态 median 阈值、Beta 后验更新、warm-up 切换），并配规范的 Algorithm 1 伪代码块
> 2. **新增 3.3.2 Theoretical Justification 小节**：引入 UCB1 / Hoeffding 不等式血统 + Thompson Sampling regret bound 作为理论锚点（配实验兜底，避免 T=10 渐近 bound 被攻击）
> 3. **统一符号系统**：新增附录 D（notation table）
> 4. **强化 A6 消融的形式化**：bandit consistency 指标（收敛 λ vs oracle 最优 λ）用公式精确定义；新增 arm 数量与 warm-up 轮数的消融
> 5. **方法论原则备注**：理论 bound 提供 motivation 与"学术严谨性"信号，但 bandit 的真正说服力来源是消融实验（尤其 consistency 指标），两者分工明确
> 6. **精简动作空间**：移除 `parallel_search`（树展开已覆盖多候选探索，"合并多源证据"可用串行轨迹表达，详见附录 B.7）；`crop`/`zoom` 标注为"待 Pilot 决定"（去留取决于 benchmark 中细粒度视觉 query 占比，可能合并为单一 `focus` 动作）

---

## 一、问题定义

### 1.1 核心科学问题

我们要解决的是这个问题：

> **给定一个开放域多模态问题（user query + 可选图像），如何设计一个检索-推理系统，使其在动作不可逆、调用有成本、证据可能被污染的环境下，做出最优的检索决策序列？**

这个问题分解为三个子问题：

**子问题 A — 检索动作的局部好坏怎么评估？**
不依赖于下游答案，能否在动作执行的当下就判断它的合理性？

**子问题 B — 当一次检索失败或返回低质量证据时，系统能否回溯并尝试替代方案？**
而不是把错误的证据当条件继续推理下去。

**子问题 C — 如何在多种检索类型（文本搜索、图像搜索、视觉操作）之间做规划，避免单一路径上的 modality blindness？**

### 1.2 当前主流工作的不足

我们调研了 2024–2026 年的相关工作，并把它们按"是否正面攻击上述三个子问题"做了分类：

| 工作 | 子问题 A | 子问题 B | 子问题 C | 核心局限 |
|------|---------|---------|---------|---------|
| **RCTS** (ICML'25) | ❌ | ⚠️ | ❌ | reward 是 rule-based 答案匹配，无 grounded 信号；树搜索建立在 flat retrieved pair list，不在动作空间 |
| **AR-MCTS** (ACL'25) | ⚠️ | ❌ | ❌ | 树搜索建立在 reasoning 空间，检索只是 expansion 时的辅助；不解决"检索决策本身的好坏"问题 |
| **MMSearch-R1** | ❌ | ❌ | ⚠️ | outcome-only RL，检索轨迹线性无回溯；reward 信号稀疏导致过程幻觉 |
| **VRAG-RL** | ❌ | ❌ | ❌ | 视觉感知动作空间但 reward 仍是 outcome-only |
| **GroundedPRM** (ICLR'26 投稿) | ✅ | N/A | N/A | 解决了 fixed-state reasoning 的 grounding 问题，**但不处理状态修改型动作**；非多模态、非 RAG |
| **VisualPRM** | ⚠️ | N/A | N/A | 多模态 PRM 但只评分 fixed-image-context 下的推理步，不评分检索动作 |

可以看到，**没有任何一个工作同时正面攻击 A+B+C**。GroundedPRM 解决了 A 在 fixed-state 下的版本，但完全没碰 RAG 场景的 state-modifying actions。这就是 EvidenceTree 要填补的空白。

### 1.3 与 RCTS 的关系（直接前作的明确差异）

RCTS 是这个方向上最相关的前作。它的两个核心组件是：
1. 用 self-consistency 生成 reasoning context 丰富 KB
2. MCTS-HR 在 Top-N retrieved pairs 上做 re-ranking

EvidenceTree 与之的本质区别：

| 维度 | RCTS | EvidenceTree |
|------|------|-------------|
| **树的状态空间** | 已检索 example 的排列组合 | 部分检索状态（query + image + evidence bundle） |
| **树的动作空间** | 选择下一个 retrieved pair 加入 context | 类型化检索动作（text_search / image_search / crop / ...） |
| **Reward 来源** | rule-based answer matching + mutual heuristic | grounded local signal + outcome（融合 GroundedPRM 思想） |
| **PRM** | 无 PRM，用规则 reward | 训练专用 action-typed PRM |
| **失败模式应对** | 失败时所有分支都错（论文 Fig 16 承认） | 通过 grounded local signal 早期识别失败分支 |
| **KB 设定** | in-domain（评估集对应的训练集） | 开放域、跨数据集 |

**最本质的差别**：RCTS 把"哪些 example 入选 context"建模成搜索问题；EvidenceTree 把"该执行什么检索动作"建模成搜索问题。前者的搜索发生在 RCTS 流程的最后一步（re-ranking），后者的搜索贯穿整个检索过程。

---

## 二、核心 Contribution

EvidenceTree 提供两个紧耦合的技术 contribution，外加一个数据集 contribution。

### Contribution 1: Action-Typed Grounded PRM

**这是什么**：一个专门为 RAG 检索动作训练的 process reward model。

**输入**：
- query（文本问题）
- image（可选）
- current_evidence_bundle（当前已收集的证据集合）
- candidate_action（候选下一步检索动作，包含类型和参数）

**输出**：
- score：预测该动作执行后整条 trajectory 最终能得到的 reward 期望
- （训练时）rationale：该动作合理/不合理的自然语言解释

**关键设计**：
- 动作类型化：每种动作（text_search / image_search / crop / zoom / answer）有独立的 grounding 验证逻辑
- 双源 reward 训练：local grounding + global outcome（沿用 GroundedPRM 原则）
- 起点：从 VisualPRM-8B 微调，继承多模态视觉推理的归纳偏置

**为什么这是 contribution**：
- VisualPRM 不评分 state-modifying actions
- GroundedPRM 不处理多模态、不处理 RAG 场景
- AR-MCTS 的 PRM 评分 reasoning steps，不评分检索动作
- 没有已发表工作训练过同时具备这三个属性的 PRM

### Contribution 2: Retrieval-Action MCTS with Self-Adjusting Modality-Coverage UCB

**这是什么**：一个把 MCTS 应用在检索动作空间的搜索算法，并在 inference 时通过 bandit 自适应调整探索行为。

**树结构**：
- 节点：部分上下文状态 (query, image, evidence_bundle)
- 边：类型化检索动作
- 叶子：answer 动作的执行

**Selection rule（Modality-Coverage UCB）**：
```
UCT(action) = Q(action) 
            + c · √(ln N(parent) / N(action))
            + λ · modality_novelty(action, current_path)
```
其中 modality_novelty 在动作引入路径上未出现过的模态/工具/粒度时给一个 bonus。

**子组件：Inference-time Self-Adjusting Search（v1.1 新增）**：

固定的 λ 值（如 0.3）无法应对不同 query 的探索需求差异——视觉密集型 query 需要激进的跨模态探索（λ 大），文本密集型 query 上跨模态探索是浪费（λ 小）。我们引入一个 inference-time 的 bandit 机制，在单 query 的 MCTS 运行中自适应调整 λ：

- 把 λ 离散化为 5 个 arm: {0.1, 0.3, 0.5, 0.7, 1.0}
- 每次 rollout 前用 Thompson Sampling 选择 λ
- Rollout 完成后根据 reward 是否高于历史 median 更新 Beta 后验
- 整个机制在单 query 内独立运行，不跨 query 共享状态——**保留 frozen-model 评估范式**

**Expansion**：节点扩展时，由 PRM 在候选动作空间上排序，选 top-k 作为 children。

**Simulation**：到达预设深度或 answer 动作时停止，用 PRM 预测 + 真实 outcome（如果可得）的混合作为 reward。

**Backup**：标准 MCTS reward 回传（不做 DAG，原因详见附录 A）。

**Termination**：早停（PRM 对 root 的 best path Q 值高于阈值）或 budget 耗尽。

**为什么这是 contribution**：
- AR-MCTS 的 MCTS 在 reasoning space，不在 action space
- RCTS 的 MCTS 在 retrieved-pair-set space，不在 action space
- Modality-Coverage UCB 是针对多模态 RAG 特有的 exploration 偏好设计的，已有 MCTS 工作没有这个机制
- Self-adjusting bandit 让 selection 行为不再是 hardcoded 先验，而是 inference-time 从自己的运行反馈中学习——这一点与 self-evolution 文献的核心思想对话，但保持 frozen-model 评估范式

### Contribution 3: ETBench-Open（数据集）

**这是什么**：一个用于评估 retrieval-aware PRM 的步骤级标注数据集，规模约 50–80K。

**为什么需要它**：
- VisualPRM400K 几乎全是数学推理 step，几乎没有检索动作样本
- AR-MCTS 数据是封闭场景的数学推理
- GroundedPRM 数据是数学+工具，不是检索动作
- 没有任何已有数据集能直接训练我们要的 PRM

**构造方法**（见第三章详述）：基于 benchmark 自带语料的 MCTS rollout + 模态特定的 grounded verification + outcome 验证。

---

## 三、方法详述

### 3.1 检索动作空间

我们定义如下类型化动作集合 𝒜：

| 动作类型 | 参数 | 描述 | 状态 |
|---------|------|------|------|
| `text_search(q)` | 文本 query | 在文本检索器上 retrieve | 必需 |
| `image_search(img_crop)` | 图像区域 | 用图像作为 query 检索 | 必需 |
| `crop(region)` | bbox 坐标 | 把当前 image 的某个区域作为新 image（改变空间注意力） | 待 Pilot 决定 |
| `zoom(region, factor)` | 区域 + 倍数 | 放大区域细节（改变分辨率） | 待 Pilot 决定 |
| `answer(text)` | 答案文本 | 终止动作 | 必需 |

**关于 crop / zoom（视觉操作动作）**：

这两个不是检索动作，而是**视觉操作动作**——不从外部语料检索，而是改变模型当前能看到的视觉区域。`crop` 框定空间范围（聚焦局部），`zoom` 提高分辨率（看清细节），实践中常配合使用（先 crop 再 zoom）。

它们的价值**强依赖于 benchmark 中"需要看局部细节"的 query 占比**（如读小字、看仪表读数、识别图中某个小区域的实体）。因此其去留**待 Pilot 阶段统计后决定**：
- 若细粒度视觉 query 占比高 → 保留 crop + zoom
- 若占比中等 → 合并为单一 `focus(region)` 动作（同时裁剪 + 放大）
- 若占比 < 10% → 暂不实现，需要时再加

**关于已砍掉的 parallel_search（v1.3 移除，详见附录 B.7）**：

早期草稿里有一个 `parallel_search(actions)` 复合动作，用于"并行执行多个检索并合并结果"。v1.3 将其移除，理由是 **tree 结构的分支展开已经提供了多候选探索能力，而真正需要"合并多源证据"的场景（如对称比较两个实体）可以用串行轨迹表达**（先搜 A 再搜 B，evidence_bundle 累积后 VLM 自然能比较）。parallel_search 带来的实现复杂度（特殊的 grounding 评分、特殊的 backup 语义、稀释探索预算）大于其收益。详细论证见附录 B.7。

### 3.2 Reward Function

**Trajectory-level reward**：

$$R(\tau) = \alpha \cdot R_{\text{local}}(\tau) + (1-\alpha) \cdot R_{\text{outcome}}(\tau)$$

其中：

**Local grounding reward**：
$$R_{\text{local}}(\tau) = \frac{1}{|\tau|} \sum_{a_i \in \tau} g(a_i, s_i)$$

`g(a, s)` 是动作 `a` 在状态 `s` 下的 grounded 评分，根据动作类型不同有不同的实现：

| 动作类型 | grounding signal `g` |
|---------|---------------------|
| `text_search(q)` | query 与原 question 的语义对齐度（独立 cross-encoder） |
| `image_search(img)` | 图像 crop 与 question 提到的视觉实体的对齐度（独立 vision-language matching model） |
| `crop` / `zoom` | 操作后区域是否包含 question 关键 entity（独立 detection / OCR 模型） |
| `answer` | 不参与 local，由 outcome 决定 |

**关键性质**：所有 `g` 都是 graded score（0–1 连续值），不是 binary。这是相对 GroundedPRM 的一个修正——RAG 场景下的"对/错"判断粒度不应该是 binary。

**Outcome reward**：

最简单的 trajectory-level 形式：
$$R_{\text{outcome}}(\tau) = \mathbb{1}[\text{answer}(\tau) \approx \text{ground truth}]$$

但**实际训练时我们用 step-conditional 的 tree-level credit**，而非把整条 trajectory 的 final outcome 均摊给所有 step。对每个节点 $(s, a)$，其 outcome label 是经过它的所有 trajectory 的成功率（Monte Carlo 估计）：

$$R_{\text{outcome}}(s, a) = \frac{1}{|\mathcal{T}(s,a)|} \sum_{\tau \in \mathcal{T}(s,a)} \mathbb{1}[\text{answer}(\tau) \approx y^*]$$

其中 $\mathcal{T}(s,a)$ 是经过节点 $(s,a)$ 的所有完整 trajectory，$y^*$ 是 ground truth。

**为什么用 tree-level credit 而非 trajectory-uniform credit**：trajectory-uniform 给好 step 和坏 step 同样的奖惩，信号 noisy；tree-level credit 让每个 step 的 label 反映"从这一步出发的真实成功潜力"，credit assignment 更精确。

对开放生成式答案，$\mathbb{1}[\cdot \approx y^*]$ 使用 LLM judge ensemble（多个 judge model 投票，降低单一 judge 的偏差）。

**默认 α=0.5**，sensitivity ablation 给三档 (0.2 / 0.5 / 0.8)。

### 3.3 Modality-Coverage UCB

Selection rule：
$$\text{UCT}(a) = Q(a) + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(a)}} + \lambda \cdot \nu(a, \pi)$$

其中 `ν(a, π)` 是动作 `a` 相对当前路径 `π` 的 modality novelty bonus：
$$\nu(a, \pi) = \mathbb{1}[\text{type}(a) \notin \text{types}(\pi)] + 0.5 \cdot \mathbb{1}[\text{granularity}(a) \notin \text{granularities}(\pi)]$$

简单来说：
- 如果路径上还没用过这个动作类型，加 1
- 如果路径上还没用过这个粒度（whole image / cropped region），再加 0.5

**为什么 modality bonus 在 selection rule 而不在 reward**：
- Reward 应该衡量"这条 trajectory 实际有多好"——是事实判断
- Selection 应该衡量"这个分支值不值得探索"——是策略判断
- 把 modality 偏好放在 selection 里，让 reward 信号保持低噪且校准良好

**默认 λ=0.3**，与 c 一起做 grid search ablation。

### 3.3.1 Inference-time Self-Adjusting Search（v1.1 新增）

固定的 λ 假设了所有 query 共享同样的最优探索偏好——这是个静态先验，是 EvidenceTree 设计中**唯一一个不依赖数据学习**的环节。我们用一个轻量 bandit 机制让这个先验在推理时自我调整。

**符号系统**：

| 符号 | 含义 |
|------|------|
| $P$ | 单 query 的 rollout 预算（默认 10） |
| $\mathcal{A} = \{\lambda_1, ..., \lambda_K\}$ | $K=5$ 个候选 λ 值，$\{0.1, 0.3, 0.5, 0.7, 1.0\}$ |
| $t \in \{1,...,P\}$ | rollout 序号 |
| $\lambda^{(t)}$ | 第 $t$ 次 rollout 选用的 λ |
| $r^{(t)}$ | 第 $t$ 次 rollout 的 PRM reward |
| $(\alpha_k^{(t)}, \beta_k^{(t)})$ | arm $k$ 在第 $t$ 步的 Beta 后验参数 |

**核心机制（形式化）**：

每个 query 推理开始时初始化独立的 Thompson Sampling bandit，先验 $\alpha_k^{(1)} = \beta_k^{(1)} = 1$（uniform prior）。

*Step 1 — 选 λ*。前 3 次 round-robin warm-up，之后用 Thompson Sampling：

$$
\lambda^{(t)} = \begin{cases} \lambda_{\sigma(t)}, & t \leq 3 \quad (\text{round-robin，覆盖 } \{0.1, 0.5, 1.0\}) \\[4pt] \lambda_{k^*}, \;\; k^* = \arg\max_{k} \theta_k^{(t)}, \;\; \theta_k^{(t)} \sim \text{Beta}(\alpha_k^{(t)}, \beta_k^{(t)}), & t > 3 \end{cases}
$$

*Step 2 — 用选定 λ 跑 rollout*，得到 reward $r^{(t)} = \text{PRM}(\pi^{(t)})$，其中 $\pi^{(t)} = \text{MCTS-Rollout}(s_0, \lambda^{(t)})$。

*Step 3 — 动态 median 阈值二值化*。定义动态阈值与成功信号：

$$
m^{(t)} = \text{median}\big(\{r^{(\tau)}\}_{\tau=1}^{t-1}\big), \qquad z^{(t)} = \mathbb{1}[r^{(t)} > m^{(t)}]
$$

*Step 4 — Beta 后验更新*（仅更新本次被选中的 arm $k^*$）：

$$
\alpha_{k^*}^{(t+1)} = \alpha_{k^*}^{(t)} + z^{(t)}, \qquad \beta_{k^*}^{(t+1)} = \beta_{k^*}^{(t)} + (1 - z^{(t)})
$$

其余 arm 后验不变：$(\alpha_k^{(t+1)}, \beta_k^{(t+1)}) = (\alpha_k^{(t)}, \beta_k^{(t)})$ for $k \neq k^*$。

**Algorithm 1: Inference-time Self-Adjusting Search**

```
Input : query q, rollout budget P, arms A = {λ_1,...,λ_K}
Output: best answer ŷ

 1: s_0 ← (q, ∅)                              # tree root
 2: (α_k, β_k) ← (1, 1)  for all k            # uniform Beta priors
 3: R ← ∅                                     # reward history
 4: for t = 1 to P do
 5:     if t ≤ 3 then
 6:         λ ← λ_{σ(t)}                       # round-robin warm-up
 7:     else
 8:         θ_k ~ Beta(α_k, β_k)  for all k    # Thompson sampling
 9:         λ ← λ_{argmax_k θ_k}
10:     π ← MCTS-Rollout(s_0, λ)              # UCT with bonus weight λ
11:     r ← PRM(π)
12:     if t > 1 then
13:         m ← median(R)
14:         z ← 1[r > m]
15:         α_{k*} ← α_{k*} + z ;  β_{k*} ← β_{k*} + (1 − z)
16:     R ← R ∪ {r}
17:     if r > τ_stop then break              # early stopping
18: return ŷ ← answer of argmax_π PRM(π)
```

**设计选择的论证**：

*为什么用 Thompson Sampling 而非 ε-greedy 或 UCB1*：TS 在 small-sample regime（$P=10$）下探索更平滑；天然支持用后验采样替代 point estimate，更鲁棒；Beta-Bernoulli 后验对小样本计算稳定。

*为什么 $K=5$ 而非 10*：$P=10$ 预算下，$K=10$ 平均每 arm 仅 1 次观测，bandit 退化为随机选择；$K=5$ 平均每 arm 2 次观测，可稳定收敛。$K=5$ 跨度 0.1–1.0 已覆盖合理范围。

*为什么用"高于历史 median"判定*：dynamic threshold 不依赖 reward 绝对标定，对 PRM calibration drift 鲁棒，自动适应不同 query 难度。

### 3.3.2 Theoretical Justification（v1.3 新增）

我们的探索机制建立在 multi-armed bandit 的理论框架上，借用两个学界公认的理论工具作为依据。

**UCB1 confidence bound（Hoeffding 血统）**：

UCT selection rule 中的 exploration 项并非经验性设计，而是源于 UCB1 算法（Auer et al., 2002）。UCB1 的选择规则为：

$$
k^{(t)} = \arg\max_{k} \left[ \hat{\mu}_k + \sqrt{\frac{2\ln t}{n_k}} \right]
$$

其中 confidence radius $\sqrt{2\ln t / n_k}$ 由 Hoeffding 不等式导出，保证真实均值以高概率落在界内。**我们 UCT 公式里的 $c\sqrt{\ln N(s)/N(s,a)}$ 正是这一 bound 的形式**——MCTS 的 UCT 本就是 UCB 在树上的推广（UCT = UCB applied to Trees，Kocsis & Szepesvári, 2006）。这给我们的 exploration 项一个有 50 年历史的统计学血统。

**Thompson Sampling regret bound**：

对于 λ 的自适应选择，Thompson Sampling 在 $K$-臂 Bernoulli 设定下享有 problem-independent regret bound（Agrawal & Goyal, 2017）：

$$
\mathbb{E}[\text{Regret}(T)] = O\left(\sqrt{KT\ln T}\right)
$$

代入我们的设定（$K=5$, $T=P=10$），累积 regret 约 $O(\sqrt{5 \cdot 10 \cdot \ln 10}) \approx O(10.7)$，为 bandit 在有限 rollout 预算内收敛到近优 arm 提供理论依据。

**重要 caveat（诚实声明）**：上述 regret bound 是 asymptotic（$T \to \infty$）结果，而我们的 $T=10$ 极小，渐近 bound 在此规模下的实际保证较弱。因此该 bound **仅提供理论 motivation，bandit 在小预算下的实际收敛行为由 §4.5 的消融实验（A6.3 收敛速度、A6.2 consistency 指标）提供 empirical evidence**。理论负责"严谨性 motivation"，实验负责"真有效"，两者分工明确，不以理论替代实验。

### 3.4 PRM 架构与训练

**架构**：基于 InternVL2.5-8B 的 VisualPRM-8B，加一个 action-conditional value head。

**输入格式**：
```
<image> {query}
[EVIDENCE_BUNDLE_START]
{evidence_1}
{evidence_2}
...
[EVIDENCE_BUNDLE_END]
[CANDIDATE_ACTION]
type: image_search
parameters: crop(0.2, 0.3, 0.5, 0.6)
[/CANDIDATE_ACTION]
```

**输出**：
- 训练时（generative）：rationale 文本 + score token
- 推理时（score-only）：直接读 score token 的 logit

**为什么训练时用生成式（generative reward model）**：

这是一个 deliberate 的设计选择，不是随意的实现细节。让 PRM 训练时同时输出 rationale 和 score，相比只输出 score 的判别式范式，有三个实质价值：

1. **Sample efficiency**：只输出 score 时，每个样本的 loss 信号是一个标量，监督稀疏。同时输出 rationale 时，每个样本提供几十个 token 的损失信号——同样的数据量，PRM 能学到更多。这对我们的低数据量目标（50–80K）至关重要。

2. **抗 reward hacking**：判别式 PRM 容易学到 spurious feature（如"长文档一律高分"）。生成式范式强迫模型"说出"判断理由——如果它给长文档打高分，rationale 必须给出理由，这种 spurious 模式在训练时更容易被识别和约束。

3. **便于失败诊断**：训练阶段如果 PRM 表现异常，可以直接读 rationale 看它的推理过程出了什么问题，而不是面对一个黑盒分数。这对调试 PRM 失败模式（哪种动作类型上失效、是否多模态 mismatch）非常有用。

**为什么推理时只取 score-only**：

推理时如果也生成完整 rationale，每次 PRM 调用会显著变慢（MCTS 一次 query 要调用 PRM 几十次）。我们的做法是：训练时的 rationale 监督已经"烧进"模型权重，推理时只读 score token 的 logit，不生成 rationale 文本。这样既享受了生成式训练的收益，又不付推理速度的代价。

**说明**：生成式 reward model 是一个公共范式（Generative Verifier、Critique-out-Loud RM 等多个工作都采用），我们在此范式下做 RAG 检索动作的适配，并验证它能否从数学推理场景迁移到开放域检索场景。

**训练流程**（三阶段）：

**Stage 1 — SFT 起步**
- 数据：50K rollout 样本，标准 next-token prediction
- 目标：让 PRM 学会输入格式 + 输出 rationale + score
- 起点：VisualPRM-8B checkpoint

**Stage 2 — Action-typed contrastive DPO**
- 数据：从 rollout 里挖掘"同状态不同动作"的对比对
- 类型：(oracle action, model action)、(text_search, image_search)、(retrieve, no_retrieve) 等
- 目标：让 PRM 在动作类型选择上有正确偏好
- 对比对的构造维度是"同状态不同动作类型"，对应检索决策场景的特点（与基于 reasoning step 前后构造对比对的做法不同）

**Stage 3 — 校准**
- 数据：held-out validation set
- 目标：用 Platt scaling / temperature scaling 把 score token logit 校准到真实 reward 期望
- 重要性：UCB 公式里的 Q 值需要有概率意义

### 3.5 ETBench-Open 数据集构造

**总规模目标**：50–80K trajectories，对应 ~300K step-level 标注。

**数据来源（benchmark 自带的固定语料）**：

我们完全在 benchmark 自带的检索语料上构造数据，**不调用任何 web API**。这一选择有三个理由：
- **可复现性**：web 内容随时间变化，他人无法精确复现
- **因果干净度**：固定语料下，性能差异完全来自算法层面，而非"是否搜得到答案"
- **论文聚焦**：避免变成"工程系统"，回归算法贡献本身

具体语料：

| Benchmark | 自带的固定语料 | 训练 query 数 |
|-----------|--------------|--------------|
| InfoSeek | Wikipedia entity 子集 (~100K entities) | ~15K |
| E-VQA | iNaturalist 子集 + Wikipedia | ~20K |
| MRAG-Bench | 自带的 multi-modal corpus | ~10K |
| OK-VQA / A-OKVQA | 自带的 Wikipedia 子集 | ~10K |
| Dyn-VQA | 自带的 dynamic VQA corpus | ~5K |
| 合计 | | ~60K queries |

每个 benchmark 内部，训练 KB 和评测 KB **共享同一份语料**——这是关键的可复现性保证。

**Rollout 策略**：

对每个 query 跑一次 MCTS（用 weak policy + uniform action prior），生成 ~10 条 trajectory，得到 ~600K trajectories。

**所有检索动作都在 benchmark 自带语料上执行**（用 BM25 + dense retriever 在语料上建立离线索引）。无外部 API 调用，无成本。

**Step-level 标注**：

每个 step（即每个动作）的标注由三部分组成：

1. **Local grounding label** `g(a, s)`：调用对应模态的 verifier 模型计算
2. **Step-conditional outcome label**：从该 step 出发的 sub-tree 的最终 outcome rate（Monte Carlo 估计）
3. **Rationale label**：该 step 合理/不合理的自然语言解释（用于生成式 PRM 训练）

前两者融合为 score label（与 reward function 公式一致）。第三个 rationale label 的来源需要单独说明。

**Rationale label 来源**：

生成式 PRM 训练要求每个样本有 (rationale, score) 对。score 由上述 grounding + outcome 计算得到，但 rationale 需要额外生成。我们的方案：

- **生成方式**：用一个强 LLM（如 Claude / GPT-4o）看每个 (state, action, score) 三元组，生成对应的 rationale 文本
- **成本估算**：约 50–80K 样本，每个生成一段 rationale，总成本约 $800
- **质量过滤规则**：rationale 必须满足 (a) 至少引用一个 evidence_id，(b) 明确提到动作类型，(c) 长度在 50–150 token 之间。约 15% 不合格的会重新生成
- **关键说明**：这个 API 调用**仅用于生成 rationale label，不用于检索**。检索全程在 benchmark 自带语料上离线进行。因此这不破坏实验可复现性——rationale 一旦生成就是固定数据集，可随数据集一起发布

**质量控制**：
- 拒绝过短轨迹（<2 步）和过长轨迹（>8 步）
- 拒绝 grounding 与 outcome 严重不一致的轨迹（可能是 noisy label）
- 人工抽样 500 条做 score label sanity check
- 人工抽样 200 条（每类动作类型）做 rationale 质量 check；若 rationale 质量低于阈值，升级到更强的生成模型或人工标注

---

## 四、实验设计

### 4.1 主战场（main paper 的核心 table）

**开放域多模态 RAG benchmarks**：

| Benchmark | 任务类型 | 评测指标 | 自带语料 |
|-----------|---------|---------|---------|
| InfoSeek | 细粒度知识 VQA | EM / F1 | Wikipedia 子集 |
| E-VQA | 百科类 VQA | EM | iNaturalist + Wikipedia |
| MRAG-Bench | 多跳推理 VQA | EM | 自带 multi-modal corpus |
| OK-VQA / A-OKVQA | Commonsense VQA | EM | Wikipedia 子集 |

**为什么不包含 Dyn-VQA 和 MMSearch 作为主表**：这两个 benchmark 的标准评测协议依赖实时 web 检索（前者需要时效性信息，后者本身是 web 搜索 benchmark）。为保持实验完全可复现，我们不在主表使用这两个，但可以放 appendix 作为"在动态语料上的额外验证"。

### 4.2 辅助战场（appendix table）

**与 RCTS 同场对比**（沿用 RCTS 论文 setup）：
- ScienceQA、MMMU、MathV
- 目的：证明 EvidenceTree 不只在新场景能打，在 RCTS 的主场也能打

放 appendix 的原因：
- ScienceQA 已经接近饱和（accuracy 90+%），improvement headroom 小
- 这些 benchmark 不是开放域 RAG，无法体现 EvidenceTree 的核心 advantage
- 评审更关心 main paper 的开放域结果

### 4.3 鲁棒性战场

**Poisoned-MRAG 风格对抗实验**：
- 向 KB 注入 5–20 对对抗性图文对（per query）
- 测量答案翻转率
- 假设：tree 搜索的回溯机制能显著降低翻转率，因为 grounded local signal 会识别被污染的证据

### 4.4 Baselines

**必须打的 baselines**：

| Baseline | 类别 | 重要性 |
|----------|------|-------|
| Vanilla RAG (PreFLMR) | Lower bound | 必须 |
| RCTS | 直接前作 | 必须 |
| AR-MCTS | 最相关 concurrent | 必须 |
| MMSearch-R1 / DeepMMSearch-R1 | Agentic RAG SOTA | 必须 |
| VisualPRM-8B + Best-of-N | 证明你不只是 PRM 改进 | 必须 |
| Zero-shot Qwen2.5-VL-7B | 无 RAG baseline | 必须 |

**Optional baselines**（看 review 反馈再加）：
- VRAG-RL（视觉感知动作）
- R1-Router（routing baseline）

### 4.5 关键 Ablation

**A1 — Reward function 的两源是否都必要**
- 只用 local grounding（α=1.0）
- 只用 outcome（α=0.0）
- 两源融合（α=0.5）
- 三档 sensitivity（α=0.2 / 0.5 / 0.8）

**A2 — Modality-Coverage UCB 是否有效**
- 标准 UCB（λ=0）
- Modality-Coverage UCB（固定 λ=0.3）
- 不同固定 λ 值的 sensitivity (0.1 / 0.3 / 0.5 / 0.7 / 1.0)

**A3 — PRM 训练阶段贡献**
- 只 Stage 1 (SFT)
- Stage 1 + Stage 2 (SFT + DPO)
- Stage 1 + Stage 2 + Stage 3 (full)

**A4 — Action 类型贡献**
- 只 text_search + answer
- 加 image_search（核心多模态动作）
- 加 crop / zoom 或 focus（仅当 Pilot 决定实现这些视觉操作动作时）

**A5 — Tree depth / rollout budget 影响**
- depth=2/3/5
- rollouts=4/10/20

**A6 — Self-Adjusting Search 是否有效（v1.1 新增，v1.3 形式化强化）**

这是 inference-time bandit 机制的关键 ablation。**bandit 是我们自己设计的组件，所有超参数（K=5、warm-up=3、λ 取值）都需要消融支撑，不能拍脑袋。** 必须包含以下五部分：

*A6.1 主对比* — Bandit vs 最佳固定 λ：
- 固定 λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}（5 个固定值各跑一遍）
- **Bandit-based adaptive λ**（我们的方法）

期望结果：Bandit 应在所有 benchmark 的平均表现上 ≥ 任意单一固定 λ，即 $\text{Acc}(\text{bandit}) \geq \max_k \text{Acc}(\text{fixed } \lambda_k)$。若 bandit 仅接近最佳固定 λ，说明机制无效——回归固定 λ。

*A6.2 Consistency 指标*（最关键的 sanity check，v1.3 形式化）：

这是回应"bandit 是否真在学习、还是纯噪声"的硬证据。对每个 query $q$，用 GT 离线反算真实最优 λ：

$$
\lambda^*_q = \arg\max_{\lambda \in \mathcal{A}} \text{Acc}\big(\text{MCTS-Rollout}(q, \lambda), \, y^*_q\big)
$$

bandit 在线收敛到的 λ：

$$
\hat{\lambda}_q = \lambda_{\,\arg\max_k \frac{\alpha_k^{(P)}}{\alpha_k^{(P)} + \beta_k^{(P)}}}
$$

定义一致性指标：

$$
\text{Consistency} = \frac{1}{|\mathcal{Q}|}\sum_{q \in \mathcal{Q}} \mathbb{1}[\hat{\lambda}_q = \lambda^*_q]
$$

**判定标准**：若 $\text{Consistency} \gg 1/K = 0.2$（随机基线），证明 bandit 真在学习；若接近 0.2，说明 bandit 没提取到信号，机制失效。

*A6.3 Convergence 速度* — Bandit 在 P 次 rollout 内的收敛行为：
- 跟踪每 rollout 后"最佳 arm 后验均值 $\alpha_k/(\alpha_k+\beta_k)$"
- 报告平均收敛步数
- 若 10 次内大部分 query 未收敛，把 default P 提到 15

*A6.4 Arm 数量敏感性*（v1.3 新增）— 验证 $K=5$ 的选择：
- 对比 $K \in \{3, 5, 10\}$
- 指标：accuracy + bandit 结果方差
- 期望：$K=10$ 因每 arm 观测不足，性能持平或更差且方差更大，验证 $K=5$ 是 $P=10$ 预算下的甜蜜点

*A6.5 Warm-up 轮数*（v1.3 新增）— 验证"前 3 次轮询"：
- 对比 warm-up ∈ {0, 3, 5}
- 期望：warm-up=0 时早期 Thompson 采样不稳定，warm-up=3 在 accuracy 和方差上最优

**形式化消融的意义**：A6.4/A6.5 把"K=5、warm-up=3"从"我们设的值"变成"消融验证的最优值"。这是回应审稿人"参数怎么定的"的标准做法——用实验支撑而非声明。

### 4.6 计算预算

**训练**：
- ETBench-Open 构造：检索部分纯本地无费用 + rationale label 生成约 $800（强 LLM API）+ 1×8 H100 节点 × 1 周
- PRM 三阶段训练：1×8 H100 节点 × 2–3 周（生成式训练比判别式略慢，rationale token 增加 forward/backward 开销）

**推理**：
- 主表评测：所有 benchmark 跑 3 个 seed
- 单个 query MCTS 推理 ~15 秒（无 API 调用，纯本地索引检索 + GPU 推理；推理时 score-only，不生成 rationale）
- 总推理预算：1×4 A100 × 2 周

**总计**：约 $800 API（仅 rationale label 生成）+ 1×8 H100 × 3–4 周 + 1×4 A100 × 2 周

服务器算力租赁场景下完全可承受。

**关于这 $800 API 成本的说明**：这是 v1.2 唯一的现金成本，且性质与 v1.0 砍掉的 web API 费用**完全不同**：
- v1.0 的 web API：用于**检索**，结果随时间变化，破坏可复现性
- v1.2 的 rationale API：仅用于**离线生成 rationale label**，一次性，生成后即固定，可随数据集发布，**不破坏可复现性**

相对 v1.0 的关键改进仍然成立：检索与推理全程无 API 依赖，实验完全可复现。

---

## 五、风险评估与对策

### 5.1 学术风险

**风险 1（最大）：被审稿人质疑为 GroundedPRM 的 RAG 版本**

应对：
- 在 introduction 第二段明确划出**conceptual 差异**——GroundedPRM 解决 fixed-state reasoning，我们解决 state-modifying actions
- 在 related work 里给 GroundedPRM 一个独立段落，讲清楚"我们借鉴它的 grounding philosophy，但 fundamentally different problem"
- 在实验里做一个 ablation：把 GroundedPRM 的训练流程直接搬到我们的数据上看效果，证明只搬流程不够，需要我们的多模态 grounding + action typing

**风险 2：被质疑为 AR-MCTS 的延伸**

应对：
- 在 related work 明确："AR-MCTS expands in reasoning space with auxiliary retrieval; we expand in retrieval action space."
- 在主表里把 AR-MCTS 作为 baseline 直接打掉

**风险 3：MCTS 在 RAG 真的有必要吗？**

如果 Best-of-N + PRM 已经接近 MCTS 性能，MCTS 这层就是冗余的。

应对：
- 必做的 ablation：Best-of-N (PRM rerank only) vs MCTS
- 如果 gap < 1 point，需要重新审视 MCTS 是否有必要——但这种情况不太可能，因为 MCTS 的真正优势在 budget-constrained tool calling 上，不在 BoN 能等价的 sample-and-pick 场景上

**风险 4：PRM 过拟合到 in-domain**

应对：
- 训练数据的 KB 和评测数据的 KB 必须不重叠（每个 benchmark 各自的 train/test 划分）
- 跨数据集的 cross-validation（在 InfoSeek 上训，在 E-VQA 上测）

### 5.2 工程风险

**风险 5：Local grounding verifier 本身不可靠**

如果 verifier 模型给 query-document alignment 打错分，整个 grounding 信号就崩了。

应对：
- 使用多个 verifier ensemble（cross-encoder + LLM judge）
- 对 verifier 本身做 calibration check（在人工标注集上验证）
- 报告 verifier accuracy 作为系统的一个 dependency

**风险 6：Open-ended answer 的 outcome 评估不可靠**

很多 RAG benchmark 答案是开放生成的，LLM judge 本身有偏差。

应对：
- LLM judge ensemble（≥3 个 judge model 投票）
- 主表用 EM/F1（更客观），LLM judge 作为辅助指标
- 对 ambiguous case 做人工抽样

**风险 7：Self-adjusting bandit 在 P=10 预算下收敛不足（v1.1 新增）**

5 个 arm 平均每 arm 2 次观测，理论上可以让 Thompson Sampling 收敛，但实际可能受 reward 噪声影响导致 bandit 选择仍接近随机。如果 ablation A6 发现 bandit 没显著优于最佳固定 λ，这个子组件失效。

应对：
- **首选诊断手段**：A6.2 的 per-query-type λ 分布分析——如果视觉 vs 文本 query 上 bandit 收敛的 λ 显著不同，机制有效；否则无效
- **如果机制无效**：直接砍掉这个子组件，论文回到固定 λ，整体仍有两个主 contribution
- **如果机制部分有效**：考虑把 P 提到 15（推理时间增加 50%，但 bandit 有更多探索机会）
- **关键原则**：这个子组件的存在不应该让主线工作变脆弱。它的 fallback 必须能干净退出

**风险 8：Rationale 质量决定 PRM 训练效果（v1.2 新增）**

生成式 PRM 训练依赖 rationale label。如果 rationale 生成器（强 LLM）对 borderline case 产生糟糕的推理，PRM 会继承这些错误——它学到的不只是错误的 score，还有错误的"推理风格"。这比判别式范式多了一个失败入口。

应对：
- 人工抽样 200 条（每类动作类型）做 rationale 质量 check
- 质量过滤规则（引用 evidence_id、提及动作类型、长度约束）过滤明显劣质 rationale
- **如果 rationale 质量整体偏低**：升级到更强的生成模型，或对关键子集做人工标注
- **极端 fallback**：如果生成式范式始终训不稳定，退回判别式 PRM（只训 score），损失 sample efficiency 但项目仍可推进——判别式 PRM + 我们的 grounded label 设计本身仍有价值

### 5.3 时间线风险

**风险 9：12 个月做完所有 contribution 压力大**

应对（分阶段截止时间）：
- Month 1–2：Pilot study（VisualPRM 直接用作 RAG re-ranker 的失败模式诊断 + 小规模 rationale 生成质量验证）
- Month 3–6：ETBench-Open 数据集构造（含 rationale label 生成）+ PRM Stage 1+2 训练
- Month 7：MCTS 算法实现 + 主表实验（固定 λ 版本）
- Month 8：集成 self-adjusting bandit + 完成 A6 ablation；若失败则回退到固定 λ
- Month 9：robustness 实验
- Month 10–11：剩余 ablation + 论文撰写
- Month 12：投稿

**Fallback 策略**：
- 如果时间紧，砍掉 robustness study（放 future work）
- 如果 PRM 训练不收敛，退化到 frozen-VisualPRM + custom UCB（依然有 contribution 2）
- 如果 MCTS 比 BoN 提升不大，重新定位为"a process reward model for retrieval actions, evaluated with BoN"——纯 PRM 方向也能 carry 一篇论文

---

## 六、贡献定位与 Narrative

### 6.1 一句话总结

> EvidenceTree 是第一个在**检索动作空间**上做 MCTS 的多模态 RAG 系统，配合一个**针对状态修改型动作**训练的 grounded PRM，以及一个 **inference-time self-adjusting 的探索机制**。

### 6.2 三段式 narrative（用于 introduction）

**段 1 — 问题**：当前多模态 RAG 系统的检索决策是 unsupervised 的——outcome-only RL 让 reward 信号稀疏，rule-based reward 让信号 noisy，没有 PRM 让 credit assignment 错位。

**段 2 — 关键观察**：GroundedPRM（concurrent work）已经证明 grounded process supervision 能大幅提升数学推理。但它的 grounding 假设了 state 是固定的——这在 RAG 场景**根本不成立**，因为每个检索动作都改变了 state。

**段 3 — 我们的方法**：EvidenceTree 把 MCTS 建立在检索动作空间，配合一个动作类型化、模态特定 grounded 的 PRM。我们进一步引入一个 inference-time self-adjusting 机制：MCTS 的探索行为通过 bandit 反馈在单 query 内动态调整，避免对所有 query 使用统一的静态探索先验。三个组件加起来解决了 state-modifying credit assignment 这个 GroundedPRM 没碰过的问题，同时让搜索行为从"hardcoded 先验"演化为"从自己的运行中学习"——这与 self-evolution 文献的核心思想对话，但保持了 frozen-model 评估范式的严谨性。

### 6.3 与各先前工作的精确差异

| 工作 | 我们继承了什么 | 我们改变了什么 |
|------|--------------|--------------|
| RCTS | MCTS for RAG 的整体框架 | 树的状态/动作空间从 retrieved-pair-set 搬到 retrieval action space；reward 从 rule-based 改为 grounded PRM |
| AR-MCTS | active retrieval 在 expansion 时的多样化作用 | 树空间从 reasoning 搬到 retrieval action |
| GroundedPRM | 双源 grounding philosophy；low-data high-grounding 训练范式 | 从 fixed-state reasoning 推广到 state-modifying actions；从 binary 改为 graded grounding；多模态 |
| VisualPRM | 起点 checkpoint；多模态视觉推理能力 | 加 action-conditional value head；训练数据从 reasoning step 改为 retrieval action |
| MMSearch-R1 | agentic 多模态 RAG 的动作空间灵感 | 从 outcome-only RL 改为 process-rewarded MCTS |
| Self-Rewarding LMs / DeepSeek-R1 / Mulberry | self-evolution 的核心理念 | 操作层次从 parameter-level 下降到 inference-time search-policy level；保持 frozen-model 评估 |

---

## 七、立项评估总结

| 维度 | 评估 |
|------|------|
| 科学问题清晰度 | ✅ 三个具体子问题（A/B/C），不空泛 |
| 与最相关 prior work 的差异 | ✅ 与 RCTS / AR-MCTS / GroundedPRM 都有明确边界 |
| Novelty 强度 | ✅ 没有已发表工作同时解决 A+B+C |
| ICLR 接受门槛 | ✅ 两个核心 contribution 都是方法级，不是工程级 |
| 与当前热点对齐 | ✅ Self-adjusting bandit 与 self-evolution 文献形成对话 |
| 实验可复现性 | ✅ 检索与推理全程无 API 依赖；rationale label 一次性离线生成后固定可发布 |
| 计算可行性（≤8B 本地 + 服务器租赁） | ✅ 仅 $800 rationale 生成费用，1×8 H100 × 3–4 周 + 1×4 A100 × 2 周 |
| 时间可行性（12 个月） | ⚠️ 紧，需要分阶段截止；self-adjusting 集成预留 1 个月 |
| 失败时的 fallback | ✅ 多档 fallback：bandit 失败回退固定 λ；生成式 PRM 失败回退判别式；PRM 整体失败回退 frozen-VisualPRM；MCTS 失败回退 PRM+BoN |
| 主要风险 | ⚠️ 被 categorize 为 "GroundedPRM 的 RAG 版本"——通过写作克制（正常引用不反复划界）+ 方法本身的 state-modifying 差异自然体现来缓解 |

---

## 附录 A：为什么不用 DAG 结构

我们曾考虑过用 DAG（两条路径汇合到同一 evidence bundle）作为搜索结构，最终决定使用 tree。三个理由：

**理由 1**：DAG 的 reward backup 规则没有标准答案。一个 reward 该如何回传到多个父节点，是 search 算法领域未解决的问题。AlphaZero 也是 tree 不是 DAG。

**理由 2**：DAG 想表达的"多源证据融合"，用 tree 的**串行轨迹**就能表达——先检索 A，再检索 B，evidence_bundle 沿路径累积，到达需要融合的节点时所有证据已在 bundle 中。这避免了 DAG 的 backup 复杂度，也不需要引入特殊的并行动作（早期草稿曾用 parallel_search 复合动作来模拟 DAG 融合，v1.3 已将其移除，理由见附录 B.7——串行表达已足够，复合动作的代价大于收益）。

**理由 3**：审稿人对 "MCTS" 标签是 friendly 的。"MCTS for X" 让审稿人在 30 秒内 categorize 工作类型，而 "DAG search for X" 会消耗审稿人的认知预算。

---

## 附录 B：未采纳的设计点（与未来工作）

以下设计点在讨论中被考虑过，最终未进入 v1.0：

**B.1 推测式 draft-verify 执行**：用 2B draft model 扩展 + 7B verifier 执行。这个机制本身是工程优化，不是研究 contribution。降级为论文的 efficiency section 一个段落。如果 latency 是瓶颈，可在 future work 里展开。

**B.2 KB 缺相似样本时的合成**：原 RCTS 在 Fig 16 暴露了"KB 没有相似样本时全部分支都错"的失败模式。我们的 grounded PRM 应该能识别这种情况（所有分支的 grounding score 都低），但**"识别"和"主动合成补救"是两件事**。后者作为 future work。

**B.3 端到端 RL 训练 retriever + PRM + policy**：这一条路也有不少近期工作（VRAG-RL、MMSearch-R1）。EvidenceTree 走的是 training-free policy + trained PRM 路线，更可控、更 reproducible。端到端 RL 留给 future work。

**B.4 Information-Diversity UCB 的更一般形式**：modality coverage 只是 information axis 多样性的一个特例。我们在 v1.0 只做 modality version，因为它最容易讲清楚和实验验证。更一般的 information-diversity UCB（覆盖 entity 多样性、attribute 多样性、tool 多样性）作为 future work。

**B.5 更激进的 Self-Evolution（v1.1 新增）**：

v1.1 的 self-adjusting bandit 是 **inference-time、search-policy-level、within-query** 的轻量自适应机制。它**故意**避开了 self-evolution 文献中更激进的几个方向，这些方向作为 future work：

| 维度 | v1.1 的设计 | 更激进的 future work |
|------|------------|--------------------|
| 调整对象 | 单一标量参数 λ | PRM 参数本身 / 整个 selection 网络 |
| 调整时机 | 推理时 | 训练-推理交错持续学习 |
| 调整范围 | 单 query 内独立 | 跨 query 持续累积经验 |
| 学习信号 | PRM 预测 reward + median 阈值 | 真实 outcome（如何获取？开放问题） |
| 评估范式 | 标准 frozen-model | online learning（评估协议需要重新设计）|

为什么不在 v1.0/v1.1 做：
- **跨 query 持续学习**会破坏 ICLR 的 frozen-model 评估范式。审稿人会立刻质疑评估公平性
- **PRM 在线 fine-tuning** 需要持续的 GPU 资源消耗，超出学生项目预算
- **无 GT 下的 reward 信号**是个开放研究问题（用 PRM 自己做 reward 会陷入循环依赖）

这些限制是真实的，不是我们"懒"。在论文 future work 章节明确指出这些方向，**既承认了局限，也为后续工作划出了清晰的研究空间**。

**B.6 NAS 视角（论文写作阶段待决定的 packaging 选项，v1.2 新增）**：

我们的 retrieval-action MCTS 与 predictor-based Neural Architecture Search 在抽象层面同构——都是 predictor-guided 的分层决策搜索。讨论中确认：

- 这个连接**不作为算法 contribution**——我们不引入任何 NAS 的算法工具（DARTS、supernet 框架等）
- 它的真实价值是一个**思考视角**：提醒我们 PRM 本质是个 performance predictor，会遇到 predictor-based search 的经典失败模式（未见区域过度自信、被 surface feature 欺骗）。这个视角已经体现在我们对 PRM 验证机制（A6 sanity check、reward hacking 检测）的重视上
- "supernet 式密集监督"这一启发其实已隐式包含在我们的 step-level supervision 设计中，无需额外组件
- "uncertainty-aware PRM"价值真实但工程代价在当前预算下偏高，列为 future work / optional ablation，不进主线

**写作决策（待定）**：是否在论文 discussion 用**一个克制的段落**说明"我们的验证机制设计受 predictor-guided search 文献启发"。该段落若写，必须包含三处主动声明：(1) 此类比不在算法上被利用，(2) 它只解释验证机制设计，(3) 不声称这是来自 NAS 的 contribution。**是否写入，留待论文初稿 framing 清晰后（Month 10–11）再决定**——届时根据论文整体"概念预算"是否还容得下这一段来判断。这是纯 packaging 决策，不影响项目设计与执行。

**B.7 parallel_search 复合动作（v1.3 移除）**：

早期动作空间里有一个 `parallel_search(actions)`，用于在单个 step 内并行执行多个检索并合并结果。它最初的设计动机是替代 DAG 结构——用一个复合动作表达"多条路径汇合到同一个 evidence bundle"，在 tree 中保留干净的 backup 语义。

**移除理由**：

1. **树展开已覆盖"多候选探索"**：MCTS 的 expansion 本身就在一个节点上生成多个 children 并通过 rollout 探索。parallel_search 想表达的"试多个检索"，树结构天然就做了。两者的区别仅在于：树分叉是"或"（OR，最终选一条路径），parallel_search 是"与"（AND，多个证据合并）。

2. **"与"语义可以用串行表达**：真正需要合并多源证据的场景（如"比较图中两个实体谁更年长"，需同时获得两者信息），可以用串行轨迹表达——先 text_search A，再 text_search B，evidence_bundle 累积后 VLM 自然能比较。代价仅是多 1–2 层深度，在 max_depth=3 预算内可接受。

3. **实现与 credit assignment 复杂度高**：parallel_search 的 grounding 评分只能用"子动作 g 值的 mean"（hacky）；它是节点内的子树，value 如何 backup 与 tree-level credit 计算冲突；它还增加动作空间大小，稀释 P=10 的探索预算。

4. **符合项目一贯的设计原则**：砍掉 cool 但 substance 不足的设计。parallel_search 听起来高级（并行融合），但它解决的问题串行已能解决，净收益为负。

**保留可能性**：如果 Pilot 或主实验阶段发现 benchmark 中有大量"对称比较"类 query，且串行表达显著低效，可重新考虑引入。但这是效率优化，不是能力必需，不进 v1 主线。

---

## 附录 C：Pilot study 设计（Month 1–2）

在投入大规模训练前，先做一个 2 周的 pilot study，验证关键假设：

**Pilot 1 — VisualPRM-8B 在 RAG 检索轨迹上的失败模式诊断**

- 取 1K InfoSeek queries（自带 Wikipedia 语料）
- 用 vanilla RAG 生成检索轨迹
- 用 VisualPRM-8B 直接给轨迹打分，与 outcome correctness 比较
- **如果 VisualPRM 已经表现好**：重新审视整个项目（见 fallback）
- **如果 VisualPRM 失败**：诊断失败模式（哪些动作类型上失败？多模态 mismatch？长上下文 degradation？）

**Pilot 2 — MCTS vs Best-of-N 的真实 gap**

- 用一个简单的 frozen scorer（VisualPRM-8B）做两种推理：
  - BoN (N=8)：独立采样 8 条 trajectory，挑 PRM 分最高的
  - MCTS (rollouts=10)：在动作空间做 MCTS
- 比较准确率与计算时间（GPU 时长）
- **如果 BoN 已经接近 MCTS**：MCTS 这一层 contribution 弱化，重新规划
- **如果 MCTS 显著优**：confirmed contribution 2 的必要性

**Pilot 3 — ETBench-Open 50 样本人工标注**

- 跑 50 个 query 的 MCTS rollout
- 人工标注 step-level grounding 的"应该是什么样"
- 与自动标注的 graded grounding 比较
- **如果一致性高**：confirmed 自动标注 pipeline 可行
- **如果一致性低**：grounding verifier 需要重新设计

这三个 pilot 的成本约 $200 + 1×4 A100 × 1 周。**任何一个 pilot 失败，都比项目做到一半失败省时省钱得多。**

---

## 附录 D：符号系统（Notation Table，v1.3 新增）

为论文撰写准备统一的符号表，避免不同章节符号冲突。

**Reward 与 PRM 相关**：

| 符号 | 含义 |
|------|------|
| $\tau$ | 一条完整 trajectory |
| $s, a$ | 状态（含 query/image/evidence_bundle）、动作 |
| $R(\tau)$ | trajectory-level reward |
| $R_{\text{local}}, R_{\text{outcome}}$ | 双源 reward 的两个分量 |
| $\alpha$ | 双源 reward 融合权重（默认 0.5） |
| $g(a,s)$ | 动作 $a$ 在状态 $s$ 下的 grounding 评分 |
| $\mathcal{T}(s,a)$ | 经过节点 $(s,a)$ 的所有 trajectory 集合 |
| $y^*$ | ground truth 答案 |
| $Q(a\mid s)$ | PRM 对动作的评分（UCT 中的 exploitation 项） |

**MCTS 与 UCT 相关**：

| 符号 | 含义 |
|------|------|
| $N(s)$ | 节点 $s$ 的访问次数 |
| $N(s,a)$ | 边 $(s,a)$ 的访问次数 |
| $c$ | exploration 常数 |
| $\nu(a,\pi)$ | 动作 $a$ 相对路径 $\pi$ 的 modality novelty |
| $\mathcal{M}(\pi), \mathcal{G}(\pi)$ | 路径上已出现的模态集合、粒度集合 |
| $w_{\text{mod}}, w_{\text{gran}}$ | 新模态/新粒度的奖励权重 |

**Self-Adjusting Bandit 相关**：

| 符号 | 含义 |
|------|------|
| $P$ | 单 query rollout 预算（默认 10） |
| $\mathcal{A}=\{\lambda_1,...,\lambda_K\}$ | $K=5$ 个候选 λ 值 |
| $t$ | rollout 序号 |
| $\lambda^{(t)}$ | 第 $t$ 次 rollout 选用的 λ |
| $r^{(t)}$ | 第 $t$ 次 rollout 的 reward |
| $(\alpha_k^{(t)}, \beta_k^{(t)})$ | arm $k$ 的 Beta 后验参数 |
| $\theta_k^{(t)}$ | 从 arm $k$ 后验采样的值 |
| $m^{(t)}$ | 动态 median 阈值 |
| $z^{(t)}$ | 二值化成功信号 |
| $\sigma(t)$ | warm-up 阶段的轮询顺序 |
| $\tau_{\text{stop}}$ | 早停阈值 |
| $\lambda^*_q, \hat{\lambda}_q$ | query $q$ 的 oracle 最优 λ、bandit 收敛 λ |

**注**：注意 $\tau$（trajectory）与 $\tau_{\text{stop}}$（早停阈值）的区别——论文撰写时若担心混淆，可把早停阈值改用其他符号（如 $\eta$）。这类符号冲突检查应在 method section 定稿时统一过一遍。
