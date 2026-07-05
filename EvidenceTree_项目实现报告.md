# EvidenceTree: Process-Rewarded MCTS for Open-Domain Multimodal RAG

## 项目实现报告 v1.5

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

> **v1.4 相对 v1.3 的修订（聚焦化）**：
> 1. **移除 self-adjusting bandit 子组件**（删除原 §3.3.1 形式化机制 + Algorithm 1 + §3.3.2 的 Thompson Sampling regret bound + §4.5 A6 全部消融 + 风险 7）。理由：PRM 的 $Q(a)$ 已是检索动作质量的核心信号，再叠一个 inference-time 的 λ 自适应旋钮会**稀释 PRM 的作用、徒增一整套消融负担**，且在 $P=10$ 预算下 Thompson Sampling 难以可信收敛。这是 Contribution 2 的一个**子组件**，移除后 Contribution 2 收敛为单一干净的"PRM 引导树搜索"，重点更集中（三个顶层 Contribution——PRM / MCTS / ETBench-Open 数据集——不变）。
> 2. **移除 Modality-Coverage UCB 的 novelty 项**（删除原 §3.3 的 $\lambda\cdot\nu(a,\pi)$ bonus 与 §4.5 A2 消融）。理由：当动作空间收敛为 `{text_search, image_search, answer}`（见 §3.1）后，可"覆盖"的模态/粒度很少，该 bonus 既无用武之地又引入 $\lambda, w_{mod}, w_{gran}$ 三个待调超参。Selection 回归**纯 UCB1**（$Q + c\sqrt{\ln N/n}$），让 PRM 的 $Q$ 成为唯一质量信号。
> 3. **Contribution 2 重定位**：从"Self-Adjusting Modality-Coverage UCB"改为"Retrieval-Action MCTS（PRM 引导的标准 UCB1 树搜索）"，多模态能力由 `image_search` 的**实体消歧**作用承载（图 query 命中实体页 → 再 text_search 查属性），而非靠 UCB 的探索偏好。
> 4. **代码同步**：删除 `mcts/bandit.py`、`uct.py` 的 `modality_novelty`、`SearchConfig` 的 `lam/w_mod/w_gran/adaptive_lambda/bandit_*`；73→59 测试全过。

> **v1.5 相对 v1.4 的修订（首次真实轨迹驱动，2026-07）**：
> 0. **背景**：DatasetConstruct 首次真实小规模试跑（5 query × 3 数据集：InfoSeek / ScienceQA / GQA）的轨迹人工审查暴露两类标签污染：(a) InfoSeek 上"答案判对但证据不支撑"（参数化蒙对）的 answer 步拿满分 outcome 标签；(b) 成功轨迹上的冗余检索步（问题已解决仍继续搜，检索结果同实体故 grounding 不低）拿"双高"标签。
> 1. **Answer-support 门控**（修 (a)，详见 §3.2）：answer 步 score 从 outcome-only 改为 $R_{\text{outcome}} \cdot (\gamma + (1-\gamma)\cdot \text{supp})$，$\gamma=0.3$。supp = 答案是否被已积累证据支撑：API judge 三档（SUPPORTED/PARTIAL/UNSUPPORTED → 1/0.5/0，temperature 0；三档而非连续值是因为 LLM 直接输出连续分数不可校准——粗而可靠优于细而失准，后续可升级为类别 token logprob 加权得到真连续值）或离线 lexical（答案内容词+数字覆盖率，数字不匹配一票否决）；判不了返回 null → 退化 outcome-only，绝不注入编造值；空 bundle → 0。保序：答错(0) < 答对无支撑(≈γ) < 答对有支撑(≈1)。`unsupported_correct` 样本**不丢弃**——留作 Stage 4.2"同状态 DPO 对"的现成负样本。
> 2. **Rationale QC 扩展**（防 GT 泄漏与 (rationale, score) 自相矛盾）：新增 (d) GT 泄漏检查——禁评测措辞（"ground truth"/"标准答案"等）、禁"可见证据/问题/动作输入之外"的 gold answer 字符串（**从证据引用答案合法，凭空知道即泄漏**）；(e) rationale 末尾必须带 `VERDICT: good|mixed|poor`，与 score 方向硬矛盾（≥0.6 配 poor、≤0.4 配 good）即拦截重生成；VERDICT 剥离存 `rationale_verdict`。生成 prompt 同步加 ex-ante 禁令（借鉴 DBAgent／Learning to Search, arXiv 2604.07146 judge prompt 的 trajectory-realism 规则）。
> 3. **冗余检索步（over-search）治理暂缓**（(b)）：候选方案为边际增益折扣 $g' = g\cdot(1-\max\cos(\text{新证据}, \text{已有 bundle}))$ + "answer-now vs 多搜一步"DPO 对；因一刀切折扣会误伤**多源佐证**（对抗污染语料恰恰依赖佐证，见 §4.3），先以 `marginal_gain` 独立字段记录不动 score，待 100–1000 条真实分布确认"冗余 vs 佐证"可分后再决策融合。
> 4. **文档收口**：删除《EvidenceTree_实现报告_for_ClaudeCode.md》（其 Stage 6 bandit、modality bonus 阶段规范与 v1.4 冲突，双文档漂移），实现规范以本报告 + 代码库现状为准；本文件更名去掉文件名中的过期版本号。
> 5. **代码同步**：`prm/verifiers.py::AnswerSupportVerifier`、`prm/data_gen.py::fuse_answer_score`、`prm/rationale_gen.py` verdict/泄漏 QC、`DatasetConstruct` config/SCHEMA/validator 同步；测试 63→72 全过，mock 四步 pipeline + validator 零违规。

> **v1.5.1 相对 v1.5 的修订（100q × 3 数据集复盘驱动，2026-07）**：
> 0. **复盘诊断**（100 query/数据集、P=10 真实运行的节点分布报告）：
>    - **InfoSeek**（表现最弱，低分节点 316/426）：病灶不在 support judge——174 个 support=0 的 answer 里 ~164 个同时也答错（unsupported_correct 仅 10），judge 与 outcome 双双指向真实失败。真正的病因在上游三处：(a) **动作配比倒挂**（image_search:text_search = 166:54，而语料纯文本，图→文 CLIP 跨模态检索是最弱通路）；(b) **检索链断裂**（轨迹均 2.2 步 = `image_search → answer`，缺"用证据中实体名 text_search 属性"的第二环；policy prompt 的 anti-fabrication 规则未放行"证据已确认的实体名"，把这条路也堵了）；(c) **轨迹多样性坍缩**（去重后仅 ~2 条/query，树不分叉 → tree credit 退化 0/1、gap 过滤变激进、DPO 兄弟对稀缺）。outcome metric 已排除嫌疑（step1 用 `infoseek_accuracy`，数值题走 range）。
>    - **ScienceQA**：unsupported_correct 105/170——答案对、judge 判不支撑。这是 **support 蕴含语义与推理型选择题的失配**：答案由题面+选项推导，证据里不会写"所以选 A"，judge 按蕴含标准判 UNSUPPORTED 是职责内行为但标准本身不适用；105 个正确答案被钉在 floor=0.3，属系统性错罚（其 rationale_qc 丢弃 25 个也多为由此产生的 verdict 矛盾）。
>    - **support=0 数量与"相乘归零"澄清**：floor 的存在使 support=0 且答对 → 0.3 而非 0（ScienceQA 中分档 120 个的主要来源即这 105 个）；score=0 仅由 outcome=0 造成。0 档本身合理且无害。
> 1. **support judge 增设第四类 `NOT_REQUIRED` → null（退回 outcome-only）**：判据写死为"答案可由题面 + 可见图像内容经感知/逻辑/常识直接推导，不依赖外部事实知识；细粒度实体事实（物种数据、日期、度量、传记）永远需要证据"。选择题 rubric 同步放宽：证据蕴含**正确选项的内容**即算支撑，不要求逐字。原始判定存入新字段 `answer_support_label`（supported/partial/unsupported/not_required/…），用于诊断 null 的成因与跨数据集分布。
> 2. **policy prompt 修检索链（针对 InfoSeek (a)(b)）**：显式放行"证据中已出现的实体名"（不算臆造）；加硬规则——实体已识别而所问属性缺失时，下一步必须提议"实体名+属性"的 text_search，不得凭实体命中直接 answer；answer 的许可条件同步加入"或题面+图像即可推导（无需外部知识）"与 NOT_REQUIRED 对齐。
> 3. **提议多样性（针对 InfoSeek (c)）**：同批候选的 text_search query 必须互异且角度不同（实体名+属性 / 关键词式 / 换措辞），且不得重发本路径已执行过的 query。
> 4. **数据配比立场（记录）**：不采纳"GQA 60% 主力"——GQA 健康源于 oracle scene-graph 语料（image_search 必中），其信念迁移不到主表战场 InfoSeek/E-VQA；GQA 定位为课程起步 + verifier 校准集，主力应是修复后的 InfoSeek。ScienceQA 保持小份额。修复后重跑 100q 验证三个指标：InfoSeek text:image 比、轨迹数/query、ScienceQA unsupported_correct。

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

### Contribution 2: Retrieval-Action MCTS（PRM 引导的树搜索）

**这是什么**：一个把 MCTS 应用在**检索动作空间**的搜索算法，由 Contribution 1 的 grounded PRM 引导，选择标准为经典 UCB1。

**树结构**：
- 节点：部分上下文状态 (query, image, evidence_bundle)
- 边：类型化检索动作
- 叶子：answer 动作的执行

**Selection rule（标准 UCB1）**：
```
UCT(action) = Q(action) + c · √(ln N(parent) / N(action))
```
其中 `Q(action)` 是 PRM 给出的动作价值（exploitation），第二项是经典 UCB1 探索项。
**没有额外的 λ·novelty 项**：PRM 的 $Q$ 已经是质量信号，再叠一个 hardcoded 的探索偏好只会稀释它、并引入待调超参（v1.4 移除，详见变更说明）。

**Expansion**：节点扩展时，由 PRM 在候选动作空间上排序，选 top-k 作为 children。

**Simulation**：到达预设深度或 answer 动作时停止，用 PRM 预测 + 真实 outcome（如果可得）的混合作为 reward。

**Backup**：标准 MCTS reward 回传（不做 DAG，原因详见附录 A）。

**Termination**：早停（PRM 对 root 的 best path Q 值高于阈值）或 budget 耗尽。

**多模态能力的来源**：不靠 UCB 的探索偏好，而靠 `image_search` 的**实体消歧**作用——纯文本 RAG 无法判定"图里这个东西是谁"，`image_search` 用图 query 命中语料中的实体页，再由 `text_search` 查其属性。这一"先图后文"的顺序由 PRM 的 $Q$ 自然学到（见 §3.1、policy prompt），是本系统多模态价值的真正载体。

**为什么这是 contribution**：
- AR-MCTS 的 MCTS 在 reasoning space，不在 action space
- RCTS 的 MCTS 在 retrieved-pair-set space，不在 action space
- 本系统把搜索建在**检索动作空间**上，每条边都改变 state（retrieval 是 state-modifying 的），并由一个**针对状态修改型动作训练的 grounded PRM** 引导——这是已有 MCTS-for-RAG 工作都没有的组合

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
| `text_search(q)` | 文本 query | 用**文本**作为 query 检索语料 | 必需 |
| `image_search(image, region?)` | 图像（可选区域） | 用**图像**作为 query 检索语料 | 必需 |
| `crop(region)` | bbox 坐标 | 把当前 image 的某个区域作为新 image（改变空间注意力） | 待 Pilot 决定 |
| `zoom(region, factor)` | 区域 + 倍数 | 放大区域细节（改变分辨率） | 待 Pilot 决定 |
| `answer(text)` | 答案文本 | 终止动作 | 必需 |

**动作空间按 query 模态切分（v1.3 定版，2026-06 收口）**：

检索动作只按 **query（检索输入）的模态**区分，而**不**按检索结果的模态区分——
`text_search` 以文本去检索，`image_search` 以图像去检索，两者查询的是**同一个统一
CLIP 语料索引**，因此一次检索返回的结果**既可能是文本侧文档、也可能是图像侧文档**
（取决于哪一侧与 query 更相似）。这把动作空间收敛为 `{text_search, image_search,
answer}` 三个。

> **设计演化记录**：v1.3 早期曾尝试把检索动作按 (query 模态 × 结果模态) 拆成 2×2
> 共四个动作（`text_search` / `text_to_image` / `image_to_text` / `image_search`）。
> 后续收口为按 query 模态切分的两个搜索动作，理由有三：(1) 在 P=10 的探索预算下，
> 四个搜索分支会把每个分支的访问次数稀释一半，伤害搜索质量；(2) "结果该是文本还是
> 图像"本就该由检索器在统一语料里按相关度自然决定，硬编码进动作类型是把检索器的工作
> 提前到动作定义里，既不必要也限制了灵活性；(3) grounding 改为按**实际返回结果**的
> 模态打分（见 §3.2），动作类型不再需要承载"结果模态"这一信息。该收口对应代码提交
> `[ActionSpace] Compress to text_search/image_search/answer`，73/73 测试通过。

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

`g(a, s)` 是动作 `a` 执行后的 grounded 评分。**关键定义（v1.3 收口）：grounding
衡量「这一步检索回的结果与 question 的相关度」，而不是「动作的 query 输入与 question
的对齐度」。** 改用结果端打分有两个直接动因：(1) `image_search` 的 query 图是状态自带、
固定的，"输入 vs question"对它没有区分度，无法据此评分；(2) 防止 policy 臆造实体已由
policy prompt 约束，grounding 该回答的问题是"这步检索到底有没有捞到有用的东西"——这只能
看结果。空结果 → `g = 0`。

打分**按每个返回结果的实际模态**进行，而非按动作类型——因为 `text_search` 和
`image_search` 查询的是同一个语料，一步的结果可能混有文本侧与图像侧文档：

| 返回结果模态 | grounding signal `g` |
|---------|---------------------|
| 文本结果 | `cos(CLIP_text(question), CLIP_text(result))` |
| 图像结果 | `cos(CLIP_text(question), CLIP_image(result))` |
| 一步多个/多模态结果 | 各结果分别打分后取 **max**（这一步的 grounding ≙ 它最相关的那条结果） |
| `answer` | 不参与 local；由 **outcome × support 门控**决定（v1.5，见本节末） |

**为什么用统一 CLIP 空间**：文本结果与图像结果都用**同一个 CLIP 模型**、对齐到**同一个
`CLIP_text(question)` 锚点**。若文本用 sentence-transformer、图像用 CLIP，两者余弦尺度不同，
PRM 会把"尺度差"误当成"动作质量差"，系统性偏向某类动作而污染训练。唯一细节：同模态
（文本-文本）余弦系统性高于跨模态（文本-图像），故两类各配一个经验 `cos band`
（文本默认 0.5/0.9、图像默认 0.15/0.32）把余弦校准到同一个"相关性 0–1"语义；band 须在
服务器上按真实余弦分布重新校准。离线/mock 模式下退化为文本结果取 question 内容词召回率、
图像结果取中性 0.5，不加载任何模型。

> `crop` / `zoom` 一旦在 Stage 0.4 后实现，其 grounding 同理按"操作后区域里检索/识别到的
> 内容 vs question"打分（detection / OCR），并入上表的结果端口径，不另起一套。

**关键性质**：所有 `g` 都是 graded score（0–1 连续值），不是 binary。这是相对 GroundedPRM 的一个修正——RAG 场景下的"对/错"判断粒度不应该是 binary。

**Outcome reward**：

最简单的 trajectory-level 形式：
$$R_{\text{outcome}}(\tau) = \mathbb{1}[\text{answer}(\tau) \approx \text{ground truth}]$$

但**实际训练时我们用 step-conditional 的 tree-level credit**，而非把整条 trajectory 的 final outcome 均摊给所有 step。对每个节点 $(s, a)$，其 outcome label 是经过它的所有 trajectory 的成功率（Monte Carlo 估计）：

$$R_{\text{outcome}}(s, a) = \frac{1}{|\mathcal{T}(s,a)|} \sum_{\tau \in \mathcal{T}(s,a)} \mathbb{1}[\text{answer}(\tau) \approx y^*]$$

其中 $\mathcal{T}(s,a)$ 是经过节点 $(s,a)$ 的所有完整 trajectory，$y^*$ 是 ground truth。

**为什么用 tree-level credit 而非 trajectory-uniform credit**：trajectory-uniform 给好 step 和坏 step 同样的奖惩，信号 noisy；tree-level credit 让每个 step 的 label 反映"从这一步出发的真实成功潜力"，credit assignment 更精确。

对开放生成式答案，$\mathbb{1}[\cdot \approx y^*]$ 使用 LLM judge ensemble（多个 judge model 投票，降低单一 judge 的偏差）。

**Answer 步的 support 门控（v1.5 新增）**：

outcome 单独决定 answer 步标签，会把"答案判对但证据不支撑"（参数化蒙对，首次真实试跑中在 InfoSeek 上实际观测到）打成满分——PRM 由此学到"无证据也可以直接 answer"，这正是 outcome-only 方法的过程幻觉问题在**标签层**的复现。v1.5 起 answer 步的标签为：

$$R_{\text{answer}}(s) = R_{\text{outcome}}(s, a_{\text{ans}}) \cdot \big(\gamma + (1-\gamma)\cdot \text{supp}(y, \mathcal{E}_s)\big), \qquad \gamma = 0.3$$

其中 $\text{supp}(y, \mathcal{E}_s) \in [0,1]$ 度量答案 $y$ 是否被状态 $s$ 已积累的证据 bundle $\mathcal{E}_s$ 支撑（API judge 或离线 lexical；判不了 = null 时退化为 outcome-only；空 bundle → 0）。API judge 为**四类判定**（v1.5.1）：SUPPORTED/PARTIAL/UNSUPPORTED → 1/0.5/0；**NOT_REQUIRED → null**——答案可由题面+可见图像经感知/逻辑/常识直接推导、不依赖外部事实知识的题（如推理型选择题），不做蕴含门控（否则会系统性错罚正确的推理答案：证据里不会写"所以选 A"）。细粒度实体事实永远不适用 NOT_REQUIRED。原始判定存 `answer_support_label` 供分布诊断。

**为什么是"乘法 + 下限"而不是纯乘法或加权和**：outcome 必须做乘法门控——答错是最干净的负信号，不能被 support 抬起（加权和会给"答错但碰上假证据"的样本 0.5 分）；下限 $\gamma$ 保住排序 **答错(0) < 答对无支撑(≈γ) < 答对有支撑(≈1)**——纯乘法会把"蒙对"压成 0、与答错无法区分，且对 support verifier 的漏判噪声过脆（转述未命中会把证据扎实的正确答案误杀成硬负样本）。

**默认 α=0.5**，sensitivity ablation 给三档 (0.2 / 0.5 / 0.8)。support 门控的 $\gamma$ 默认 0.3，可做 sensitivity 检查。

### 3.3 UCB1 Selection

Selection rule（经典 UCB1，无额外 novelty 项）：
$$\text{UCT}(a) = Q(a) + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(a)}}$$

- 第一项 $Q(a)$：PRM 给出的动作价值（exploitation）；边未访问时用 PRM prior。
- 第二项：经典 UCB1 探索项（Hoeffding 血统，见 §3.3.1）。

**为什么不加 modality-coverage 的 $\lambda\cdot\nu(a,\pi)$ 项**（v1.4 移除）：动作空间收敛为 `{text_search, image_search, answer}`（§3.1）后，可"覆盖"的模态/粒度极少，该 bonus 无用武之地；而它会引入 $\lambda, w_{mod}, w_{gran}$ 三个待调超参，并与 PRM 的 $Q$ 抢"该探索哪个分支"的决定权、稀释 PRM 信号。让 $Q$ 做唯一质量信号、UCB1 做标准探索，更干净也更少消融负担。

**默认 $c=1.0$**，做 sensitivity ablation。

### 3.3.1 UCB1 的理论血统

UCT selection rule 的 exploration 项并非经验设计，而是源于 UCB1（Auer et al., 2002）：

$$
k^{(t)} = \arg\max_{k} \left[ \hat{\mu}_k + \sqrt{\frac{2\ln t}{n_k}} \right]
$$

其中 confidence radius $\sqrt{2\ln t / n_k}$ 由 Hoeffding 不等式导出。**我们 UCT 里的 $c\sqrt{\ln N(s)/N(s,a)}$ 正是这一 bound 的形式**——MCTS 的 UCT 本就是 UCB 在树上的推广（UCT = UCB applied to Trees，Kocsis & Szepesvári, 2006），给 exploration 项一个有数十年血统的统计学依据。

> **v1.4 移除**：v1.1–v1.3 曾在此处引入一个 inference-time 的 Thompson Sampling bandit 自适应调 λ（原 §3.3.1 形式化机制 + Algorithm 1）及其 regret bound（原 §3.3.2 Theoretical Justification）。已整体删除——PRM 的 $Q$ 已是核心质量信号，bandit 稀释其作用、徒增 §4.5 A6 一整套消融，且在 $P=10$ 预算下难以可信收敛。详见开头 v1.4 变更说明。

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

- **生成方式**：用一个强 LLM（如 Claude / GPT-4o / DeepSeek）看每个 (state, action, score) 三元组，生成对应的 rationale 文本
- **成本估算**：约 50–80K 样本，每个生成一段 rationale，总成本约 $800
- **质量过滤规则（v1.5 扩展至五条）**：rationale 必须满足 (a) 至少引用一个**本样本可见**的 evidence_id，(b) 明确提到动作类型，(c) 长度在 30–150 token 之间，(d) **无 GT 泄漏**——不含评测措辞（"ground truth"/"标准答案"等），不含可见证据/问题/动作输入之外的 gold answer 字符串（从证据引用答案合法，凭空知道即泄漏；泄漏样本会教 PRM 以"我知道正确答案"的姿态推理，而推理时它没有这个信息，train/inference 分布错位），(e) 末尾带 `VERDICT: good|mixed|poor` 且与 score 方向无硬矛盾（score≥0.6 不得 poor、≤0.4 不得 good——防止 (rationale, score) 自相矛盾的训练对教出"说一套打一套"的 PRM）。不合格的重新生成（至多 3 次）
- **关键说明**：这个 API 调用**仅用于生成 rationale label，不用于检索**。检索全程在 benchmark 自带语料上离线进行。因此这不破坏实验可复现性——rationale 一旦生成就是固定数据集，可随数据集一起发布

**质量控制**：
- 拒绝过短轨迹（<2 步）和过长轨迹（>8 步）
- 拒绝 grounding 与 outcome 严重不一致的轨迹（可能是 noisy label）
- "答对但证据不支撑"的 answer 样本**不丢弃**：由 support 门控压低分数并标记 `unsupported_correct`，留作 Stage 4.2 同状态 DPO 对的负样本（v1.5）
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

**A2 — UCB 探索常数 c 的敏感性**
- 纯贪心 PRM（c=0）
- 标准 UCB1（c ∈ {0.5, 1.0, 2.0}）
- 关注点：树搜索相对 greedy / Best-of-N 的增益是否稳健（呼应 Stage 0.2）

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

**A6 — （已移除，v1.4）**

self-adjusting bandit 已删除，故其全套消融（A6.1 主对比 / A6.2 consistency / A6.3 收敛 / A6.4 arm 数 / A6.5 warm-up）一并取消。这正是移除该组件的动机之一：少一个自设计组件，就少一整套必须自证的消融负担。

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
- v1.5 新增的 answer-support judge 同属此依赖：其三档判定（SUPPORTED/PARTIAL/UNSUPPORTED）应在 Stage 0.3 的人工一致性检查中一并校验；三分类的 agreement 也比连续分数更易度量

**风险 6：Open-ended answer 的 outcome 评估不可靠**

很多 RAG benchmark 答案是开放生成的，LLM judge 本身有偏差。

应对：
- LLM judge ensemble（≥3 个 judge model 投票）
- 主表用 EM/F1（更客观），LLM judge 作为辅助指标
- 对 ambiguous case 做人工抽样

**风险 7（已移除，v1.4）**：原"self-adjusting bandit 在 P=10 下收敛不足"风险随 bandit 组件一并删除。

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
- Month 7：MCTS 算法实现（PRM 引导 UCB1）+ 主表实验
- Month 8：搜索侧消融（A2 c 敏感性、A5 depth/rollout 预算）+ Stage 0.2 复核（MCTS vs BoN）
- Month 9：robustness 实验
- Month 10–11：剩余 ablation + 论文撰写
- Month 12：投稿

**Fallback 策略**：
- 如果时间紧，砍掉 robustness study（放 future work）
- 如果 PRM 训练不收敛，退化到 frozen-VisualPRM + 标准 UCB1（依然有 contribution 2）
- 如果 MCTS 比 BoN 提升不大，重新定位为"a process reward model for retrieval actions, evaluated with BoN"——纯 PRM 方向也能 carry 一篇论文

---

## 六、贡献定位与 Narrative

### 6.1 一句话总结

> EvidenceTree 是第一个在**检索动作空间**上做 MCTS 的多模态 RAG 系统，由一个**针对状态修改型检索动作**训练的 grounded PRM 引导搜索（标准 UCB1）。

### 6.2 三段式 narrative（用于 introduction）

**段 1 — 问题**：当前多模态 RAG 系统的检索决策是 unsupervised 的——outcome-only RL 让 reward 信号稀疏，rule-based reward 让信号 noisy，没有 PRM 让 credit assignment 错位。

**段 2 — 关键观察**：GroundedPRM（concurrent work）已经证明 grounded process supervision 能大幅提升数学推理。但它的 grounding 假设了 state 是固定的——这在 RAG 场景**根本不成立**，因为每个检索动作都改变了 state。

**段 3 — 我们的方法**：EvidenceTree 把 MCTS 建立在检索动作空间，配合一个动作类型化、模态特定 grounded 的 PRM，由标准 UCB1 引导（PRM 的 $Q$ 是唯一质量信号）。两个组件解决了 state-modifying credit assignment 这个 GroundedPRM 没碰过的问题——把 grounded process supervision 从 fixed-state reasoning 推广到每一步都改变 state 的检索动作，并保持 frozen-model 评估范式的严谨性。

### 6.3 与各先前工作的精确差异

| 工作 | 我们继承了什么 | 我们改变了什么 |
|------|--------------|--------------|
| RCTS | MCTS for RAG 的整体框架 | 树的状态/动作空间从 retrieved-pair-set 搬到 retrieval action space；reward 从 rule-based 改为 grounded PRM |
| AR-MCTS | active retrieval 在 expansion 时的多样化作用 | 树空间从 reasoning 搬到 retrieval action |
| GroundedPRM | 双源 grounding philosophy；low-data high-grounding 训练范式 | 从 fixed-state reasoning 推广到 state-modifying actions；从 binary 改为 graded grounding；多模态 |
| VisualPRM | 起点 checkpoint；多模态视觉推理能力 | 加 action-conditional value head；训练数据从 reasoning step 改为 retrieval action |
| MMSearch-R1 | agentic 多模态 RAG 的动作空间灵感 | 从 outcome-only RL 改为 process-rewarded MCTS |

---

## 七、立项评估总结

| 维度 | 评估 |
|------|------|
| 科学问题清晰度 | ✅ 三个具体子问题（A/B/C），不空泛 |
| 与最相关 prior work 的差异 | ✅ 与 RCTS / AR-MCTS / GroundedPRM 都有明确边界 |
| Novelty 强度 | ✅ 没有已发表工作同时解决 A+B+C |
| ICLR 接受门槛 | ✅ 两个核心 contribution 都是方法级，不是工程级 |
| 与当前热点对齐 | ✅ grounded process reward + multimodal RAG + test-time search 均为当前活跃方向 |
| 实验可复现性 | ✅ 检索与推理全程无 API 依赖；rationale label 一次性离线生成后固定可发布 |
| 计算可行性（≤8B 本地 + 服务器租赁） | ✅ 仅 $800 rationale 生成费用，1×8 H100 × 3–4 周 + 1×4 A100 × 2 周 |
| 时间可行性（12 个月） | ⚠️ 紧，需要分阶段截止 |
| 失败时的 fallback | ✅ 多档 fallback：生成式 PRM 失败回退判别式；PRM 整体失败回退 frozen-VisualPRM；MCTS 失败回退 PRM+BoN |
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

**B.5 Self-Evolution 作为 future work（v1.4 重写）**：

v1.1–v1.3 曾用一个 inference-time 的 self-adjusting bandit 体现"轻量自我调整"，v1.4 已移除（稀释 PRM、消融负担重、$P=10$ 难收敛）。"自我进化"方向**保留为 future work，但落点改到训练时**——比推理时调一个标量 λ 更有 substance 的，是一个 **ReST / STaR 式的自举闭环**：用当前 PRM 引导 MCTS 跑 rollout → outcome 验证挑成功轨迹 → tree-level credit 重新打标 → 重训得到更强 PRM，证明跨轮单调提升。它不与 PRM 抢戏，而正是"让 PRM 自己变强"。

| 维度 | 本项目 v1（主线） | self-evolution future work |
|------|------------------|---------------------------|
| 调整对象 | PRM 一次性训练后 frozen | PRM 参数随轮次自举更新 |
| 调整时机 | 训练后固定 | 训练-推理交错（多轮） |
| 学习信号 | 数据集固定标签 | 自生成轨迹 + outcome 验证重标 |
| 评估范式 | 标准 frozen-model | 需设计防 collapse 的多轮评估协议 |
| 前置验证 | — | 先在 1K 子集验"第 2 轮 > 第 1 轮"，过了再放量 |

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
| $c$ | UCB1 exploration 常数 |

**搜索其它符号**：

| 符号 | 含义 |
|------|------|
| $P$ | 单 query rollout 预算（默认 10） |
| $\tau_{\text{stop}}$ | 早停阈值（best rollout reward 超过即停） |

**注**：$\tau$（trajectory）与 $\tau_{\text{stop}}$（早停阈值）符号相近，论文定稿时可把早停阈值改用 $\eta$ 避免混淆。
