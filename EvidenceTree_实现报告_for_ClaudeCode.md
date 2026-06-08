# EvidenceTree 实现报告（for Claude Code）

> 本文档是给 Claude Code 的**实现规范**，不是设计论证文档。
> 目标：把项目切成可独立验证的阶段，每个阶段有明确的输入、输出、验收标准。
> **核心原则：每个阶段必须能独立跑通并验证后，再进入下一阶段。不要一次实现多个阶段。**

---

## 0. 项目一句话概述

EvidenceTree 是一个多模态 RAG 系统：在**检索动作空间**上做 MCTS 搜索，由一个 **grounded PRM** 给每个检索动作打分引导搜索，并在推理时用 **Thompson Sampling bandit** 自适应调整探索强度 λ。

数据流：
- **训练线**：benchmark 自带语料 → weak policy 跑 rollout → step 打标 → 训练 PRM
- **推理线**：用户 query → MCTS（PRM 引导 + bandit 调 λ）→ 输出最优答案

---

## 1. 技术栈与环境约束

| 项 | 选择 |
|----|------|
| 语言 | Python 3.10+ |
| 深度学习框架 | PyTorch 2.x |
| PRM 基座 | VisualPRM-8B（从 HuggingFace 加载，微调） |
| 训练框架 | transformers + peft (LoRA) + trl (DPO) |
| 推理加速 | vLLM（policy model 生成动作时用） |
| 检索 | BM25 (rank-bm25) + dense retriever (sentence-transformers / CLIP) |
| 向量索引 | FAISS |
| 实验追踪 | wandb |
| 配置管理 | hydra 或 yaml + argparse |

**环境约束（重要）**：
- 算力：本地 ≤8B 模型 + 租赁服务器（1×8 H100 训练 / 1×4 A100 推理）
- **检索全程离线**：所有检索在 benchmark 自带语料的本地索引上做，**不调用任何 web API**
- 唯一的外部 API：用强 LLM（Claude/GPT-4o）离线生成 rationale label（一次性）

---

## 2. 建议目录结构

```
evidencetree/
├── configs/                  # 所有超参配置
│   ├── data.yaml
│   ├── prm_train.yaml
│   └── mcts.yaml
├── data/
│   ├── corpus/               # benchmark 自带语料的本地索引
│   ├── trajectories/         # 生成的 rollout 数据
│   └── etbench_open/         # 最终训练数据集
├── src/
│   ├── actions/              # 检索动作定义与执行
│   │   ├── action_space.py   # 动作类型（核心 3 个 + 视觉操作待定）
│   │   ├── retrievers.py     # BM25 + dense retriever 封装
│   │   └── executor.py       # 动作执行器
│   ├── prm/                  # PRM 相关
│   │   ├── model.py          # PRM 模型封装（VisualPRM-8B）
│   │   ├── verifiers.py      # 模态特定 grounding verifier
│   │   ├── data_gen.py       # 训练数据生成
│   │   ├── rationale_gen.py  # 强 LLM 生成 rationale
│   │   └── train.py          # 三阶段训练
│   ├── mcts/                 # 搜索相关
│   │   ├── node.py           # 树节点
│   │   ├── uct.py            # UCT selection + modality bonus
│   │   ├── bandit.py         # Thompson Sampling bandit
│   │   └── search.py         # 主搜索循环
│   ├── eval/                 # 评测
│   │   ├── benchmarks.py     # benchmark 加载
│   │   └── metrics.py        # EM/F1/consistency 等
│   └── utils/
├── scripts/                  # 入口脚本
│   ├── build_index.py
│   ├── gen_trajectories.py
│   ├── gen_rationale.py
│   ├── train_prm.py
│   └── run_inference.py
├── tests/                    # 每个模块的单元测试
└── pilot/                    # Pilot study 脚本（最先做）
```

---

## 3. 分阶段实现计划

> **每个阶段标注：目标 / 依赖 / 产出 / 验收标准。**
> **关键：Stage 0（Pilot）必须最先做，它决定整个项目是否值得继续。**

---

### Stage 0 — Pilot Study（最先做，1–2 周）

**这是 go/no-go 决策阶段，不是热身。如果 pilot 失败，需要重新审视项目而非继续。**

#### Stage 0.1 — VisualPRM 失败模式诊断

- **目标**：验证 frozen VisualPRM-8B 直接用作 RAG 检索轨迹打分器时，到底差在哪
- **依赖**：无（最先做）
- **步骤**：
  1. 加载 1K InfoSeek queries（含自带 Wikipedia 语料）
  2. 用 vanilla RAG（BM25 检索 top-5 + LLM 生成）跑出检索轨迹
  3. 用 frozen VisualPRM-8B 给每条轨迹打分
  4. 对比 PRM 评分与真实 outcome correctness
- **产出**：一份诊断报告（PRM 评分与 outcome 的相关性、失败模式分类）
- **验收标准**：
  - ✅ 如果 VisualPRM 评分与 outcome 相关性低（如 Spearman < 0.3）→ 证明需要微调，**项目假设成立，继续**
  - ⚠️ 如果 VisualPRM 已经表现好（相关性 > 0.6）→ **停下来重新审视项目**（可能不需要微调 PRM）

#### Stage 0.2 — MCTS vs Best-of-N 的 gap

- **目标**：验证 MCTS 这一层是否真的比简单的 Best-of-N 强，决定 Contribution 2 是否成立
- **步骤**：
  1. 用 frozen VisualPRM-8B 做两种推理：
     - BoN (N=8)：独立采样 8 条 trajectory，挑 PRM 分最高
     - MCTS (rollouts=10)：在动作空间做简化版 MCTS
  2. 比较准确率与 GPU 时间
- **验收标准**：
  - ✅ MCTS 显著优于 BoN → Contribution 2 必要性确认
  - ⚠️ BoN 接近 MCTS → MCTS 这层 contribution 弱化，需重新规划

#### Stage 0.3 — 50 样本人工标注一致性

- **目标**：验证自动 grounding 标注 pipeline 是否可靠
- **步骤**：跑 50 个 query 的 rollout，人工标注 step-level grounding 应该是什么样，与自动标注比较
- **验收标准**：
  - ✅ 一致性高 → 自动标注 pipeline 可行
  - ⚠️ 一致性低 → grounding verifier 需要重新设计

#### Stage 0.4 — crop/zoom 需求统计（决定动作空间）

- **目标**：统计目标 benchmark 中"需要看图像局部细节"的 query 占比，决定是否实现 crop/zoom
- **步骤**：
  1. 从每个目标 benchmark 抽样 100–200 个 query
  2. 人工/LLM 辅助标注：该 query 是否需要看图像局部（读小字、看仪表读数、识别图中某小区域实体）才能回答
  3. 统计占比
- **决策标准**：
  - 占比高（> 25%）→ 实现 crop + zoom 两个独立动作
  - 占比中等（10–25%）→ 合并为单一 `focus(region)` 动作
  - 占比低（< 10%）→ 不实现，动作空间保持 {text_search, image_search, answer}
- **产出**：动作空间的最终定版，供 Stage 1 实现

**Stage 0 总验收**：四个 pilot 都通过 → 进入 Stage 1。任何一个亮红灯 → 暂停，与导师讨论。

---

### Stage 1 — 检索动作空间与执行器（1 周）

- **目标**：实现检索动作的定义和执行，建立本地检索索引
- **依赖**：Stage 0 通过
- **动作空间（v1.3 精简版）**：

| 动作 | 参数 | 说明 | 状态 |
|------|------|------|------|
| `text_search(query)` | 文本 query | 文本检索 | **必需，先做** |
| `image_search(img_crop)` | 图像区域 | 图像检索 | **必需，先做** |
| `answer(text)` | 答案文本 | 终止动作 | **必需，先做** |
| `crop(region)` | bbox | 聚焦图像局部 | 待 Pilot 决定 |
| `zoom(region, factor)` | 区域+倍数 | 放大细节 | 待 Pilot 决定 |

- **实现内容**：
  1. `action_space.py`：先实现 3 个必需动作（text_search / image_search / answer）的数据结构
     - **crop / zoom 暂不实现**：在 Stage 0 Pilot 统计 benchmark 中"需要看局部细节"的 query 占比后再决定：占比高→实现 crop+zoom；中等→合并为单一 `focus(region)`；< 10%→不实现
     - **不要实现 parallel_search**：树展开已覆盖多候选探索，"合并多源证据"用串行轨迹表达即可（先搜 A 再搜 B，evidence_bundle 累积后比较）
  2. `retrievers.py`：BM25 + dense retriever 封装，建在 benchmark 自带语料上
  3. `executor.py`：给定动作 + 当前状态，执行并返回更新后的 evidence_bundle
  4. `build_index.py`：对每个 benchmark 自带语料建 FAISS 索引（离线，一次性）
- **产出**：可以对任意 query 执行核心动作，返回检索结果
- **验收标准**：
  - ✅ 单元测试：每种动作类型能正确执行并更新 evidence_bundle
  - ✅ `text_search("Taylor Swift songs")` 能从本地索引返回相关文档
  - ✅ `image_search` 能用图像 query 返回相关图文对
  - ✅ evidence_bundle 状态正确累积（执行多个动作后包含所有证据）
  - ✅ 动作空间设计为可扩展（之后若 Pilot 决定加 crop/zoom/focus，不需重构）

---

### Stage 2 — Grounding Verifiers（1 周）

- **目标**：实现模态特定的 grounding verifier，给每个动作打 local grounding 分
- **依赖**：Stage 1
- **实现内容**：`verifiers.py`，按动作类型实现：

| 动作类型 | verifier 实现 | 状态 |
|---------|--------------|------|
| `text_search` | cross-encoder 算 query 与原 question 的语义对齐度 | 必需 |
| `image_search` | CLIP / vision-language matching 算图像 crop 与 question 视觉实体对齐度 | 必需 |
| `answer` | 不参与 local，由 outcome 决定 | 必需 |
| `crop` / `zoom` | detection / OCR 模型判断操作后区域是否含 question 关键 entity | 仅当 Stage 1 决定实现这些动作时才做 |

- **注**：不实现 parallel_search 的 verifier（该动作已移除）

- **关键**：所有 grounding 是 **graded score（0–1 连续值）**，不是 binary
- **产出**：`g(action, state) -> float` 函数
- **验收标准**：
  - ✅ 给一个明显合理的动作（如已知歌手后搜其歌曲）打高分
  - ✅ 给一个明显不合理的动作（如无证据时直接 answer）打低分
  - ✅ Stage 0.3 的人工标注一致性在这里复测通过

---

### Stage 3 — 训练数据生成（ETBench-Open）（2–3 周）

- **目标**：生成 50–80K 条带 step-level 标注的 trajectory
- **依赖**：Stage 1, 2
- **实现内容**：
  1. `gen_trajectories.py`：用 weak policy（如 Qwen2.5-VL-7B）在每个 query 上跑 MCTS rollout，生成 ~600K trajectory
  2. `data_gen.py`：给每个 step 打标
     - local grounding label：调 Stage 2 的 verifier
     - outcome label：**tree-level credit**（经过该节点的所有 trajectory 的成功率，Monte Carlo 估计）
  3. `rationale_gen.py`：用强 LLM（Claude/GPT-4o）给每个样本生成 rationale
     - 质量过滤：必须引用 ≥1 个 evidence_id、提到动作类型、长度 50–150 token
  4. 质量控制：拒绝过短(<2 步)/过长(>8 步)/grounding-outcome 严重不一致的轨迹
- **产出**：ETBench-Open 数据集（每条样本含 state, action, rationale, score）
- **验收标准**：
  - ✅ 数据集规模达到 50–80K samples
  - ✅ 人工抽样 200 条检查 rationale 质量合格
  - ✅ 人工抽样 500 条检查 score label sanity
  - ✅ 数据格式可直接喂给 Stage 4 训练

> **可选探索（来自 D²Evo 借鉴）**：difficulty-aware 子采样——用 weak PRM 评分与 grounding label 的差距分层采样（保留中等差距的样本，它们学习信号最强）。**先在 1K 子集验证 |Δ| metric 是否预测学习增益，成功再全量。失败则跳过，用均匀采样。**

---

### Stage 4 — PRM 三阶段训练（2–3 周）

- **目标**：从 VisualPRM-8B 微调出 EvidenceTree-PRM
- **依赖**：Stage 3
- **实现内容**：`train.py`，三阶段：

| 阶段 | 方法 | 目标 |
|------|------|------|
| Stage 4.1 | SFT | 学会四元组输入格式 + 输出 rationale+score（生成式） |
| Stage 4.2 | Action-typed DPO | 挖掘"同状态不同动作"对比对，学正确的动作偏好 |
| Stage 4.3 | Calibration | Platt/temperature scaling 把 score logit 校准到真实 reward 期望 |

- **关键设计**：
  - **训练时 generative**（输出 rationale + score），**推理时 score-only**（只读 score token logit，不生成 rationale）
  - 用 LoRA 微调降低显存
- **产出**：训练好的 PRM checkpoint
- **验收标准**：
  - ✅ Stage 4.1 后：PRM 能对四元组输入输出格式正确的 rationale+score
  - ✅ Stage 4.2 后：PRM 在"同状态不同动作"对比对上偏好正确率提升
  - ✅ Stage 4.3 后：PRM score 校准良好（calibration curve 接近对角线）
  - ✅ 推理时 score-only 模式速度达标（单次评分 < 100ms）
  - ✅ **关键**：在 held-out 集上，PRM 评分与 outcome 相关性显著高于 Stage 0.1 的 frozen baseline

---

### Stage 5 — MCTS 搜索（固定 λ 版本）（1–2 周）

- **目标**：实现完整 MCTS 搜索，先用固定 λ（暂不加 bandit）
- **依赖**：Stage 1, 4
- **实现内容**：
  1. `node.py`：树节点（state, children, visit count, value）
  2. `uct.py`：UCT selection rule
     ```
     UCT(a|s) = Q(a|s) + c·√(ln N(s) / N(s,a)) + λ·ν(a,π)
     ```
     - `Q(a|s)`：PRM 评分（exploitation）
     - 第二项：经典 exploration
     - `ν(a,π)`：modality novelty bonus（路径上新模态 +w_mod，新粒度 +w_gran）
  3. `search.py`：标准 MCTS 四阶段（selection → expansion → simulation → backup）
     - expansion 时：用 policy LLM 生成候选动作，PRM 评分，选 top-k 作 children
     - 默认 P=10 rollout，max_depth=3
     - 早停：best path 的 PRM Q 值 > 阈值
- **产出**：给定 query，返回最优 answer
- **验收标准**：
  - ✅ 完整搜索流程跑通，能输出答案
  - ✅ 在 Stage 0.2 的样本上，MCTS 性能复现/超过 pilot 结果
  - ✅ modality bonus 生效：能观察到跨模态探索行为
  - ✅ 固定 λ 下，在主 benchmark（InfoSeek/E-VQA/MRAG-Bench）上跑出主表结果

---

### Stage 6 — Self-Adjusting Bandit（1 周）

- **目标**：在 MCTS 上加 Thompson Sampling bandit 自适应调 λ
- **依赖**：Stage 5
- **实现内容**：`bandit.py`，按 Algorithm 1：
  1. 5 个 arm：λ ∈ {0.1, 0.3, 0.5, 0.7, 1.0}，每 arm 一个 Beta(1,1) 先验
  2. 前 3 次 round-robin warm-up（覆盖 {0.1, 0.5, 1.0}）
  3. 之后每次 rollout 前 Thompson 采样选 λ
  4. rollout 后：reward 高于历史 median → 该 arm α+1，否则 β+1
  5. bandit 状态**单 query 独立**，不跨 query 共享
- **关键代码逻辑**：
  ```
  for t in range(P):
      if t < 3: λ = round_robin[t]
      else: λ = argmax_k(Beta(α_k, β_k).sample())
      π = mcts_rollout(root, λ)
      r = PRM(π)
      if t > 0:
          z = 1 if r > median(history) else 0
          α[k*] += z; β[k*] += (1-z)
      history.append(r)
  ```
- **产出**：带自适应 λ 的完整推理
- **验收标准**：
  - ✅ bandit 正确运行，λ 在 rollout 间变化
  - ✅ 单 query 独立性验证（不同 query 的 bandit 状态互不影响）
  - ✅ 集成后整体推理流程不崩

---

### Stage 7 — 评测与消融（2 周）

- **目标**：完成主表实验 + 关键消融
- **依赖**：Stage 5, 6
- **实现内容**：
  1. `benchmarks.py`：加载 InfoSeek / E-VQA / MRAG-Bench / OK-VQA
  2. `metrics.py`：EM / F1 / consistency 指标
  3. Baselines：Vanilla RAG / RCTS / AR-MCTS / VisualPRM+BoN / zero-shot Qwen2.5-VL-7B
  4. 消融实验：

| 消融 | 内容 | 验收关注点 |
|------|------|----------|
| A6.1 | bandit vs 固定 λ ∈ {0.1...1.0} | bandit ≥ 最佳固定 λ |
| **A6.2** | **consistency 指标**（bandit 收敛 λ vs oracle 最优 λ） | **远高于随机 0.2 → bandit 真在学习** |
| A6.3 | 收敛速度（P 次内是否收敛） | 10 次内收敛，否则提到 P=15 |
| A6.4 | arm 数量 K ∈ {3,5,10} | K=5 是甜蜜点 |
| A6.5 | warm-up ∈ {0,3,5} | warm-up=3 最优 |

- **产出**：主表 + 消融表 + 多 seed 方差报告
- **验收标准**：
  - ✅ 主表：EvidenceTree 在主 benchmark 上超过 baselines
  - ✅ **A6.2 consistency 显著 > 0.2**（这是 bandit 有效性的硬证据）
  - ✅ 所有结果报告 3 seed 的 mean ± std
  - ✅ **特别检查**：adaptive λ 的结果方差是否比固定 λ 显著更大（若是，说明 bandit 在放大噪声而非提取信号 → 需诊断）

---

## 4. 关键实现注意点（Claude Code 必读）

### 4.1 容易踩的坑

1. **tree-level credit 不要写成 trajectory-uniform**：outcome label 是"经过该节点的所有 trajectory 成功率"，不是"整条 trajectory 的 final reward 均摊给所有 step"。这是常见 bug。

2. **训练 generative，推理 score-only**：训练时模型输出 rationale+score；推理时只读 score token 的 logit，**不要生成 rationale 文本**（否则 MCTS 几十次 PRM 调用会极慢）。

3. **bandit 单 query 独立**：每个 query 初始化新的 bandit，**不要跨 query 共享状态**（共享会破坏 frozen-model 评估范式）。

4. **检索全程离线**：所有检索在本地 FAISS 索引上，**不要调用任何 web API**。唯一外部 API 是 rationale 生成（离线一次性）。

5. **modality bonus 在 selection，不在 reward**：ν(a,π) 加在 UCT 公式里，**不要加进 reward function**（reward 是事实判断，selection 是策略判断，混了会污染 reward）。

6. **不要实现 parallel_search**：树展开（expansion 生成多 children）已经覆盖"探索多个候选动作"。需要合并多源证据时用串行轨迹（先搜 A 再搜 B）。parallel_search 的特殊 backup 语义会和 tree-level credit 冲突。

7. **动作空间从最小集起步**：先实现 {text_search, image_search, answer} 三个，跑通整条链路后，再根据 Stage 0.4 的统计决定是否加 crop/zoom/focus。不要一开始就实现全部动作——动作越多，P=10 预算下每个分支被探索越少。

### 4.2 fallback 链（如果某个组件失败）

按这个顺序降级，每一级仍可推进项目：
- bandit 失败（A6.2 consistency 接近 0.2）→ 回退固定 λ
- 生成式 PRM 训不稳 → 回退判别式 PRM（只训 score，损失 sample efficiency）
- PRM 整体不如 frozen baseline → 回退 frozen-VisualPRM（重新审视项目）
- MCTS 不如 BoN → 回退 PRM+BoN

### 4.3 每阶段结束时的自检

每个 Stage 完成后，Claude Code 应该：
1. 跑该 Stage 的单元测试
2. 对照"验收标准"逐条检查
3. 把验收结果记录下来（哪些通过、哪些有问题）
4. **只有验收全过，才进入下一 Stage**

---

## 5. 阶段依赖图

```
Stage 0 (Pilot) ──┬──> Stage 1 (动作空间) ──> Stage 2 (verifier) ──> Stage 3 (数据生成) ──> Stage 4 (PRM 训练)
                  │                                                                              │
                  └──> [go/no-go 决策]                                                          ↓
                                                                          Stage 5 (MCTS 固定λ) ──> Stage 6 (bandit) ──> Stage 7 (评测+消融)
```

关键路径：Stage 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7，基本是线性的。
Stage 5 可以在 Stage 4 进行中并行准备（先用 frozen PRM 搭框架，PRM 训好后替换）。

---

## 6. 起步建议

**第一步做什么**：直接从 Stage 0.1 开始——加载 VisualPRM-8B 和 1K InfoSeek，跑诊断。这一步成本最低、信息量最大，且决定整个项目是否继续。

**不要做什么**：不要一上来就搭完整框架。先用 pilot 验证核心假设，再逐阶段建。
