# 轨迹如何变成数据集：四步流水线全流程讲解（含 v1.5 support 门控）

这份文档回答一个问题：**一条 MCTS 轨迹从被跑出来，到成为（或没成为）ETBench-Open
里的训练样本，中间到底发生了什么？**

格式模仿 `longest_trajectories_simple_cn.md`：每个例子逐步看，但这次的视角不是
"这条轨迹合不合理"，而是"**pipeline 会怎么处置它的每一步**"。所有数值为示意。

---

## 第 0 部分：先回答最关键的问题——什么会被丢弃？

三个容易混淆的概念，先分清：

| 概念 | 含义 | 进不进数据集 |
|---|---|---|
| **低分样本**（步骤打了低 score） | 坏检索步、失败轨迹上的步骤 | **进**。PRM 是打分器，必须见过坏动作长什么样才能给坏动作打低分。全是高分样本的数据集训不出 PRM |
| **DPO rejected**（偏好对里的败方） | 同状态对比中"不该选"的那个动作，如 `unsupported_correct` 的 answer 步 | **进**。它带着压低的 score 和标记留在数据集里，Stage 4.2 挖对时作为 rejected 方使用。"rejected"是它在**一对**里的角色，不是它在**数据集**里的去留 |
| **Step 4 硬过滤丢弃** | 见下面三条规则 | **不进**。这是唯一真正的删除 |

**Step 4 的三条丢弃规则（唯一的"不进数据集"名单）**：

1. **轨迹级**：步数 < 2 或 > 8 → 整条轨迹的所有样本丢弃（`quality.min_steps/max_steps`）；
2. **样本级**：非 answer 步的 `|local_grounding − outcome_credit| > 0.7` → 该样本丢弃
   （grounding 和 outcome 严重打架 = 疑似 noisy label；answer 步 local=null 不受此规则约束）；
3. **样本级**：rationale 重生成 3 次仍不过 QC → 该样本丢弃。

除此之外，**所有轨迹——成功的、失败的、蒙对的——都会变成样本进入数据集**。
唯一的轨迹级预处理是同 query 内动作序列完全相同的去重（Step 1）。

---

## 第 1 部分：Step 1 —— 轨迹是怎么被跑出来的

对每个 query，policy VLM（弱策略，如 Qwen2.5-VL）在 MCTS 里跑 **P=10 次 rollout**：

```
每次 rollout：
  selection   从根出发，按 UCT 选已扩展的分支往下走
  expansion   到达叶子时，policy 提议候选动作（text_search 的 query 由它拟、
              image_search 用状态自带的图、answer 起草答案），执行后成为新节点
  simulation  贪心走到 answer 或深度上限（max_depth=3）
  backup      这条轨迹的启发式引导分回传到路径上的节点
```

要点：

- **10 次 rollout 的轨迹全部收集**——不是只留最好的一条。失败轨迹是 PRM 的
  负样本来源；同一状态下的不同分支是 tree-level credit 和 DPO 对的原料。
- 数据生成时**关闭早停**（`early_stop_q: 2.0`），就是为了让树多探索、多分叉。
- `gen_reward` 只是生成期的引导分（启发式），**不是训练标签**，Step 2 会重新打标。

一个 query 跑完后，树可能长这样（示意）：

```
root "这栋建筑正式开放日期是哪一天？" (图: Pro Football Hall of Fame)
├── text_search("opening date Pro Football Hall of Fame")   ← 6 条轨迹经过
│   ├── answer("September 7, 1963")                          ← 4 条（全对）
│   ├── image_search(原图)                                    ← 2 条
│   │   └── answer("September 7, 1963")                      ← 2 条（全对）
└── image_search(原图)                                        ← 4 条轨迹经过
    ├── answer("1963")                                        ← 1 条（对）
    └── text_search("Canton Ohio museum")                     ← 3 条
        └── answer("in 1962")                                 ← 3 条（全错）
```

---

## 第 2 部分：Step 2 —— 每一步拿到什么标签

每个 step 变成一个独立样本，带三个（answer 步四个）标签：

| 标签 | 谁打的 | 含义 |
|---|---|---|
| `local_grounding` | CLIP verifier | 这一步**检索回的结果**与问题的相关度（0–1；answer 步 = null） |
| `outcome_credit` | tree-level credit | 经过**相同动作前缀**的所有轨迹的成功率（MC 均值） |
| `answer_support`（仅 answer 步，v1.5） | support verifier | 答案是否被已积累证据支撑（判不了 = null；judge 判 NOT_REQUIRED——题面+图像即可推导、无需外部知识——也 = null，v1.5.1，防止蕴含门控错罚推理题的正确答案） |
| `score` | 融合公式 | 非 answer 步：`α·local + (1−α)·outcome`；answer 步：`outcome × (0.3 + 0.7·support)` |

**tree-level credit 用上面那棵树算给你看**（这是"不是均摊"的意义所在）：

- 前缀 `text_search("opening date...")`：6 条经过、6 条全对 → credit = **1.0**
- 前缀 `image_search(原图)`：4 条经过、1 条对 → credit = **0.25**
- 前缀 `image_search → text_search("Canton Ohio museum")`：3 条经过全错 → credit = **0.0**

同一个 query，"先文搜"和"先图搜"两个第一步拿到完全不同的 outcome 信号——
这是 PRM 学"这个状态该选哪类动作"的核心监督。如果按经典 bug 把整条轨迹的
final reward 均摊，同一轨迹里的好步和坏步会拿一样的分。

### 例 A：健康轨迹（InfoSeek，全部保留）

**问题**：这栋建筑正式开放日期是哪一天？ **答案**：September 7, 1963 ✅

| step | 动作 | local | outcome | support | score | 去向 |
|---|---|---|---|---|---|---|
| 0 | `text_search("opening date PFHOF")`，检索回含开放日期的段落 | 0.85 | 1.0 | — | 0.93 | **保留**（高分正样本） |
| 1 | `image_search(原图)`，检索回同实体的奖项/成员信息 | 0.55 | 1.0 | — | 0.78 | **保留**（gap=0.45 < 0.7） |
| 2 | `answer("September 7, 1963")` | null | 1.0 | **1.0**（日期在 [e0] 里） | **1.0** | **保留**（理想 answer 正样本） |

注意 step 1 是那个"多余的 image_search"：当前标签体系下它仍拿 0.78 的不低分数
（同实体结果 grounding 不低 + 躺在成功轨迹上）。这正是 v1.5 变更说明里**暂缓处理**
的 over-search 问题——`marginal_gain` 字段记录、待真实分布再决策，本文档如实标注。

### 例 B：蒙对轨迹（InfoSeek，v1.5 的主角）

**问题**：这只鸟通常最大能长到多少克？ **标准答案**：~153g（容差内 145 被判对）
**模型答案**：Up to 145 grams ✅（但证据里从没出现过任何克数）

设这条路径独占（n_through=1，outcome=1.0）：

| step | 动作 | local | outcome | support | score | 去向 |
|---|---|---|---|---|---|---|
| 0 | `text_search("how heavy mourning dove")`，检索回鸟种比较、无数值 | 0.15 | 1.0 | — | 0.58 | **丢弃**（gap=0.85 > 0.7，规则 2） |
| 1 | `image_search(鸟图)`，检索回麻雀、鸟叫声等 | 0.10 | 1.0 | — | 0.55 | **丢弃**（gap=0.90 > 0.7） |
| 2 | `answer("Up to 145 grams")` | null | 1.0 | **0.0**（145 不在任何证据里，数字门控一票否决） | **0.3** | **保留** + `unsupported_correct: true` |

三层机制在这条轨迹上各司其职：

- 两个烂检索步被 **gap 过滤**丢掉（grounding 说没用、outcome 说成功，打架 = 疑似噪声）；
- answer 步 **v1.5 之前**会拿 score=1.0（outcome-only）——PRM 学到"无证据也能 answer 满分"；
  **v1.5 之后**被 support 门控压到 0.3，且**保留**在数据集里；
- `unsupported_correct: true` 标记它为 Stage 4.2 DPO 的 rejected 候选（见第 5 部分）。

### 例 C：失败轨迹（也进数据集——这是负样本，不是垃圾）

**问题**：同例 A。这是树里 `image_search → text_search("Canton Ohio museum") → answer("in 1962")` 那 3 条之一。

| step | 动作 | local | outcome | support | score | 去向 |
|---|---|---|---|---|---|---|
| 0 | `image_search(原图)` | 0.55 | 0.25 | — | 0.40 | **保留**（gap=0.30；中低分：这个状态先图搜不是好选择） |
| 1 | `text_search("Canton Ohio museum")`，检索回城市信息、无开放日期 | 0.45 | 0.0 | — | 0.23 | **保留**（gap=0.45；低分负样本） |
| 2 | `answer("in 1962")` | null | 0.0 | 0.0 | **0.0** | **保留**（最干净的负样本：答错 → 乘法门控归零） |

这条轨迹一个样本都没被丢——它们全是 PRM 需要的**打低分教材**。如果失败轨迹
不进数据集，PRM 只见过好动作，推理时给什么都打高分，MCTS 就退化成随机搜索。

---

## 第 3 部分：Step 3 —— rationale 与五条 QC

强 LLM（DeepSeek）看每个样本的 (state, action, score)，写一段解释 score 的
rationale，末尾带 `VERDICT: good|mixed|poor`。五条 QC（不过则重生成，至多 3 次）：

| 规则 | 拦什么 |
|---|---|
| (a) 引用 ≥1 个可见 evidence_id | 空谈不引证 |
| (b) 提到动作类型 | 答非所问 |
| (c) 30–150 token | 太敷衍 / 太啰嗦 |
| (d) **GT 泄漏** | 评测措辞（"ground truth"/"标准答案"）；或 gold answer 出现在正文里、却不在可见证据/问题/动作输入中 |
| (e) **VERDICT 与 score 方向一致** | score≥0.6 配 poor、score≤0.4 配 good 的硬矛盾 |

用例 B 的 answer 步（score=0.3）示范：

**会通过的 rationale**（判断从证据出发，方向与 0.3 一致）：

> This answer step commits to "Up to 145 grams", but neither [e0] nor [e1]
> contains any weight figure — the retrieved passages only compare dove
> species qualitatively. Answering without numeric support from the evidence
> bundle is premature here.
> VERDICT: poor

**会被拦下重生成的 rationale**：

> ...the true weight is around **153 grams**, so this answer step is close...
> VERDICT: mixed

拦截原因：153 是 gold answer 派生信息，**不在任何可见证据里**——rationale
"凭空知道"了答案（规则 d）。训练时喂这种样本，PRM 会学到"以知道正确答案的
姿态做判断"，而推理时它根本没有这个信息。

注意合法情形的区分：例 A 的 rationale 写 "[e0] states the Hall opened on
September 7, 1963" **完全合法**——答案字符串来自被引用的证据。"从证据引用"
和"凭空知道"由 QC 自动区分（gold 字符串是否在可见文本中）。

---

## 第 4 部分：Step 4 —— 过滤、划分、体检

1. 按第 0 部分的三条规则丢弃；
2. 按 **query** 划 train/val（默认 5% val，同 query 的样本绝不跨集——防泄漏）；
3. 产出 `stats.json`：各规则丢弃计数、动作类型分布、score 均值等。

**丢弃分布是第一道体检**：`rationale_qc` 丢弃率异常高 → 换更强的生成模型；
`grounding_outcome_gap` 丢弃率异常高 → verifier 或 outcome 判定有问题，先诊断再放量。

一个值得知道的细节：gap 过滤在 **n_through=1** 的独占路径上最激进（outcome 是
0/1 极值，gap 容易超 0.7，如例 B 的两个检索步）；在共享前缀上（credit 是分数均值）
则温和得多（如例 C 的 step 0，outcome=0.25）。这是设计使然——独占路径上
grounding 与结果极值打架，恰恰是最没把握的标签。

---

## 第 5 部分：DPO 对是从数据集里"配"出来的，不是数据集之外的东西

Stage 4.2 训练时，挖对脚本按 `(query_id, 动作前缀)` 分组——也就是树上的**同一个
节点**——把同状态下的兄弟动作配成偏好对：

```
状态：例 B 的 step 2 前（证据里没有任何克数）
├── answer("Up to 145 grams")     score=0.3, unsupported_correct=true  ← rejected
└── text_search("mourning dove weight grams")（兄弟分支，检索到了体重段落，score=0.8）← chosen
```

```
状态：例 A / 例 C 共同的根节点
├── text_search("opening date PFHOF")   credit=1.0   ← chosen
└── image_search(原图)                   credit=0.25  ← rejected
```

两个动作**都是数据集里的正常样本**，各自带着自己的 score。"chosen/rejected"
只是它们在这一对里的角色。DPO 用这种对教 PRM 排序：
第一对教"**证据不足时，检索 > answer**"（何时不该停），
第二对教"**这个状态先文搜 > 先图搜**"（动作类型偏好）。
这正是 MCTS expansion 每次面对的比较题——监督信号和使用场景同构。

所以回到你的问题：**rejected 的轨迹/样本不会不进数据集。恰恰相反——它们是
数据集里最有教学价值的部分，被专门标记出来以免在配对时找不到。** 真正消失的
只有第 0 部分那三条规则丢掉的东西：太短/太长的轨迹、标签自相矛盾的检索步、
rationale 写不合格的样本。

---

## 附：样本去向速查

```
一条轨迹的一个 step
│
├─ 轨迹步数 <2 或 >8 ────────────────→ ✗ 整条丢弃
├─ 非 answer 步且 |local−outcome|>0.7 ─→ ✗ 该样本丢弃
├─ rationale 3 次仍不过 QC ───────────→ ✗ 该样本丢弃
│
└─ 其余全部 ─→ ✓ 进入 train/val
     ├─ 高分样本：教 PRM "什么是好动作"
     ├─ 低分样本（含失败轨迹全部步骤）：教 PRM "什么是坏动作"
     └─ unsupported_correct 标记样本：score 已压至 ≈0.3，
        Stage 4.2 作为 DPO 对的 rejected 方复用
```
