# ETBench-Open — 数据集规范（schema spec + dataset card 模板）

> 本文件是 ETBench-Open 的**发布标准**。所有由 `run_pipeline.py` 产出的数据都应通过
> `validate_dataset.py` 校验；对外发布前补齐下方 **Dataset Card** 的每一节。
> 目标：让数据集达到顶会公开数据集的可复现 / 可审计 / 可再分发标准。

---

## 1. 记录 schema（字段字典）

数据集是 **step-level** 的：一条记录 = 一条轨迹中的一个检索/回答步骤，是 PRM 的一个训练样本。
JSONL，每行一个 UTF-8 JSON 对象。

### 1.1 训练样本记录（`train.jsonl` / `val.jsonl` / `test.jsonl`）

| 字段 | 类型 | 可空 | 取值/约束 | 含义 |
|------|------|------|-----------|------|
| `sample_id` | str | 否 | 全局唯一，`<traj_id>#s<step_index>` | 样本主键 |
| `traj_id` | str | 否 | `<query_id>#t<k>` | 所属轨迹 |
| `query_id` | str | 否 | benchmark 内唯一 | 所属 query；**划分 train/val/test 的单位** |
| `question` | str | 否 | 非空 | 原始问题 |
| `image_path` | str \| null | 是 | **相对 key 或 image_id**（禁止本机绝对路径） | query 图，纯文本 query 为 null |
| `state.actions_before` | list[str] | 否 | 可空列表 | 本步之前已执行的动作（`type(input)` 串） |
| `state.evidence_before` | list[obj] | 否 | 见 §1.3 | 本步之前累积的证据快照 |
| `action.type` | str | 否 | `text_search` \| `image_search` \| `answer` | 动作类型 |
| `action.input` | str | 否 | 非空 | text_search=query；image_search=区域描述；answer=答案文本 |
| `observation_evidence` | list[obj] | 否 | 见 §1.3，answer 步为空 | 本步检索回的证据 |
| `local_grounding` | float \| null | 是 | `[0,1]`；**当且仅当** `action.type=="answer"` 时为 null | 检索结果 vs 问题的 grounded 分（统一 CLIP 空间，按结果模态打分取 max） |
| `outcome_credit` | float | 否 | `[0,1]` | tree-level credit：经过该节点的所有轨迹的 outcome 成功率（MC 估计） |
| `n_traj_through` | int | 否 | `>=1` | 经过该节点的轨迹数（credit 的样本量） |
| `alpha` | float | 否 | `[0,1]` | 双源融合权重 |
| `answer_support` | float \| null | 是 | `[0,1]`；**仅 answer 步**可为数值，非 answer 步恒 null | 答案是否被已积累证据支撑（lexical/API judge；判不了或 NOT_REQUIRED = null） |
| `answer_support_label` | str \| null | 是 | `supported`/`partial`/`unsupported`/`not_required`/`no_evidence`/`lexical`/`unverifiable`/`unparseable` | support 判定的原始类别（诊断 null 的成因：无需外部证据 vs 判不了） |
| `support_floor` | float | 否 | `[0,1]`，默认 0.3 | answer 步融合下限（见 score 公式） |
| `unsupported_correct` | bool | 否 | — | answer 步且 `outcome>=0.5` 且 `support<=阈值` → true（参数化蒙对；**不丢弃**，留作 DPO 负样本） |
| `gold_answers` | list | 否 | 同 §1.2 | 标准答案（QC 泄漏检查与后续分析用；PRM 输入不含此字段） |
| `score` | float | 否 | `[0,1]` | 非 answer 步：`alpha*local + (1-alpha)*outcome`；answer 步：`outcome*(floor+(1-floor)*support)`，support=null 时退化为 outcome | 训练目标分 |
| `rationale` | str | 否 | 非空，30–150 token | 强 LLM 解释该 score 的理由（VERDICT 行已剥离） |
| `rationale_verdict` | str \| null | 是 | `good` \| `mixed` \| `poor` | rationale 的结论方向（与 score 方向一致性 QC 用） |
| `rationale_backend` | str | 否 | `api` \| `mock` | rationale 来源 |
| `rationale_attempts` | int | 否 | `>=1` | QC 重生成次数 |
| `rationale_qc_pass` | bool | 否 | 发布集应**恒为 true** | 是否通过 QC |
| `rationale_qc_reasons` | list[str] | 否 | pass 时为空 | 未过原因 |
| `gen_provenance`* | obj | 否 | 见 §1.4 | **发布需补**：生成出处（policy/rationale 模型版本、种子） |

\* `gen_provenance` 当前 pipeline 未写入，发布前需补（见 §3）。

### 1.2 轨迹记录（中间产物 `*_trajectories.jsonl`，可选随发布以供复现）

| 字段 | 类型 | 约束 | 含义 |
|------|------|------|------|
| `traj_id` / `query_id` / `question` / `image_path` | — | 同上 | — |
| `gold_answers` | list | 非空 | 标准答案（string-type=字符串列表；value-type=`{wikidata, range:[lo,hi]}`，见 §4） |
| `rollout_t` | int | `>=0` | 第几次 rollout |
| `gen_reward` | float | — | 生成期引导分（**非标签**，仅记录） |
| `final_answer` | str | — | 该轨迹答案 |
| `outcome_em` | float | `{0,1}` 或 `[0,1]` | 答案正确性（见 §4 关于 metric 的说明） |
| `steps` | list[obj] | 2–8 条 | 每步 `{step_index, action_type, action_input, evidence[], (region, image_path)}` |

### 1.3 Evidence 对象

| 字段 | 类型 | 含义 |
|------|------|------|
| `evidence_id` | str | 轨迹内唯一，`e0,e1,...`（被 rationale 引用） |
| `doc_id` | str | 语料 doc 主键（轨迹 evidence 内含；样本快照可省） |
| `title` | str | 文档标题，可空串 |
| `text` | str | 文档/片段正文 |
| `image_path` | str\|null | 图像侧结果的相对 key（图像 grounding 用） |
| `score` | float | 检索器相似度（轨迹内含；样本快照可省） |

### 1.4 `gen_provenance`（发布必填）

记录**实际使用**的 policy：正式数据在服务器上用**本地部署的 Qwen2.5-VL-7B**生成（开发期
冒烟用 DashScope 的 `qwen3-vl-flash` API，因 qwen2.5-vl 已在 DashScope 下线）。`policy_deployment`
区分 `local` / `api`，避免把开发期 API 模型误记为正式模型。

```json
{"policy_model":"Qwen2.5-VL-7B-Instruct","policy_deployment":"local","policy_snapshot":"2026-06",
 "retriever":"bm25+clip-ViT-B-32","rationale_model":"deepseek-v4-pro","rationale_deployment":"api",
 "mcts":{"rollouts":10,"max_depth":3,"c_uct":1.0},"outcome_metric":"infoseek_relaxed",
 "seed":0,"pipeline_commit":"<git-sha>","dataset_version":"v0.1"}
```

---

## 2. 不变量（validator 强制检查）

1. `sample_id` 全局唯一；`traj_id` 一致；`<traj_id>#s<n>` 与 `step_index` 对齐。
2. 分数域：`local_grounding/outcome_credit/score/alpha/answer_support ∈ [0,1]`；`local_grounding is null ⇔ action.type=="answer"`；`answer_support` 仅 answer 步可为数值。
3. `score == alpha*local + (1-alpha)*outcome`（非 answer 步）；answer 步 `score == outcome*(floor+(1-floor)*support)`（`support=null` 时 `== outcome`），容差 1e-6。
4. 轨迹长度 ∈ `[min_steps, max_steps]`（默认 2–8）——这是**轨迹级前置门**（在 Step 4 对完整轨迹施加）。注意：per-sample 的 grounding-gap 过滤会合法地让某些轨迹在最终文件里只剩 1 个样本；该样本仍是有效的独立 (state, action, label) 节点，因此**最终文件里某轨迹样本数 < min_steps 不算违规**（仅作 info 报告）。
5. `|local - outcome| <= max_grounding_outcome_gap`（默认 0.7）——发布集中不应有越界样本。
6. `rationale_qc_pass == true`，且 rationale 满足：引用 ≥1 个**本样本可见**的 `evidence_id`、提到 `action.type`、30–150 token、**无 GT 泄漏**（不含 "ground truth"/"标准答案" 等评测措辞；不含"可见证据/问题/动作输入之外"的 gold answer 字符串——从证据里引用答案合法，凭空知道答案即泄漏）、`rationale_verdict` 与 score 方向无硬矛盾（score≥0.6 不得 poor，score≤0.4 不得 good）。
7. **划分纯净**：同一 `query_id` 不跨 train/val/test。
8. **可再分发**：`image_path` 不得是本机绝对路径（必须是相对 key / image_id / null）。
9. `n_traj_through >= 1`。
10.（发布级）每条含 `gen_provenance`；随集附 `manifest.json`（每 split 行数 + sha256）。

---

## 3. 发布产物清单（release bundle）

```
etbench_open/
├── train.jsonl  val.jsonl  test.jsonl     # 样本（schema §1.1）
├── corpus.jsonl                            # 语料 doc_id -> {text,title,image_id}（证据可溯源）
├── images/  或  image_urls.jsonl           # 图片：能再分发则打包，否则给 URL+下载脚本（见 §5）
├── stats.json                              # 分布报告（§4 扩展）
├── manifest.json                           # 行数 + 每文件 sha256 + dataset_version
├── SCHEMA.md  DATASET_CARD.md  LICENSE      # 规范 + 卡片 + 许可
└── load_dataset.py                          # HF datasets 加载脚本（Features 对齐 §1.1）
```

---

## 4. 关于 outcome 标签（重要：透明、非黑盒）

InfoSeek 答案分两类，必须**分类型**用其**官方 rule-based 容差**判定，禁止裸 `exact_match`：

- **STRING 类**：`gold_answers` 是可接受字符串列表 → 归一化后任一匹配即 1（现 EM 对此正确）。
- **VALUE（数值）类**：`gold` 是 `{wikidata: v, range: [lo, hi]}` → **解析预测中的数值，落在 `[lo,hi]` 内即 1**。
  当前 `metrics.exact_match` 把答案字符串与该 dict 的 repr 做串比较，**对数值题恒为 0**（已验证：现有真实集 96% outcome≈0 即源于此）。这是确定性、可复现的规则修复，**不是 LLM-judge**，与项目"grounded、不引黑盒"的主张一致。

`stats.json` 发布版需含：train/val/test 行数、动作类型分布、平均步数、`score/local/outcome` 直方图、
QC 通过率、各 drop 原因计数、**outcome 正例率（按答案类型分别报告）**、Stage 0.3 的 50 样本人工一致性。

---

## 5. Dataset Card 模板（DATASET_CARD.md，发布前逐节补全）

```markdown
# ETBench-Open
## Motivation        为什么造、解决什么（retrieval-aware PRM 缺标注数据）
## Composition       样本/轨迹数、字段（指向 SCHEMA.md）、答案类型分布、多模态占比
## Collection        来源 = InfoSeek/OVEN/Wikipedia；policy/rationale 模型+版本+日期；MCTS 配置；种子
## Preprocessing     tree-level credit、grounding 打分、QC 过滤规则、划分策略
## Uses              训练/评估 retrieval-action PRM；**不适用**：当通用 QA 训练集
## Distribution      许可（见下）；图片以 URL/ID 分发；版本号 + changelog
## Maintenance       维护者、联系方式、更新计划、勘误渠道
## Ethics & License  Wikipedia 图片 CC-BY-SA 等源许可传递；派生集许可；PII 声明（百科类，低风险但需说明）
```

**许可红线**：Wikipedia/OVEN 图片通常**不可直接再分发** → 发布 `image_url`/`image_id` + 下载脚本，
不打包图片字节；并在卡片中显式传递源许可。
