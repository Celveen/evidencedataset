# DatasetConstruct — ETBench-Open 数据集构建

把 benchmark 自带语料 + MCTS rollout 变成带 step-level 标注的 PRM 训练数据集
（实现报告 Stage 3）。四步流水线：

```
Step 1  policy VLM(API) 在每个 query 上跑 MCTS  →  trajectories（动作序列）
Step 2  每个 step：verifier 打 local 分 + GT 算 outcome 分（tree-level credit）→ score label
Step 3  强 LLM 看 (state, action, score) → 生成解释 score 的 rationale
Step 4  质量过滤 + train/val 划分 → ETBench-Open
```

算法逻辑在主包里（`src/evidencetree/prm/{verifiers,data_gen,rationale_gen}.py`、
`src/evidencetree/mcts/`），本文件夹只放流水线编排脚本 + 配置 + 文档。

---

## 快速开始

```bash
conda activate RAG

# 本地冒烟（合成数据，零 API 调用，~10 秒）：验证四步链路
python DatasetConstruct/run_pipeline.py --mock

# 真实运行（需先配 API key 和原始数据，见下文）
python DatasetConstruct/run_pipeline.py --n 100        # 先 100 个 query 试跑！
python DatasetConstruct/run_pipeline.py --steps 3,4    # 只跑指定步骤
python DatasetConstruct/run_pipeline.py --force        # 忽略已有输出重做
```

每一步也可单独运行：`python DatasetConstruct/step1_gen_trajectories.py --mock` 等。

---

## 检索动作空间（按 query 模态切分）

检索动作只按 **query（检索输入）的模态**区分，两个搜索动作查询的是**同一个统一
CLIP 语料索引**，结果既可能是文本侧、也可能是图像侧文档：

| 动作 | query 模态 | 检索器 | grounding 打分 |
|---|---|---|---|
| `text_search` | 文本 | BM25/Dense（或统一 CLIP 文本塔） | 按返回结果模态打分（见下） |
| `image_search` | 图像 | 统一 CLIP（图 query 检索混合语料） | 按返回结果模态打分（见下） |
| `answer` | — | — | 不打 local（由 outcome 决定） |

启用图检索：`config.yaml` 的 `retriever.image_search: true`（真实模式生效）。
**现实约束**：`image_search` 需要**语料里有图像**（当前 Wikipedia 语料是纯文本，
image 子索引为空 → 该动作执行但返回 []），且需要 **query 图（OVEN）**。补齐图像数据
前，图检索执行但无结果。`crop` / `zoom` / `focus` 暂不做（待 Stage 0.4 统计决定）。

> 早期版本曾把检索动作按 (query 模态 × 结果模态) 拆成 2×2 四个
> （`text_search` / `text_to_image` / `image_to_text` / `image_search`），后收口为按
> query 模态切分的两个搜索动作 —— 结果模态由检索器在统一语料里按相关度自然决定，不再
> 硬编码进动作类型。详见根目录 `EvidenceTree_项目实现报告.md` §3.1。

## 看清轨迹：inspector 与 trace

轨迹 JSONL 太长难判断各部分是否正常时，用 inspector 渲染紧凑摘要（动作类型、
检索命中数+片段、三个分数、rationale+QC）：

```bash
python DatasetConstruct/inspect_trajectories.py data/trajectories/infoseek_rationales.jsonl --n 5
python DatasetConstruct/inspect_trajectories.py <file> --action image_search    # 只看某动作
python DatasetConstruct/inspect_trajectories.py <file> --query <query_id> --full # 某 query 全文
```

确定性追踪一条完整 MCTS 轨迹（selection→expansion→simulation→backup，图文场景，无需下载）：

```bash
python DatasetConstruct/trace_trajectory.py
```

---

## API key 填在哪

```bash
cp DatasetConstruct/.env.example DatasetConstruct/.env   # 然后编辑 .env
```

`.env` 已被 .gitignore 忽略，**不会被提交**。已 export 的环境变量优先于 .env。

| 环境变量 | 用在哪一步 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | Step 1 policy VLM | 默认配置走 DashScope 的 OpenAI 兼容端点跑 Qwen2.5-VL-7B（报告指定的 weak policy） |
| `DEEPSEEK_API_KEY` | Step 3 rationale；verifier=api 时的 judge | 强 LLM 默认 **deepseek-v4-pro**（OpenAI 兼容端点 `https://api.deepseek.com/v1`） |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | 可选 | 若把 provider 换回 Claude / GPT-4o |

换模型/端点改 [config.yaml](config.yaml) 的 `policy` / `rationale.generation` 块：
`provider`（anthropic 或 openai——openai 覆盖一切 OpenAI 兼容端点）、`model`、
`base_url`、`api_key_env`（指定读哪个环境变量）。多模态 query 的图像会自动
base64 上传（anthropic image block / openai image_url）。

---

## 原始数据放在哪

放到 `data/corpus/infoseek/`（与主项目共用，路径在 config.yaml `data.data_dir`；
整个 `data/` 都被 .gitignore 忽略，大文件不会误提交）：

```
data/corpus/infoseek/
├── raw/                          # InfoSeek 官方原始标注（已下载，2026-06-12）
│   ├── infoseek_val.jsonl        # 73,620 条，带 answer/answer_eval —— 实际评测用它
│   ├── infoseek_test.jsonl       # 347,980 条，官方未公开答案（仅 leaderboard 用）
│   ├── infoseek_val_withkb.jsonl # query -> Wikidata entity 映射（val 共 1,794 个实体）
│   └── infoseek_human.jsonl      # 人工 set，无公开答案
├── infoseek_queries.jsonl        # ← prepare_infoseek.py 由 raw/ 转换生成（已生成）
└── infoseek_corpus.jsonl         # 检索语料（待构建，见下）
```

官方下载源（GCS，`raw/` 里的文件即来自这里）：
`http://storage.googleapis.com/gresearch/open-vision-language/infoseek/<文件名>`

格式转换（raw → pipeline 格式，可重复跑）：

```bash
python DatasetConstruct/prepare_infoseek.py
# 若已下载 OVEN 图像：--images-dir /path/to/oven_images（按 <image_id>.jpg 查找）
```

**检索语料构建**（一次性离线，已提供脚本）：

```bash
python DatasetConstruct/build_corpus.py        # Wikidata 实体 -> Wikipedia 全文 -> 切块
```

val 涉及 1,794 个唯一实体；脚本批量解析 enwiki 标题、抓取页面 plaintext
（限速友好 + 断点续跑，缓存在 `raw/wiki_pages.jsonl`），切成 ~1200 字符的
段落 chunk 写入 `infoseek_corpus.jsonl`。语料一次抓取后固定，可随数据集发布；
运行时检索只打本地索引，不违反离线约束。

**仍待办——图像**：InfoSeek 的图来自 OVEN（`image_id` 如 `oven_04990048`），
需按 [OVEN 仓库](https://github.com/open-vision-language/oven) 的
image_downloads 说明下载（量大，建议直接下到 GPU 服务器）。没有图像时
`image_path=null`，文本-only 链路可先跑（Step 1 的 VLM 调用自动退化为纯文本）。

检索全程在本地 BM25/FAISS 索引上进行，**绝不调用 web 检索 API**——这是项目
可复现性的硬约束（报告 §4.1.4）。

---

## 各步骤的产物：放哪、什么格式

mock 模式的产物自动加 `mock_` 前缀（Step 4 则写进 `mock/` 子目录），不会污染真实数据。
所有产物均被 .gitignore 忽略（大文件不进 git）。

### Step 1 → `data/trajectories/infoseek_trajectories.jsonl`

每行一条 trajectory：

```json
{
  "traj_id": "q_123#t0",
  "query_id": "q_123",
  "question": "In which city is the Eiffel Tower located?",
  "image_path": null,
  "gold_answers": ["Paris"],
  "rollout_t": 0,
  "lambda": 0.3,
  "gen_reward": 0.85,
  "final_answer": "Paris",
  "outcome_em": 1.0,
  "steps": [
    {"step_index": 0, "action_type": "text_search",
     "action_input": "Eiffel Tower city",
     "evidence": [{"evidence_id": "e0", "doc_id": "d7", "title": "Eiffel Tower",
                    "text": "...", "score": 6.1}]},
    {"step_index": 1, "action_type": "answer", "action_input": "Paris", "evidence": []}
  ]
}
```

- 每个 query 收集**所有** rollout 的轨迹（≤ P 条，默认 P=10；同 query 内动作
  序列完全相同的去重，开关 `quality.dedupe_within_query`）。
- `gen_reward` 是生成期的引导分（启发式 scorer），**不是训练标签**。
- **断点续跑**：重跑时已有 `query_id` 自动跳过。
- `image_search` 步会额外带 `"region"`（归一化 bbox 或 null）与 `"image_path"`
  （实际作为检索 query 的图），供 Step 2 的 CLIP verifier 打分；其
  `action_input` 用 `describe()` 形式（含 region），保证不同聚焦算不同节点。

### Step 2 → `data/trajectories/infoseek_scored.jsonl`

每行一个 step-level 训练样本：

```json
{
  "sample_id": "q_123#t0#s0",
  "traj_id": "q_123#t0", "query_id": "q_123",
  "question": "...", "image_path": null,
  "state": {"actions_before": [], "evidence_before": []},
  "action": {"type": "text_search", "input": "Eiffel Tower city"},
  "observation_evidence": [{"evidence_id": "e0", "title": "...", "text": "..."}],
  "local_grounding": 0.92,
  "outcome_credit": 0.6,
  "n_traj_through": 5,
  "alpha": 0.5,
  "score": 0.76
}
```

- `local_grounding`：verifier 的 graded 分（0–1 连续）= **该步检索回的结果与
  question 的相关性**（不是动作输入与 question）；空结果为 0；**answer 步为 null**
  （不参与 local，由 outcome 决定）。
- `outcome_credit`：**tree-level credit** —— 同 query 内经过相同动作前缀的
  所有轨迹的成功率（`n_traj_through` 条的 Monte Carlo 均值）。**不是**把整条
  轨迹的 final reward 均摊给每个 step（那是报告 §4.1.1 点名的经典 bug）。
- `answer_support`（仅 answer 步）：答案是否被**已积累证据**支撑（`verifier.support`
  配置：api judge / lexical / off）。堵"答对但证据不支撑"（参数化蒙对）被打满分的
  口子——否则 PRM 会学到"无证据也可以直接 answer"。**null 的两种成因**（原始判定存
  `answer_support_label`）：判不了（unverifiable/unparseable），或 judge 判
  **NOT_REQUIRED**——题面+图像即可推导、无需外部知识（如 ScienceQA 推理选择题），
  对这类题做蕴含门控会系统性错罚正确答案（v1.5.1）。null 一律退回 outcome-only。
- `unsupported_correct`：answer 步且 `outcome>=0.5` 且 `support<=阈值` → true。
  **不丢弃**——这些是 Stage 4.2"同状态 DPO 对"的现成负样本。
- `score`：非 answer 步 `= alpha*local + (1-alpha)*outcome`（α 在 `verifier.alpha`）；
  answer 步 `= outcome*(floor+(1-floor)*support)`，保序
  答错(0) < 答对无支撑(≈floor=0.3) < 答对有支撑(≈1)；support=null 退化为 outcome。

### Step 3 → `data/trajectories/infoseek_rationales.jsonl`

在 Step 2 样本上追加：

```json
{
  "rationale": "This text_search step ... the retrieved passage [e0] ...",
  "rationale_backend": "api",
  "rationale_attempts": 1,
  "rationale_qc_pass": true,
  "rationale_qc_reasons": []
}
```

断点续跑：已有 `sample_id` 跳过；每 50 条落盘一次，中断损失小。

### Step 4 → `data/etbench_open/`

```
data/etbench_open/
├── train.jsonl    # 最终训练集（Step 3 的全字段）
├── val.jsonl      # 验证集（按 query 划分，默认 5%，同 query 不跨集防泄漏）
└── stats.json     # 输入/保留/各规则丢弃数、动作类型分布、score 均值
```

---

## 质量控制（在哪改、怎么控）

阈值全部在 [config.yaml](config.yaml) 的 `quality` / `rationale` 块：

| 规则 | 默认 | 配置项 | 出处 |
|---|---|---|---|
| 轨迹过短/过长整条丢弃 | 2 ≤ 步数 ≤ 8 | `quality.min_steps` / `max_steps` | 报告 §3.5 |
| grounding↔outcome 严重不一致的样本丢弃（noisy label） | \|local−outcome\| > 0.7 | `quality.max_grounding_outcome_gap` | 报告 §3.5 |
| rationale 必须：引用 ≥1 个样本内可见的 evidence_id；明确提到动作类型；长度 30–150 token | — | 代码 `prm/rationale_gen.py::check_rationale` | 报告 §3.5 |
| rationale **GT 泄漏检查**：不得含 "ground truth"/"标准答案" 等评测措辞；不得含可见证据/问题/动作输入之外的 gold answer 字符串（从证据引用答案合法，凭空知道即泄漏） | — | 同上 | DBAgent judge prompt 的 trajectory-realism 规则 |
| rationale 须以 `VERDICT: good\|mixed\|poor` 结尾且**与 score 方向无硬矛盾**（score≥0.6 不得 poor、≤0.4 不得 good）；VERDICT 行剥离存入 `rationale_verdict` | — | 同上 | 防 (rationale, score) 自相矛盾的训练对 |
| QC 不过自动重生成 | 至多 3 次 | `rationale.max_attempts` | 报告预估 ~15% 重生成 |

人工抽检（报告要求，pipeline 不替代）：跑完后抽 500 条查 score 合理性、
每类动作各抽 200 条查 rationale 质量。`stats.json` 里的丢弃分布是第一道体检：
`rationale_qc` 丢弃率异常高 → 换更强的生成模型；`grounding_outcome_gap`
丢弃率异常高 → verifier 或 outcome 评估有问题，先诊断再放量。

---

## 成本与放量建议

API 调用量级（真实模式）：

- **Step 1** ≈ n_queries × P × (1–2) 次 policy 调用（提议动作 + 起草答案）
- **Step 2** ≈ 0（统一 CLIP verifier 本地推理，无 API 成本）
- **Step 3** ≈ 保留样本数 × ~1.2 次强 LLM 调用（含重生成）

报告预算 rationale 约 $800（50–80K 样本）。**强烈建议放量路径**：
`--mock` 验证链路 → `--n 100` 真实试跑（检查 stats.json 与抽样质量）→
`--n 1000` → 全量。SDK 自带限速重试；中断直接重跑同一命令即可续上。

注意：报告原设计 weak policy 是**本地 vLLM 跑 Qwen2.5-VL-7B**（零 API 成本）。
当前按需求用 API 形式实现；全量 60K query × 10 rollout 的 Step 1 API 费用
不可忽视，放量前先用 100/1000 试跑估算单价，或届时切回服务器本地 policy
（改 `policy.backend: hf` 即可，接口不变）。

---

## verifier 现状（诚实声明）

**grounding 衡量「检索回的结果 vs 问题」的相关性**，不是「动作输入 vs 问题」。
原因：防臆造已由 policy prompt 解决，且不做 focus → 图像动作的 query 图固定、
「输入 vs 问题」对它们无区分度；改看结果才能反映「这步检索有没有用」。空结果 → 0。

**统一 CLIP 空间打分（`backend: clip`，正式版）**：文本结果与图像结果都用
**同一个 CLIP 模型**、对齐到**同一个 `CLIP_text(question)` 锚点**——
- 文本侧结果：CLIP 文本塔，`cos(CLIP_text(question), CLIP_text(结果))`
- 图像侧结果：CLIP 图像塔，`cos(CLIP_text(question), CLIP_image(结果))`，取所有结果 max

（`text_search` 与 `image_search` 查同一语料，一步的结果可能混有两种模态，各自打分后取 max。）

**为什么必须统一模型**：若文本用 sentence-transformer、图像用 CLIP，两者余弦尺度不同，
PRM 会把"尺度差"误当成"动作质量差"，从而系统性偏向某类动作 → 污染训练。统一 CLIP +
同一 question 锚点消除这个偏差。唯一细节：同模态（文本-文本）余弦系统性高于跨模态
（文本-图像），所以两类各配一个 `cos band` 把余弦校准到同一个"相关性 0–1"语义：
`text_cos_lo/hi`（默认 0.5/0.9）、`image_cos_lo/hi`（默认 0.15/0.32）。**band 是经验值，
服务器上务必按真实余弦分布重新校准。**

`backend: lexical`（mock/离线）：文本结果用 question 内容词召回率、图像结果中性 0.5，
不加载任何模型（冒烟测试用）。

> **依赖**：`clip` 需要 `sentence-transformers` + `pillow`（已在 `requirements/models.txt`）。
> 服务器可把 `clip_model` 换成 OpenCLIP ViT-L，或注入与 `ClipImageRetriever` 共享的
> CLIP 实例避免重复加载。
> **仍待办**：band 真实分布校准；Stage 0.3（50 样本人工一致性）通过后再全量放量 Step 2。
