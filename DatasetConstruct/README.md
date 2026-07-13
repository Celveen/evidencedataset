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

### 本地 Qwen2.5-VL-7B

模型默认部署在 `models/Qwen2.5-VL-7B-Instruct/`，独立环境位于
`.venv-qwen25vl/`。启动 OpenAI 兼容服务：

```bash
DatasetConstruct/start_qwen25vl.sh
```

服务地址为 `http://127.0.0.1:8000/v1`。单次图片问答：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/qwen25vl_infer.py \
  --image /path/to/image.jpg --prompt "Describe this image."
```

服务已注册为用户级 systemd 服务：

```bash
systemctl --user status qwen25vl
systemctl --user restart qwen25vl
systemctl --user stop qwen25vl
journalctl --user -u qwen25vl -f
```

运行数据构造时使用完整的本地配置：

```bash
python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml --n 100
```

该配置默认开启 `image_search`：使用 CPU 上的
`sentence-transformers/clip-ViT-B-32` 对 `corpus.jsonl` 中带
`image_path` 的文档建图像索引，每次返回 3 个结果。检索出的图片会连同原始
query 图片一起交给 Qwen2.5-VL；首次运行会从 Hugging Face 下载约 578 MB
的 CLIP 模型。

`ocr(image, region?)` 已作为可选 MCTS 动作保留，但默认关闭。它不是 Top-K
检索器，而是对当前 query/evidence 图像直接读取文字，只返回一条 OCR evidence。
启用方式：

```bash
--set retriever.ocr.enabled=true
```

OCR action 依赖系统安装的 `tesseract` CLI。由于 OCR 没有可比较的 Top-K
候选结果，Step 2 不给它 local grounding，`score=outcome_credit`；同时样本
会带 `score_source=outcome_only` 和较低的 `label_weight`，训练时可用这个权重
避免它压过有 local 分的动作。

---

## API key 填在哪

```bash
cp DatasetConstruct/.env.example DatasetConstruct/.env   # 然后编辑 .env
```

`.env` 已被 .gitignore 忽略，**不会被提交**。已 export 的环境变量优先于 .env。

| 环境变量 | 用在哪一步 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | Step 1 policy VLM | 默认配置走 DashScope 的 OpenAI 兼容端点跑 Qwen2.5-VL-7B（报告指定的 weak policy） |
| `LOCAL_QWEN_API_KEY` | 本地 Qwen policy VLM | 本地服务不校验 key，保留非空占位值 `local` 即可 |
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

### 其他已下载数据集

ScienceQA 和 MRAG-Bench 可先转换少量样本：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/prepare_other_datasets.py \
  --dataset all --n 5
```

转换结果位于 `data/corpus/<dataset>/`，统一使用：

```text
queries.jsonl  # query_id, question, gold_answers, image_path, metadata
corpus.jsonl   # doc_id, title, text, optional image_path
assets/        # 抽取的查询图/页面图；MRAG corpus 直接引用外盘完整图像库
```

运行其中一个数据集：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml --n 1 --force \
  --set benchmark=scienceqa \
  --set data.data_dir=data/corpus/scienceqa \
  --set output.trajectories=data/trajectories/scienceqa_trajectories.jsonl \
  --set output.scored=data/trajectories/scienceqa_scored.jsonl \
  --set output.rationales=data/trajectories/scienceqa_rationales.jsonl \
  --set output.dataset_dir=data/etbench_open/scienceqa
```

当前适配能力：

| 数据集 | 当前适配 |
|---|---|
| ScienceQA | 查询图只交给 Qwen；lecture/hint 作为文本语料，不把查询图复制进检索库 |
| MRAG-Bench | 18,984 张无标签候选图进入 CLIP/FAISS；15 张查询图及其内容重复副本已排除 |

MRAG 的首次运行会建立
`data/corpus/mrag_bench/clip_index`，后续直接复用 43 MB 的 FAISS 缓存。

2026-06-17 起，SlideVQA 和 M3DocVQA 的适配、OCR 动作和专用测试入口已从
当前管道移除。之后 OCR 作为通用动作重新加入，但不恢复这两个数据集的专用
适配逻辑。历史生成数据可能仍在 `data/` 中，但默认脚本不再引用它们。

小样本结果位于 `data/pipeline_<N>q/<dataset>/`。问题级命中表示同一问题至少
有一条 rollout 精确匹配正确答案；这些小样本只用于验证适配和管道，不代表完整
benchmark 准确率。

作者版 CLIP image grounding 在本轮的实际分布：

- MRAG-Bench：19 个 `image_search`，均值 `0.8002`，范围 `0.3663–1.0`。

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
- `image_search` 步额外保存 `region` 和实际查询 `image_path`。Step 2 使用
  query 图像（或 region）与 question 做 CLIP 图文对齐；不同 region 的
  `action_input` 也不同，因此会被视为不同的 tree-credit 节点。
- `ocr` 步保存实际读取的图像/region，并把识别出的文本作为一条 evidence。
  OCR 不返回 Top-K，因此没有 local grounding，只使用 tree-level outcome credit。

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

- `local_grounding`：verifier 的 graded 分（0–1 连续）；**answer 步为 null**
  （不参与 local，由 outcome 决定）。
  - `text_search`：搜索 query 与原问题的内容词对齐度。
  - `image_search`：query 图像/region 与问题文本的 CLIP cosine，经
    `[verifier.cos_lo, verifier.cos_hi]` 线性映射到 `[0,1]`。
    这不是检索结果相似度的平均值；缺少图片或 CLIP 失败时回退到 `0.5`。
  - `ocr`：为 null。OCR 当前只产生单条识别结果，没有 Top-K 候选可比较。
- `outcome_credit`：**tree-level credit** —— 同 query 内经过相同动作前缀的
  所有轨迹的成功率（`n_traj_through` 条的 Monte Carlo 均值）。**不是**把整条
  轨迹的 final reward 均摊给每个 step（那是报告 §4.1.1 点名的经典 bug）。
- `score = alpha*local + (1-alpha)*outcome`，α 在 `verifier.alpha` 配置。

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
| rationale 必须：引用 ≥1 个样本内可见的 evidence_id；明确提到动作类型；长度 50–150 whitespace word | — | 代码 `prm/rationale_gen.py::check_rationale` | 报告 §3.5 |
| QC 不过自动重生成 | 至多 3 次 | `rationale.max_attempts` | 报告预估 ~15% 重生成 |

人工抽检（报告要求，pipeline 不替代）：跑完后抽 500 条查 score 合理性、
每类动作各抽 200 条查 rationale 质量。`stats.json` 里的丢弃分布是第一道体检：
`rationale_qc` 丢弃率异常高 → 换更强的生成模型；`grounding_outcome_gap`
丢弃率异常高 → verifier 或 outcome 评估有问题，先诊断再放量。

---

## 成本与放量建议

API 调用量级（真实模式）：

- **Step 1** ≈ n_queries × P × (1–2) 次 policy 调用（提议动作 + 起草答案）
- **Step 2** ≈ 0（lexical verifier 免费；改 `verifier.backend: api` 才走 judge）
- **Step 3** ≈ 保留样本数 × ~1.2 次强 LLM 调用（含重生成）

报告预算 rationale 约 $800（50–80K 样本）。**强烈建议放量路径**：
`--mock` 验证链路 → `--n 100` 真实试跑（检查 stats.json 与抽样质量）→
`--n 1000` → 全量。SDK 自带限速重试；中断直接重跑同一命令即可续上。

当前 `config.qwen25vl.yaml` 通过本机 OpenAI 兼容服务调用
Qwen2.5-VL-7B，Step 1 不产生外部 API 费用；DeepSeek rationale 仍会产生费用。

---

## verifier 现状

verifier 按动作类型分两路：

- `text_search` 使用 `verifier.backend`：`lexical` 为离线内容词对齐，
  `api` 为 LLM judge。
- `ocr` 不使用 local verifier，局部分为 null，最终分数只看 outcome credit。
- `image_search` 使用 `verifier.image_backend`：
  - `clip`：作者实现的 CLIP 图文对齐评分。计算 query 图像/region 与 question
    的 cosine，再通过 `cos_lo`、`cos_hi` 校准到 `[0,1]`。
  - `neutral`：固定 `0.5`，用于 mock 或不希望加载 CLIP 的场景。

默认模型为 `clip-ViT-B-32`。本地 Qwen 配置让 Step 2 CLIP 在 CPU 上运行，
避免与 Qwen2.5-VL-7B 争用显存。
