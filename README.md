# EvidenceTree

多模态 RAG 系统：在**检索动作空间**上做 MCTS 搜索（标准 UCB1），由一个 **grounded PRM**
给每个检索动作打分引导搜索——PRM 的 Q 是唯一质量信号。

> 设计与实现规范见 [`EvidenceTree_项目实现报告.md`](EvidenceTree_项目实现报告.md)（当前 v1.5）。
> **核心纪律：每个阶段必须独立跑通并验证后，才进入下一阶段。**

## 数据流

- **训练线**：benchmark 自带语料 → weak policy 跑 rollout → step 打标 → 训练 PRM
- **推理线**：用户 query → MCTS（PRM 引导的 UCB1）→ 输出最优答案

## 快速开始

```bash
# 1. 激活 conda 虚拟环境 RAG（位于 /opt/anaconda3/envs/RAG，Python 3.11）
conda activate RAG
# 若 shell 里 conda activate 不可用，直接用环境内解释器也行：
#   /opt/anaconda3/envs/RAG/bin/python ...

# 2. 安装依赖（本地装到 models 档；server 档仅 GPU 服务器需要）
pip install -r requirements/models.txt
pip install -e .

# 3. 跑通第一步 Stage 0.1（mock 模式，无需 GPU / 无需下载模型）
python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml --mock

# 4. 跑 MCTS 搜索框架（mock 数据 + 启发式 scorer，离线）
python scripts/run_inference.py --config configs/mcts.yaml --mock

# 5. 跑测试
pytest

# 6. 数据集构建（ETBench-Open，4 步 pipeline；mock 无需 API/下载）
python DatasetConstruct/run_pipeline.py --mock
python DatasetConstruct/trace_trajectory.py                 # 在图文场景上确定性追踪一条 MCTS 轨迹
# 轨迹看不清时用 inspector 渲染紧凑摘要：
python DatasetConstruct/inspect_trajectories.py <轨迹.jsonl> --n 5
```

数据集构建（policy VLM + 强 LLM 走 API、检索动作、质量控制、如何放量）详见
[`DatasetConstruct/README.md`](DatasetConstruct/README.md)；
从一条轨迹到一条训练样本的逐步图解见
[`DatasetConstruct/PIPELINE_WALKTHROUGH_CN.md`](DatasetConstruct/PIPELINE_WALKTHROUGH_CN.md)。
真实运行（VisualPRM-8B + 1K InfoSeek）需在 GPU 服务器上，详见
[`pilot/README.md`](pilot/README.md)。

## 论文（AAAI 投稿稿）

[`paper/`](paper/) 下是 AAAI 格式的论文初稿：

- `evidencetree.tex` / `evidencetree.pdf` — 正文（当前预印版式仅模仿 AAAI 双栏外观，正式提交前需换官方 author kit）
- `references.bib` — 参考文献
- `NUMBERS_SOURCES.md` — 表格中每个基线数字的出处与核实等级（✅/☑/⚠）

**论文纪律**：我方结果全部留 "–"（训练未完成，投稿前必须填入真实数字）；
基线数字全部引自原发表论文，逐格可溯源。本地编译：`pdflatex → bibtex → pdflatex ×2`。

## 依赖分档

| 文件 | 用途 | 适用 |
|------|------|------|
| `requirements/base.txt` | 轻量纯 Python（含 BM25、metrics、API 客户端） | 本地，mock 冒烟 |
| `requirements/models.txt` | torch / transformers / faiss 等 | 真实跑模型 |
| `requirements/server.txt` | vLLM / peft / trl / wandb | GPU 服务器训练推理 |

## 目录结构

```
src/evidencetree/
├── actions/      检索动作（按 query 模态切分：text_search / image_search + answer，统一 CLIP 语料）+ 执行器 + 检索器
├── prm/          PRM verifier（grounding + answer-support judge）、数据生成、rationale 生成与 QC、训练（Stage 2-4）
├── mcts/         树搜索（node / UCT / proposer，纯 UCB1，Stage 5）
├── eval/         benchmark 加载与指标
├── generation/   可配置生成后端（mock / HF / API）
└── utils/        config / logging
pilot/            Stage 0 Pilot 脚本（最先做，go/no-go 决策）
DatasetConstruct/ ETBench-Open 数据集构建 pipeline（Stage 3，API policy + 强 LLM）
paper/            AAAI 论文稿（tex / bib / PDF + 基线数字溯源）
scripts/          各阶段入口 CLI
configs/          超参配置
tests/            单元测试
```

## 阶段进度

- [x] **框架搭建** — 目录骨架 + 依赖 + 虚拟环境
- [x] **Stage 0.1** — VisualPRM 失败模式诊断 pipeline（mock 可跑通；real 待服务器）
- [~] Stage 0.2 — MCTS vs Best-of-N gap（pilot 脚本 `pilot/stage0_2_mcts_vs_bon.py` 已实现，mock 跑通；真实 verdict 待 GPU 服务器：frozen VisualPRM + policy 模型 + 真实 InfoSeek）
- [ ] Stage 0.3 — 50 样本人工标注一致性
- [ ] Stage 0.4 — crop/zoom 需求统计（决定动作空间）
- [x] Stage 1 — 检索动作空间与执行器（按 query 模态切分：text_search / image_search + answer，查询同一统一 CLIP 语料；executor + BM25/dense/CLIP 检索 + build_index）
- [~] Stage 2 — Grounding Verifiers — **接口 + lexical/API-judge 后端 + answer-support judge（SUPPORTED/PARTIAL/UNSUPPORTED/NOT_REQUIRED 四档，含离线 lexical fallback）已实现**；正式 cross-encoder/CLIP verifier 待 GPU 服务器（Stage 0.3 一致性验证后替换）
- [~] Stage 3 — 训练数据生成（ETBench-Open）— **构建 pipeline 已实现**（[DatasetConstruct/](DatasetConstruct/README.md)，4 步全流程 + tree-level credit + answer-support 门控 + rationale QC（VERDICT 一致性 / gold-answer 防泄漏），mock 跑通）；真实数据生成待 API key + 原始数据
- [ ] Stage 4 — PRM 三阶段训练
- [~] Stage 5 — MCTS 搜索（PRM-guided UCB1）— **框架已实现**（node/UCT/四阶段循环，mock 验证通过）；真实验收（benchmark 主表）待 PRM
  - ⚠️ Self-Adjusting Bandit 与 Modality-Coverage UCB 已**移除**（与导师讨论后：额外的 λ·novelty 项稀释 PRM 的 Q、徒增消融；搜索回归纯 UCB1，让 PRM 的 Q 成为唯一质量信号）
- [ ] Stage 7 — 评测与消融
- [~] 论文 — **AAAI 格式初稿完成**（[paper/](paper/)：正文 + 附录 A–G + 预注册消融表；基线数字逐格溯源于 `NUMBERS_SOURCES.md`）；我方结果格待 Stage 4/7 完成后填入

> PRM 的**训练代码**（Stage 4）尚未实现；Stage 2/3 的 verifier 与数据生成代码已就位。
> 搜索框架通过可插拔的 `TrajectoryScorer` 接口先用 mock/启发式 scorer 运行，PRM 训好后直接替换。

> ⚠️ Stage 0（Pilot）是 go/no-go 决策阶段。四个 pilot 全过才进入 Stage 1；任何一个亮红灯，暂停与导师讨论。
