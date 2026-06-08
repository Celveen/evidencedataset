# EvidenceTree

多模态 RAG 系统：在**检索动作空间**上做 MCTS 搜索，由一个 **grounded PRM** 给每个检索
动作打分引导搜索，并在推理时用 **Thompson Sampling bandit** 自适应调整探索强度 λ。

> 实现规范见 [`EvidenceTree_实现报告_for_ClaudeCode.md`](EvidenceTree_实现报告_for_ClaudeCode.md)。
> **核心纪律：每个阶段必须独立跑通并验证后，才进入下一阶段。**

## 数据流

- **训练线**：benchmark 自带语料 → weak policy 跑 rollout → step 打标 → 训练 PRM
- **推理线**：用户 query → MCTS（PRM 引导 + bandit 调 λ）→ 输出最优答案

## 快速开始

```bash
# 1. 创建虚拟环境（conda 有权限问题，用 venv）
python3 -m venv RAG
source RAG/bin/activate          # Windows: RAG\Scripts\activate

# 2. 安装依赖（本地冒烟只需 base）
pip install -U pip
pip install -r requirements/base.txt
pip install -e .

# 3. 跑通第一步 Stage 0.1（mock 模式，无需 GPU / 无需下载模型）
python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml --mock

# 4. 跑测试
pytest
```

真实运行（VisualPRM-8B + 1K InfoSeek）需在 GPU 服务器上，详见
[`pilot/README.md`](pilot/README.md)。

## 依赖分档

| 文件 | 用途 | 适用 |
|------|------|------|
| `requirements/base.txt` | 轻量纯 Python（含 BM25、metrics、API 客户端） | 本地，mock 冒烟 |
| `requirements/models.txt` | torch / transformers / faiss 等 | 真实跑模型 |
| `requirements/server.txt` | vLLM / peft / trl / wandb | GPU 服务器训练推理 |

## 目录结构

```
src/evidencetree/
├── actions/      检索动作定义与执行（Stage 1）
├── prm/          PRM 模型、verifier、数据生成、训练（Stage 2-4）
├── mcts/         树搜索、UCT、bandit（Stage 5-6）
├── eval/         benchmark 加载与指标
├── generation/   可配置生成后端（mock / HF / API）
└── utils/        config / logging
pilot/            Stage 0 Pilot 脚本（最先做，go/no-go 决策）
scripts/          各阶段入口 CLI
configs/          超参配置
tests/            单元测试
```

## 阶段进度

- [x] **框架搭建** — 目录骨架 + 依赖 + 虚拟环境
- [x] **Stage 0.1** — VisualPRM 失败模式诊断 pipeline（mock 可跑通；real 待服务器）
- [ ] Stage 0.2 — MCTS vs Best-of-N gap
- [ ] Stage 0.3 — 50 样本人工标注一致性
- [ ] Stage 0.4 — crop/zoom 需求统计（决定动作空间）
- [ ] Stage 1 — 检索动作空间与执行器
- [ ] Stage 2 — Grounding Verifiers
- [ ] Stage 3 — 训练数据生成（ETBench-Open）
- [ ] Stage 4 — PRM 三阶段训练
- [ ] Stage 5 — MCTS 搜索（固定 λ）
- [ ] Stage 6 — Self-Adjusting Bandit
- [ ] Stage 7 — 评测与消融

> ⚠️ Stage 0（Pilot）是 go/no-go 决策阶段。四个 pilot 全过才进入 Stage 1；任何一个亮红灯，暂停与导师讨论。
