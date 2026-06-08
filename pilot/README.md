# Pilot — Stage 0（go/no-go 决策阶段）

> ⚠️ Stage 0 不是热身，是决定项目是否继续的 go/no-go 阶段。四个 pilot 全过才进 Stage 1。

## Stage 0.1 — VisualPRM 失败模式诊断 ✅（已实现）

验证 frozen VisualPRM-8B 直接给 RAG 检索轨迹打分时，评分与真实 outcome 的相关性。

### 本地冒烟（mock，无需 GPU / 无需下载）

```bash
source RAG/bin/activate
python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml --mock
```

- 用合成的 InfoSeek 风格小数据 + mock 生成器 + mock PRM。
- 走完整条 pipeline（数据加载 → BM25 → 生成 → outcome → PRM 打分 → 相关性分析），
  在 `data/pilot_reports/` 产出 `stage0_1_mock_*.json` 与 `.md` 报告。
- 目的：验证**代码链路与分析逻辑**正确，不是验证真实结论。mock PRM 的相关性由
  `configs/pilot.yaml` 的 `prm.mock_correlation` 控制（默认 0.2，模拟"frozen baseline 弱相关"）。

### 真实运行（在 GPU 服务器上）

1. 装模型依赖：
   ```bash
   pip install -r requirements/models.txt
   ```
2. 准备 InfoSeek 数据为 JSONL，放到 `data/corpus/infoseek/`：
   - `infoseek_queries.jsonl`：每行 `{"query_id","question","gold_answers":[...],"image_path"?}`
   - `infoseek_corpus.jsonl`：每行 `{"doc_id","text","title"?,"image_path"?}`
3. 选生成后端（编辑 `configs/pilot.yaml` 的 `generation.backend`）：
   - `hf`：本地 HF 模型（如 `Qwen/Qwen2.5-VL-7B-Instruct`）
   - `api`：Claude/GPT-4o（设 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY`）
4. 去掉 `--mock` 跑真实诊断：
   ```bash
   python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml
   # 也可临时覆盖：--set generation.backend=api --n 1000
   ```

> **注意**：`prm/model.py::_score_real` 中读取 score-token 的方式是一个合理默认，
> **必须在服务器上对照真实 VisualPRM-8B 的 I/O 模板校验**（该逻辑被隔离在单个方法里）。

### 验收标准

| Spearman(PRM, outcome) | 结论 |
|------------------------|------|
| `< 0.3` | ✅ frozen PRM 弱相关 → 需要微调，**项目假设成立，继续** |
| `> 0.6` | ⚠️ frozen PRM 已经好 → **停下重新审视项目** |
| `[0.3, 0.6]` | 灰区 → 与导师讨论 |

报告里同时给 Pearson、AUC、正确/错误轨迹的平均 PRM 分、以及 TP/FP/FN/TN 失败模式分桶
（重点看 **FP：PRM 高估错误轨迹** 的占比）。

---

## Stage 0.2 / 0.3 / 0.4 ⬜（待实现）

- **0.2** MCTS vs Best-of-N 的 gap（决定 Contribution 2）
- **0.3** 50 样本人工标注一致性（决定自动 grounding 标注是否可靠）
- **0.4** crop/zoom 需求统计（决定动作空间最终定版）

这三个 pilot 在 Stage 0.1 验收通过后再实现。
