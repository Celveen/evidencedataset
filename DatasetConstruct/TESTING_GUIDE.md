# EvidenceTree 数据集管道测试指南

本文档说明如何测试当前保留的数据集、调整测试数量，以及开关 Qwen-VL、
DeepSeek、CLIP image grounding 和 OCR action。

> 2026-06-17 起，SlideVQA 和 M3DocVQA 已从当前管道中移除。历史生成数据
> 可能仍在 `data/` 目录里，但默认测试和数据准备不再引用它们。OCR 作为通用
> MCTS 动作保留，默认关闭，不恢复这两个数据集的专用适配。

所有命令默认在仓库根目录执行：

```bash
cd /home/wenke/SQJ/code/EvidenceTree
```

推荐使用本地 Qwen 环境：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py --help
```

## 1. 测试前检查

检查 Qwen-VL 服务：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果没启动：

```bash
DatasetConstruct/start_qwen25vl.sh
```

或使用 systemd：

```bash
systemctl --user restart qwen25vl
journalctl --user -u qwen25vl -f
```

关闭后台 Qwen-VL：

```bash
systemctl --user stop qwen25vl
systemctl --user status qwen25vl
```

如果是手动启动的普通后台进程：

```bash
pgrep -af 'serve_qwen25vl|qwen25vl|vllm'
pkill -f 'serve_qwen25vl.py'
```

如果端口仍被占用：

```bash
lsof -i :8000
kill <PID>
```

检查 API key：

```bash
sed -E 's/(KEY|TOKEN)=.*/\1=<redacted>/' DatasetConstruct/.env
```

至少应包含：

```bash
LOCAL_QWEN_API_KEY=local
DEEPSEEK_API_KEY=你的key
```

检查当前保留的数据集：

```bash
for d in infoseek scienceqa mrag_bench; do
  echo "== $d =="
  wc -l data/corpus/$d/queries.jsonl data/corpus/$d/corpus.jsonl
done
```

## 2. 单个数据集测试

基本模板：

```bash
DATASET=scienceqa
N=15

.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml \
  --n "$N" --force \
  --set benchmark="$DATASET" \
  --set data.data_dir="data/corpus/$DATASET" \
  --set mcts.rollouts=2 \
  --set output.trajectories="data/pipeline_${N}q/$DATASET/trajectories.jsonl" \
  --set output.scored="data/pipeline_${N}q/$DATASET/scored.jsonl" \
  --set output.rationales="data/pipeline_${N}q/$DATASET/rationales.jsonl" \
  --set output.dataset_dir="data/pipeline_${N}q/$DATASET/dataset"
```

输出目录结构：

```text
data/pipeline_15q/scienceqa/
├── trajectories.jsonl
├── scored.jsonl
├── rationales.jsonl
└── dataset/
    ├── train.jsonl
    ├── val.jsonl
    └── stats.json
```

## 3. 测试不同数量

只改 `N` 即可：

```bash
N=5    # 快速冒烟
N=15   # 小样本检查
N=100  # 放量前检查
```

增加每题 MCTS 轨迹数：

```bash
--set mcts.rollouts=5
```

## 4. 一次重测所有保留数据集

```bash
N=5
ROOT="data/pipeline_${N}q"
SUMMARY="$ROOT/summary.md"

for DATASET in infoseek scienceqa mrag_bench; do
  echo "===== running $DATASET ($N queries) ====="
  .venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
    --config DatasetConstruct/config.qwen25vl.yaml \
    --n "$N" --force \
    --set benchmark="$DATASET" \
    --set data.data_dir="data/corpus/$DATASET" \
    --set mcts.rollouts=5 \
    --set output.trajectories="$ROOT/$DATASET/trajectories.jsonl" \
    --set output.scored="$ROOT/$DATASET/scored.jsonl" \
    --set output.rationales="$ROOT/$DATASET/rationales.jsonl" \
    --set output.dataset_dir="$ROOT/$DATASET/dataset"
done

echo "===== summary ====="
PYTHONPATH=src .venv-qwen25vl/bin/python \
  DatasetConstruct/evaluate_pipeline_outputs.py \
  --root "$ROOT" \
  --soft-threshold 0.75 \
  --examples 5 \
  --output "$SUMMARY"

echo "Summary saved to $SUMMARY"
```

## 5. 开关 Qwen-VL

默认使用 `DatasetConstruct/config.qwen25vl.yaml`：

```yaml
policy:
  backend: api
  model: Qwen2.5-VL-7B-Instruct
  base_url: http://127.0.0.1:8000/v1
```

Step 1 中 Qwen-VL 负责：

- MCTS 动作提议；
- 根据检索证据生成候选答案；
- 读取 query 图片和检索出的 evidence 图片。

关闭 Qwen-VL、改用 mock policy：

```bash
--set policy.backend=mock
```

完整例子：

```bash
DATASET=scienceqa
N=5

.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml \
  --n "$N" --force \
  --set benchmark="$DATASET" \
  --set data.data_dir="data/corpus/$DATASET" \
  --set policy.backend=mock \
  --set rationale.backend=mock \
  --set verifier.image_backend=neutral \
  --set output.trajectories="data/pipeline_mock_${N}q/$DATASET/trajectories.jsonl" \
  --set output.scored="data/pipeline_mock_${N}q/$DATASET/scored.jsonl" \
  --set output.rationales="data/pipeline_mock_${N}q/$DATASET/rationales.jsonl" \
  --set output.dataset_dir="data/pipeline_mock_${N}q/$DATASET/dataset"
```

注意：`--mock` 参数会切到合成 InfoSeek 数据；真实数据但不用 Qwen 时，使用
`--set policy.backend=mock`。

## 6. 开关 DeepSeek

DeepSeek 默认只用于 Step 3 rationale。当前 Step 2 默认是 lexical verifier，
不会调用 DeepSeek。

关闭 DeepSeek rationale：

```bash
--set rationale.backend=mock
```

只跑前两步：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml \
  --steps 1,2 \
  --n 15 --force \
  --set benchmark=scienceqa \
  --set data.data_dir=data/corpus/scienceqa
```

只重跑 DeepSeek rationale 和过滤：

```bash
.venv-qwen25vl/bin/python DatasetConstruct/run_pipeline.py \
  --config DatasetConstruct/config.qwen25vl.yaml \
  --steps 3,4 --force \
  --set benchmark=scienceqa \
  --set data.data_dir=data/corpus/scienceqa \
  --set output.trajectories=data/pipeline_15q/scienceqa/trajectories.jsonl \
  --set output.scored=data/pipeline_15q/scienceqa/scored.jsonl \
  --set output.rationales=data/pipeline_15q/scienceqa/rationales.jsonl \
  --set output.dataset_dir=data/pipeline_15q/scienceqa/dataset
```

## 7. 开关 CLIP 与 image_search

关闭 image_search：

```bash
--set retriever.image.enabled=false
```

这会禁用 MCTS 中的 `image_search` 动作。对 MRAG-Bench 影响最大。

保留 image_search，但关闭 CLIP step-level 打分：

```bash
--set verifier.image_backend=neutral
```

此时 `image_search` 的局部分固定为 `0.5`。

使用 CLIP step-level 打分：

```yaml
verifier:
  image_backend: clip
```

作用：给 `image_search` 动作打局部分，判断 query 图像/区域与问题文本是否相关。

启用跨模态动作空间：

```bash
--set retriever.cross_modal=true
```

启用后会额外允许：

```text
text_to_image(query)          # 文本搜图像
image_to_text(image, region?) # 图像搜文本
```

注意：

- 默认关闭，不会改变普通测试结果。
- 当前只对本地 CLIP 语料生效；- `text_to_image` / `image_search` 需要 corpus 里有图像；`image_to_text` 需要
  query 图像和可索引文本 corpus。

## 8. 开关 OCR action

OCR 默认关闭：

```yaml
retriever:
  ocr:
    enabled: false
```

临时启用：

```bash
--set retriever.ocr.enabled=true
```

当前项目已经配置了本地 Tesseract，不依赖系统 PATH：

```yaml
retriever:
  ocr:
    enabled: true
    command: /home/wenke/SQJ/code/EvidenceTree/.tools/tesseract/bin/tesseract
```

检查 OCR 是否可用：

```bash
/home/wenke/SQJ/code/EvidenceTree/.tools/tesseract/bin/tesseract --version
```

它在 MCTS 里表示：

```text
ocr(image, region?)
```

含义是读取当前 query 图像，或者最近一次 evidence 里的图像。它不是 Top-K
检索器，只返回一条 OCR 识别文本。因此 Step 2 中：

- `local_grounding=null`
- `score=outcome_credit`
- `score_source=outcome_only`
- `label_weight=1-alpha`，默认 alpha=0.5，所以 OCR 样本训练权重为 0.5

训练时建议用 `label_weight` 加权 loss。这样 OCR 这种缺 local 分的样本仍能学习
“最终有没有帮助”，但不会和 `text_search/image_search` 这种双信号标签同等强度地
影响模型。

## 9. 看结果

看 Step 4 过滤结果：

```bash
cat data/pipeline_15q/scienceqa/dataset/stats.json
```

汇总保留数据集结果：

```bash
PYTHONPATH=src .venv-qwen25vl/bin/python \
  DatasetConstruct/evaluate_pipeline_outputs.py \
  --root data/pipeline_15q \
  --soft-threshold 0.75 \
  --examples 5 \
  --output data/pipeline_15q/summary.md
```

`soft_match` 的规则：

- normalized EM 命中；
- 或答案短语互相包含；
- 或 token F1 大于等于 `--soft-threshold`。

soft-match 只是快速人工排查辅助，不等于最终语义评测。

## 10. 当前动作空间

当前 MCTS 动作：

- `text_search(query)`：文本检索；
- `image_search(image/region)`：图像检索，依赖 CLIP/官方图像索引；
- `text_to_image(query)`：文本搜图像，需 `retriever.cross_modal=true`；
- `image_to_text(image/region)`：图像搜文本，需 `retriever.cross_modal=true`；
- `ocr(image/region)`：读取单张图或区域中的文字，默认关闭；
- `answer(text)`：终止并给出答案。

SlideVQA/M3DocVQA 专用的 PDF/PPT 适配逻辑仍然移除。
