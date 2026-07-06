# 论文表格数字溯源与核实状态

> 论文纪律：我方结果全部留 "--"（训练未完成）；基线数字全部来自真实发表论文。
> 本文档记录每个数字的出处与核实等级，投稿前按"待复核"清单人工过一遍 PDF。

核实等级：✅ = 从官方 repo/eval-kit/论文表直接核实；☑ = 多个独立来源交叉引用一致；⚠ = 单一来源提取，投稿前需对照原 PDF 复核。

## Table 1 — InfoSeek（val，官方 relaxed accuracy，Un-Q/Un-E/All）

| 行 | 数字 | 出处 | 等级 |
|---|---|---|---|
| BLIP-2 (Flan-T5-XXL) | 12.7/12.3/12.5 | InfoSeek 官方 eval repo（github.com/edchengg/infoseek_eval，README：12.74/12.28/12.51） | ✅ |
| InstructBLIP | 8.9/7.4/8.1 | 同上（8.89/7.38/8.06） | ✅ |
| LLaVA-1.5-7B | 9.6/9.4/9.5 | Wiki-LLaVA (CVPRW'24) Table 2；ReflectiVA/DBAgent 表交叉一致 | ☑ |
| GPT-4V | 15.0/14.3/14.6 | Learning-to-Search (arXiv 2604.07146) Table 1 编集（本人读过原表） | ☑ |
| Qwen2.5-VL-7B zero-shot | 22.8/24.1/23.7 | 同上 Table 1（本人读过原表） | ✅ |
| Wiki-LLaVA | 30.1/27.8/28.9 | Caffagni et al. CVPRW 2024 Table 2（All=28.9 为常引均值，原表核对） | ☑（All ⚠） |
| PreFLMR (RA-VQAv2) | --/--/30.7 | Lin et al. ACL 2024 Table 7（30.65；作者自注 setup 早于 InfoSeek 正式版，caption 已声明） | ✅ |
| EchoSight (LLaMA-3-8B) | --/--/31.3 | Yan & Xie EMNLP'24 Findings 摘要 + 官方 repo README | ✅ |
| RoRA-VLM | 25.1/27.3/-- | Qi et al. arXiv 2410.08876 | ✅ |
| ReflectiVA | 40.4/39.8/40.1 | Cocchi et al. CVPR 2025；Wiki-R1 引"prior SOTA 40.1"交叉确认 | ☑ |
| Wiki-PRF | 43.3/42.7/42.8 | 官方 repo README（github.com/cqu-student/Wiki-PRF；NeurIPS 2025，arXiv 2510.14605） | ✅ |
| Wiki-R1 | 47.8/--/44.1 | arXiv 2603.05256（ICLR 2026）摘要/项目页；Un-E 未见 → 留 -- | ☑ |
| DBAgent (SFT: InfoSeek) | 46.5/51.0/49.9 | arXiv 2604.07146 Table 1（本人通读原 PDF） | ✅ |

**明确不报 InfoSeek 的相关方法**（已核实，不入表）：AR-MCTS（benchmark 为 MathVista/We-Math/GAOKAO-MM）、RCTS（ScienceQA/MMMU/MathVista/VizWiz/VSR）。
**未入表原因记录**：MMSearch-R1 的 InfoSeek 用 LLM-judge + 抽样子集，与标准 val 协议不可比；mR2AG 40.6/39.8/40.2 为二手引用且与 ReflectiVA 数字高度相似，风险大；ReAG (CVPR 2026) 两次提取数字冲突（50.3/48.2 vs "+3.7 over ReflectiVA"），待复核 arXiv 2511.22715 Table 2 后可补。

## Table 2 — ScienceQA（full test accuracy）

| 行 | 数字 | 出处 | 等级 |
|---|---|---|---|
| Human | 88.40 | Lu et al. NeurIPS 2022（官方 leaderboard README） | ✅ |
| MM-CoT (Large) | 91.68 | Zhang et al. TMLR 2024（leaderboard 核实） | ✅ |
| LLaVA (FT, Vicuna-13B) | 90.92 | Liu et al. NeurIPS 2023 Table 7 | ✅ |
| Zero-shot Qwen2-VL-7B / InternVL2-8B | 80.33 / 93.00 | RCTS 论文主表（官方 repo 结果表图直读） | ✅ |
| Vanilla RAG (Qwen2-VL-7B) | 86.68 | 同上 | ✅ |
| RCTS Qwen2-VL-7B / InternVL2-8B | 91.44 / 94.20 | 同上（ICML 2025 Spotlight，arXiv 2506.07785） | ✅ |

**协议注意**（caption 已声明）：AR-MCTS 不评 ScienceQA；LLaVA-1.5 66.8 / InstructBLIP 60.5 等为 SQA-IMG 子集口径，混表会被审稿人抓，已排除。

## Table 3 — GQA（test-dev balanced accuracy）

| 行 | 数字 | 出处 | 等级 |
|---|---|---|---|
| BLIP-2 (Vicuna-13B) | 41.0 | LLaVA-1.5 (CVPR 2024) Table 1 转引（caption 已注明 ∗）；注意 Qwen-VL README 版本为 32.3（prompt 协议不同） | ⚠ |
| InstructBLIP (Vicuna-7B) | 49.2 | InstructBLIP 论文 Table 1 | ☑ |
| Qwen-VL-Chat / Qwen-VL | 57.5 / 59.3 | 官方 QwenLM/Qwen-VL README（直接抓取） | ✅ |
| LLaVA-1.5-7B / 13B | 62.0 / 63.3 | 官方 MODEL_ZOO.md（直接抓取；注意不是 67.0） | ✅ |

**已核实**：RCTS/AR-MCTS 均不报 GQA；多模态 RAG 方法基本无 GQA 结果（caption 已按"感知证据探针"定位）。

## 投稿前待人工复核清单

1. GPT-4V InfoSeek 15.0/14.3/14.6 的最初出处（目前引 DBAgent Table 1 编集口径）；
2. Wiki-LLaVA All=28.9 对照 CVPRW 原表；
3. BLIP-2 GQA 41.0 vs 32.3 二选一并统一 caption 措辞；
4. references.bib 中 `wikir12026` 的作者名单为占位符（agent 被代理挡住没拿到），需从 arXiv 2603.05256 补全；
5. RoRA-VLM 的 venue（ICCV 2025 Workshop 具体名）与 9 人作者名单；
6. GroundedPRM / VisualPRM / DeepMMSearch-R1 / HiPRAG / TreePS-RAG / Poisoned-MRAG / Learning-to-Search 目前均为 arXiv preprint 引用，投稿时再查一次是否已被接收；
7. 换 AAAI 官方模板（aaai2027.sty + aaai27.bst，author-year 引用制），本稿 preamble 仅为格式模拟。
