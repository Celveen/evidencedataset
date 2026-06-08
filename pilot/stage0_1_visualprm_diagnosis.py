"""Stage 0.1 — VisualPRM failure-mode diagnosis (go/no-go pilot).

Pipeline (implementation report §Stage 0.1):
    1. Load N InfoSeek queries + their own corpus.
    2. Vanilla RAG: BM25 top-k -> LLM generates an answer -> build a trajectory.
    3. Outcome correctness: EM/F1 of the answer vs. gold.
    4. Frozen VisualPRM-8B scores each trajectory (score-only).
    5. Correlation analysis: Spearman(PRM score, outcome) + failure-mode buckets.
       -> Verdict: Spearman < 0.3  => fine-tuning needed, hypothesis holds, CONTINUE.
                   Spearman > 0.6  => STOP and reconsider the project.

Modes:
    --mock   self-contained: synthetic data + mock generator + mock PRM. No GPU,
             no downloads. Validates the whole pipeline + analysis with signal.
    (real)   loads VisualPRM-8B + real InfoSeek; run on the GPU server.

Usage:
    python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml --mock
    python pilot/stage0_1_visualprm_diagnosis.py --config configs/pilot.yaml \
        --set generation.backend=api --set generation.provider=anthropic
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make ``src`` importable when run as a plain script (no install needed).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.actions.retrievers import BM25Retriever  # noqa: E402
from evidencetree.eval import benchmarks, metrics  # noqa: E402
from evidencetree.generation import build_generator  # noqa: E402
from evidencetree.prm.model import Trajectory, TrajectoryStep, VisualPRMScorer  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("pilot.stage0_1")


# --------------------------------------------------------------------------- #
# Component construction
# --------------------------------------------------------------------------- #
def build_components(cfg: dict[str, Any], mock: bool):
    """Build (queries, corpus, retriever, generator, scorer) from config."""
    data_cfg = cfg.get("data", {})
    n_queries = int(cfg.get("n_queries", 1000))

    log.info("Loading %s (n=%d, mock=%s)...", data_cfg.get("benchmark"), n_queries, mock)
    queries, corpus = benchmarks.load_infoseek(
        n=n_queries,
        mock=mock,
        data_dir=data_cfg.get("data_dir"),
        seed=int(data_cfg.get("seed", 0)),
    )
    log.info("Loaded %d queries, %d corpus docs.", len(queries), len(corpus))

    retriever = BM25Retriever().build(corpus)

    gen_cfg = dict(cfg.get("generation", {}))
    if mock:
        gen_cfg["backend"] = "mock"
    generator = build_generator(gen_cfg)
    log.info("Generation backend: %s", gen_cfg.get("backend"))

    prm_cfg = dict(cfg.get("prm", {}))
    scorer = VisualPRMScorer(
        model_name=prm_cfg.get("model_name", "OpenGVLab/VisualPRM-8B"),
        mock=mock,
        device=prm_cfg.get("device"),
        mock_correlation=float(prm_cfg.get("mock_correlation", 0.2)),
        mock_seed=int(prm_cfg.get("mock_seed", 0)),
    )
    log.info("PRM scorer: %s (mock=%s)", scorer.model_name, mock)
    return queries, corpus, retriever, generator, scorer


# --------------------------------------------------------------------------- #
# Vanilla RAG + outcome
# --------------------------------------------------------------------------- #
def run_vanilla_rag(query, retriever, generator, top_k: int, mock: bool) -> Trajectory:
    """BM25 top-k retrieval -> generate answer -> assemble a trajectory."""
    hits = retriever.search(query.question, top_k=top_k)
    context = [h.text for h in hits]

    # The mock generator may consume an eval-only ``reference`` hint; real
    # backends ignore it (we don't pass it for them).
    gen_kwargs = {"reference": query.gold_answers} if mock else {}
    answer = generator.generate(query.question, context, **gen_kwargs)

    steps = [
        TrajectoryStep(
            action_type="text_search",
            action_input=query.question,
            observation=" | ".join(f"{h.title}: {h.text}" for h in hits),
        ),
        TrajectoryStep(action_type="answer", action_input=answer),
    ]
    return Trajectory(question=query.question, steps=steps, final_answer=answer)


def compute_outcome(answer: str, gold_answers, metric: str) -> float:
    if metric == "f1":
        return metrics.f1_score(answer, gold_answers)
    return metrics.exact_match(answer, gold_answers)


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
def diagnose(cfg: dict[str, Any], mock: bool) -> dict[str, Any]:
    queries, corpus, retriever, generator, scorer = build_components(cfg, mock)
    top_k = int(cfg.get("retriever", {}).get("top_k", 5))
    outcome_metric = cfg.get("outcome", {}).get("metric", "exact_match")

    records: list[dict[str, Any]] = []
    for i, query in enumerate(queries):
        traj = run_vanilla_rag(query, retriever, generator, top_k, mock)
        outcome = compute_outcome(traj.final_answer, query.gold_answers, outcome_metric)
        # Mock PRM consumes an eval-only outcome hint; real PRM never sees it.
        prm_score = scorer.score(traj, outcome_hint=outcome if mock else None)
        records.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "gold": query.gold_answers,
                "answer": traj.final_answer,
                "outcome": outcome,
                "prm_score": prm_score,
            }
        )
        if (i + 1) % max(1, len(queries) // 10) == 0:
            log.info("  diagnosed %d/%d", i + 1, len(queries))

    analysis = analyze([r["prm_score"] for r in records], [r["outcome"] for r in records])
    return {"records": records, "analysis": analysis, "outcome_metric": outcome_metric}


def analyze(prm_scores: list[float], outcomes: list[float]) -> dict[str, Any]:
    """Correlation between PRM scores and outcomes + failure-mode buckets."""
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    scores = np.asarray(prm_scores, dtype=float)
    outs = np.asarray(outcomes, dtype=float)
    n = len(scores)

    spearman = _safe_corr(spearmanr, scores, outs)
    pearson = _safe_corr(pearsonr, scores, outs)

    # Failure-mode buckets (split PRM by median; outcome by 0.5).
    median_score = float(np.median(scores)) if n else 0.0
    buckets = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for s, o in zip(scores, outs):
        high_prm = s >= median_score
        correct = o >= 0.5
        if correct and high_prm:
            buckets["tp"] += 1          # PRM rightly favors a correct trajectory
        elif correct and not high_prm:
            buckets["fn"] += 1          # PRM under-scores a correct trajectory
        elif not correct and high_prm:
            buckets["fp"] += 1          # PRM over-scores a wrong trajectory (key failure)
        else:
            buckets["tn"] += 1

    # Discrimination: AUC = P(PRM ranks a correct above a wrong) when outcome binary.
    auc = _binary_auc(scores, outs)

    correct_mask = outs >= 0.5
    mean_score_correct = float(scores[correct_mask].mean()) if correct_mask.any() else None
    mean_score_wrong = (
        float(scores[~correct_mask].mean()) if (~correct_mask).any() else None
    )

    return {
        "n": int(n),
        "outcome_accuracy": float(outs.mean()) if n else 0.0,
        "spearman": spearman,
        "pearson": pearson,
        "auc": auc,
        "median_prm_score": median_score,
        "mean_prm_score_correct": mean_score_correct,
        "mean_prm_score_wrong": mean_score_wrong,
        "buckets": buckets,
    }


def _safe_corr(fn, a, b) -> float | None:
    """Return correlation coefficient, or None if undefined (e.g. constant input)."""
    import numpy as np

    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    coef = fn(a, b)[0]
    return None if (coef != coef) else float(coef)  # NaN guard


def _binary_auc(scores, outcomes) -> float | None:
    """ROC-AUC when outcomes are binary; None otherwise."""
    import numpy as np

    binary = np.isin(np.unique(outcomes), [0.0, 1.0]).all()
    pos = outcomes >= 0.5
    if not binary or pos.all() or (~pos).all():
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(pos.astype(int), scores))
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# Verdict + reporting
# --------------------------------------------------------------------------- #
def verdict(spearman: float | None, thresholds: dict[str, Any]) -> tuple[str, str]:
    low = float(thresholds.get("needs_finetune_below", 0.3))
    high = float(thresholds.get("already_good_above", 0.6))
    if spearman is None:
        return "INCONCLUSIVE", "Spearman undefined (constant scores/outcomes). Inspect data."
    if spearman < low:
        return "CONTINUE", (
            f"Spearman {spearman:.3f} < {low}: frozen VisualPRM correlates weakly "
            "with outcome -> fine-tuning is justified. Project hypothesis holds."
        )
    if spearman > high:
        return "STOP", (
            f"Spearman {spearman:.3f} > {high}: frozen VisualPRM already tracks "
            "outcome well -> STOP and reconsider whether PRM fine-tuning is needed."
        )
    return "GREY", (
        f"Spearman {spearman:.3f} in [{low}, {high}]: ambiguous. Discuss with advisor; "
        "consider more queries or inspecting failure buckets."
    )


def write_report(result: dict[str, Any], cfg: dict[str, Any], mock: bool) -> Path:
    report_dir = Path(cfg.get("output", {}).get("report_dir", "data/pilot_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "mock" if mock else "real"
    base = report_dir / f"stage0_1_{tag}_{ts}"

    analysis = result["analysis"]
    decision, rationale = verdict(analysis["spearman"], cfg.get("thresholds", {}))
    payload = {
        "stage": "0.1 — VisualPRM failure-mode diagnosis",
        "mode": tag,
        "timestamp": ts,
        "config": cfg,
        "analysis": analysis,
        "verdict": {"decision": decision, "rationale": rationale},
        "records": result["records"],
    }
    base.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base.with_suffix(".md").write_text(
        _render_markdown(payload, result["outcome_metric"]), encoding="utf-8"
    )
    return base


def _render_markdown(payload: dict[str, Any], outcome_metric: str) -> str:
    a = payload["analysis"]
    b = a["buckets"]
    v = payload["verdict"]

    def fmt(x):
        return "n/a" if x is None else f"{x:.3f}"

    disclaimer = []
    if payload["mode"] == "mock":
        disclaimer = [
            "> ⚠️ **MOCK 模式**：合成数据 + 合成 PRM 分，仅用于验证 pipeline 与分析逻辑。",
            "> 下面的相关性与 verdict **不是真实诊断结论**（样本极少、PRM 非真实模型）。",
            "> 真实结论需在 GPU 服务器上去掉 `--mock`、用 n=1000 真实数据跑。",
            "",
        ]

    return "\n".join(
        [
            f"# Stage 0.1 诊断报告 ({payload['mode']})",
            "",
            *disclaimer,
            f"- 时间: {payload['timestamp']}",
            f"- 样本数 N: {a['n']}",
            f"- outcome 指标: {outcome_metric}",
            f"- vanilla RAG 准确率: {fmt(a['outcome_accuracy'])}",
            "",
            "## PRM ↔ outcome 相关性",
            "",
            "| 指标 | 值 |",
            "|------|----|",
            f"| Spearman | {fmt(a['spearman'])} |",
            f"| Pearson | {fmt(a['pearson'])} |",
            f"| AUC (binary outcome) | {fmt(a['auc'])} |",
            f"| 平均 PRM 分（answer 正确） | {fmt(a['mean_prm_score_correct'])} |",
            f"| 平均 PRM 分（answer 错误） | {fmt(a['mean_prm_score_wrong'])} |",
            "",
            "## 失败模式分桶（PRM 中位数划分 × outcome）",
            "",
            "| | outcome 正确 | outcome 错误 |",
            "|---|---|---|",
            f"| PRM 高 | TP={b['tp']} | **FP={b['fp']}**（PRM 高估错误轨迹）|",
            f"| PRM 低 | FN={b['fn']}（PRM 低估正确轨迹）| TN={b['tn']} |",
            "",
            "## 结论",
            "",
            f"**Decision: {v['decision']}**",
            "",
            v["rationale"],
            "",
            "> 验收标准（实现报告 Stage 0.1）：Spearman < 0.3 → 继续；> 0.6 → 停下重审。",
        ]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0.1 VisualPRM diagnosis pilot.")
    p.add_argument("--config", default="configs/pilot.yaml", help="YAML config path.")
    p.add_argument("--mock", action="store_true", help="Run self-contained mock mode.")
    p.add_argument("--n", type=int, default=None, help="Override n_queries.")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override config, e.g. --set generation.backend=api (repeatable).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg["n_queries"] = args.n

    log.info("=== Stage 0.1 diagnosis (mock=%s) ===", args.mock)
    if args.mock:
        log.warning(
            "MOCK 模式：合成数据 + 合成 PRM，verdict 仅验证 pipeline，非真实诊断结论。"
        )
    result = diagnose(cfg, mock=args.mock)
    base = write_report(result, cfg, mock=args.mock)

    a = result["analysis"]
    decision, rationale = verdict(a["spearman"], cfg.get("thresholds", {}))
    sp = "n/a" if a["spearman"] is None else f"{a['spearman']:.3f}"
    log.info("Spearman(PRM, outcome) = %s | accuracy = %.3f", sp, a["outcome_accuracy"])
    log.info("VERDICT: %s — %s", decision, rationale)
    log.info("Report written: %s.{json,md}", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
