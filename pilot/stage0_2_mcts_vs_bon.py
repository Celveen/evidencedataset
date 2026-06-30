"""Stage 0.2 — MCTS vs Best-of-N gap (go/no-go pilot for Contribution 2).

Question (implementation report §Stage 0.2): does putting a *tree* over the
retrieval-action space (UCT selection + tree-shared statistics + modality
bonus) actually beat plain Best-of-N sampling, when BOTH are guided by the SAME
frozen VisualPRM-8B? If BoN ≈ MCTS, the MCTS layer (Contribution 2) is not
pulling its weight and must be rethought before any PRM is trained.

Both methods share proposer / executor / PRM / max_depth. The ONLY difference:
    Best-of-N : N independent stochastic rollouts from the root, pick argmax PRM.
                No tree, no UCT, no cross-rollout statistics.
    MCTS      : the real searcher — selection→expansion→simulation→backup, with
                UCB1 over the PRM's Q, sharing one tree across P rollouts.

Because a tree makes *more* PRM calls than BoN, accuracy alone is not a fair
verdict — we report PRM-call counts and wall time too, so a win can be read as
"better answers" vs. "merely more compute".

Modes:
    --mock   self-contained: synthetic data + mock generator + offline overlap
             scorer. No GPU, no downloads. Validates the harness end-to-end.
    (real)   frozen VisualPRM-8B + policy backend + real InfoSeek; GPU server.

Usage:
    python pilot/stage0_2_mcts_vs_bon.py --config configs/pilot_stage0_2.yaml --mock
    python pilot/stage0_2_mcts_vs_bon.py --config configs/pilot_stage0_2.yaml \
        --set generation.backend=hf --n 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Make ``src`` importable when run as a plain script (no install needed).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidencetree.actions import ActionExecutor, BM25Retriever  # noqa: E402
from evidencetree.actions.action_space import SearchState  # noqa: E402
from evidencetree.eval import benchmarks, metrics  # noqa: E402
from evidencetree.generation import build_generator  # noqa: E402
from evidencetree.mcts import HeuristicProposer, MCTSSearcher, SearchConfig  # noqa: E402
from evidencetree.mcts.proposer import LLMProposer  # noqa: E402
from evidencetree.mcts.search import state_to_trajectory  # noqa: E402
from evidencetree.prm.model import HeuristicOverlapScorer, VisualPRMScorer  # noqa: E402
from evidencetree.utils import config as cfgutil  # noqa: E402
from evidencetree.utils import get_logger  # noqa: E402

log = get_logger("pilot.stage0_2")


# --------------------------------------------------------------------------- #
# A scorer wrapper that counts PRM calls, so the MCTS/BoN comparison can be
# read against compute, not just accuracy.
# --------------------------------------------------------------------------- #
class CountingScorer:
    """Wraps any TrajectoryScorer and counts ``score`` invocations."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0

    def score(self, trajectory, **kwargs: Any) -> float:
        self.calls += 1
        return float(self.inner.score(trajectory, **kwargs))

    def reset(self) -> None:
        self.calls = 0


# --------------------------------------------------------------------------- #
# Component construction (shared by both methods)
# --------------------------------------------------------------------------- #
def build_components(cfg: dict[str, Any], mock: bool):
    data_cfg = cfg.get("data", {})
    n_queries = int(cfg.get("n_queries", 200))
    queries, corpus = benchmarks.load_infoseek(
        n=n_queries, mock=mock,
        data_dir=data_cfg.get("data_dir"), seed=int(data_cfg.get("seed", 0)),
    )
    log.info("Loaded %d queries, %d corpus docs.", len(queries), len(corpus))

    retriever = BM25Retriever().build(corpus)
    executor = ActionExecutor(
        text_retriever=retriever,
        top_k=int(cfg.get("retriever", {}).get("top_k", 5)),
    )

    gen_cfg = dict(cfg.get("generation", {}))
    if mock:
        gen_cfg["backend"] = "mock"
    generator = build_generator(gen_cfg)

    # Mock threads the eval-only gold hint into answer drafting so generator
    # accuracy is controlled (same device as Stage 0.1); real runs never do.
    answer_kwargs_fn = None
    if mock:
        gold_by_q = {q.question: q.gold_answers for q in queries}
        answer_kwargs_fn = lambda s: {"reference": gold_by_q.get(s.question)}  # noqa: E731
    if mock:
        proposer = HeuristicProposer(generator, answer_kwargs_fn=answer_kwargs_fn)
    else:
        # Real run: policy LLM proposes actions (temperature>0 gives BoN diversity),
        # with the heuristic proposer as a parse-failure fallback.
        proposer = LLMProposer(generator, fallback=HeuristicProposer(generator))

    prm_cfg = dict(cfg.get("prm", {}))
    scorer_kind = prm_cfg.get("scorer") or ("overlap" if mock else "visualprm")
    if scorer_kind == "overlap":
        base_scorer = HeuristicOverlapScorer()
    else:
        base_scorer = VisualPRMScorer(
            model_name=prm_cfg.get("model_name", "OpenGVLab/VisualPRM-8B"),
            mock=mock, device=prm_cfg.get("device"),
        )
    scorer = CountingScorer(base_scorer)
    log.info("Scorer: %s | generation backend: %s | proposer: %s",
             scorer_kind, gen_cfg.get("backend"), type(proposer).__name__)
    return queries, executor, proposer, scorer


# --------------------------------------------------------------------------- #
# Best-of-N: N independent stochastic rollouts, pick argmax PRM. No tree.
# --------------------------------------------------------------------------- #
def sample_trajectory(
    root: SearchState, proposer, executor, max_depth: int, rng: random.Random
) -> SearchState:
    """One stochastic root→terminal rollout (random choice among proposals)."""
    state = root
    while not state.is_terminal and state.depth < max_depth:
        candidates = proposer.propose(state, k=3)
        if not candidates:
            break
        action = candidates[rng.randrange(len(candidates))]
        state = executor.execute(state, action)
    if not state.is_terminal:
        state = executor.execute(state, proposer.propose_answer(state))
    return state


def best_of_n(
    question: str, image_path: str | None, *, proposer, executor, scorer,
    n_samples: int, max_depth: int, rng: random.Random,
) -> tuple[str, float]:
    """Return (best answer, best PRM reward) over N independent samples."""
    root = SearchState(question=question, image_path=image_path)
    best_answer, best_reward = "", float("-inf")
    for _ in range(n_samples):
        terminal = sample_trajectory(root, proposer, executor, max_depth, rng)
        reward = scorer.score(state_to_trajectory(terminal))
        if reward > best_reward:
            best_reward, best_answer = reward, terminal.final_answer or ""
    return best_answer, best_reward


# --------------------------------------------------------------------------- #
# Evaluation loop: run both methods on every query
# --------------------------------------------------------------------------- #
def evaluate(cfg: dict[str, Any], mock: bool) -> dict[str, Any]:
    queries, executor, proposer, scorer = build_components(cfg, mock)
    outcome_metric = cfg.get("outcome", {}).get("metric", "exact_match")
    em = (lambda a, g: metrics.f1_score(a, g)) if outcome_metric == "f1" \
        else (lambda a, g: metrics.exact_match(a, g))

    search_cfg = SearchConfig.from_dict(cfg)
    searcher = MCTSSearcher(executor, proposer, scorer, config=search_cfg)

    bon_cfg = cfg.get("bon", {})
    n_samples = int(bon_cfg.get("n_samples", 8))
    max_depth = int(search_cfg.max_depth)
    rng = random.Random(int(bon_cfg.get("seed", 0)))

    records: list[dict[str, Any]] = []
    agg = {
        "bon": {"correct": 0.0, "calls": 0, "secs": 0.0},
        "mcts": {"correct": 0.0, "calls": 0, "secs": 0.0},
    }
    for i, q in enumerate(queries):
        # --- Best-of-N ---
        scorer.reset()
        t0 = time.perf_counter()
        bon_ans, bon_reward = best_of_n(
            q.question, q.image_path, proposer=proposer, executor=executor,
            scorer=scorer, n_samples=n_samples, max_depth=max_depth, rng=rng,
        )
        bon_secs, bon_calls = time.perf_counter() - t0, scorer.calls
        bon_em = em(bon_ans, q.gold_answers)

        # --- MCTS ---
        scorer.reset()
        t0 = time.perf_counter()
        res = searcher.search(q.question, image_path=q.image_path)
        mcts_secs, mcts_calls = time.perf_counter() - t0, scorer.calls
        mcts_em = em(res.answer, q.gold_answers)

        agg["bon"]["correct"] += bon_em
        agg["bon"]["calls"] += bon_calls
        agg["bon"]["secs"] += bon_secs
        agg["mcts"]["correct"] += mcts_em
        agg["mcts"]["calls"] += mcts_calls
        agg["mcts"]["secs"] += mcts_secs

        records.append({
            "query_id": q.query_id, "question": q.question, "gold": q.gold_answers,
            "bon": {"answer": bon_ans, "outcome": bon_em, "reward": bon_reward,
                    "prm_calls": bon_calls},
            "mcts": {"answer": res.answer, "outcome": mcts_em,
                     "reward": res.best_reward, "prm_calls": mcts_calls,
                     "rollouts": res.rollouts_run},
        })
        if (i + 1) % max(1, len(queries) // 10) == 0:
            log.info("  evaluated %d/%d", i + 1, len(queries))

    n = len(records)
    summary = {
        "n": n, "outcome_metric": outcome_metric, "n_samples_bon": n_samples,
        "rollouts_mcts": search_cfg.rollouts,
        "acc_bon": agg["bon"]["correct"] / n if n else 0.0,
        "acc_mcts": agg["mcts"]["correct"] / n if n else 0.0,
        "prm_calls_bon": agg["bon"]["calls"] / n if n else 0.0,
        "prm_calls_mcts": agg["mcts"]["calls"] / n if n else 0.0,
        "secs_bon": agg["bon"]["secs"] / n if n else 0.0,
        "secs_mcts": agg["mcts"]["secs"] / n if n else 0.0,
    }
    summary["mcts_minus_bon"] = summary["acc_mcts"] - summary["acc_bon"]
    summary["prm_call_ratio"] = (
        summary["prm_calls_mcts"] / summary["prm_calls_bon"]
        if summary["prm_calls_bon"] else None
    )
    return {"summary": summary, "records": records}


# --------------------------------------------------------------------------- #
# Verdict + reporting
# --------------------------------------------------------------------------- #
def verdict(summary: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, str]:
    margin = float(thresholds.get("mcts_wins_margin", 0.03))
    delta = summary["mcts_minus_bon"]
    ratio = summary.get("prm_call_ratio")
    cost = "" if ratio is None else f" (MCTS uses {ratio:.1f}× the PRM calls)"
    if delta >= margin:
        return "CONTINUE", (
            f"MCTS beats Best-of-N by {delta:+.3f} ≥ {margin}{cost}: the tree adds "
            "value over independent sampling → Contribution 2 justified. Re-check "
            "the win still holds when compute-matched (raise N until PRM calls tie)."
        )
    if delta <= -margin:
        return "RED", (
            f"Best-of-N beats MCTS by {-delta:+.3f}{cost}: the tree hurts → fall "
            "back to PRM+BoN (report §4.2) and reconsider Contribution 2."
        )
    return "GREY", (
        f"MCTS ≈ Best-of-N (Δ={delta:+.3f}, |Δ|<{margin}){cost}: the tree is not "
        "pulling its weight. Before training a PRM, either strengthen the search "
        "(richer action space / deeper budget) or demote MCTS in the story."
    )


def write_report(result: dict[str, Any], cfg: dict[str, Any], mock: bool) -> Path:
    report_dir = Path(cfg.get("output", {}).get("report_dir", "data/pilot_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "mock" if mock else "real"
    base = report_dir / f"stage0_2_{tag}_{ts}"
    decision, rationale = verdict(result["summary"], cfg.get("thresholds", {}))
    payload = {
        "stage": "0.2 — MCTS vs Best-of-N gap",
        "mode": tag, "timestamp": ts, "config": cfg,
        "summary": result["summary"],
        "verdict": {"decision": decision, "rationale": rationale},
        "records": result["records"],
    }
    base.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    base.with_suffix(".md").write_text(_render_markdown(payload), encoding="utf-8")
    return base


def _render_markdown(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    v = payload["verdict"]
    ratio = "n/a" if s["prm_call_ratio"] is None else f"{s['prm_call_ratio']:.1f}×"
    disclaimer = []
    if payload["mode"] == "mock":
        disclaimer = [
            "> ⚠️ **MOCK 模式**：合成数据 + 离线 overlap scorer，仅验证对比 harness。",
            "> 下面的准确率与 verdict **不是真实结论**——真实结论需在 GPU 服务器上去掉",
            "> `--mock`，用 frozen VisualPRM-8B + policy 模型 + 真实 InfoSeek 跑。",
            "",
        ]
    return "\n".join([
        f"# Stage 0.2 — MCTS vs Best-of-N ({payload['mode']})",
        "",
        *disclaimer,
        f"- 时间: {payload['timestamp']}",
        f"- 样本数 N: {s['n']} | 指标: {s['outcome_metric']}",
        f"- BoN N={s['n_samples_bon']} | MCTS rollouts={s['rollouts_mcts']}",
        "",
        "## 主对比",
        "",
        "| 方法 | 准确率 | 平均 PRM 调用/query | 平均耗时(s)/query |",
        "|------|--------|--------------------|-------------------|",
        f"| Best-of-N | {s['acc_bon']:.3f} | {s['prm_calls_bon']:.1f} | {s['secs_bon']:.3f} |",
        f"| MCTS | {s['acc_mcts']:.3f} | {s['prm_calls_mcts']:.1f} | {s['secs_mcts']:.3f} |",
        f"| **Δ (MCTS−BoN)** | **{s['mcts_minus_bon']:+.3f}** | PRM 调用比 {ratio} | |",
        "",
        "> 公平性提醒：MCTS 天然比 BoN 多调 PRM。若 MCTS 仅靠更多算力取胜，请提高 N",
        "> 直到两者 PRM 调用数持平后再比准确率（compute-matched comparison）。",
        "",
        "## 结论",
        "",
        f"**Decision: {v['decision']}**",
        "",
        v["rationale"],
        "",
        "> 验收标准（实现报告 Stage 0.2）：MCTS 显著优于 BoN → Contribution 2 必要性确认；",
        "> BoN 接近 MCTS → MCTS 这层 contribution 弱化，需重新规划。",
    ])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0.2 MCTS vs Best-of-N pilot.")
    p.add_argument("--config", default="configs/pilot_stage0_2.yaml")
    p.add_argument("--mock", action="store_true", help="Self-contained mock mode.")
    p.add_argument("--n", type=int, default=None, help="Override n_queries.")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="Override config, e.g. --set generation.backend=hf (repeatable).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = cfgutil.load_config(args.config, overrides=args.overrides)
    if args.n is not None:
        cfg["n_queries"] = args.n

    log.info("=== Stage 0.2 MCTS vs Best-of-N (mock=%s) ===", args.mock)
    if args.mock:
        log.warning("MOCK 模式：合成数据 + 离线 scorer，仅验证 harness，非真实结论。")
    result = evaluate(cfg, mock=args.mock)
    base = write_report(result, cfg, mock=args.mock)

    s = result["summary"]
    decision, rationale = verdict(s, cfg.get("thresholds", {}))
    log.info("acc: BoN=%.3f  MCTS=%.3f  Δ=%+.3f | PRM calls/q: BoN=%.1f MCTS=%.1f",
             s["acc_bon"], s["acc_mcts"], s["mcts_minus_bon"],
             s["prm_calls_bon"], s["prm_calls_mcts"])
    log.info("VERDICT: %s — %s", decision, rationale)
    log.info("Report written: %s.{json,md}", base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
