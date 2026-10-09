"""Try a rationale model on a handful of samples before spending a budget on it.

Sends N samples (from a scored/sampled JSONL, or three built-in InfoSeek-style
examples) through the configured ``rationale.generation`` block with the real
five-rule QC and regeneration loop, and prints each rationale, its verdict, QC
result, attempts and latency, then the pass rate. Use it whenever the
rationale model or provider changes (e.g. switching to Kimi). Costs at most
N x ``rationale.max_attempts`` API calls.

With ``--batch group`` (the step-5 default) siblings — samples from the same
state — share one request, exactly as step 5 sends them.

Usage:
    python DatasetConstruct/try_rationale_model.py --provider kimi_code
    python DatasetConstruct/try_rationale_model.py --provider kimi_code --batch node
    python DatasetConstruct/try_rationale_model.py --input data/construction/infoseek/scored.jsonl --n 10
    python DatasetConstruct/try_rationale_model.py --set rationale.generation.model=<model> --n 3
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

from common import apply_provider, load_env, provider_names, read_jsonl, resolve

from evidencetree.utils import config as cfgutil

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

_HOF = {"evidence_id": "e0", "title": "Pro Football Hall of Fame", "text": "", "image_path": "x.jpg"}
_OPENING = {
    "evidence_id": "e1", "title": "Pro Football Hall of Fame",
    "text": "The Pro Football Hall of Fame is the hall of fame for professional American "
            "football, located in Canton, Ohio. It opened on September 7, 1963, with 17 "
            "charter inductees.",
}
_OFF_TOPIC = {
    "evidence_id": "e1", "title": "Canton, Ohio",
    "text": "Canton is a city in Stark County, Ohio, known for its steel industry.",
}


def builtin_samples() -> list[dict[str, Any]]:
    """A good search, a weak search, and a supported answer (paper Figure 2)."""
    q = "When did this building officially open?"
    base = {"question": q, "gold_answers": ["September 7, 1963"], "query_id": "demo"}
    image_state = {"actions_before": ["image_search(query image)"], "evidence_before": [_HOF]}
    return [
        dict(base, sample_id="demo#t0#s1", parent_key="after-image", state=image_state,
             action={"type": "text_search", "input": "Pro Football Hall of Fame opening date"},
             observation_evidence=[_OPENING], score=0.88),
        dict(base, sample_id="demo#t1#s1", parent_key="after-image", state=image_state,
             action={"type": "text_search", "input": "Canton Ohio"},
             observation_evidence=[_OFF_TOPIC], score=0.21),
        dict(base, sample_id="demo#t0#s2", parent_key="after-lookup",
             state={"actions_before": ["image_search(query image)",
                                       "text_search(Pro Football Hall of Fame opening date)"],
                    "evidence_before": [_HOF, _OPENING]},
             action={"type": "answer", "input": "September 7, 1963"},
             observation_evidence=[], score=1.0),
    ]


def main(argv=None) -> int:
    from step5_gen_rationales import build_rationale_generator

    from evidencetree.prm.rationale_gen import group_samples

    p = argparse.ArgumentParser(description="Try the rationale model on a few samples.")
    p.add_argument("--benchmark", default="infoseek")
    p.add_argument("--config", default=None)
    p.add_argument("--input", default=None, help="scored/sampled JSONL to draw samples from")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mock", action="store_true", help="template backend (no API)")
    p.add_argument("--provider", default=None, help=f"endpoint preset: {', '.join(provider_names())}")
    p.add_argument("--batch", choices=("group", "node"), default="group",
                   help="group = siblings share one request (as step 5 does)")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args(argv)

    load_env()
    cfg = cfgutil.load_config(args.config or CONFIG_DIR / f"{args.benchmark}.yaml",
                              overrides=args.overrides)
    cfg = apply_provider(cfg, "rationale", args.provider)
    gen_cfg = (cfg.get("rationale") or {}).get("generation") or {}
    if args.input:
        rows = list(read_jsonl(resolve(args.input)))
        samples = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    else:
        samples = builtin_samples()[: args.n]
    try:
        gen = build_rationale_generator(cfg, args.mock)
    except (EnvironmentError, ImportError) as exc:
        print(f"cannot build the rationale model: {exc}\n"
              f"  key     -> DatasetConstruct/.env  ({gen_cfg.get('api_key_env')}=...)\n"
              f"  preset  -> DatasetConstruct/configs/providers/{args.provider or '<name>'}.yaml")
        return 2
    print(f"model: {'mock' if args.mock else gen_cfg.get('model')} @ "
          f"{'-' if args.mock else gen_cfg.get('base_url')} ({gen_cfg.get('provider')})\n")

    units = group_samples(samples, 6) if args.batch == "group" else [[s] for s in samples]
    print(f"{len(samples)} samples in {len(units)} request group(s), batch={args.batch}\n")
    n_pass = n_attempts = done = 0
    t_start = time.time()
    for unit in units:
        t0 = time.time()
        try:
            outs = gen.generate_group(unit) if len(unit) > 1 else [gen.generate_for(unit[0])]
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[{unit[0]['sample_id']}...] ERROR {type(exc).__name__}: {exc}\n")
            continue
        elapsed = time.time() - t0
        for s, out in zip(unit, outs):
            done += 1
            n_pass += int(out["rationale_qc_pass"])
            n_attempts += out["rationale_attempts"]
            qc = "PASS" if out["rationale_qc_pass"] else "FAIL " + "; ".join(out["rationale_qc_reasons"])
            print(f"[{s['sample_id']}] {s['action']['type']}({s['action']['input']}) "
                  f"score={s['score']:.2f}")
            print(f"  verdict={out['rationale_verdict']} QC={qc} attempts={out['rationale_attempts']} "
                  f"mode={out.get('rationale_mode')} group took {elapsed:.1f}s")
            print(f"  {out['rationale']}\n")
    if done:
        print(f"QC pass {n_pass}/{done} | mean attempts {n_attempts / done:.2f} | "
              f"wall time {time.time() - t_start:.1f}s")
    return 0 if done and n_pass == done else 1


if __name__ == "__main__":
    raise SystemExit(main())
