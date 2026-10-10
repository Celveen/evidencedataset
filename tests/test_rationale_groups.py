"""Group-mode rationales (siblings share one request) and provider presets."""

import sys
from pathlib import Path

import pytest

from evidencetree.prm.rationale_gen import (
    RationaleGenerator,
    check_rationale,
    group_samples,
    parse_group_output,
    sibling_leak,
)

_DC = Path(__file__).resolve().parents[1] / "DatasetConstruct"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from common import apply_provider, provider_names  # noqa: E402

_SHARED = {"actions_before": ["image_search(query image)"],
           "evidence_before": [{"evidence_id": "e0", "title": "Hall of Fame", "text": ""}]}


def _sib(i, atype, title, score):
    obs = [] if atype == "answer" else [
        {"evidence_id": "e1", "title": title, "text": f"{title} page text."}]
    return {"sample_id": f"q#t{i}#s1", "query_id": "q", "parent_key": "p",
            "question": "When did this building open?", "gold_answers": ["1963"],
            "state": _SHARED, "action": {"type": atype, "input": f"input {i}"},
            "observation_evidence": obs, "score": score}


def _rationale(atype, cite, extra="", verdict="good"):
    body = (f"This {atype} step is judged from the question and the evidence [{cite}] "
            f"only. {extra} " + "It keeps the search focused on the asked attribute and "
            "adds relevant material to the bundle. " * 3).strip()
    return f"{body}\nVERDICT: {verdict}"


class ScriptedGenerator:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, docs, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def test_group_samples_and_parse():
    a, b, c = _sib(1, "text_search", "Opening", 0.9), _sib(2, "text_search", "Canton", 0.2), \
        dict(_sib(3, "answer", "", 1.0), parent_key="other")
    groups = group_samples([a, c, b], max_group=6)
    assert [[s["sample_id"] for s in g] for g in groups] == [
        [a["sample_id"], b["sample_id"]], [c["sample_id"]]]
    assert len(group_samples([a, b], max_group=1)) == 2
    text = "[[C2]]\nsecond\nVERDICT: poor\n[[C1]]\nfirst\nVERDICT: good\n[[C9]] junk"
    assert parse_group_output(text, 2) == {2: "second\nVERDICT: poor", 1: "first\nVERDICT: good"}


def test_group_prompt_shares_state_once_and_lists_each_candidate():
    a, b = _sib(1, "text_search", "Opening", 0.9), _sib(2, "image_search", "Canton", 0.2)
    prompt = RationaleGenerator(backend="mock").build_group_prompt([a, b])
    assert prompt.count("When did this building open?") == 1
    assert prompt.count("[e0] Hall of Fame") == 1           # shared evidence sent once
    assert "[[C1]]" in prompt and "[[C2]]" in prompt
    assert "input 1" in prompt and "input 2" in prompt and "0.90" in prompt and "0.20" in prompt


def test_generate_group_one_request_when_all_blocks_pass():
    a, b = _sib(1, "text_search", "Opening", 0.9), _sib(2, "text_search", "Canton", 0.2)
    reply = (f"[[C1]]\n{_rationale('text_search', 'e1')}\n"
             f"[[C2]]\n{_rationale('text_search', 'e1', verdict='poor')}")
    gen = ScriptedGenerator([reply])
    out = RationaleGenerator(backend="api", generator=gen).generate_group([a, b])
    assert len(gen.prompts) == 1
    assert [o["rationale_mode"] for o in out] == ["group", "group"]
    assert all(o["rationale_qc_pass"] and o["rationale_attempts"] == 1 for o in out)
    assert [o["rationale_verdict"] for o in out] == ["good", "poor"]


def test_generate_group_falls_back_for_missing_or_leaking_blocks():
    a, b, c = (_sib(1, "text_search", "Opening Day", 0.9), _sib(2, "text_search", "Canton", 0.2),
               _sib(3, "image_search", "Stadium", 0.5))
    leaking = _rationale("text_search", "e1", extra="Unlike Opening Day results,", verdict="poor")
    reply = f"[[C1]]\n{_rationale('text_search', 'e1')}\n[[C2]]\n{leaking}"   # C3 missing
    singles = [_rationale("text_search", "e1", verdict="poor"),
               _rationale("image_search", "e1", verdict="mixed")]
    gen = ScriptedGenerator([reply] + singles)
    out = RationaleGenerator(backend="api", generator=gen).generate_group([a, b, c])
    assert len(gen.prompts) == 3                              # 1 group + 2 fallbacks
    assert [o["rationale_mode"] for o in out] == ["group", "group_fallback", "group_fallback"]
    assert out[1]["rationale_attempts"] == 2 and all(o["rationale_qc_pass"] for o in out)


def test_sibling_leak_rule():
    a, b = _sib(1, "text_search", "Opening Day", 0.9), _sib(2, "text_search", "Canton", 0.2)
    body = "This text_search step found nothing useful, while Opening Day pages did [e1]."
    assert sibling_leak(body, b, [a, b]) == "Opening Day"
    assert sibling_leak(body, a, [a, b]) is None               # its own evidence is fine
    ok, reasons = check_rationale(_rationale("text_search", "e1", extra="Opening Day"), b, [a, b])
    assert not ok and any("another candidate" in r for r in reasons)


def test_apply_provider_swaps_endpoint_keys_only():
    cfg = {"rationale": {"backend": "api", "generation": {
        "provider": "openai", "model": "deepseek-v4-pro", "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY", "extra_body": {"thinking": {"type": "disabled"}},
        "system_prompt": "SYS", "temperature": 0.3}}}
    out = apply_provider(cfg, "rationale", "kimi_code")
    gen = out["rationale"]["generation"]
    assert gen["api_key_env"] == "KIMI_CODE_API_KEY" and "extra_body" not in gen
    assert gen["system_prompt"] == "SYS" and gen["temperature"] == 0.3
    assert out["rationale"]["provider_preset"] == "kimi_code"
    assert cfg["rationale"]["generation"]["model"] == "deepseek-v4-pro"   # input untouched
    judged = apply_provider({}, "judge", "deepseek")
    assert judged["verifier"]["support"]["generation"]["extra_body"]["thinking"]["type"] == "disabled"
    assert apply_provider(cfg, "rationale", None) is cfg
    with pytest.raises(FileNotFoundError, match="available"):
        apply_provider(cfg, "rationale", "no_such_provider")
    assert {"kimi_code", "kimi_code_anthropic", "moonshot", "deepseek",
            "local_qwen3_coder", "local_qwen3_instruct"} <= set(provider_names())
