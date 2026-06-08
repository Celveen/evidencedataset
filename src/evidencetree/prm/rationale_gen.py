"""Rationale generation with a strong LLM (offline, one-time).

Placeholder — implemented in **Stage 3**.

Uses the configurable ``generation`` API backend (Claude / GPT-4o). The ONLY
sanctioned external API in the project; all retrieval stays offline.

Quality filter: each rationale must cite >=1 evidence_id, mention the action
type, and be 50-150 tokens.
"""

from __future__ import annotations

_STAGE = "Stage 3 — 训练数据生成（ETBench-Open）"


def generate_rationales(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"generate_rationales is not implemented yet (planned for {_STAGE})."
    )
