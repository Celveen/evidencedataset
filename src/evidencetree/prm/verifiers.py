"""Modality-specific grounding verifiers: g(action, state) -> float in [0, 1].

Placeholder — implemented in **Stage 2**.

    text_search  -> cross-encoder: query vs. original question alignment
    image_search -> CLIP / VL matching: image crop vs. question visual entities
    answer       -> not scored locally (decided by outcome)
    crop / zoom  -> detection / OCR (only if Stage 1 implements those actions)

All grounding scores are GRADED (continuous 0-1), never binary.
"""

from __future__ import annotations

_STAGE = "Stage 2 — Grounding Verifiers"


def grounding_score(action, state) -> float:  # pragma: no cover
    raise NotImplementedError(
        f"grounding_score is not implemented yet (planned for {_STAGE})."
    )
