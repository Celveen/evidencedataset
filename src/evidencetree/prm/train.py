"""Three-stage PRM training from VisualPRM-8B.

Placeholder — implemented in **Stage 4**.

    4.1 SFT             learn the 4-tuple input format + generative rationale+score
    4.2 Action-typed DPO mine same-state/different-action pairs, learn preferences
    4.3 Calibration      Platt / temperature scaling of score logit -> reward

Key: train GENERATIVE (rationale + score); infer SCORE-ONLY (read score-token
logit, do not generate rationale text). Use LoRA to fit in memory.
"""

from __future__ import annotations

_STAGE = "Stage 4 — PRM 三阶段训练"


def train(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"PRM training is not implemented yet (planned for {_STAGE})."
    )
