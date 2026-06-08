"""Action executor: given (action, state) -> updated evidence_bundle.

Placeholder — implemented in **Stage 1**.

The executor runs an action via the retrievers and accumulates results into an
``evidence_bundle`` that grows monotonically across a trajectory (search A, then
search B -> bundle contains both).
"""

from __future__ import annotations

_STAGE = "Stage 1 — 检索动作空间与执行器"


class ActionExecutor:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"ActionExecutor is not implemented yet (planned for {_STAGE})."
        )
