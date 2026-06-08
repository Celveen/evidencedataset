"""Retrieval action space definitions.

Placeholder — implemented in **Stage 1**.

v1.3 minimal action set (implement these three first):
    text_search(query)     — text retrieval                [required]
    image_search(img_crop) — image retrieval               [required]
    answer(text)           — terminal action               [required]

Deferred (decided by Stage 0.4 statistics):
    crop(region) / zoom(region, factor)  -> or merged focus(region)

Do NOT add parallel_search: tree expansion already covers multi-candidate
exploration, and its backup semantics would conflict with tree-level credit.
The design must stay extensible so adding crop/zoom/focus later needs no rewrite.
"""

from __future__ import annotations

_STAGE = "Stage 1 — 检索动作空间与执行器"


def __getattr__(name: str):  # pragma: no cover - guard for premature use
    raise NotImplementedError(
        f"action_space.{name} is not implemented yet (planned for {_STAGE})."
    )
