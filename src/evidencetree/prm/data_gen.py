"""Training-data step labelling.

Placeholder — implemented in **Stage 3**.

For each step, produce:
    * local grounding label  -> call Stage 2 verifiers
    * outcome label          -> TREE-LEVEL credit: success rate of all
                                trajectories passing through this node (Monte
                                Carlo). NOT trajectory-uniform final-reward
                                averaging (that is a common bug).
"""

from __future__ import annotations

_STAGE = "Stage 3 — 训练数据生成（ETBench-Open）"


def label_steps(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        f"label_steps is not implemented yet (planned for {_STAGE}). "
        "Remember: outcome label is tree-level credit, not trajectory-uniform."
    )
