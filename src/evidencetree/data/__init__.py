"""Dataset-level utilities shared by data preparation and construction."""

from .splits import SplitResult, SplitSpec, read_ids, select_splits, stable_rank, write_ids

__all__ = [
    "SplitResult",
    "SplitSpec",
    "read_ids",
    "select_splits",
    "stable_rank",
    "write_ids",
]
