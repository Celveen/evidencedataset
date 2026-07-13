"""Helpers for model-proposed image regions."""

from __future__ import annotations

import math
from typing import Sequence


def safe_crop_box(
    image_size: tuple[int, int],
    region: Sequence[float] | None,
) -> tuple[int, int, int, int] | None:
    """Convert a model-proposed region into a safe PIL crop box.

    The action space asks models for normalized boxes in ``[0, 1]``. In
    practice VLMs sometimes return pixel coordinates or slightly out-of-range
    values. This helper accepts both normalized and pixel-like boxes, clamps
    them to the image boundary, and returns ``None`` for unusable boxes.
    """

    if region is None or len(region) != 4:
        return None

    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    try:
        x1, y1, x2, y2 = (float(value) for value in region)
    except (TypeError, ValueError):
        return None

    values = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values):
        return None

    if max(abs(value) for value in values) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = max(0, min(width, int(round(left))))
    right = max(0, min(width, int(round(right))))
    top = max(0, min(height, int(round(top))))
    bottom = max(0, min(height, int(round(bottom))))

    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right, bottom
