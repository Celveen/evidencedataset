"""Deterministic stand-in CLIP encoders for mock runs (no model downloads).

Both towers project into ONE 128-dim space, so cross-modal similarity is real
and predictable:

* image tower — a 64-bin color histogram (4 levels per RGB channel);
* text tower  — color words from :data:`MOCK_COLORS` hit the SAME color bins,
  every other token is hashed into dims 64..127 (bag of words).

The mock benchmark (``evidencetree.eval.benchmarks``) paints each entity in
one of these colors, so ``image_search`` identifies an entity by color and a
text query naming the color can retrieve the entity's image cross-modally —
the same mechanics the real CLIP index provides, at unit-test cost.
"""

from __future__ import annotations

import re
import zlib
from typing import Sequence

import numpy as np

DIM = 128
_COLOR_DIMS = 64
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# name -> RGB, every entry in a distinct 4-level color bin (never white/black).
MOCK_COLORS: dict[str, tuple[int, int, int]] = {
    "crimson": (255, 0, 0),
    "lime": (0, 255, 0),
    "navy": (0, 0, 255),
    "amber": (255, 170, 0),
    "violet": (170, 0, 255),
    "teal": (0, 170, 170),
    "olive": (85, 85, 0),
    "maroon": (85, 0, 0),
    "coral": (255, 85, 85),
    "indigo": (85, 0, 170),
    "ochre": (170, 85, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "jade": (0, 170, 85),
    "rust": (170, 0, 0),
    "plum": (170, 85, 170),
    "sand": (255, 255, 85),
    "mint": (85, 255, 170),
    "slate": (85, 85, 170),
    "rose": (255, 85, 170),
}


def color_bin(rgb: Sequence[float]) -> int:
    """4 levels per channel -> one of 64 bins."""
    r, g, b = (min(3, int(v) // 64) for v in rgb[:3])
    return r * 16 + g * 4 + b


_WHITE_BIN = color_bin((255, 255, 255))


def mock_image_encoder(images) -> np.ndarray:
    """64-bin color histogram (white background de-emphasized)."""
    vecs = np.zeros((len(images), DIM), dtype="float32")
    for i, image in enumerate(images):
        pixels = np.asarray(image.convert("RGB").resize((16, 16)), dtype="float32")
        for px in pixels.reshape(-1, 3):
            vecs[i, color_bin(px)] += 1.0
        vecs[i, _WHITE_BIN] *= 0.1
    return vecs


def mock_text_encoder(texts) -> np.ndarray:
    """Color words hit the image tower's bins; other tokens hash to 64..127."""
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for tok in _TOKEN_RE.findall(str(text).lower()):
            if tok in MOCK_COLORS:
                vecs[i, color_bin(MOCK_COLORS[tok])] += 3.0
            vecs[i, _COLOR_DIMS + zlib.crc32(tok.encode()) % (DIM - _COLOR_DIMS)] += 1.0
    return vecs
