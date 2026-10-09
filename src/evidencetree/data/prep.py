"""Helpers shared by the data-preparation scripts and the local fetch kit.

Only the standard library and Pillow are used, so the fetch kit can run on a
plain laptop Python.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Pack paragraphs greedily into chunks of at most ~``max_chars``.

    A paragraph longer than ``max_chars`` is split at sentence ends, and a
    sentence longer than that is hard-wrapped at whitespace.
    """
    pieces: list[str] = []
    for para in (p.strip() for p in str(text).split("\n") if p.strip()):
        if len(para) <= max_chars:
            pieces.append(para)
            continue
        for sent in _SENTENCE_END_RE.split(para):
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                pieces.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                pieces.append(sent)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def save_image(source, dest: str | Path, max_side: int = 768, quality: int = 90) -> Path:
    """Re-encode an image (bytes, path or file object) as RGB JPEG whose longer
    side is at most ``max_side``. Writes atomically; returns ``dest``."""
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as img:
        img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        tmp = dest.with_name(dest.name + ".part")
        img.save(tmp, "JPEG", quality=quality)
    tmp.replace(dest)
    return dest


def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()
