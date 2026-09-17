"""Stitch multiple images into labeled contact sheets for vision.

Global rule: when a vision request involves multiple images, stitch them into
as few sheets as possible (one request per sheet) instead of one request per
image. Each sheet holds at most ``MAX_TILES_PER_SHEET`` (9 = 3x3) tiles with
``P1..Pn`` index labels burned in; larger batches split into multiple sheets.
When stitched detail is insufficient (tiny text/faces), callers split the
batch into 2+ sheets rather than falling back to per-image calls.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from hermes_constants import get_hermes_dir

MAX_TILES_PER_SHEET = 9
TILE_W = 768
TILE_H = 768


@dataclass
class StitchedSheet:
    """One contact sheet on disk plus the 1-based source labels it carries."""

    path: Path
    sources: list[str]
    labels: list[str]


def chunk_sources(
    sources: Sequence[str], max_tiles: int = MAX_TILES_PER_SHEET
) -> list[list[str]]:
    """Split ``sources`` into sheet-sized chunks (order preserved)."""
    max_tiles = max(1, int(max_tiles or MAX_TILES_PER_SHEET))
    return [list(sources[i : i + max_tiles]) for i in range(0, len(sources), max_tiles)]


def sheet_grid(n: int) -> tuple[int, int]:
    """Grid (cols, rows) for ``n`` tiles, capped at 3 columns for readability."""
    if n <= 0:
        return (0, 0)
    if n <= 3:
        return (n, 1)
    cols = 3
    rows = (n + cols - 1) // cols
    return (cols, rows)


def _require_pillow():
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ValueError(
            "Stitching multiple images requires Pillow (`pip install Pillow`); "
            "retry with a single image instead."
        ) from exc
    return Image, ImageDraw


async def stitch_images(
    sources: Sequence[str],
    task_id: Optional[str] = None,
    *,
    labels: bool = True,
    max_tiles: int = MAX_TILES_PER_SHEET,
    start_index: int = 1,
) -> list[StitchedSheet]:
    """Resolve ``sources`` and stitch into contact sheet PNGs.

    Returns one :class:`StitchedSheet` per chunk (single sheet for <=9
    inputs). Sheets live under the vision cache dir; the CALLER owns cleanup.
    Labels are global 1-based (``P<k>``) continuing from ``start_index`` so a
    split batch keeps stable references across sheets.
    """
    from io import BytesIO

    from tools.image_source import ResolveContext, resolve_image_source

    srcs = [s for s in (sources or []) if isinstance(s, str) and s.strip()]
    if not srcs:
        raise ValueError("No images to stitch.")
    Image, ImageDraw = _require_pillow()

    # Resolve all inputs first so a bad source fails before any sheet is written.
    decoded: list = []
    for src in srcs:
        resolved = await resolve_image_source(
            src.strip(), ResolveContext(task_id=task_id)
        )
        try:
            img = Image.open(BytesIO(resolved.data))
            img.load()
        except Exception as exc:
            raise ValueError(
                f"Could not decode image for stitching: {src[:120]} ({exc})"
            ) from exc
        if img.mode not in ("RGB", "L"):
            try:
                img = img.convert("RGB")
            except Exception as exc:
                raise ValueError(
                    f"Could not convert image for stitching: {src[:120]} ({exc})"
                ) from exc
        decoded.append((src, img))

    out_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets: list[StitchedSheet] = []
    index = int(start_index or 1)
    for chunk_idx, chunk in enumerate(
        chunk_sources([s for s, _ in decoded], max_tiles)
    ):
        members = decoded[
            chunk_idx * int(max_tiles or MAX_TILES_PER_SHEET) : chunk_idx
            * int(max_tiles or MAX_TILES_PER_SHEET)
            + len(chunk)
        ]
        cols, rows = sheet_grid(len(members))
        sheet = Image.new("RGB", (cols * TILE_W, rows * TILE_H), "white")
        draw = ImageDraw.Draw(sheet)
        sheet_labels: list[str] = []
        for pos, (src, img) in enumerate(members):
            label = f"P{index}"
            sheet_labels.append(label)
            index += 1
            tile = img.copy()
            tile.thumbnail((TILE_W, TILE_H))
            ox = (pos % cols) * TILE_W + (TILE_W - tile.width) // 2
            oy = (pos // cols) * TILE_H + (TILE_H - tile.height) // 2
            sheet.paste(tile, (ox, oy))
            if labels:
                lx, ly = (pos % cols) * TILE_W + 8, (pos // cols) * TILE_H + 8
                # High-contrast tag without external fonts.
                draw.rectangle([lx, ly, lx + 52, ly + 24], fill="black")
                draw.text((lx + 6, ly + 4), label, fill="white")
        out_path = out_dir / f"stitched_{uuid.uuid4().hex[:12]}.png"
        sheet.save(out_path, format="PNG")
        sheets.append(
            StitchedSheet(path=out_path, sources=list(chunk), labels=sheet_labels)
        )
    return sheets


def describe_sheets(sheets: Sequence[StitchedSheet]) -> str:
    """One-line provenance note mapping labels to sources (paths truncated)."""
    bits = []
    for sheet in sheets:
        for label, src in zip(sheet.labels, sheet.sources):
            bits.append(f"{label}={src[:120]}")
    return "; ".join(bits)
