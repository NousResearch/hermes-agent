"""Self-localization action markers for computer_use captures.

Codex-style self-localization: cua-driver draws a tinted agent cursor on the
*user's physical screen*, but that overlay never lands in the PNG the model
receives — so after a click/drag the model cannot see where its own action
went. Game loops, drags, and precise manipulation fall apart because the model
has no visual confirmation of its landing point.

This module composites a lightweight marker onto the *follow-up* capture PNG
(the one produced by ``capture_after=True``) at the resolved landing point of
the immediately preceding action:

  * click / double_click / right/middle click → a translucent ringed crosshair
    centered on the click point.
  * drag → a start ring, an end ring, and a connecting arrow shaft so the model
    sees the whole gesture.

Design constraints (mirrors AGENTS.md "narrow waist"):
  * Pure, dependency-light, and *lazy* about Pillow. Pillow is an optional
    dependency; when it is not importable the compositor degrades to a no-op
    (returns the original bytes unchanged) so the capture path never breaks.
  * The marker is drawn with a bright outline and a translucent fill so it is
    unmistakable without blanketing the UI underneath.
  * This is the *last action's landing point*, semantically distinct from the
    SOM overlay's numbered "next-click candidates" — the two coexist.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

Point = Tuple[int, int]


@dataclass
class ActionMarker:
    """Describes where the previous action landed, for compositing.

    Exactly one geometry is used depending on ``kind``:
      * kind="click" (and double/right/middle) → ``point`` is the landing pixel.
      * kind="drag" → ``start`` and ``end`` are the gesture endpoints.
    """

    kind: str = "click"
    point: Optional[Point] = None
    start: Optional[Point] = None
    end: Optional[Point] = None
    label: str = ""


# High-contrast red-orange; the RGBA alpha keeps the fill translucent so the
# underlying UI stays legible. Outline is fully opaque for a crisp edge.
_OUTLINE_RGBA = (255, 40, 40, 255)
_FILL_RGBA = (255, 60, 60, 70)
_SHAFT_RGBA = (255, 40, 40, 220)
_CLICK_RADIUS = 16
_DRAG_ENDPOINT_RADIUS = 11


def pillow_available() -> bool:
    """True iff Pillow can be imported. Used to gate compositing/tests."""
    try:
        import PIL  # noqa: F401

        return True
    except Exception:
        return False


def _has_geometry(marker: ActionMarker) -> bool:
    if marker.kind == "drag":
        return marker.start is not None and marker.end is not None
    return marker.point is not None


def overlay_action_marker(png_bytes: bytes, marker: ActionMarker) -> bytes:
    """Return ``png_bytes`` with ``marker`` drawn on top.

    Degrades gracefully to the original bytes when:
      * Pillow is unavailable,
      * the marker carries no usable coordinates,
      * or the input is not a decodable image.

    Never raises for a drawing problem — a broken marker must not sink the
    capture that carries it.
    """
    if not png_bytes or not _has_geometry(marker):
        return png_bytes
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # Pillow not installed — self-localization is a nice-to-have, so we
        # silently return the untouched screenshot.
        return png_bytes

    try:
        base = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        # Not a real/parseable image — hand back exactly what we got.
        return png_bytes

    try:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        if marker.kind == "drag":
            _draw_drag(draw, marker.start, marker.end)
        else:
            _draw_click(draw, marker.point)

        composited = Image.alpha_composite(base, overlay).convert("RGB")
        out = io.BytesIO()
        composited.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        # Any drawing/encoding failure → fall back to the original screenshot.
        return png_bytes


def _draw_click(draw, point: Optional[Point]) -> None:
    if point is None:
        return
    x, y = int(point[0]), int(point[1])
    r = _CLICK_RADIUS
    # Translucent filled ring.
    draw.ellipse([x - r, y - r, x + r, y + r], fill=_FILL_RGBA,
                 outline=_OUTLINE_RGBA, width=3)
    # Crosshair through the exact landing point.
    draw.line([x - r - 6, y, x + r + 6, y], fill=_OUTLINE_RGBA, width=2)
    draw.line([x, y - r - 6, x, y + r + 6], fill=_OUTLINE_RGBA, width=2)
    # Center dot so the precise pixel is unambiguous.
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=_OUTLINE_RGBA)


def _draw_drag(draw, start: Optional[Point], end: Optional[Point]) -> None:
    if start is None or end is None:
        return
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    re = _DRAG_ENDPOINT_RADIUS
    # Connecting shaft.
    draw.line([x0, y0, x1, y1], fill=_SHAFT_RGBA, width=3)
    _draw_arrow_head(draw, (x0, y0), (x1, y1))
    # Start ring (hollow) and end ring (filled) to encode direction.
    draw.ellipse([x0 - re, y0 - re, x0 + re, y0 + re],
                 outline=_OUTLINE_RGBA, width=3)
    draw.ellipse([x1 - re, y1 - re, x1 + re, y1 + re],
                 fill=_FILL_RGBA, outline=_OUTLINE_RGBA, width=3)


def _draw_arrow_head(draw, start: Point, end: Point) -> None:
    import math

    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    head_len = 14.0
    head_w = 8.0
    # Base of the arrow head, stepped back from the tip.
    bx, by = x1 - ux * head_len, y1 - uy * head_len
    # Perpendicular unit vector.
    px, py = -uy, ux
    left = (bx + px * head_w, by + py * head_w)
    right = (bx - px * head_w, by - py * head_w)
    draw.polygon([(x1, y1), left, right], fill=_OUTLINE_RGBA)
