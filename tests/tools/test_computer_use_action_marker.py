"""Tests for the computer_use self-localization action marker.

Codex-style self-localization: after an action (click / drag / ...), the
follow-up capture PNG has a marker drawn at the resolved landing point so the
model can *see* where its own action landed, instead of only seeing a tinted
cursor on the user's physical screen.

These tests exercise the pure compositing layer
(`tools.computer_use.action_marker`) — no cua-driver, no live screen. The
marker is drawn onto an in-memory PNG and the result is decoded back to
pixels to prove the marker is really there.
"""

from __future__ import annotations

import base64
import io

import pytest

from tools.computer_use.action_marker import (
    ActionMarker,
    overlay_action_marker,
    pillow_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_png(width: int = 200, height: int = 160) -> bytes:
    """A plain white PNG so any marker pixel is trivially detectable."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _decode(png_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _has_non_white_near(img, cx: int, cy: int, radius: int = 24) -> bool:
    """True if any non-white pixel exists within `radius` of (cx, cy)."""
    w, h = img.size
    px = img.load()
    for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            if px[x, y] != (255, 255, 255):
                return True
    return False


pytestmark = pytest.mark.skipif(
    not pillow_available(),
    reason="Pillow not installed — marker compositing degrades to a no-op",
)


# ---------------------------------------------------------------------------
# Click marker
# ---------------------------------------------------------------------------

def test_click_marker_draws_pixels_at_landing_point():
    base = _white_png()
    marker = ActionMarker(kind="click", point=(100, 80), label="click")

    out = overlay_action_marker(base, marker)

    assert out != base, "compositing should change the PNG bytes"
    img = _decode(out)
    # The crosshair passes exactly through the landing point.
    assert _has_non_white_near(img, 100, 80, radius=2), (
        "expected marker pixels at the exact click landing point"
    )


def test_click_marker_leaves_far_corner_untouched():
    base = _white_png()
    marker = ActionMarker(kind="click", point=(100, 80))

    out = overlay_action_marker(base, marker)
    img = _decode(out)

    # A corner far from the marker must stay white (marker doesn't blanket
    # the whole image / hide the UI).
    assert not _has_non_white_near(img, 4, 4, radius=3)


def test_click_marker_preserves_image_dimensions():
    base = _white_png(200, 160)
    out = overlay_action_marker(base, ActionMarker(kind="click", point=(50, 50)))
    img = _decode(out)
    assert img.size == (200, 160)


# ---------------------------------------------------------------------------
# Drag marker
# ---------------------------------------------------------------------------

def test_drag_marker_draws_along_the_path():
    base = _white_png()
    marker = ActionMarker(kind="drag", start=(20, 20), end=(180, 140), label="drag")

    out = overlay_action_marker(base, marker)
    img = _decode(out)

    # Endpoints marked.
    assert _has_non_white_near(img, 20, 20, radius=6), "drag start not marked"
    assert _has_non_white_near(img, 180, 140, radius=6), "drag end not marked"
    # A midpoint along the line is drawn (the connecting arrow shaft).
    assert _has_non_white_near(img, 100, 80, radius=6), "drag shaft not drawn"


# ---------------------------------------------------------------------------
# Graceful degradation & guards
# ---------------------------------------------------------------------------

def test_marker_with_no_coordinates_returns_original():
    base = _white_png()
    # A click marker with no point (e.g. element bounds were unresolvable)
    # must not raise and must return the image unchanged.
    out = overlay_action_marker(base, ActionMarker(kind="click", point=None))
    assert out == base


def test_out_of_bounds_point_does_not_raise():
    base = _white_png(100, 100)
    marker = ActionMarker(kind="click", point=(999, 999))
    # Should be a no-throw; drawing partially or not at all is fine.
    out = overlay_action_marker(base, marker)
    assert isinstance(out, (bytes, bytearray))


def test_invalid_png_bytes_returns_original():
    junk = b"not-a-real-png"
    out = overlay_action_marker(junk, ActionMarker(kind="click", point=(10, 10)))
    assert out == junk
