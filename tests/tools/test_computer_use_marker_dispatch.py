"""E2E test for computer_use self-localization marker through the dispatch path.

Exercises `tools.computer_use.tool._dispatch` end-to-end: a click with
`capture_after=True` must return a multimodal envelope whose screenshot has a
marker composited at the click's landing point. The image is decoded back to
pixels to prove the marker really lands where the action did — not mocked.
"""

from __future__ import annotations

import base64
import io

import pytest

from unittest.mock import patch

from tools.computer_use.action_marker import ActionMarker, pillow_available
from tools.computer_use.backend import CaptureResult


def _white_png_b64(w: int = 200, h: int = 160) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _StubBackend:
    """Minimal backend that records a click marker and returns a white PNG.

    Mirrors the contract the tool layer depends on: click() records a pending
    marker, capture() returns a plain screenshot, consume_action_marker()
    hands the marker over one-shot.
    """

    def __init__(self):
        self._last_app = "StubApp"
        self._pending = None

    def click(self, *, element=None, x=None, y=None, button="left",
              click_count=1, modifiers=None):
        from tools.computer_use.backend import ActionResult

        self._pending = ActionMarker(kind="click", point=(x, y), label="click")
        return ActionResult(ok=True, action="click", message="clicked")

    def capture(self, mode="som", app=None):
        return CaptureResult(
            mode=mode, width=200, height=160,
            png_b64=_white_png_b64(), elements=[], app="StubApp",
            window_title="w", png_bytes_len=100, image_mime_type="image/png",
        )

    def consume_action_marker(self):
        m, self._pending = self._pending, None
        return m


pytestmark = pytest.mark.skipif(
    not pillow_available(), reason="Pillow not installed"
)


@pytest.fixture(autouse=True)
def _no_aux_vision_routing():
    """Keep the multimodal envelope: don't reroute the screenshot to aux vision.

    The follow-up capture returns a native multimodal image envelope only when
    aux-vision routing is off. In CI/dev the ambient config may enable it,
    which would flatten the image into text and hide the marker pixels we're
    asserting on.
    """
    with patch(
        "tools.computer_use.tool._should_route_through_aux_vision",
        return_value=False,
    ):
        yield


def _decode(b64: str):
    from PIL import Image

    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _non_white_near(img, cx, cy, radius=3) -> bool:
    w, h = img.size
    px = img.load()
    for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
        for y in range(max(0, cy - radius), min(h, cy + radius + 1)):
            if px[x, y] != (255, 255, 255):
                return True
    return False


def test_click_capture_after_marks_landing_point():
    from tools.computer_use.tool import _dispatch

    backend = _StubBackend()
    resp = _dispatch(backend, "click", {
        "coordinate": [120, 90],
        "capture_after": True,
    })

    assert isinstance(resp, dict) and resp.get("_multimodal")
    url = resp["content"][1]["image_url"]["url"]
    b64 = url.split("base64,", 1)[1]
    img = _decode(b64)
    # Marker crosshair passes through the exact click point.
    assert _non_white_near(img, 120, 90, radius=2), "no marker at click point"
    # Far corner untouched.
    assert not _non_white_near(img, 3, 3, radius=2)


def test_show_action_marker_false_leaves_screenshot_clean():
    from tools.computer_use.tool import _dispatch

    backend = _StubBackend()
    resp = _dispatch(backend, "click", {
        "coordinate": [120, 90],
        "capture_after": True,
        "show_action_marker": False,
    })

    assert isinstance(resp, dict) and resp.get("_multimodal")
    url = resp["content"][1]["image_url"]["url"]
    b64 = url.split("base64,", 1)[1]
    img = _decode(b64)
    # No marker → the whole white image stays white.
    assert not _non_white_near(img, 120, 90, radius=30)


def test_marker_is_one_shot_not_restamped():
    from tools.computer_use.tool import _dispatch

    backend = _StubBackend()
    # First click stamps a marker and consumes it.
    _dispatch(backend, "click", {"coordinate": [50, 50], "capture_after": True})
    # A plain capture afterwards must NOT carry a stale marker.
    assert backend.consume_action_marker() is None
