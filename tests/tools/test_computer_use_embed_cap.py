"""Regression for #133596 — the native ``computer_use`` embed was uncapped.

``_capture_response`` embedded ``cap.png_b64`` directly in the multimodal
envelope: a high-resolution desktop capture (e.g. ~14 MB base64) exceeded the
provider's per-image ceiling (Anthropic 5 MB — "Image base64 size … exceeds
API limit") and every retry re-sent the identical oversized payload. The embed
now goes through the same proactive resize browser/native vision already apply
(``vision.embed_target_bytes`` + 1568px cap, JPEG re-encode when shrinking),
while the original persists untouched on disk at ``screenshot_path``.
"""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import patch

import pytest

from tools.computer_use.backend import CaptureResult, UIElement


def _png_b64(width: int, height: int) -> str:
    from PIL import Image

    out = BytesIO()
    Image.new("RGB", (width, height), color=(30, 60, 120)).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _make_capture(png_b64: str, width: int, height: int) -> CaptureResult:
    return CaptureResult(
        mode="som",
        width=width,
        height=height,
        png_b64=png_b64,
        elements=[UIElement(index=0, role="AXButton", label="Sign in", bounds=(10, 20, 80, 30))],
        app="Safari",
        window_title="GitHub – Issue #133596",
        png_bytes_len=len(base64.b64decode(png_b64, validate=False)),
    )


def _url_dimensions(data_url: str) -> tuple[int, int]:
    from PIL import Image

    raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
    with Image.open(BytesIO(raw)) as img:
        return img.size


@pytest.fixture
def tmp_hermes_home(tmp_path, monkeypatch):
    """Cache writes (screenshot persistence) land under tmp_path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants

    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    def _fake_get(subdir, _legacy=None):
        target = cache_root / subdir
        target.mkdir(parents=True, exist_ok=True)
        return target

    with patch.object(hermes_constants, "get_hermes_dir", _fake_get):
        yield tmp_path


class TestNativeEmbedCap:
    def test_oversized_capture_is_downscaled_in_envelope(self, tmp_hermes_home):
        """A capture wider than the 1568px embed cap is downscaled for the model while the
        persisted original keeps full resolution, and the scale lands in meta + summary."""
        from tools.computer_use import tool as cu_tool

        cap = _make_capture(_png_b64(3000, 1800), width=3000, height=1800)
        with patch.object(cu_tool, "_should_route_through_aux_vision", return_value=False):
            resp = cu_tool._capture_response(cap)

        assert resp.get("_multimodal") is True
        image_part = next(p for p in resp["content"] if p.get("type") == "image_url")
        new_w, new_h = _url_dimensions(image_part["image_url"]["url"])
        assert max(new_w, new_h) <= 1568
        scale = resp["meta"]["image_scale"]
        assert (scale["orig_width"], scale["orig_height"]) == (3000, 1800)
        assert (scale["new_width"], scale["new_height"]) == (new_w, new_h)
        assert "downscaled" in resp["content"][0]["text"]
        # The persisted original is untouched: full resolution on disk.
        from PIL import Image

        with Image.open(resp["meta"]["screenshot_path"]) as persisted:
            assert persisted.size == (3000, 1800)

    def test_small_capture_embedded_unchanged(self, tmp_hermes_home):
        """Under both caps the embed is the original PNG with no scale meta or note."""
        from tools.computer_use import tool as cu_tool

        b64 = _png_b64(400, 300)
        cap = _make_capture(b64, width=400, height=300)
        with patch.object(cu_tool, "_should_route_through_aux_vision", return_value=False):
            resp = cu_tool._capture_response(cap)

        image_part = next(p for p in resp["content"] if p.get("type") == "image_url")
        assert image_part["image_url"]["url"] == f"data:image/png;base64,{b64}"
        assert "image_scale" not in resp["meta"]
        assert "downscaled" not in resp["content"][0]["text"]

    def test_resize_failure_fails_open_to_raw_bytes(self, tmp_hermes_home):
        """A resize-path failure must not drop the screenshot: the raw base64 is embedded
        exactly as before the fix."""
        from tools.computer_use import tool as cu_tool

        b64 = _png_b64(3000, 1800)
        cap = _make_capture(b64, width=3000, height=1800)
        with patch.object(cu_tool, "_should_route_through_aux_vision", return_value=False), \
             patch("tools.vision_tools._resize_image_for_vision", side_effect=RuntimeError("boom")):
            resp = cu_tool._capture_response(cap)

        image_part = next(p for p in resp["content"] if p.get("type") == "image_url")
        assert image_part["image_url"]["url"] == f"data:image/png;base64,{b64}"
        assert "image_scale" not in resp["meta"]
