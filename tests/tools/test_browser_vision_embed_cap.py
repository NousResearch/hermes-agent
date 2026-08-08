"""Regression tests for the embed-time byte/pixel cap in ``browser_vision``.

When the active main model supports native vision, ``browser_vision`` attaches
the screenshot directly to the conversation as a multimodal tool result. That
image is baked into immutable history and re-sent on every subsequent turn, so
Anthropic's per-image limits (5 MB base64 OR 8000px per side) apply to it
permanently — an oversized embed wedges the session with a non-retryable 400.

A full-page screenshot can be tall and low-byte (e.g. 1200×12000 at 0.06 MB):
it passes any byte check but trips the pixel ceiling. ``browser_vision`` must
mirror ``_vision_analyze_native`` and proactively resize down to the embed
target (4 MB / 7900px) BEFORE embedding, on EITHER limit.

These tests fail without the proactive cap and skip when Pillow is unavailable
(the dimension resize is a no-op without it — the documented soft-dep posture).
"""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest

from tools.browser_tool import browser_vision
from tools.vision_tools import _EMBED_MAX_DIMENSION, _EMBED_TARGET_BYTES


def _decode_data_url(data_url: str) -> bytes:
    assert data_url.startswith("data:image/"), data_url[:40]
    return base64.b64decode(data_url.split(",", 1)[1])


def _run_browser_vision_native(screenshot_path):
    """Invoke browser_vision forcing the native fast path with a fixed screenshot."""
    def _fake_run_browser_command(*args, **kwargs):
        return {"success": True, "data": {"path": str(screenshot_path)}}

    with patch("tools.browser_tool._is_camofox_mode", return_value=False), \
            patch("tools.browser_tool._get_browser_engine", return_value="chrome"), \
            patch("tools.browser_tool._run_browser_command",
                  side_effect=_fake_run_browser_command), \
            patch("tools.vision_tools._should_use_native_vision_fast_path",
                  return_value=True):
        return browser_vision("describe what you see")


def _extract_embedded_url(result) -> str:
    assert isinstance(result, dict), f"expected multimodal envelope, got {type(result).__name__}: {str(result)[:200]}"
    assert result.get("_multimodal") is True
    return next(
        p["image_url"]["url"]
        for p in result["content"]
        if p.get("type") == "image_url"
    )


def test_tall_low_byte_screenshot_resized_under_pixel_cap(tmp_path):
    """A tall, small-byte full-page screenshot must be downscaled under 8000px.

    This is the parity gap: the byte check alone lets a 200×9000 PNG (a few KB)
    slip into immutable history at 9000px, and Anthropic rejects it with a
    non-retryable 400 that bricks the session.
    """
    Image = pytest.importorskip("PIL.Image")

    shot = tmp_path / "tall.png"
    Image.new("RGB", (200, 9000), "white").save(shot, format="PNG")
    # Sanity: low byte size but over the pixel ceiling — only the pixel arm fires.
    assert shot.stat().st_size * 4 // 3 < _EMBED_TARGET_BYTES, "test image unexpectedly large in bytes"
    assert max(Image.open(shot).size) > _EMBED_MAX_DIMENSION

    result = _run_browser_vision_native(shot)
    url = _extract_embedded_url(result)

    with Image.open(io.BytesIO(_decode_data_url(url))) as embedded:
        assert max(embedded.size) <= _EMBED_MAX_DIMENSION, (
            f"embedded screenshot is {embedded.size} — longest side exceeds the "
            f"{_EMBED_MAX_DIMENSION}px embed cap and would wedge the session on Anthropic"
        )


def test_oversized_byte_screenshot_resized_under_embed_cap(tmp_path):
    """A heavy-byte screenshot must be downscaled under the 4 MB embed byte cap."""
    Image = pytest.importorskip("PIL.Image")

    shot = tmp_path / "heavy.png"
    # Noisy PNG that base64-encodes well over the embed byte cap (won't compress).
    Image.effect_noise((2600, 2600), 80).convert("RGB").save(shot, format="PNG")
    assert shot.stat().st_size * 4 // 3 > _EMBED_TARGET_BYTES, "test image not big enough in bytes"

    result = _run_browser_vision_native(shot)
    url = _extract_embedded_url(result)

    assert len(url) <= _EMBED_TARGET_BYTES, (
        f"embedded screenshot {len(url) / 1024 / 1024:.1f} MB exceeds the embed cap "
        f"{_EMBED_TARGET_BYTES / 1024 / 1024:.0f} MB — would wedge sessions on Anthropic"
    )


def test_small_screenshot_embedded_unchanged(tmp_path):
    """A small in-bounds screenshot is embedded as-is (no needless resize)."""
    Image = pytest.importorskip("PIL.Image")

    shot = tmp_path / "small.png"
    Image.new("RGB", (320, 240), "white").save(shot, format="PNG")
    original = shot.read_bytes()

    result = _run_browser_vision_native(shot)
    url = _extract_embedded_url(result)

    assert _decode_data_url(url) == original, "in-bounds screenshot should not be re-encoded"
