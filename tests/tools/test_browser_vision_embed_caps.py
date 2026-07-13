"""browser_vision must apply the proactive embed caps on the native fast path.

The native fast path bakes the screenshot into conversation history, where it
is re-sent on every subsequent turn. vision_analyze has capped its embeds
since the wedged-session incident (Anthropic rejects >5 MB / >8000 px per side
with a non-retryable 400), but browser_vision embedded full-page screenshots
at full resolution — arbitrarily tall pages OOM local vision models during
prefill (observed: repeated Metal kIOGPUCommandBufferCallbackErrorOutOfMemory
on an MLX VLM) and wedge Anthropic sessions.
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised on minimal CI images
    Image = None

from tools.vision_tools import _EMBED_TARGET_BYTES


# Minimal valid 1x1 PNG bytes.
_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _run_browser_vision_with_screenshot(png_writer):
    """Drive the real browser_vision with the browser daemon faked out.

    ``png_writer(path)`` writes the test PNG to the screenshot path the tool
    chose; the fake _run_browser_command reports success with that path, so
    everything downstream (encode, embed caps, envelope build) is real code.
    """
    from tools import browser_tool

    def _fake_run_browser_command(task_id, command, args, **kwargs):
        assert command == "screenshot"
        shot_path = args[-1]
        png_writer(shot_path)
        return {"success": True, "data": {"path": shot_path}}

    with (
        patch.object(browser_tool, "_is_camofox_mode", return_value=False),
        patch.object(browser_tool, "_is_local_backend", return_value=True),
        patch.object(browser_tool, "_get_browser_engine", return_value="chrome"),
        patch.object(
            browser_tool, "_run_browser_command",
            side_effect=_fake_run_browser_command,
        ),
        patch(
            "tools.vision_tools._should_use_native_vision_fast_path",
            return_value=True,
        ),
    ):
        return browser_tool.browser_vision("what is on the page?")


@pytest.mark.skipif(Image is None, reason="Pillow not installed — resize is a no-op")
def test_oversized_screenshot_embeds_under_cap(tmp_path):
    def _write_big(path):
        Image.effect_noise((2600, 2600), 80).convert("RGB").save(path, format="PNG")

    result = _run_browser_vision_with_screenshot(_write_big)

    assert isinstance(result, dict), f"expected multimodal envelope, got: {str(result)[:200]}"
    assert result.get("_multimodal") is True
    url = next(
        p["image_url"]["url"] for p in result["content"] if p.get("type") == "image_url"
    )
    assert len(url) <= _EMBED_TARGET_BYTES, (
        f"embedded screenshot {len(url) / 1024 / 1024:.1f} MB exceeds embed cap "
        f"{_EMBED_TARGET_BYTES / 1024 / 1024:.0f} MB — re-sent every turn, this "
        f"wedges Anthropic sessions and OOMs local vision models"
    )
    assert result.get("meta", {}).get("screenshot_path")


def test_small_screenshot_embeds_at_full_resolution(tmp_path):
    """Guard against over-eager capping: an under-cap screenshot must embed
    as its full-resolution encoding, byte for byte."""

    def _write_tiny(path):
        with open(path, "wb") as fh:
            fh.write(_TINY_PNG)

    result = _run_browser_vision_with_screenshot(_write_tiny)

    assert isinstance(result, dict), f"expected multimodal envelope, got: {str(result)[:200]}"
    assert result.get("_multimodal") is True
    url = next(
        p["image_url"]["url"] for p in result["content"] if p.get("type") == "image_url"
    )
    expected_b64 = base64.b64encode(_TINY_PNG).decode("ascii")
    assert url == f"data:image/png;base64,{expected_b64}"
