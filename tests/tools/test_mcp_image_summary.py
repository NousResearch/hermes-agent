"""Tests for the MCP ImageContent → auxiliary vision summary bridge.

The bridge (``_summarize_mcp_image`` in ``tools/mcp_tool.py``) attaches a
best-effort text summary to every ``MEDIA:<path>`` tag produced from an MCP
``ImageContent`` block, so pure-text main models can "see" the image through
their auxiliary vision model. It must be fail-open: no misconfiguration,
timeout, or vision failure may ever break the original MEDIA tag.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _png_bytes() -> bytes:
    """Minimal valid 1x1 PNG (accepted by the image-cache format sniff)."""
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )


def _media_tag(tmp_path) -> str:
    """Cache a real PNG through the same helper the MCP pipeline uses."""
    import tools.mcp_tool as mt

    block = SimpleNamespace(
        data=base64.b64encode(_png_bytes()).decode("ascii"),
        mimeType="image/png",
    )
    tag = mt._cache_mcp_image_block(block)
    assert tag.startswith("MEDIA:"), "test setup: image must cache"
    return tag


def _vision_cfg(**overrides) -> dict:
    cfg = {"provider": "custom:mlx", "model": "/local/Mage-VL-8bit"}
    cfg.update(overrides)
    return {"auxiliary": {"vision": cfg}}


@pytest.fixture(autouse=True)
def _clean_cache():
    import tools.mcp_tool as mt

    mt._MCP_IMAGE_SUMMARY_CACHE.clear()
    yield
    mt._MCP_IMAGE_SUMMARY_CACHE.clear()


@pytest.mark.asyncio
async def test_no_vision_config_is_noop(tmp_path, monkeypatch):
    """Without auxiliary.vision configured the bridge returns "" and never
    calls the vision tool — a zero-footprint no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)
    fake = AsyncMock()
    with patch("hermes_cli.config.load_config", return_value={}), patch(
        "tools.vision_tools.vision_analyze_tool", fake
    ):
        assert await mt._summarize_mcp_image(tag) == ""
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_opt_out_flag_disables_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)
    fake = AsyncMock()
    with patch(
        "hermes_cli.config.load_config",
        return_value=_vision_cfg(summarize_mcp_images=False),
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mt._summarize_mcp_image(tag) == ""
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_summary_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)
    fake = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "A red pixel."})
    )
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        summary = await mt._summarize_mcp_image(tag)
    assert summary == "[图片内容摘要] A red pixel."
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_cache_prevents_reanalysis(tmp_path, monkeypatch):
    """The same cached image path is summarized at most once per process."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)
    fake = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "first"})
    )
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        r1 = await mt._summarize_mcp_image(tag)
        r2 = await mt._summarize_mcp_image(tag)
    assert r1 == r2 == "[图片内容摘要] first"
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_vision_failure_is_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)

    # success=False
    fake = AsyncMock(return_value=json.dumps({"success": False, "analysis": "err"}))
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mt._summarize_mcp_image(tag) == ""

    # malformed JSON
    fake = AsyncMock(return_value="not json")
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mt._summarize_mcp_image(tag) == ""

    # exception
    fake = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mt._summarize_mcp_image(tag) == ""


@pytest.mark.asyncio
async def test_vision_timeout_is_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    tag = _media_tag(tmp_path)

    async def _hang(*a, **k):
        await asyncio.sleep(30)
        return "never"

    fake = AsyncMock(side_effect=_hang)
    with patch(
        "hermes_cli.config.load_config",
        return_value=_vision_cfg(mcp_summary_timeout=0.2),
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mt._summarize_mcp_image(tag) == ""


@pytest.mark.asyncio
async def test_non_media_or_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool as mt

    assert await mt._summarize_mcp_image("not-a-media-tag") == ""
    assert await mt._summarize_mcp_image("MEDIA:/no/such/file.png") == ""
