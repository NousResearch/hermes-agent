"""Tests for the MCP ImageContent → auxiliary vision summary bridge.

The bridge (``_summarize_mcp_image`` in ``tools/mcp_tool_content.py``) attaches a
best-effort text summary to every ``MEDIA:<path>`` tag produced from an MCP
``ImageContent`` block, so pure-text main models can "see" the image through
their auxiliary vision model. It must be fail-open: no misconfiguration,
timeout, or vision failure may ever break the original MEDIA tag.

Rebased onto the post-refactor module layout: the image-cache helpers and the
summary bridge now live in ``tools/mcp_tool_content.py`` (the monolithic
``tools/mcp_tool.py`` was decomposed), and the summary is wired into
``_render_content_blocks`` in ``tools/mcp_tool_handlers.py``.
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
    import tools.mcp_tool_content as mtc

    block = SimpleNamespace(
        data=base64.b64encode(_png_bytes()).decode("ascii"),
        mimeType="image/png",
    )
    tag = mtc._cache_mcp_image_block(block)
    assert tag.startswith("MEDIA:"), "test setup: image must cache"
    return tag


def _vision_cfg(**overrides) -> dict:
    cfg = {"provider": "custom:mlx", "model": "/local/Mage-VL-8bit"}
    cfg.update(overrides)
    return {"auxiliary": {"vision": cfg}}


@pytest.fixture(autouse=True)
def _clean_cache():
    import tools.mcp_tool_content as mtc

    mtc._MCP_IMAGE_SUMMARY_CACHE.clear()
    yield
    mtc._MCP_IMAGE_SUMMARY_CACHE.clear()


@pytest.mark.asyncio
async def test_no_vision_config_is_noop(tmp_path, monkeypatch):
    """Without auxiliary.vision configured the bridge returns "" and never
    calls the vision tool — a zero-footprint no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)
    fake = AsyncMock()
    with patch("hermes_cli.config.load_config", return_value={}), patch(
        "tools.vision_tools.vision_analyze_tool", fake
    ):
        assert await mtc._summarize_mcp_image(tag) == ""
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_opt_out_flag_disables_bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)
    fake = AsyncMock()
    with patch(
        "hermes_cli.config.load_config",
        return_value=_vision_cfg(summarize_mcp_images=False),
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mtc._summarize_mcp_image(tag) == ""
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_returns_summary_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)
    fake = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "A red pixel."})
    )
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        summary = await mtc._summarize_mcp_image(tag)
    assert summary == "[图片内容摘要] A red pixel."
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_cache_prevents_reanalysis(tmp_path, monkeypatch):
    """The same cached image path is summarized at most once per process."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)
    fake = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "first"})
    )
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        r1 = await mtc._summarize_mcp_image(tag)
        r2 = await mtc._summarize_mcp_image(tag)
    assert r1 == r2 == "[图片内容摘要] first"
    fake.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_is_bounded_lru(tmp_path, monkeypatch):
    """The summary cache is a bounded LRU: once the cap is hit the
    least-recently-used entry is evicted, and an evicted image is re-analyzed
    on its next appearance instead of growing the cache without bound.

    Regression for the review finding that ``_MCP_IMAGE_SUMMARY_CACHE`` was an
    unbounded ``dict`` that grew monotonically over a long-lived gateway
    process lifetime.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    old_max = mtc._MCP_IMAGE_SUMMARY_CACHE_MAX
    mtc._MCP_IMAGE_SUMMARY_CACHE_MAX = 2
    try:
        tag_a = _media_tag(tmp_path)
        tag_b = _media_tag(tmp_path)
        tag_c = _media_tag(tmp_path)
        assert tag_a != tag_b != tag_c

        fake = AsyncMock(
            side_effect=[
                json.dumps({"success": True, "analysis": "A"}),
                json.dumps({"success": True, "analysis": "B"}),
                json.dumps({"success": True, "analysis": "C"}),
                json.dumps({"success": True, "analysis": "A-again"}),
            ]
        )
        with patch(
            "hermes_cli.config.load_config", return_value=_vision_cfg()
        ), patch("tools.vision_tools.vision_analyze_tool", fake):
            assert await mtc._summarize_mcp_image(tag_a) == "[图片内容摘要] A"
            assert await mtc._summarize_mcp_image(tag_b) == "[图片内容摘要] B"
            # A is now the least-recently-used entry (cap=2)...
            assert await mtc._summarize_mcp_image(tag_c) == "[图片内容摘要] C"
            # ...so A was evicted and is re-analyzed, while B stays cached.
            assert await mtc._summarize_mcp_image(tag_a) == "[图片内容摘要] A-again"
        assert fake.await_count == 4
        assert len(mtc._MCP_IMAGE_SUMMARY_CACHE) == 2
        # Cache keys are the bare image paths (MEDIA: prefix stripped).
        assert tag_a[len("MEDIA:"):] in mtc._MCP_IMAGE_SUMMARY_CACHE
        assert tag_c[len("MEDIA:"):] in mtc._MCP_IMAGE_SUMMARY_CACHE
        assert tag_b[len("MEDIA:"):] not in mtc._MCP_IMAGE_SUMMARY_CACHE
    finally:
        mtc._MCP_IMAGE_SUMMARY_CACHE_MAX = old_max
        mtc._MCP_IMAGE_SUMMARY_CACHE.clear()


@pytest.mark.asyncio
async def test_vision_failure_is_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)

    # success=False
    fake = AsyncMock(return_value=json.dumps({"success": False, "analysis": "err"}))
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mtc._summarize_mcp_image(tag) == ""

    # malformed JSON
    fake = AsyncMock(return_value="not json")
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mtc._summarize_mcp_image(tag) == ""

    # exception
    fake = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mtc._summarize_mcp_image(tag) == ""


@pytest.mark.asyncio
async def test_vision_timeout_is_fail_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    tag = _media_tag(tmp_path)

    async def _hang(*a, **k):
        await asyncio.sleep(30)
        return "never"

    fake = AsyncMock(side_effect=_hang)
    with patch(
        "hermes_cli.config.load_config",
        return_value=_vision_cfg(mcp_summary_timeout=0.2),
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        assert await mtc._summarize_mcp_image(tag) == ""


@pytest.mark.asyncio
async def test_non_media_or_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.mcp_tool_content as mtc

    assert await mtc._summarize_mcp_image("not-a-media-tag") == ""
    assert await mtc._summarize_mcp_image("MEDIA:/no/such/file.png") == ""


# ---------------------------------------------------------------------------
# Wiring tests: the summary must actually reach the model through the handler
# render path (rebase addition — the bridge is invoked from
# ``_render_content_blocks`` in tools/mcp_tool_handlers.py).
# ---------------------------------------------------------------------------


def _tool_result(blocks):
    result = SimpleNamespace(content=blocks)
    return result


@pytest.mark.asyncio
async def test_render_content_blocks_appends_summary(tmp_path, monkeypatch):
    """``_render_content_blocks`` appends the vision summary right after the
    MEDIA tag for image blocks."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _render_content_blocks

    tag = _media_tag(tmp_path)
    fake = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "wired through"})
    )
    block = SimpleNamespace(data=None, mimeType=None, type="image")  # render via pre-cached tag path is internal; use real block instead
    real_block = SimpleNamespace(
        data=base64.b64encode(_png_bytes()).decode("ascii"),
        mimeType="image/png",
    )
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        text, usable = await _render_content_blocks(
            _tool_result([real_block]), "srv"
        )
    assert "MEDIA:" in text
    assert "[图片内容摘要] wired through" in text
    assert usable == 1


@pytest.mark.asyncio
async def test_render_content_blocks_fail_open_keeps_media_tag(tmp_path, monkeypatch):
    """When summarization fails the MEDIA tag still renders (fail-open)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_tool_handlers import _render_content_blocks

    real_block = SimpleNamespace(
        data=base64.b64encode(_png_bytes()).decode("ascii"),
        mimeType="image/png",
    )
    fake = AsyncMock(side_effect=RuntimeError("vision down"))
    with patch(
        "hermes_cli.config.load_config", return_value=_vision_cfg()
    ), patch("tools.vision_tools.vision_analyze_tool", fake):
        text, usable = await _render_content_blocks(
            _tool_result([real_block]), "srv"
        )
    assert "MEDIA:" in text
    assert "图片内容摘要" not in text
    assert usable == 1
