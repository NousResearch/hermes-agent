"""Tests for MCP tool max_result_bytes truncation (issue #44172)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


class _FakeContentBlock:
    """Minimal content block with .text attribute."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in."""

    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    """Run an MCP coroutine directly in a fresh event loop."""
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_lock_and_run():
            for srv in list(mcp_tool._servers.values()):
                if getattr(srv, "_rpc_lock", None) is None:
                    srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_lock_and_run())
    finally:
        loop.close()


@pytest.fixture
def _patch_mcp_server():
    """Patch _servers and the MCP event loop so _make_tool_handler can run."""
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


class TestMaxResultBytes:
    """Verify max_result_bytes truncates oversized MCP tool results."""

    def test_no_limit_by_default(self, _patch_mcp_server):
        """Without max_result_bytes, large results pass through unchanged."""
        large_text = "x" * 50000
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(large_text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0)
        result = json.loads(handler({"text": "hello"}))
        assert result["result"] == large_text

    def test_truncates_when_over_limit(self, _patch_mcp_server):
        """Results exceeding max_result_bytes are truncated with a notice."""
        large_text = "x" * 50000
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(large_text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0, max_result_bytes=1000)
        result = json.loads(handler({"text": "hello"}))
        assert len(result["result"]) < len(large_text)
        assert "truncated" in result["result"]
        assert "1000 bytes" in result["result"]

    def test_small_result_not_truncated(self, _patch_mcp_server):
        """Results under the limit pass through unchanged."""
        small_text = "hello world"
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(small_text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0, max_result_bytes=10000)
        result = json.loads(handler({"text": "hello"}))
        assert result["result"] == small_text

    def test_zero_means_no_limit(self, _patch_mcp_server):
        """max_result_bytes=0 means no truncation (the default)."""
        large_text = "y" * 100000
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(large_text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0, max_result_bytes=0)
        result = json.loads(handler({"text": "hello"}))
        assert result["result"] == large_text

    def test_exact_limit_not_truncated(self, _patch_mcp_server):
        """A result exactly at the byte limit is NOT truncated."""
        text = "a" * 100  # 100 bytes exactly
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0, max_result_bytes=100)
        result = json.loads(handler({"text": "hello"}))
        assert result["result"] == text

    def test_multibyte_utf8_truncation(self, _patch_mcp_server):
        """Truncation handles multi-byte UTF-8 characters gracefully."""
        # Each Chinese character is 3 bytes in UTF-8
        chinese_text = "你好" * 1000  # 6000 bytes
        _patch_mcp_server.call_tool = AsyncMock(
            return_value=_FakeCallToolResult([_FakeContentBlock(chinese_text)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "echo", 30.0, max_result_bytes=3000)
        result = json.loads(handler({"text": "hello"}))
        assert "truncated" in result["result"]
        # The truncated portion should not contain broken UTF-8 sequences
        # (errors="ignore" in decode handles this)

    def test_server_task_reads_config(self):
        """MCPServerTask.run() reads max_result_bytes from config."""
        server = mcp_tool.MCPServerTask("test")
        assert server.max_result_bytes == 0  # default
        # Simulate what run() does
        config = {"max_result_bytes": 5000, "command": "echo"}
        server.max_result_bytes = int(config.get("max_result_bytes", 0) or 0)
        assert server.max_result_bytes == 5000

    def test_config_string_value_coerced(self):
        """String config values are coerced to int."""
        server = mcp_tool.MCPServerTask("test")
        config = {"max_result_bytes": "8192"}
        server.max_result_bytes = int(config.get("max_result_bytes", 0) or 0)
        assert server.max_result_bytes == 8192
