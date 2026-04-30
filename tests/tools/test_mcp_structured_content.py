"""Tests for MCP tool structuredContent preservation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


class _FakeContentBlock:
    """Minimal content block with .text and .type attributes."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in.

    Uses camelCase ``structuredContent`` / ``isError`` to match the real
    MCP SDK Pydantic model (``mcp.types.CallToolResult``).
    """

    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro, timeout=30):
    """Run an MCP coroutine directly in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        # `_rpc_lock` must be created inside the loop that awaits it, or asyncio
        # raises "attached to a different loop". Build it here and attach it to
        # whatever fake server is currently registered under _servers.
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
    # `_rpc_lock` is acquired by _make_tool_handler's call path (mcp_tool.py
    # ~L2008) to serialize JSON-RPC against the server — build it inside the
    # fresh loop that _fake_run_on_mcp_loop spins up, not at fixture import.
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


class TestStructuredContentPreservation:
    """Ensure structuredContent from CallToolResult is forwarded."""

    def test_text_only_result(self, _patch_mcp_server):
        """When no structuredContent, result is text-only (existing behaviour)."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("hello")],
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data == {"result": "hello"}

    def test_both_content_and_structured(self, _patch_mcp_server):
        """When both content and structuredContent are present, combine them."""
        session = _patch_mcp_server
        payload = {"value": "secret-123", "revealed": True}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("OK")],
                structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        # content is the primary result, structuredContent is supplementary
        assert data["result"] == "OK"
        assert data["structuredContent"] == payload

    def test_both_content_and_structured_desktop_commander(self, _patch_mcp_server):
        """Real-world case: Desktop Commander returns file text in content,
        metadata in structuredContent.  Agent must see file contents."""
        session = _patch_mcp_server
        file_text = "import os\nprint('hello')\n"
        metadata = {"fileName": "main.py", "filePath": "/tmp/main.py", "fileType": "python"}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock(file_text)],
                structuredContent=metadata,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data["result"] == file_text
        assert data["structuredContent"] == metadata

    def test_structured_content_none_falls_back_to_text(self, _patch_mcp_server):
        """When structuredContent is explicitly None, fall back to text."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                structuredContent=None,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data == {"result": "done"}

    def test_empty_text_with_structured_content(self, _patch_mcp_server):
        """When content blocks are empty but structuredContent exists."""
        session = _patch_mcp_server
        payload = {"status": "ok", "data": [1, 2, 3]}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[],
                structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data["result"] == payload
