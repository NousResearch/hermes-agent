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


class _FakeImageContentBlock:
    """Minimal image block with MCP ImageContent-like attributes."""

    def __init__(self):
        self.type = "image"
        self.data = "not-base64"
        self.mimeType = "image/png"


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in.

    Uses camelCase ``structuredContent`` / ``isError`` to match the real
    MCP SDK Pydantic model (``mcp.types.CallToolResult``).
    """

    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
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


def _call_tool_result(session, content, structured):
    session.call_tool = AsyncMock(
        return_value=_FakeCallToolResult(
            content=content,
            structuredContent=structured,
        )
    )
    handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
    return json.loads(handler({}))


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

    def test_fastmcp_semantic_duplicate_omits_structured_content(self, _patch_mcp_server):
        """FastMCP dict tools mirror structuredContent as one JSON TextContent."""
        payload = {"value": "secret-123", "revealed": True, "items": [1, 2]}
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock(json.dumps(payload))],
            payload,
        )
        assert data == {"result": json.dumps(payload)}

    def test_duplicate_with_different_whitespace_and_key_formatting(self, _patch_mcp_server):
        """Semantic equality, not byte equality, controls the narrow dedupe."""
        payload = {"b": [2, 3], "a": {"nested": True}}
        text = json.dumps({"a": {"nested": True}, "b": [2, 3]}, indent=2)
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock(text)],
            payload,
        )
        assert data == {"result": text}

    def test_supplementary_human_file_text_remains_combined(self, _patch_mcp_server):
        """File/body text plus metadata is not a duplicate FastMCP payload."""
        file_text = "import os\nprint('hello')\n"
        metadata = {"fileName": "main.py", "filePath": "/tmp/main.py", "fileType": "python"}
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock(file_text)],
            metadata,
        )
        assert data["result"] == file_text
        assert data["structuredContent"] == metadata

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

    def test_multiple_text_blocks_remain_combined(self, _patch_mcp_server):
        """Multiple text blocks may carry extra context; do not dedupe."""
        payload = {"value": 1}
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock(json.dumps(payload)), _FakeContentBlock("human note")],
            payload,
        )
        assert data["result"] == f"{json.dumps(payload)}\nhuman note"
        assert data["structuredContent"] == payload

    def test_image_or_media_content_remains_combined(self, _patch_mcp_server):
        """Image/MEDIA content must keep structured metadata attached."""
        payload = {"value": 1}
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock(json.dumps(payload)), _FakeImageContentBlock()],
            payload,
        )
        assert data["result"] == json.dumps(payload)
        assert data["structuredContent"] == payload

    def test_media_text_content_remains_combined(self, _patch_mcp_server):
        """Already-materialized MEDIA tags are not treated as duplicate JSON text."""
        payload = {"image": "metadata"}
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock("MEDIA:/tmp/render.png")],
            payload,
        )
        assert data["result"] == "MEDIA:/tmp/render.png"
        assert data["structuredContent"] == payload

    def test_json_string_semantics_are_deduped(self, _patch_mcp_server):
        """JSON-native scalar semantics still dedupe when values are equal."""
        data = _call_tool_result(
            _patch_mcp_server,
            [_FakeContentBlock('"2026-07-13T12:34:56+00:00"')],
            "2026-07-13T12:34:56+00:00",
        )
        assert data == {"result": '"2026-07-13T12:34:56+00:00"'}

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
