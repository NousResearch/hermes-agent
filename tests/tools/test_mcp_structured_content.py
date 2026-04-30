"""Tests for MCP tool structuredContent preservation."""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


class _FakeContentBlock:
    """Minimal content block with .text and .type attributes."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeImageBlock:
    """Minimal MCP ImageContent stand-in."""

    def __init__(self, data: str, mime_type: str = "image/png"):
        self.type = "image"
        self.data = data
        self.mimeType = mime_type


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in."""

    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


class _FakeAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _fake_run_on_mcp_loop(coro, timeout=30):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def _patch_mcp_server():
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=_FakeAsyncLock())
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


class TestStructuredContentPreservation:
    def test_text_only_result(self, _patch_mcp_server):
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
        assert data["result"] == "OK"
        assert data["structuredContent"] == payload

    def test_both_content_and_structured_desktop_commander(self, _patch_mcp_server):
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

    def test_image_content_is_saved_and_reported(self, _patch_mcp_server, tmp_path, monkeypatch):
        session = _patch_mcp_server
        png_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/w8AAgMBgJ/l8QAAAABJRU5ErkJggg=="
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeImageBlock(png_base64)],
            )
        )

        created_files = []
        real_mkstemp = mcp_tool.tempfile.mkstemp

        def _mkstemp_in_tmpdir(*args, **kwargs):
            fd, path = real_mkstemp(dir=tmp_path, *args, **kwargs)
            created_files.append(Path(path))
            return fd, path

        monkeypatch.setattr(mcp_tool.tempfile, "mkstemp", _mkstemp_in_tmpdir)

        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        raw = handler({})
        data = json.loads(raw)

        assert "Media saved:" in data["result"]
        assert "image/png" in data["result"]
        assert len(created_files) == 1
        saved = created_files[0]
        assert saved.exists()
        assert saved.suffix == ".png"
        assert saved.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
