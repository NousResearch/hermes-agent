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

    def __init__(self, content, is_error=False, structuredContent=None, meta=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent
        # Real SDK exposes the wire ``_meta`` field as ``.meta`` (Pydantic alias).
        self.meta = meta


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


class TestMetaPassthrough:
    """Server ``_meta`` is surfaced, minus protocol-reserved keys.

    Ported from MoonshotAI/kimi-code#2596/#2600.
    """

    def test_vendor_meta_passes_through(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                meta={"com.example/handoff": {"url": "https://x"}},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data["result"] == "done"
        assert data["_meta"] == {"com.example/handoff": {"url": "https://x"}}

    def test_reserved_meta_keys_dropped(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                meta={
                    "modelcontextprotocol.io/progress": 1,
                    "tools.mcp.com/trace": "x",
                    "com.example.mcp/vendor": "keep",  # trailing mcp label = vendor ns
                    "unprefixed": "keep",
                },
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data["_meta"] == {
            "com.example.mcp/vendor": "keep",
            "unprefixed": "keep",
        }

    def test_all_reserved_meta_omits_field(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                meta={"mcp.io/internal": True},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {"result": "done"}

    def test_meta_with_structured_content(self, _patch_mcp_server):
        """With usable text, structuredContent is suppressed but _meta rides."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("txt")],
                structuredContent={"ok": True},
                meta={"com.example/k": "v"},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {
            "result": "txt",
            "_meta": {"com.example/k": "v"},
        }

    def test_non_serializable_meta_dropped(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                meta={"com.example/obj": object()},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {"result": "done"}

    def test_non_dict_meta_ignored(self, _patch_mcp_server):
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("done")],
                meta="not-a-dict",
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {"result": "done"}


class TestReservedMetaKeyPredicate:
    def test_reserved_prefixes(self):
        assert mcp_tool._is_reserved_mcp_meta_key("modelcontextprotocol.io/x")
        assert mcp_tool._is_reserved_mcp_meta_key("mcp.dev/x")
        assert mcp_tool._is_reserved_mcp_meta_key("tools.mcp.com/x")

    def test_vendor_and_unprefixed_not_reserved(self):
        assert not mcp_tool._is_reserved_mcp_meta_key("com.example.mcp/x")  # trailing label
        assert not mcp_tool._is_reserved_mcp_meta_key("com.example/x")
        assert not mcp_tool._is_reserved_mcp_meta_key("plain-key")
        assert not mcp_tool._is_reserved_mcp_meta_key("/leading-slash")


class TestContentStructuredArbitration:
    """content and structuredContent are alternatives — never both.

    Ported from MoonshotAI/kimi-code#3234: spec-following servers render
    their data into content (verbatim dual-emit or a faithful human
    reorganisation), so forwarding both sent the same information twice.
    """

    def test_dual_emit_suppresses_structured(self, _patch_mcp_server):
        """Verbatim dual-emit servers: model receives content only."""
        session = _patch_mcp_server
        payload = {"items": [1, 2, 3]}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock(json.dumps(payload))],
                structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {"result": json.dumps(payload)}

    def test_prose_summary_suppresses_structured(self, _patch_mcp_server):
        """Lossy prose summaries also win — no heuristic is attempted."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("3 item(s) found")],
                structuredContent={"items": [1, 2, 3]},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data == {"result": "3 item(s) found"}

    def test_whitespace_only_content_falls_back(self, _patch_mcp_server):
        """Whitespace-only text is not usable content — fallback fires."""
        session = _patch_mcp_server
        payload = {"status": "ok"}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("   \n")],
                structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data["structuredContent"] == payload

    def test_structured_only_still_surfaced(self, _patch_mcp_server):
        """structuredContent-only servers keep working (#2596 fix preserved)."""
        session = _patch_mcp_server
        payload = {"only": "structured"}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[],
                structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data["result"] == payload


class TestDroppedBlockNotice:
    """Unsupported content blocks surface a drop notice to the model.

    Ported from MoonshotAI/kimi-code#3227.
    """

    def test_unsupported_block_renders_notice(self, _patch_mcp_server):
        session = _patch_mcp_server
        # NOTE: no `uri` — a uri'd block without .resource is rendered as a
        # resource link by _render_mcp_resource_block, not dropped.
        weird = SimpleNamespace(
            type="hologram",
            mimeType="application/x-hologram",
            size=1234,
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[weird])
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert "[MCP content dropped: unsupported block" in data["result"]
        assert "type=hologram" in data["result"]
        assert "mimeType=application/x-hologram" in data["result"]
        assert "size=1234" in data["result"]

    def test_drop_notice_does_not_suppress_structured(self, _patch_mcp_server):
        """A drop notice is not usable content — structured fallback fires."""
        session = _patch_mcp_server
        weird = SimpleNamespace(type="hologram")
        payload = {"real": "data"}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[weird], structuredContent=payload,
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        data = json.loads(handler({}))
        assert data["structuredContent"] == payload
        assert "[MCP content dropped" in data["result"]

    def test_notice_helper_minimal_block(self):
        notice = mcp_tool._render_mcp_dropped_block_notice(
            SimpleNamespace(), "mystery"
        )
        assert notice == "[MCP content dropped: unsupported block (type=mystery)]"
