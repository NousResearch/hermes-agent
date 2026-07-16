"""Tests for MCP tool EmbeddedResource (UIResource) block handling.

An ``EmbeddedResource`` (MCP ``type: "resource"``) carries its payload under
``block.resource`` — NOT at the top level. Before this fix, Hermes' MCP client
only extracted ``block.text`` and image blocks, so a spec-correct ``UIResource``
(the MCP-UI card shape) was silently dropped: the model and any ``post_tool_call``
hook both received an empty result. These tests pin the corrected behaviour —
the resource's text flows into ``result`` and the structured resource surfaces
under ``resources`` with ``{uri, mimeType, text, _meta}`` intact.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


class _FakeContentBlock:
    """Minimal text content block with .text and .type attributes."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


class _FakeResource:
    """Stand-in for TextResourceContents / BlobResourceContents.

    The real mcp python types expose ``_meta`` as the ``.meta`` attribute
    (aliased to ``_meta`` on the wire), so this mirrors that.
    """

    def __init__(self, uri=None, mimeType=None, text=None, blob=None, meta=None):
        self.uri = uri
        self.mimeType = mimeType
        if text is not None:
            self.text = text
        if blob is not None:
            self.blob = blob
        self.meta = meta


class _FakeEmbeddedResource:
    """Minimal EmbeddedResource stand-in: type='resource' + .resource, no top-level .text."""

    def __init__(self, resource):
        self.type = "resource"
        self.resource = resource


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in (camelCase to match the SDK model)."""

    def __init__(self, content, is_error=False, structuredContent=None):
        self.content = content
        self.isError = is_error
        self.structuredContent = structuredContent


def _fake_run_on_mcp_loop(coro_or_factory, timeout=30):
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
    fake_session = MagicMock()
    fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)
    with patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


UI_HTML = "<div style='padding:16px'>Google Wave — 2009</div>"
UI_META = {"mcpui.dev/ui-preferred-frame-size": ["400px", "520px"]}


class TestEmbeddedResourceHandling:
    """A UIResource EmbeddedResource must survive transport, not be dropped."""

    def test_ui_resource_text_flows_into_result(self, _patch_mcp_server):
        """The resource's HTML text reaches the model-facing ``result``."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://research/google-wave",
            mimeType="text/html;profile=mcp-app",
            text=UI_HTML,
            meta=UI_META,
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[_FakeEmbeddedResource(res)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        # Was "" before the fix — now carries the card HTML.
        assert data["result"] == UI_HTML

    def test_ui_resource_structure_surfaced_under_resources(self, _patch_mcp_server):
        """The structured {uri, mimeType, text, _meta} is preserved for plugins."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://research/google-wave",
            mimeType="text/html;profile=mcp-app",
            text=UI_HTML,
            meta=UI_META,
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[_FakeEmbeddedResource(res)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        assert "resources" in data
        assert len(data["resources"]) == 1
        entry = data["resources"][0]
        assert entry["uri"] == "ui://research/google-wave"
        assert entry["mimeType"] == "text/html;profile=mcp-app"
        assert entry["text"] == UI_HTML
        assert entry["_meta"] == UI_META

    def test_blob_resource_contributes_structure_only(self, _patch_mcp_server):
        """A blob resource has no text — it surfaces structurally but doesn't pollute result."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://x/y",
            mimeType="text/html;profile=mcp-app",
            blob="PGgxPkhpPC9oMT4=",  # base64 of <h1>Hi</h1>
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[_FakeEmbeddedResource(res)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        assert data["result"] == ""  # no text blocks
        assert data["resources"][0]["blob"] == "PGgxPkhpPC9oMT4="
        assert "text" not in data["resources"][0]

    def test_mixed_text_and_resource_blocks(self, _patch_mcp_server):
        """A plain text block plus a UIResource: text joins result, resource surfaces too."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://research/x",
            mimeType="text/html;profile=mcp-app",
            text=UI_HTML,
            meta=UI_META,
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeContentBlock("Dropped a primer."), _FakeEmbeddedResource(res)]
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        assert "Dropped a primer." in data["result"]
        assert UI_HTML in data["result"]
        assert data["resources"][0]["uri"] == "ui://research/x"

    def test_resource_without_meta_omits_meta_key(self, _patch_mcp_server):
        """No _meta on the resource → no _meta key in the surfaced entry (clean shape)."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://x/y",
            mimeType="text/html;profile=mcp-app",
            text="<p>hi</p>",
            meta=None,
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[_FakeEmbeddedResource(res)])
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        assert "_meta" not in data["resources"][0]

    def test_resource_plus_structured_content_coexist(self, _patch_mcp_server):
        """structuredContent and embedded resources both surface without clobbering."""
        session = _patch_mcp_server
        res = _FakeResource(
            uri="ui://x/y", mimeType="text/html;profile=mcp-app", text=UI_HTML, meta=UI_META
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeEmbeddedResource(res)],
                structuredContent={"status": "ok"},
            )
        )
        handler = mcp_tool._make_tool_handler("test-server", "create_primer", 30.0)
        data = json.loads(handler({}))
        assert data["result"] == UI_HTML
        assert data["structuredContent"] == {"status": "ok"}
        assert data["resources"][0]["uri"] == "ui://x/y"
