"""Regression tests for MCP EmbeddedResource block handling.

Background
==========
MCP servers — most visibly QMD's ``mcp_qmd_get`` / ``mcp_qmd_multi_get`` —
ship a document's content back as an ``EmbeddedResource`` content block
(``type="resource"``) inside the ``CallToolResult``. The previous handler
in ``tools/mcp_tool.py`` iterated content blocks looking only for
``block.text`` and image blocks, silently dropping EmbeddedResource. The
agent saw ``{"result": ""}`` for a successful lookup, while QMD's
``mcp_qmd_query`` worked fine because it returns plain ``TextContent``
blocks. The fix adds a focused ``_extract_embedded_resource_parts``
helper and threads it into both the success path and the error path.

What this file locks in
-----------------------
- EmbeddedResource with TextResourceContents is surfaced in ``result``.
- Multiple blocks (TextContent + EmbeddedResource) all flow through.
- The error path merges EmbeddedResource text into the error message.
- BlobResourceContents with an image MIME → ``MEDIA:<path>`` tag.
- BlobResourceContents with a text-ish MIME (json/xml/yaml/text) → text.
- BlobResourceContents with a non-textish binary MIME → ``[omitted]`` marker.
- A non-EmbeddedResource block still produces no parts (helper is a no-op).
- structuredContent is preserved alongside the embedded resource text.
- Missing MCP SDK on the import path doesn't NameError (defensive).
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_tool


# ---------------------------------------------------------------------------
# Fixtures + helpers (mirrors tests/tools/test_mcp_structured_content.py)
# ---------------------------------------------------------------------------


class _FakeTextContentBlock:
    """Plain ``TextContent`` block — the shape the old handler knew about."""

    def __init__(self, text: str, block_type: str = "text"):
        self.text = text
        self.type = block_type


def _build_embedded_resource_block(uri: str, text: str, mime_type: str = "text/plain"):
    """Build a real MCP SDK ``EmbeddedResource`` block with a TextResource.

    Falls back to a duck-typed SimpleNamespace if the SDK import fails —
    keeps the high-level handler tests running even in stripped envs.
    """
    try:
        from mcp.types import EmbeddedResource, TextResourceContents

        resource = TextResourceContents(uri=uri, text=text, mimeType=mime_type)
        return EmbeddedResource(type="resource", resource=resource)
    except ImportError:
        # Duck-typed fallback matching the helper's geta\r getters.
        return SimpleNamespace(
            type="resource",
            resource=SimpleNamespace(uri=uri, text=text, mimeType=mime_type),
        )


def _build_blob_embedded_resource_block(uri: str, blob_b64: str, mime_type: str):
    try:
        from mcp.types import BlobResourceContents, EmbeddedResource

        resource = BlobResourceContents(uri=uri, blob=blob_b64, mimeType=mime_type)
        return EmbeddedResource(type="resource", resource=resource)
    except ImportError:
        return SimpleNamespace(
            type="resource",
            resource=SimpleNamespace(uri=uri, blob=blob_b64, mimeType=mime_type),
        )


class _FakeCallToolResult:
    """Minimal CallToolResult stand-in.

    camelCase attributes to match the real MCP SDK Pydantic model.
    """

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
    with patch.dict(mcp_tool._servers, {"qmd": fake_server}), \
         patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
        yield fake_session


# ---------------------------------------------------------------------------
# End-to-end through _make_tool_handler
# ---------------------------------------------------------------------------


class TestEmbeddedResourceInResult:
    """A successful ``mcp_qmd_get`` should surface the document body."""

    def test_text_embedded_resource_returns_document_body(self, _patch_mcp_server):
        """TextResourceContents (the QMD shape) must land in result text.

        Without the fix this assertion is empty: ``{"result": ""}``.
        """
        session = _patch_mcp_server
        doc_body = "# Jane's Diary\n\nDay 1: ships in the harbor.\n"
        embedded = _build_embedded_resource_block(
            "qmd://docs/janes-diary",
            doc_body,
            "text/plain",
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[embedded])
        )
        handler = mcp_tool._make_tool_handler("qmd", "qmd_get", 30.0)
        raw = handler({"uri": "qmd://docs/janes-diary"})
        data = json.loads(raw)
        assert doc_body in data["result"], (
            f"document body missing from result: {data!r}"
        )

    def test_uri_header_included_when_available(self, _patch_mcp_server):
        """Multi-resource calls should keep resources distinguishable.

        The helper prefixes the document body with ``# <uri>`` so the
        agent can tell multiple embedded resources apart.
        """
        session = _patch_mcp_server
        blocks = [
            _build_embedded_resource_block(
                "qmd://docs/a", "==alpha==\nfirst body", "text/plain",
            ),
            _build_embedded_resource_block(
                "qmd://docs/b", "==bravo==\nsecond body", "text/plain",
            ),
        ]
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=blocks)
        )
        handler = mcp_tool._make_tool_handler("qmd", "multi_get", 30.0)
        raw = handler({"uris": ["qmd://docs/a", "qmd://docs/b"]})
        data = json.loads(raw)
        text = data["result"]
        assert "qmd://docs/a" in text, f"missing URI a header: {text!r}"
        assert "qmd://docs/b" in text, f"missing URI b header: {text!r}"
        assert "first body" in text and "second body" in text

    def test_mixed_text_and_embedded_resource_concatenate(self, _patch_mcp_server):
        """A typical tool result has prose around the document body."""
        session = _patch_mcp_server
        doc_body = "doc body content here"
        blocks = [
            _FakeTextContentBlock("Header prose\n"),
            _build_embedded_resource_block("qmd://x", doc_body, "text/plain"),
            _FakeTextContentBlock("Footer prose\n"),
        ]
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=blocks)
        )
        handler = mcp_tool._make_tool_handler("qmd", "qmd_get", 30.0)
        raw = handler({"uri": "qmd://x"})
        data = json.loads(raw)
        text = data["result"]
        assert "Header prose" in text
        assert doc_body in text
        assert "Footer prose" in text
        # Order is preserved by the loop order.
        assert text.index("Header prose") < text.index(doc_body)
        assert text.index(doc_body) < text.index("Footer prose")

    def test_structured_content_preserved_with_embedded_resource(
        self, _patch_mcp_server,
    ):
        """QMD-style ``mcp_qmd_get`` may carry metadata alongside the body."""
        session = _patch_mcp_server
        metadata = {"id": "abc", "score": 0.97, "tags": ["nostalgia"]}
        blocks = [
            _build_embedded_resource_block("qmd://x", "body text", "text/plain"),
        ]
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=blocks, structuredContent=metadata,
            )
        )
        handler = mcp_tool._make_tool_handler("qmd", "qmd_get", 30.0)
        raw = handler({"uri": "qmd://x"})
        data = json.loads(raw)
        assert "body text" in data["result"]
        assert data["structuredContent"] == metadata


class TestEmbeddedResourceOnErrorPath:
    """An error result that arrives as EmbeddedResource must surface text."""

    def test_error_path_includes_embedded_resource_text(self, _patch_mcp_server):
        """Some servers return ``isError=True`` with the failure reason
        inside an EmbeddedResource. The agent should see the reason."""
        session = _patch_mcp_server
        embedded = _build_embedded_resource_block(
            "qmd://docs/missing",
            "doc not found in collection",
            "text/plain",
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[embedded], is_error=True)
        )
        handler = mcp_tool._make_tool_handler("qmd", "qmd_get", 30.0)
        raw = handler({"uri": "qmd://docs/missing"})
        data = json.loads(raw)
        assert "error" in data
        assert "doc not found in collection" in data["error"], (
            f"reason missing from error envelope: {data!r}"
        )


class TestBlobResourceContents:
    """Binary blobs land inside BlobResourceContents — needs safe handling."""

    def test_image_blob_returns_media_tag(self, _patch_mcp_server, tmp_path, monkeypatch):
        """An image blob goes through the existing image pipeline."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Minimal 1x1 transparent PNG.
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        embedded = _build_blob_embedded_resource_block(
            "qmd://rendered/page-1.png",
            base64.b64encode(png).decode("ascii"),
            "image/png",
        )
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[embedded])
        )
        handler = mcp_tool._make_tool_handler("qmd", "render", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert "MEDIA:" in data["result"], f"expected MEDIA: tag, got {data!r}"
        # Result should also include the URI header so the agent can
        # match render output to its source.
        assert "qmd://rendered/page-1.png" in data["result"]

    def test_text_blob_decoded_safely(self, _patch_mcp_server):
        """A text-ish blob (json/xml/yaml/text) decodes and lands in result."""
        session = _patch_mcp_server
        payload = '{"items": [1, 2, 3], "ok": true}'
        embedded = _build_blob_embedded_resource_block(
            "qmd://export/manifest.json",
            base64.b64encode(payload.encode("utf-8")).decode("ascii"),
            "application/json",
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[embedded])
        )
        handler = mcp_tool._make_tool_handler("qmd", "export", 30.0)
        raw = handler({})
        data = json.loads(raw)
        text = data["result"]
        assert "items" in text and "1, 2, 3" in text

    def test_binary_blob_emits_omitted_marker(self, _patch_mcp_server):
        """A non-textish binary blob (e.g. application/pdf) returns a
        clear marker instead of crashing or silently dropping."""
        session = _patch_mcp_server
        embedded = _build_blob_embedded_resource_block(
            "qmd://file/report.pdf",
            base64.b64encode(b"%PDF-1.4 fake bytes").decode("ascii"),
            "application/pdf",
        )
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[embedded])
        )
        handler = mcp_tool._make_tool_handler("qmd", "fetch_pdf", 30.0)
        raw = handler({})
        data = json.loads(raw)
        text = data["result"]
        assert "omitted" in text.lower()
        assert "application/pdf" in text or "pdf" in text.lower()
        # The URI should still be referenced so the agent can route
        # around the binary.
        assert "qmd://file/report.pdf" in text


# ---------------------------------------------------------------------------
# Helper-level unit tests (no server, no event loop)
# ---------------------------------------------------------------------------


class TestExtractEmbeddedResourcePartsHelper:
    """Direct helper-level tests that don't depend on the SDK SDK being
    installed. These mirror the focus of the ImageContent tests."""

    def test_non_embedded_block_returns_empty(self):
        """A plain TextContent has a .text attr, but it's not an
        EmbeddedResource — should produce no parts (the caller's loop
        handles it via the existing text branch)."""
        from tools.mcp_tool import _extract_embedded_resource_parts

        block = SimpleNamespace(text="hello")  # not type=resource
        assert _extract_embedded_resource_parts(block) == []

    def test_handles_missing_sdk_types_gracefully(self, monkeypatch):
        """If MCP SDK types aren't importable (e.g., cron-only install),
        the helper must return ``[]`` rather than ``NameError``.

        This is the regression for the missing module-level initializer.
        """
        from tools import mcp_tool
        monkeypatch.setattr(
            mcp_tool, "_MCP_EMBEDDED_RESOURCE_TYPES", False, raising=False,
        )
        # Even if EmbeddedResource reference is None, the helper short-
        # circuits cleanly.
        from tools.mcp_tool import _extract_embedded_resource_parts

        fake = SimpleNamespace(type="resource", resource=SimpleNamespace(uri="x", text="hi"))
        assert _extract_embedded_resource_parts(fake) == []

    def test_looks_textish_classifier(self):
        from tools.mcp_tool import _looks_textish

        assert _looks_textish("text/plain")
        assert _looks_textish("TEXT/HTML")  # case insensitive
        assert _looks_textish("application/json")
        assert _looks_textish("application/json; charset=utf-8")
        assert _looks_textish("application/yaml")
        assert not _looks_textish("application/pdf")
        assert not _looks_textish("image/png")
        assert not _looks_textish("")  # empty → falsy


# ---------------------------------------------------------------------------
# Existing-behaviour preservation (regression guards)
# ---------------------------------------------------------------------------


class TestExistingBehaviorPreserved:
    """Make sure the fix didn't regress the existing happy paths."""

    def test_plain_text_block_still_works(self, _patch_mcp_server):
        """No embedded resource — plain text still flows through unchanged."""
        session = _patch_mcp_server
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(
                content=[_FakeTextContentBlock("OK")],
            )
        )
        handler = mcp_tool._make_tool_handler("qmd", "ping", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data == {"result": "OK"}

    def test_empty_content_with_structured_falls_back(self, _patch_mcp_server):
        """The pre-existing structuredContent fall-back still fires when
        there are no content blocks AND no embedded resources."""
        session = _patch_mcp_server
        payload = {"status": "ok"}
        session.call_tool = AsyncMock(
            return_value=_FakeCallToolResult(content=[], structuredContent=payload)
        )
        handler = mcp_tool._make_tool_handler("qmd", "q", 30.0)
        raw = handler({})
        data = json.loads(raw)
        assert data["result"] == payload
