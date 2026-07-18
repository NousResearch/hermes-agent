"""Tests for MCP ResourceLink / EmbeddedResource / AudioContent handling.

Regression coverage for a customer report (2026-07): non-image binary
resources returned through MCP resource blocks were silently dropped from
tool results, so a PDF-returning MCP tool appeared to return metadata only.
"""

import base64
import json
from types import SimpleNamespace

import pytest


PDF_BYTES = b"%PDF-1.4 fake pdf payload for tests"


def _blob_resource(data: bytes, uri="slack://files/F123/report.pdf", mime="application/pdf"):
    return SimpleNamespace(
        uri=uri,
        mimeType=mime,
        blob=base64.b64encode(data).decode("ascii"),
        text=None,
    )


def _embedded(resource):
    return SimpleNamespace(type="resource", resource=resource)


@pytest.fixture()
def doc_cache(tmp_path, monkeypatch):
    """Point the document cache at a temp dir."""
    import gateway.platforms.base as base

    monkeypatch.setattr(base, "DOCUMENT_CACHE_DIR", tmp_path)
    monkeypatch.setenv("HERMES_DOCUMENT_CACHE_DIR", str(tmp_path))
    # _resolve_cache_dir consults the module constant; patching the constant
    # is sufficient because the import-default comparison detects the change.
    return tmp_path


class TestRenderResourceBlock:
    def test_embedded_pdf_blob_is_materialized(self, doc_cache):
        from tools.mcp_tool import _render_mcp_resource_block

        out = _render_mcp_resource_block(_embedded(_blob_resource(PDF_BYTES)), "slack")
        assert "saved to" in out
        assert "application/pdf" in out
        # Extract path and verify bytes round-trip
        path = out.split("saved to ", 1)[1].split(" (", 1)[0]
        with open(path, "rb") as fh:
            assert fh.read() == PDF_BYTES
        assert "report.pdf" in path

    def test_embedded_text_resource_is_inlined(self):
        from tools.mcp_tool import _render_mcp_resource_block

        res = SimpleNamespace(uri="mem://notes", mimeType="text/plain", text="hello world", blob=None)
        assert _render_mcp_resource_block(_embedded(res), "srv") == "hello world"

    def test_resource_link_preserves_uri_and_points_at_reader(self):
        from tools.mcp_tool import _render_mcp_resource_block

        link = SimpleNamespace(
            type="resource_link",
            uri="slack://files/F123",
            name="report.pdf",
            mimeType="application/pdf",
        )
        out = _render_mcp_resource_block(link, "slack")
        assert "slack://files/F123" in out
        # Must be the real wire name (mcp__<server>__read_resource), not a
        # made-up "<server>_read_resource" the agent can't actually call.
        assert "mcp__slack__read_resource" in out
        assert "report.pdf" in out

    def test_oversized_blob_fails_explicitly_without_writing(self, doc_cache, monkeypatch):
        import tools.mcp_tool as m

        monkeypatch.setattr(m, "_MCP_RESOURCE_MAX_BYTES", 8)
        out = m._render_mcp_resource_block(_embedded(_blob_resource(PDF_BYTES)), "srv")
        assert "too large" in out
        assert not list(doc_cache.glob("doc_*"))

    def test_malformed_base64_fails_explicitly(self):
        from tools.mcp_tool import _render_mcp_resource_block

        res = SimpleNamespace(uri="x://y", mimeType="application/pdf", blob="!!!not-base64!!!", text=None)
        out = _render_mcp_resource_block(_embedded(res), "srv")
        assert "could not be decoded" in out

    def test_non_resource_block_returns_empty(self):
        from tools.mcp_tool import _render_mcp_resource_block

        assert _render_mcp_resource_block(SimpleNamespace(type="text", text="hi"), "srv") == ""

    def test_path_traversal_uri_is_neutralized(self, doc_cache):
        from tools.mcp_tool import _render_mcp_resource_block

        res = _blob_resource(PDF_BYTES, uri="evil://host/../../etc/passwd")
        out = _render_mcp_resource_block(_embedded(res), "srv")
        assert "saved to" in out
        path = out.split("saved to ", 1)[1].split(" (", 1)[0]
        assert str(doc_cache) in path
        assert "/etc/passwd" not in path


class TestResourceFilename:
    def test_uri_last_segment_used(self):
        from tools.mcp_tool import _mcp_resource_filename

        assert _mcp_resource_filename("slack://f/ABC/quarterly.pdf", "application/pdf") == "quarterly.pdf"

    def test_fallback_to_mime_extension(self):
        from tools.mcp_tool import _mcp_resource_filename

        name = _mcp_resource_filename("", "application/pdf")
        assert name.endswith(".pdf")

    def test_dotdot_rejected(self):
        from tools.mcp_tool import _mcp_resource_filename

        assert _mcp_resource_filename("x://y/..", "application/pdf") != ".."

    def test_control_chars_stripped(self):
        from tools.mcp_tool import _mcp_resource_filename

        name = _mcp_resource_filename("x://h/report.pdf%0Ainjected%1b[31m", "application/pdf")
        assert "\n" not in name and "\x1b" not in name

    def test_long_filename_capped_preserving_extension(self):
        from tools.mcp_tool import _mcp_resource_filename

        name = _mcp_resource_filename("x://h/" + "a" * 500 + ".pdf", "application/pdf")
        assert len(name) <= 150
        assert name.endswith(".pdf")


class TestPreDecodeSizeCap:
    def test_oversized_b64_rejected_before_decode(self, monkeypatch):
        import tools.mcp_tool as m

        monkeypatch.setattr(m, "_MCP_RESOURCE_MAX_B64_CHARS", 16)
        res = SimpleNamespace(
            uri="x://y/big.pdf", mimeType="application/pdf",
            blob="A" * 100, text=None,
        )
        called = []
        monkeypatch.setattr(base64, "b64decode", lambda *a, **k: called.append(1))
        out = m._render_mcp_resource_block(_embedded(res), "srv")
        assert "too large" in out
        assert not called


class TestAudioBlock:
    def test_non_audio_returns_empty(self):
        from tools.mcp_tool import _cache_mcp_audio_block

        block = SimpleNamespace(data=base64.b64encode(b"x").decode(), mimeType="application/pdf")
        assert _cache_mcp_audio_block(block) == ""

    def test_audio_block_cached_as_media(self, tmp_path, monkeypatch):
        import gateway.platforms.base as base
        from tools.mcp_tool import _cache_mcp_audio_block

        monkeypatch.setattr(base, "AUDIO_CACHE_DIR", tmp_path)
        block = SimpleNamespace(
            data=base64.b64encode(b"RIFFfakewav").decode(),
            mimeType="audio/wav",
        )
        out = _cache_mcp_audio_block(block)
        assert out.startswith("MEDIA:")


class TestToolResultLoopOrdering:
    def test_mixed_blocks_preserve_order(self, doc_cache):
        """Simulate the tool-result block loop with text + pdf resource."""
        from tools.mcp_tool import (
            _cache_mcp_image_block,
            _cache_mcp_audio_block,
            _render_mcp_resource_block,
        )

        blocks = [
            SimpleNamespace(type="text", text="File ID: F123\nMIME Type: application/pdf"),
            _embedded(_blob_resource(PDF_BYTES)),
        ]
        parts = []
        for block in blocks:
            if getattr(block, "text", None):
                parts.append(block.text)
                continue
            tag = _cache_mcp_image_block(block) or _cache_mcp_audio_block(block)
            if tag:
                parts.append(tag)
                continue
            rendered = _render_mcp_resource_block(block, "slack")
            if rendered:
                parts.append(rendered)
        assert len(parts) == 2
        assert parts[0].startswith("File ID")
        assert "saved to" in parts[1]

    def test_existing_image_behavior_unchanged(self):
        from tools.mcp_tool import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(b"some bytes").decode("ascii"),
            mimeType="application/pdf",
        )
        assert _cache_mcp_image_block(block) == ""


class TestErrorPathResourceText:
    """isError payloads must surface EmbeddedResource text, not drop it."""

    @pytest.fixture()
    def _handler(self, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch as mock_patch

        from tools import mcp_tool

        fake_session = MagicMock()
        fake_server = SimpleNamespace(session=fake_session, _rpc_lock=None)

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

        with mock_patch.dict(mcp_tool._servers, {"test-server": fake_server}), \
             mock_patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_fake_run_on_mcp_loop):
            fake_session.call_tool = AsyncMock()
            yield fake_session, mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)

    def test_error_embedded_resource_text_surfaced(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        res = SimpleNamespace(uri="mem://err", mimeType="text/plain",
                              text="quota exceeded for workspace W1", blob=None)
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[_embedded(res)], isError=True, structuredContent=None,
        ))
        data = json.loads(handler({}))
        assert "quota exceeded for workspace W1" in data["error"]

    def test_error_mixed_text_and_resource(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        res = SimpleNamespace(uri="mem://err", mimeType="text/plain",
                              text=" — details in resource", blob=None)
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="tool failed"), _embedded(res)],
            isError=True, structuredContent=None,
        ))
        data = json.loads(handler({}))
        assert "tool failed" in data["error"]
        assert "details in resource" in data["error"]

    def test_error_with_no_text_blocks_falls_back(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[], isError=True, structuredContent=None,
        ))
        data = json.loads(handler({}))
        assert data["error"] == "MCP tool returned an error"
