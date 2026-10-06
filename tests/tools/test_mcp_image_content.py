"""Regression tests for MCP ImageContent block handling.

Background
==========
MCP tool results may include ``ImageContent`` blocks (screenshots from
Playwright / Blockbench / Puppeteer / any server that returns renders).
The tool result handler in ``tools/mcp_tool.py`` used to iterate content
blocks looking only for ``block.text`` — image blocks were silently dropped
and the agent saw an empty result. Distilled from @c3115644151's PR #17915
and @gnanirahulnutakki's PR #10848 (both too stale to cherry-pick); this
test file locks in #10848's approach of plumbing the bytes through
Hermes' existing ``cache_image_from_bytes`` so a ``MEDIA:<path>`` tag
goes back to the agent and through to messaging adapters that render
images natively.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace


def _png_bytes():
    """Return a minimal valid PNG byte sequence.

    Hermes' ``cache_image_from_bytes`` has a format-sniff guard that rejects
    non-image payloads — use a real PNG signature so the test exercises the
    full pipeline instead of the reject path.
    """
    # 1x1 transparent PNG
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )


class TestMimeExtension:
    def test_maps_jpeg_variants_to_jpg(self):
        from tools.mcp_tool_content import _mcp_image_extension_for_mime_type
        assert _mcp_image_extension_for_mime_type("image/jpeg") == ".jpg"
        assert _mcp_image_extension_for_mime_type("image/jpg") == ".jpg"
        assert _mcp_image_extension_for_mime_type("IMAGE/JPEG") == ".jpg"
        assert _mcp_image_extension_for_mime_type("image/jpeg; charset=utf-8") == ".jpg"


    def test_unknown_defaults_to_png(self):
        from tools.mcp_tool_content import _mcp_image_extension_for_mime_type
        assert _mcp_image_extension_for_mime_type("") == ".png"
        assert _mcp_image_extension_for_mime_type("image/unheard-of-format") == ".png"


class TestCacheMcpImageBlock:
    def test_returns_media_tag_for_valid_image_block(self, tmp_path, monkeypatch):
        """A well-formed ImageContent block with valid PNG bytes caches
        to the image dir and the helper returns a ``MEDIA:<path>`` tag."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(_png_bytes()).decode("ascii"),
            mimeType="image/png",
        )
        tag = _cache_mcp_image_block(block)
        assert tag.startswith("MEDIA:"), f"expected MEDIA: tag, got {tag!r}"
        # The cached file should be in Hermes' image cache dir
        from gateway.platforms.base import get_image_cache_dir
        cache_dir = str(get_image_cache_dir().resolve())
        assert tag.startswith(f"MEDIA:{cache_dir}"), (
            f"cached file not under HERMES_HOME image cache dir. "
            f"tag={tag!r}, cache_dir={cache_dir!r}"
        )
        # And it should exist + have the PNG bytes
        path = tag[len("MEDIA:"):]
        with open(path, "rb") as fh:
            assert fh.read() == _png_bytes()

    def test_returns_empty_when_block_is_not_an_image(self, tmp_path, monkeypatch):
        """Non-image MIME types shouldn't trigger caching."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(b"some bytes").decode("ascii"),
            mimeType="application/pdf",
        )
        assert _cache_mcp_image_block(block) == ""

    def test_returns_empty_when_block_has_no_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        block = SimpleNamespace(data=None, mimeType="image/png")
        assert _cache_mcp_image_block(block) == ""


    def test_returns_empty_when_bytes_dont_look_like_an_image(self, tmp_path, monkeypatch):
        """``cache_image_from_bytes`` has a format sniff; if the claimed
        ``image/png`` is actually an HTML error page, the cache raises and
        we log + drop rather than propagate."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(b"<html>error</html>").decode("ascii"),
            mimeType="image/png",
        )
        assert _cache_mcp_image_block(block) == ""

    def test_handles_jpeg(self, tmp_path, monkeypatch):
        """JPEG signature should also be accepted."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        # minimal JPEG SOI marker + filler
        jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
        block = SimpleNamespace(
            data=base64.b64encode(jpeg).decode("ascii"),
            mimeType="image/jpeg",
        )
        tag = _cache_mcp_image_block(block)
        assert tag.startswith("MEDIA:")
        assert tag.endswith(".jpg"), f"expected .jpg extension, got {tag!r}"


class TestNativeImageAttach:
    """An MCP tool registered the real way and dispatched through the registry hands a vision route the
    pixels via the shared native gate; the image is re-encoded from the sniffed cache file within the embed
    budget, never the server's raw payload."""

    def _call(self, monkeypatch, tmp_path, cfg, image: bytes = b"", mime: str = "image/png"):
        import asyncio
        import io
        from unittest.mock import AsyncMock, patch
        from PIL import Image
        from tools import mcp_tool
        from tools.mcp_tool_registration import _register_server_tools
        from tools.registry import ToolRegistry

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        if not image:
            buf = io.BytesIO()
            Image.new("RGB", (3000, 2000), (20, 90, 200)).save(buf, "PNG")  # past the 1568 px embed edge
            image = buf.getvalue()
        result = SimpleNamespace(isError=False, structuredContent=None, meta=None, content=[
            SimpleNamespace(type="text", text="snapshot"),
            SimpleNamespace(type="image", data=base64.b64encode(image).decode(), mimeType=mime)])
        server = mcp_tool.MCPServerTask("srv")
        server._tools = [SimpleNamespace(name="snap", description="screenshot", inputSchema=None)]
        server.session = SimpleNamespace(call_tool=AsyncMock(return_value=result))

        def run(factory, timeout=30):
            async def go():
                server._rpc_lock = asyncio.Lock()
                return await factory()
            return asyncio.run(go())

        registry = ToolRegistry()
        with patch("tools.registry.registry", registry), \
             patch.dict(mcp_tool._servers, {"srv": server}), \
             patch("tools.mcp_tool_loop._run_on_mcp_loop", side_effect=run), \
             patch("hermes_cli.config.load_config", return_value=cfg), \
             patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"), \
             patch("agent.auxiliary_client._read_main_model", return_value="claude-opus-4-5"), \
             patch("agent.image_routing._lookup_supports_vision", return_value=True):
            assert "mcp__srv__snap" in _register_server_tools("srv", server, {})
            return registry.dispatch("mcp__srv__snap", {})

    def test_vision_route_gets_a_resized_envelope_and_keeps_the_media_path(self, tmp_path, monkeypatch):
        out = self._call(monkeypatch, tmp_path, {})
        assert isinstance(out, dict) and out["_multimodal"] is True
        url = out["content"][1]["image_url"]["url"]
        from tools.vision_tools_history_budget import resolve_embed_target_bytes
        assert url.startswith("data:image/jpeg;base64,") and len(url) <= resolve_embed_target_bytes()
        assert "MEDIA:" in out["text_summary"]

    def test_image_input_mode_text_or_an_undecodable_image_keeps_the_string_result(self, tmp_path, monkeypatch):
        assert "MEDIA:" in self._call(monkeypatch, tmp_path, {"agent": {"image_input_mode": "text"}})
        # A valid JPEG header over a truncated pixel stream passes the cache's and the sniff's header checks.
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.effect_noise((256, 256), 64).convert("RGB").save(buf, "JPEG")
        truncated = self._call(monkeypatch, tmp_path, {}, image=buf.getvalue()[: len(buf.getvalue()) // 2],
                               mime="image/jpeg")
        assert isinstance(truncated, str) and "MEDIA:" in truncated
