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
import json
from types import SimpleNamespace

import pytest


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


    def test_bytes_the_cache_rejects_become_an_inline_notice(self, tmp_path, monkeypatch):
        """``cache_image_from_bytes`` has a format sniff; if the claimed
        ``image/png`` is actually an HTML error page, the cache raises and
        we log + report the block inline rather than propagate or drop it."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(b"<html>error</html>").decode("ascii"),
            mimeType="image/png",
        )
        tag = _cache_mcp_image_block(block)
        assert not tag.startswith("MEDIA:") and "image/png" in tag

    @pytest.mark.parametrize(("kind", "mime", "data"), [
        ("image", "image/svg+xml", base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode()),
        ("image", "image/png", "!!!notbase64"),
        ("audio", "audio/wav", "!!!notbase64"),
        # Only non-alphabet characters, or valid groups behind them: a lenient decoder yields b"" or
        # garbage bytes, which the audio cache (no format sniff) would store as a MEDIA file.
        ("audio", "audio/wav", "!!!!"),
        ("audio", "audio/wav", "!!!!abcd"),
    ])
    def test_media_only_result_that_cannot_be_cached_is_visible_to_the_model(
        self, tmp_path, monkeypatch, kind, mime, data,
    ):
        """A tool whose only output is a media block Hermes cannot turn into a MEDIA tag must
        not read as a successful empty result: the model has to learn the block existed."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_handlers import _render_call_tool_result

        result = SimpleNamespace(content=[SimpleNamespace(type=kind, data=data, mimeType=mime)],
                                 isError=False, structuredContent=None, meta=None)
        rendered = json.loads(_render_call_tool_result(result, "diagrams"))["result"]
        assert kind in rendered and mime in rendered

    @pytest.mark.parametrize("block", [
        SimpleNamespace(type="image", data=base64.b64encode(b"<svg/>").decode(), mimeType="image/svg+xml"),
        SimpleNamespace(type="audio", data="!!!notbase64", mimeType="audio/wav"),
        SimpleNamespace(type="resource", resource=SimpleNamespace(
            uri="x://r.bin", mimeType="application/octet-stream", blob="!!!notbase64", text=None)),
    ], ids=["image-cache-rejects", "audio-bad-base64", "resource-bad-base64"])
    def test_failure_notice_does_not_displace_structured_content(self, tmp_path, monkeypatch, block):
        """A failure notice says a block existed, not what it held. Like the unsupported-block
        notice, it must not count as rendered content, or the over-cap structuredContent that an
        otherwise-empty result carries is dropped in its favour."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS
        from tools.mcp_tool_handlers import _render_call_tool_result

        def render(content):
            return json.loads(_render_call_tool_result(SimpleNamespace(
                content=content, isError=False, meta=None,
                structuredContent={"rows": ["x" * 100] * (_MCP_HARD_RESULT_CAP_CHARS // 50)}), "srv"))

        structured_alone = render([])["result"]
        with_notice = render([block])
        assert with_notice.get("structuredContent") == structured_alone
        assert "[MCP " in with_notice["result"]

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
