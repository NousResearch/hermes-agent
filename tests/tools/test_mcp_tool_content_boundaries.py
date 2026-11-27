"""Boundary regressions for MCP content decoding."""

import base64
from types import SimpleNamespace

import pytest

from tools import mcp_tool_content


@pytest.mark.parametrize(
    ("mime_type", "render", "cache_subdir"),
    [
        ("image/png", mcp_tool_content._cache_mcp_image_block, "images"),
        ("audio/wav", mcp_tool_content._cache_mcp_audio_block, "audio"),
    ],
)
def test_malformed_base64_media_is_never_cached(
    tmp_path, monkeypatch, mime_type, render, cache_subdir
):
    """Invalid wire data must not become a successful MEDIA cache entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    block = SimpleNamespace(data="%%%%", mimeType=mime_type)

    assert render(block) == ""
    cache_dir = tmp_path / "cache" / cache_subdir
    assert not cache_dir.exists() or [path for path in cache_dir.iterdir() if path.is_file()] == []


@pytest.mark.parametrize(
    ("mime_type", "render", "payload"),
    [
        (
            "image/png",
            mcp_tool_content._cache_mcp_image_block,
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
        ),
        ("audio/wav", mcp_tool_content._cache_mcp_audio_block, b"RIFFfakewav"),
    ],
)
def test_line_folded_base64_media_preserves_decoded_bytes(
    tmp_path, monkeypatch, mime_type, render, payload
):
    """ASCII whitespace used to line-fold base64 must not change its payload."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    encoded = base64.b64encode(payload).decode("ascii")
    line_folded = f"\t{encoded[:8]}\r\n{encoded[8:16]} {encoded[16:]}\n"

    block = SimpleNamespace(data=line_folded, mimeType=mime_type)

    tag = render(block)
    assert tag.startswith("MEDIA:")
    cached_path = tag.removeprefix("MEDIA:")
    assert cached_path.startswith(str(tmp_path))
    with open(cached_path, "rb") as cached_file:
        assert cached_file.read() == payload
