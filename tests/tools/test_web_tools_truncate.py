"""Unit tests for the truncate-and-store web_extract path (no LLM).

Covers convert_base64_images_to_links, _truncate_with_footer, _store_full_text,
_get_extract_char_limit, and the end-to-end web_extract_tool truncation behavior.
"""
import asyncio
import json
import os
from unittest.mock import patch

import pytest

import tools.web_tools as wt


class TestImageConversion:
    def test_markdown_base64_image_keeps_alt_drops_blob(self):
        blob = "A" * 5000
        text = f"before ![a cat]( data:image/png;base64,{blob}) after"
        out = wt.convert_base64_images_to_links(text)
        assert "[IMAGE: a cat]" in out
        assert "base64" not in out
        assert blob not in out
        assert "before" in out and "after" in out

    def test_markdown_base64_image_no_alt(self):
        out = wt.convert_base64_images_to_links("x ![](data:image/jpeg;base64,QQ==) y")
        assert "[IMAGE]" in out
        assert "base64" not in out

    def test_real_http_image_links_preserved(self):
        text = "see ![logo](https://example.com/logo.png) here"
        out = wt.convert_base64_images_to_links(text)
        # Real image URLs must survive so the agent can inspect them.
        assert "![logo](https://example.com/logo.png)" in out

    def test_bare_and_parenthesised_base64_become_placeholder(self):
        blob = "Z" * 3000
        bare = wt.convert_base64_images_to_links(f"data:image/gif;base64,{blob}")
        assert bare == "[IMAGE]"
        paren = wt.convert_base64_images_to_links(f"(data:image/gif;base64,{blob})")
        assert paren == "[IMAGE]"


class TestTruncation:
    def test_short_content_returned_whole(self):
        content = "# Title\n\nshort body\n"
        out, truncated = wt._truncate_with_footer(content, "https://e.com", 15000)
        assert out == content
        assert truncated is False

    def test_long_content_truncated_with_footer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        body = "\n".join(f"line {i} " + "x" * 50 for i in range(2000))
        out, truncated = wt._truncate_with_footer(body, "https://example.com/page", 4000)
        assert truncated is True
        assert "[TRUNCATED]" in out
        assert "Full text saved to:" in out
        assert "read_file" in out
        # Head and tail are both present (first and last lines survive).
        assert "line 0 " in out
        assert "line 1999 " in out
        # The omitted middle is gone.
        assert "line 1000 " not in out
        # Sent text is bounded near the budget (+ footer overhead).
        assert len(out) < 4000 + 2000

    def test_truncation_stores_full_text_readable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        body = "UNIQUE_MIDDLE_MARKER\n" + ("\n".join(f"row {i}" for i in range(5000)))
        out, truncated = wt._truncate_with_footer(body, "https://example.com/doc", 3000)
        assert truncated is True
        # Extract the stored path from the footer and confirm full text is there.
        path_line = next(ln for ln in out.splitlines() if "Full text saved to:" in ln)
        stored_path = path_line.split("Full text saved to:", 1)[1].strip()
        assert os.path.exists(stored_path)
        full = open(stored_path).read()
        assert "UNIQUE_MIDDLE_MARKER" in full
        assert "row 2500" in full  # the omitted-middle row is in the stored file


class TestCharLimitConfig:
    def test_default_when_unset(self):
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert wt._get_extract_char_limit() == wt.DEFAULT_EXTRACT_CHAR_LIMIT

    def test_config_override(self):
        with patch("tools.web_tools._load_web_config", return_value={"extract_char_limit": 40000}):
            assert wt._get_extract_char_limit() == 40000

    def test_clamps_floor(self):
        with patch("tools.web_tools._load_web_config", return_value={"extract_char_limit": 100}):
            assert wt._get_extract_char_limit() == 2000

    def test_bad_value_falls_back(self):
        with patch("tools.web_tools._load_web_config", return_value={"extract_char_limit": "nope"}):
            assert wt._get_extract_char_limit() == wt.DEFAULT_EXTRACT_CHAR_LIMIT


class TestEndToEnd:
    def test_web_extract_truncates_large_page_no_llm(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        big = "\n".join(f"para {i} " + "y" * 80 for i in range(3000))

        class FakeProvider:
            name = "fake"
            display_name = "Fake"

            def supports_extract(self):
                return True

            async def extract(self, urls, **kwargs):
                return [{"url": urls[0], "title": "Big Page", "content": big,
                         "raw_content": big, "metadata": {}}]

        with patch("tools.web_tools._ensure_web_plugins_loaded"), \
             patch("tools.web_tools._get_extract_backend", return_value="fake"), \
             patch("tools.web_tools.async_is_safe_url", new=_AsyncTrue()), \
             patch("agent.web_search_registry.resolve_fallback_chain",
                   return_value=[FakeProvider()]), \
             patch("agent.web_search_registry.get_active_extract_provider",
                   return_value=FakeProvider()), \
             patch("agent.web_search_registry.get_provider", return_value=FakeProvider()):
            result = json.loads(asyncio.new_event_loop().run_until_complete(
                wt.web_extract_tool(["https://example.com/big"], char_limit=5000)
            ))

        assert "results" in result
        content = result["results"][0]["content"]
        assert "[TRUNCATED]" in content
        assert "Full text saved to:" in content
        # No LLM was involved: para 0 (head) and the last para (tail) are verbatim.
        assert "para 0 " in content
        assert "para 2999 " in content


def _make_awaitable(value):
    async def _coro(*a, **k):
        return value
    return _coro()


class _AsyncTrue:
    """Async callable that always returns True (re-awaitable per call)."""
    async def __call__(self, *a, **k):
        return True
