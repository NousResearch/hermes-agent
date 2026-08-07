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


    def test_truncation_stores_full_text_readable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        monkeypatch.setenv("TERMINAL_ENV", "local")
        body = "UNIQUE_MIDDLE_MARKER\n" + ("\n".join(f"row {i}" for i in range(5000)))
        out, truncated = wt._truncate_with_footer(body, "https://example.com/doc", 3000)
        assert truncated is True
        # Extract the stored path from the footer and confirm full text is there.
        path_line = next(ln for ln in out.splitlines() if "Full text saved to:" in ln)
        stored_path = path_line.split("Full text saved to:", 1)[1].strip()
        assert stored_path.startswith(str(tmp_path / ".hermes"))
        assert os.path.exists(stored_path)
        full = open(stored_path).read()
        assert "UNIQUE_MIDDLE_MARKER" in full
        assert "row 2500" in full  # the omitted-middle row is in the stored file

    def test_truncation_uses_docker_visible_cache_path(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        body = "\n".join(f"row {i}" for i in range(5000))

        out, truncated = wt._truncate_with_footer(
            body, "https://example.com/doc", 3000
        )

        assert truncated is True
        stored_files = list((hermes_home / "cache" / "web").glob("example.com-*.md"))
        assert len(stored_files) == 1
        visible_path = f"/root/.hermes/cache/web/{stored_files[0].name}"
        assert f"Full text saved to: {visible_path}" in out
        assert f'read_file path="{visible_path}"' in out
        assert str(hermes_home) not in out
        assert "row 2500" in stored_files[0].read_text(encoding="utf-8")


class TestCharLimitConfig:
    def test_default_when_unset(self):
        with patch("tools.web_tools._load_web_config", return_value={}):
            assert wt._get_extract_char_limit() == wt.DEFAULT_EXTRACT_CHAR_LIMIT


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
