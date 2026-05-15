"""Tests for Crawl4AI-powered deep crawl tool."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "crawl4ai_tool.py"
SPEC = importlib.util.spec_from_file_location("crawl4ai_tool_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
crawl4ai_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = crawl4ai_tool
SPEC.loader.exec_module(crawl4ai_tool)


@pytest.mark.asyncio
async def test_crawl4ai_deep_crawl_returns_trimmed_pages(monkeypatch):
    """A successful Crawl4AI run should return compact page records."""

    class FakeStrategy:
        def __init__(self, max_depth, include_external, max_pages):
            self.max_depth = max_depth
            self.include_external = include_external
            self.max_pages = max_pages

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMarkdown:
        raw_markdown = "# Example Domain\n\nThis domain is for examples."

    class FakeResult:
        url = "https://example.com"
        success = True
        markdown = FakeMarkdown()
        html = "<h1>Example Domain</h1>"
        metadata = {"title": "Example Domain"}
        links = {
            "internal": [
                {"href": "https://example.com/about"},
                {"href": "https://example.com/docs"},
            ],
            "external": [{"href": "https://iana.org/domains/reserved"}],
        }

    class FakeCrawler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def arun(self, url, config):
            assert url == "https://example.com"
            assert config.kwargs["deep_crawl_strategy"].max_depth == 2
            assert config.kwargs["deep_crawl_strategy"].max_pages == 3
            assert config.kwargs["deep_crawl_strategy"].include_external is False
            assert config.kwargs["excluded_tags"] == ["script", "style"]
            assert config.kwargs["remove_forms"] is True
            assert config.kwargs["page_timeout"] == 12345
            return [FakeResult()]

    monkeypatch.setattr(crawl4ai_tool, "is_safe_url", lambda url: True)
    monkeypatch.setattr(crawl4ai_tool, "check_website_access", lambda url: None)
    monkeypatch.setattr(crawl4ai_tool, "_current_python_has_crawl4ai", lambda: True)
    monkeypatch.setattr(crawl4ai_tool, "_load_crawl4ai_classes", lambda: (FakeCrawler, FakeConfig, FakeStrategy))

    raw = await crawl4ai_tool.crawl4ai_deep_crawl_tool(
        "https://example.com",
        max_depth=2,
        max_pages=3,
        include_external=False,
        excluded_tags=["script", "style"],
        remove_forms=True,
        page_timeout_ms=12345,
        max_content_chars_per_page=1000,
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["tool"] == "crawl4ai_deep_crawl"
    assert payload["crawl"]["pages_returned"] == 1
    assert payload["results"] == [
        {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "# Example Domain\n\nThis domain is for examples.",
            "links": {
                "internal": ["https://example.com/about", "https://example.com/docs"],
                "external": ["https://iana.org/domains/reserved"],
            },
            "error": None,
        }
    ]


@pytest.mark.asyncio
async def test_crawl4ai_deep_crawl_blocks_private_urls(monkeypatch):
    monkeypatch.setattr(crawl4ai_tool, "is_safe_url", lambda url: False)

    raw = await crawl4ai_tool.crawl4ai_deep_crawl_tool("http://127.0.0.1:8000")

    payload = json.loads(raw)
    assert payload["success"] is False
    assert "private or internal" in payload["error"]


def test_check_crawl4ai_available_uses_packaged_import():
    assert isinstance(crawl4ai_tool.check_crawl4ai_available(), bool)


def test_parse_subprocess_stdout_ignores_crawl4ai_progress_logs():
    raw = (
        "[INIT].... → Crawl4AI 0.8.6\n"
        "[FETCH]... ↓ https://example.com\n"
        "__HERMES_CRAWL4AI_JSON__:{\"success\": true, \"results\": []}\n"
    )

    assert crawl4ai_tool._parse_subprocess_stdout(raw) == {"success": True, "results": []}
