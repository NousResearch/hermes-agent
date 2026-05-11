"""Tests for the web_extract fallback chain."""
from __future__ import annotations

import json
from typing import List

import pytest


class TestGetExtractFallbackBackends:
    def test_empty_when_unconfigured(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        assert web_tools._get_extract_fallback_backends("firecrawl") == []

    def test_filters_unavailable_backends(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": ["exa", "parallel"]},
        )

        def fake_available(name: str) -> bool:
            return name == "exa"

        monkeypatch.setattr(web_tools, "_is_backend_available", fake_available)
        assert web_tools._get_extract_fallback_backends("firecrawl") == ["exa"]

    def test_excludes_primary(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": ["exa", "firecrawl"]},
        )
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda name: True)
        assert web_tools._get_extract_fallback_backends("firecrawl") == ["exa"]

    def test_dedupes(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": ["exa", "exa", "EXA"]},
        )
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda name: True)
        assert web_tools._get_extract_fallback_backends("firecrawl") == ["exa"]

    def test_non_list_value_returns_empty(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": "exa"},
        )
        assert web_tools._get_extract_fallback_backends("firecrawl") == []

    def test_ignores_search_only_backends(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": ["brave-free", "exa"]},
        )
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda name: True)
        assert web_tools._get_extract_fallback_backends("firecrawl") == ["exa"]

    def test_preserves_order(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_fallback_backends": ["parallel", "exa"]},
        )
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda name: True)
        assert web_tools._get_extract_fallback_backends("firecrawl") == ["parallel", "exa"]


@pytest.mark.asyncio
class TestWebExtractFallbackBehavior:
    _OK_RESULT = [
        {
            "url": "https://example.com",
            "title": "Example",
            "content": "hello",
            "raw_content": "hello",
        }
    ]

    async def test_no_fallback_returns_primary_response(self, monkeypatch):
        from tools import web_tools

        call_log: List[str] = []

        async def fake_run(backend, safe_urls, format):
            call_log.append(backend)
            return self._OK_RESULT

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "exa")
        monkeypatch.setattr(web_tools, "_get_extract_fallback_backends", lambda exclude: [])
        monkeypatch.setattr(web_tools, "_run_extract_backend", fake_run)

        out = json.loads(
            await web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
        )
        assert call_log == ["exa"]
        assert out["results"][0]["url"] == "https://example.com"

    async def test_falls_back_when_primary_fails(self, monkeypatch):
        from tools import web_tools

        responses = [RuntimeError("primary fail"), self._OK_RESULT]
        call_order: List[str] = []

        async def fake_run(backend, safe_urls, format):
            call_order.append(backend)
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "exa")
        monkeypatch.setattr(
            web_tools,
            "_get_extract_fallback_backends",
            lambda exclude: ["parallel"],
        )
        monkeypatch.setattr(web_tools, "_run_extract_backend", fake_run)

        out = json.loads(
            await web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
        )
        assert call_order == ["exa", "parallel"]
        assert out["results"][0]["url"] == "https://example.com"

    async def test_returns_last_failure_when_all_backends_fail(self, monkeypatch):
        from tools import web_tools

        async def fake_run(backend, safe_urls, format):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "exa")
        monkeypatch.setattr(
            web_tools,
            "_get_extract_fallback_backends",
            lambda exclude: ["parallel"],
        )
        monkeypatch.setattr(web_tools, "_run_extract_backend", fake_run)

        out = json.loads(
            await web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
        )
        assert "rate limited" in out["error"]

    async def test_no_safe_urls_skips_backend_calls(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "is_safe_url", lambda url: False)
        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "exa")
        monkeypatch.setattr(web_tools, "_get_extract_fallback_backends", lambda exclude: ["parallel"])

        async def fake_run(backend, safe_urls, format):
            raise AssertionError("_run_extract_backend should not be called")

        monkeypatch.setattr(web_tools, "_run_extract_backend", fake_run)

        out = json.loads(
            await web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
        )
        assert out["results"][0]["error"] == "Blocked: URL targets a private or internal network address"

    async def test_search_only_error_passes_through(self, monkeypatch):
        from tools import web_tools

        async def fake_run(backend, safe_urls, format):
            raise ValueError(
                "SearXNG is a search-only backend and cannot extract URL content. "
                "Set web.extract_backend to firecrawl, tavily, exa, or parallel."
            )

        monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "searxng")
        monkeypatch.setattr(web_tools, "_get_extract_fallback_backends", lambda exclude: [])
        monkeypatch.setattr(web_tools, "_run_extract_backend", fake_run)

        out = json.loads(
            await web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
        )
        assert "search-only backend" in out["error"] or "search-only" in out["error"]
