"""Tests for configurable web search/extract fallback chains.

Covers:
- _get_fallback_backends() — config parsing (list, comma/space string,
  bracketed string, dedup, primary removal, empty/whitespace handling)
- _run_search_fallbacks() — primary fails → first healthy fallback wins;
  all fail → primary error surfaced; empty chain → primary result unchanged;
  unavailable / unregistered / search-incapable candidates skipped
- web_search_tool() end-to-end — brave-free 402 → exa fallback returns results
- _run_extract_fallbacks() — primary raises → fallback result returned;
  all raise → primary exception re-raised; per-URL errors do NOT trigger fallback
- web_extract_tool() end-to-end — primary raises → fallback extract used
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# _get_fallback_backends — config parsing
# ---------------------------------------------------------------------------


class TestGetFallbackBackends:
    def test_list_value_preserved_in_order(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily"]})
        assert web_tools._get_fallback_backends("search") == ["exa", "tavily"]

    def test_comma_separated_string_split(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": "exa, tavily"})
        assert web_tools._get_fallback_backends("search") == ["exa", "tavily"]

    def test_bracketed_string_parsed(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": "[exa, tavily]"})
        assert web_tools._get_fallback_backends("search") == ["exa", "tavily"]

    def test_missing_key_yields_empty_chain(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        assert web_tools._get_fallback_backends("search") == []

    def test_null_value_yields_empty_chain(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": None})
        assert web_tools._get_fallback_backends("search") == []

    def test_primary_dropped_from_chain(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["brave-free", "exa"]})
        assert web_tools._get_fallback_backends("search", primary="brave-free") == ["exa"]

    def test_duplicates_collapsed_first_wins(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily", "exa"]})
        assert web_tools._get_fallback_backends("search") == ["exa", "tavily"]

    def test_whitespace_and_empty_entries_discarded(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["  ", "", "exa"]})
        assert web_tools._get_fallback_backends("search") == ["exa"]

    def test_case_normalized(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["EXA", "Tavily"]})
        assert web_tools._get_fallback_backends("search") == ["exa", "tavily"]

    def test_extract_capability_reads_its_own_key(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "search_fallbacks": ["exa"],
            "extract_fallbacks": ["tavily"],
        })
        assert web_tools._get_fallback_backends("extract") == ["tavily"]


# ---------------------------------------------------------------------------
# _run_search_fallbacks — dispatch behavior
# ---------------------------------------------------------------------------


class _FakeSearchProvider:
    """Minimal registered-shaped provider for fallback dispatch tests."""

    def __init__(self, name, result=None, available=True, supports=True, raises=None):
        self._name = name
        self._result = result
        self._available = available
        self._supports = supports
        self._raises = raises
        self.calls = 0

    @property
    def name(self):
        return self._name

    def supports_search(self):
        return self._supports

    def is_available(self):
        return self._available

    def search(self, query, limit=5):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._result


class TestRunSearchFallbacks:
    def _registry(self, providers):
        table = {p.name: p for p in providers}
        return lambda name: table.get(name)

    def test_first_healthy_fallback_wins(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily"]})
        exa = _FakeSearchProvider("exa", result={"success": True, "data": {"web": [{"title": "from exa"}]}})
        tavily = _FakeSearchProvider("tavily", result={"success": True, "data": {"web": []}})
        primary_result = {"success": False, "error": "Brave Search returned HTTP 402"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([exa, tavily]),
        )
        assert out["success"] is True
        assert out["data"]["web"][0]["title"] == "from exa"
        assert exa.calls == 1
        assert tavily.calls == 0  # never reached

    def test_skips_failing_fallback_to_next(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily"]})
        exa = _FakeSearchProvider("exa", result={"success": False, "error": "exa down"})
        tavily = _FakeSearchProvider("tavily", result={"success": True, "data": {"web": [{"title": "ok"}]}})
        primary_result = {"success": False, "error": "primary dead"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([exa, tavily]),
        )
        assert out["success"] is True
        assert exa.calls == 1
        assert tavily.calls == 1

    def test_all_fail_returns_primary_error(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa"]})
        exa = _FakeSearchProvider("exa", result={"success": False, "error": "exa down"})
        primary_result = {"success": False, "error": "Brave Search returned HTTP 402"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([exa]),
        )
        # Primary's error is more actionable than the fallback's.
        assert out is primary_result
        assert "402" in out["error"]

    def test_empty_chain_returns_primary_untouched(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        primary_result = {"success": False, "error": "boom"}
        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([]),
        )
        assert out is primary_result

    def test_unavailable_fallback_skipped(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily"]})
        exa = _FakeSearchProvider("exa", available=False)
        tavily = _FakeSearchProvider("tavily", result={"success": True, "data": {"web": []}})
        primary_result = {"success": False, "error": "primary dead"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([exa, tavily]),
        )
        assert out["success"] is True
        assert exa.calls == 0  # skipped — not available
        assert tavily.calls == 1

    def test_unregistered_fallback_skipped(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["ghost", "tavily"]})
        tavily = _FakeSearchProvider("tavily", result={"success": True, "data": {"web": []}})
        primary_result = {"success": False, "error": "primary dead"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([tavily]),  # "ghost" not registered
        )
        assert out["success"] is True
        assert tavily.calls == 1

    def test_raising_fallback_treated_as_failure(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"search_fallbacks": ["exa", "tavily"]})
        exa = _FakeSearchProvider("exa", raises=RuntimeError("kaboom"))
        tavily = _FakeSearchProvider("tavily", result={"success": True, "data": {"web": []}})
        primary_result = {"success": False, "error": "primary dead"}

        out = web_tools._run_search_fallbacks(
            "q", 5, primary_result, primary="brave-free",
            _wsp_get_provider=self._registry([exa, tavily]),
        )
        assert out["success"] is True
        assert tavily.calls == 1


# ---------------------------------------------------------------------------
# web_search_tool end-to-end — brave 402 → exa fallback
# ---------------------------------------------------------------------------


class TestWebSearchToolFallbackE2E:
    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        register_all_web_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_brave_402_falls_back_to_exa(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "search_backend": "brave-free",
            "search_fallbacks": ["exa"],
        })
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        # Brave returns a 402 failure; exa returns results.
        brave_fail = {"success": False, "error": "Brave Search returned HTTP 402"}
        exa_ok = {"success": True, "data": {"web": [{"title": "exa hit", "url": "https://x", "description": "", "position": 1}]}}

        from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
        from plugins.web.exa.provider import ExaWebSearchProvider
        monkeypatch.setattr(BraveFreeWebSearchProvider, "search", lambda self, q, limit=5: brave_fail)
        monkeypatch.setattr(ExaWebSearchProvider, "search", lambda self, q, limit=5: exa_ok)

        result = json.loads(web_tools.web_search_tool("test", limit=5))
        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "exa hit"

    def test_no_fallback_configured_surfaces_brave_error(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "search_backend": "brave-free",
        })
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        brave_fail = {"success": False, "error": "Brave Search returned HTTP 402"}
        from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
        monkeypatch.setattr(BraveFreeWebSearchProvider, "search", lambda self, q, limit=5: brave_fail)

        result = json.loads(web_tools.web_search_tool("test", limit=5))
        assert result["success"] is False
        assert "402" in result["error"]

    def test_empty_successful_result_does_not_fallback(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "search_backend": "brave-free",
            "search_fallbacks": ["exa"],
        })
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSAkey123")
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        brave_empty = {"success": True, "data": {"web": []}}
        from plugins.web.brave_free.provider import BraveFreeWebSearchProvider
        from plugins.web.exa.provider import ExaWebSearchProvider
        monkeypatch.setattr(BraveFreeWebSearchProvider, "search", lambda self, q, limit=5: brave_empty)

        exa_called = {"hit": False}

        def _exa_search(self, q, limit=5):
            exa_called["hit"] = True
            return {"success": True, "data": {"web": [{"title": "should not appear"}]}}

        monkeypatch.setattr(ExaWebSearchProvider, "search", _exa_search)

        result = json.loads(web_tools.web_search_tool("test", limit=5))
        assert result["success"] is True
        assert result["data"]["web"] == []
        assert exa_called["hit"] is False  # empty-but-successful = no fallback


# ---------------------------------------------------------------------------
# _run_extract_fallbacks — dispatch behavior
# ---------------------------------------------------------------------------


class _FakeExtractProvider:
    def __init__(self, name, result=None, available=True, supports=True, raises=None):
        self._name = name
        self._result = result
        self._available = available
        self._supports = supports
        self._raises = raises
        self.calls = 0

    @property
    def name(self):
        return self._name

    def supports_extract(self):
        return self._supports

    def is_available(self):
        return self._available

    def extract(self, urls, **kwargs):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._result


class TestRunExtractFallbacks:
    def _registry(self, providers):
        table = {p.name: p for p in providers}
        return lambda name: table.get(name)

    def test_fallback_result_returned_when_primary_raises(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_fallbacks": ["tavily"]})
        tavily = _FakeExtractProvider("tavily", result=[{"url": "https://x", "title": "t", "content": "c"}])
        primary = _FakeExtractProvider("firecrawl")
        primary_exc = RuntimeError("firecrawl unreachable")

        out = asyncio.get_event_loop().run_until_complete(
            web_tools._run_extract_fallbacks(
                ["https://x"], None, primary, primary_exc,
                _wsp_get_provider=self._registry([tavily]),
            )
        )
        assert out[0]["title"] == "t"
        assert tavily.calls == 1

    def test_all_raise_reraises_primary(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_fallbacks": ["tavily"]})
        tavily = _FakeExtractProvider("tavily", raises=RuntimeError("tavily down"))
        primary = _FakeExtractProvider("firecrawl")
        primary_exc = RuntimeError("firecrawl unreachable")

        with pytest.raises(RuntimeError, match="firecrawl unreachable"):
            asyncio.get_event_loop().run_until_complete(
                web_tools._run_extract_fallbacks(
                    ["https://x"], None, primary, primary_exc,
                    _wsp_get_provider=self._registry([tavily]),
                )
            )

    def test_empty_chain_reraises_primary(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        primary = _FakeExtractProvider("firecrawl")
        primary_exc = RuntimeError("primary dead")

        with pytest.raises(RuntimeError, match="primary dead"):
            asyncio.get_event_loop().run_until_complete(
                web_tools._run_extract_fallbacks(
                    ["https://x"], None, primary, primary_exc,
                    _wsp_get_provider=self._registry([]),
                )
            )

    def test_unavailable_fallback_skipped(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_fallbacks": ["exa", "tavily"]})
        exa = _FakeExtractProvider("exa", available=False)
        tavily = _FakeExtractProvider("tavily", result=[{"url": "https://x", "title": "t", "content": "c"}])
        primary = _FakeExtractProvider("firecrawl")
        primary_exc = RuntimeError("primary dead")

        out = asyncio.get_event_loop().run_until_complete(
            web_tools._run_extract_fallbacks(
                ["https://x"], None, primary, primary_exc,
                _wsp_get_provider=self._registry([exa, tavily]),
            )
        )
        assert out[0]["title"] == "t"
        assert exa.calls == 0
        assert tavily.calls == 1


# ---------------------------------------------------------------------------
# web_extract_tool end-to-end — primary raises → fallback used
# ---------------------------------------------------------------------------


class TestWebExtractToolFallbackE2E:
    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        register_all_web_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_primary_raises_falls_back_to_tavily(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "extract_backend": "exa",
            "extract_fallbacks": ["tavily"],
        })
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly")

        async def _allow_ssrf(_url):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        from plugins.web.exa.provider import ExaWebSearchProvider
        from plugins.web.tavily.provider import TavilyWebSearchProvider

        def _exa_raise(self, urls, **kwargs):
            raise RuntimeError("exa backend down")

        monkeypatch.setattr(ExaWebSearchProvider, "extract", _exa_raise)
        monkeypatch.setattr(
            TavilyWebSearchProvider, "extract",
            lambda self, urls, **kwargs: [{"url": urls[0], "title": "tavily", "content": "body", "raw_content": "body"}],
        )

        result = json.loads(asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        ))
        # Fallback content reached the post-processing pipeline.
        assert any("tavily" in (r.get("title") or "") for r in result["results"])
