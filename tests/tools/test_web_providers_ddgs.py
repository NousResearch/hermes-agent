"""Tests for the DuckDuckGo (ddgs) web search provider.

Covers:
- DDGSWebSearchProvider.is_available() — reflects package importability
- DDGSWebSearchProvider.search() — happy path, missing package, runtime error
- Result normalization (title, url, description, position)
- DDGSWebSearchProvider.extract() — happy path, per-URL error, missing package
- _is_backend_available("ddgs") / _get_backend() integration
- web_extract with ddgs backend (no longer search-only)
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from tests.tools.conftest import register_all_web_providers


def _install_fake_ddgs(
    monkeypatch,
    *,
    text_results=None,
    text_raises=None,
    extract_results=None,
    extract_raises=None,
):
    """Install a stub ``ddgs`` module in sys.modules for the duration of a test.

    ``text_results``: iterable of dicts to yield from DDGS().text(...).
    ``text_raises``: if set, DDGS().text raises this exception instead.
    ``extract_results``: dict mapping URL -> {title, content} returned from
        DDGS().extract(...). Yields nothing on miss.
    ``extract_raises``: if set, DDGS().extract raises this exception instead.
    """
    fake = types.ModuleType("ddgs")

    class _FakeDDGS:
        def __enter__(self):
            return self
        def __exit__(self, *_a):
            return False
        def text(self, query, max_results=5):
            if text_raises is not None:
                raise text_raises
            for hit in (text_results or []):
                yield hit
        def extract(self, url, fmt=None):
            # ddgs 9.14 returns a dict with keys {url, content}. Earlier
            # versions yielded a generator of single-key dicts; the test
            # fixtures target the modern contract.
            if extract_raises is not None:
                raise extract_raises
            if extract_results and url in extract_results:
                return extract_results[url]
            # Match real ddgs behavior: empty dict on miss (the provider
            # treats the empty content as "no content extracted" rather
            # than an error).
            return {}

    fake.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake)
    return fake


# ---------------------------------------------------------------------------
# DDGSWebSearchProvider unit tests
# ---------------------------------------------------------------------------


class TestDDGSProviderIsConfigured:
    def test_configured_when_package_importable(self, monkeypatch):
        _install_fake_ddgs(monkeypatch)
        # Drop any cached ``plugins.web.ddgs.provider`` so is_configured re-imports ddgs fresh
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().is_available() is True

    def test_not_configured_when_package_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        # Block the import so ``import ddgs`` raises ImportError even if the package is actually installed
        import builtins
        orig_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().is_available() is False

    def test_provider_name(self):
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().name == "ddgs"

    def test_implements_web_search_provider(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert issubclass(DDGSWebSearchProvider, WebSearchProvider)


class TestDDGSProviderSearch:
    def test_happy_path_normalizes_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "href": "https://a.example.com", "body": "desc A"},
            {"title": "B", "href": "https://b.example.com", "body": "desc B"},
            {"title": "C", "href": "https://c.example.com", "body": "desc C"},
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 3
        assert web[0] == {"title": "A", "url": "https://a.example.com", "description": "desc A", "position": 1}
        assert web[2]["position"] == 3

    def test_accepts_url_key_as_fallback_for_href(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": "A", "url": "https://a.example.com", "body": "desc A"},
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)

        assert result["success"] is True
        assert result["data"]["web"][0]["url"] == "https://a.example.com"

    def test_limit_is_respected(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[
            {"title": f"R{i}", "href": f"https://r{i}.example.com", "body": ""}
            for i in range(10)
        ])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=3)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 3

    def test_missing_package_returns_failure(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import builtins
        orig_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "ddgs" in result["error"].lower()

    def test_runtime_error_returns_failure(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_raises=RuntimeError("rate limited 202"))
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("q", limit=5)
        assert result["success"] is False
        assert "rate limited" in result["error"] or "failed" in result["error"].lower()

    def test_empty_results(self, monkeypatch):
        _install_fake_ddgs(monkeypatch, text_results=[])
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        result = DDGSWebSearchProvider().search("nothing", limit=5)
        assert result["success"] is True
        assert result["data"]["web"] == []


# ---------------------------------------------------------------------------
# Integration: _is_backend_available / _get_backend / check_web_api_key
# ---------------------------------------------------------------------------


class TestDDGSBackendWiring:
    def test_is_backend_available_true_when_package_importable(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._is_backend_available("ddgs") is True

    def test_is_backend_available_false_when_package_missing(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        assert web_tools._is_backend_available("ddgs") is False

    def test_configured_backend_accepted(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "ddgs"

    def test_ddgs_trails_paid_providers_in_auto_detect(self, monkeypatch):
        """Exa (priority) should win over ddgs in auto-detect."""
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "exa"

    def test_auto_detect_picks_ddgs_as_last_resort(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})
        for key in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
                    "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL", "BRAVE_SEARCH_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools._get_backend() == "ddgs"

    def test_check_web_api_key_true_when_ddgs_configured(self, monkeypatch):
        from tools import web_tools
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        assert web_tools.check_web_api_key() is True


# ---------------------------------------------------------------------------
# ddgs extract: provider implements extract() (ddgs package >= 8.0)
# ---------------------------------------------------------------------------


class TestDDGSProviderExtract:
    """Unit tests for the extract() method on the provider itself."""

    def test_extract_returns_normalized_content(self, monkeypatch):
        # DDGS.extract() returns {url, content} per URL — no title field. The
        # provider normalizes this into the hermes registry shape with a
        # blank title (the registry contract reserves the field but the
        # upstream package doesn't populate it).
        _install_fake_ddgs(
            monkeypatch,
            extract_results={
                "https://example.com": {
                    "url": "https://example.com",
                    "content": "Hello, world.",
                }
            },
        )
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        results = DDGSWebSearchProvider().extract(["https://example.com"])

        assert len(results) == 1
        r = results[0]
        assert r["url"] == "https://example.com"
        assert r["title"] == ""  # DDGS doesn't surface a title
        assert r["content"] == "Hello, world."
        assert r["raw_content"] == "Hello, world."
        assert r["metadata"]["source"] == "ddgs"
        assert r["metadata"]["length"] == len("Hello, world.")
        assert "error" not in r

    def test_extract_multiple_urls(self, monkeypatch):
        # With real ddgs 9.14, a URL that returns empty content (or empty
        # dict from a miss) still surfaces as a result entry — the
        # provider doesn't error. The test verifies all three URLs are
        # returned in order, with the known-good ones populated and the
        # unknown one having empty content.
        _install_fake_ddgs(
            monkeypatch,
            extract_results={
                "https://a.example.com": {"url": "https://a.example.com", "content": "AAA"},
                "https://b.example.com": {"url": "https://b.example.com", "content": "BBB"},
            },
        )
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        results = DDGSWebSearchProvider().extract(
            ["https://a.example.com", "https://b.example.com", "https://c.example.com"]
        )

        urls = [r["url"] for r in results]
        assert urls == [
            "https://a.example.com",
            "https://b.example.com",
            "https://c.example.com",
        ]
        assert results[0]["content"] == "AAA"
        assert results[1]["content"] == "BBB"
        assert results[2]["content"] == ""  # empty dict from miss
        assert "error" not in results[2]

    def test_extract_runtime_error_per_url(self, monkeypatch):
        # When the package raises on extract, every URL gets an error entry
        # rather than the whole call blowing up.
        _install_fake_ddgs(monkeypatch, extract_raises=RuntimeError("rate limited"))
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        results = DDGSWebSearchProvider().extract(["https://example.com"])

        assert len(results) == 1
        assert "error" in results[0]
        assert "rate limited" in results[0]["error"]

    def test_extract_missing_package(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "ddgs", raising=False)
        monkeypatch.delitem(sys.modules, "plugins.web.ddgs.provider", raising=False)
        import builtins
        orig_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "ddgs":
                raise ImportError("blocked for test")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        from plugins.web.ddgs.provider import DDGSWebSearchProvider

        results = DDGSWebSearchProvider().extract(["https://example.com"])

        assert len(results) == 1
        assert "error" in results[0]
        assert "ddgs" in results[0]["error"].lower()

    def test_supports_extract_returns_true(self):
        from plugins.web.ddgs.provider import DDGSWebSearchProvider
        assert DDGSWebSearchProvider().supports_extract() is True


class TestDDGSExtractIntegration:
    """web_extract end-to-end with the ddgs backend installed and selected.

    Replaces the previous TestDDGSSearchOnlyErrors class which asserted that
    ddgs was search-only — the underlying ``ddgs`` Python package gained an
    extract() method in 8.0, and the provider now exposes it.
    """

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

    def test_web_extract_returns_ddgs_content(self, monkeypatch):
        import asyncio
        from tools import web_tools

        _install_fake_ddgs(
            monkeypatch,
            extract_results={
                "https://example.com": {
                    "url": "https://example.com",
                    "content": "Example body text.",
                }
            },
        )
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "ddgs"})
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: True)
        monkeypatch.setattr(web_tools, "_is_tool_gateway_ready", lambda: False)

        async def _allow_ssrf(_url: str) -> bool:
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_ssrf)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        result_str = asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
        result = json.loads(result_str)
        # On success the dispatcher returns {"results": [...]} (no "success"
        # key — the presence of "results" with content IS success). On
        # failure, the response is {"success": False, "error": "..."}.
        assert "results" in result
        assert "error" not in result
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com"
        assert "Example body text." in result["results"][0]["content"]
