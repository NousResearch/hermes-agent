"""Tests for the web tools provider architecture.

Covers:
- WebSearchProvider / WebExtractProvider ABC enforcement
- Per-capability backend selection (_get_search_backend, _get_extract_backend)
- Backward compatibility (web.backend still works as shared fallback)
- Config keys merge correctly via DEFAULT_CONFIG
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestWebProviderABCs:
    """The unified WebSearchProvider ABC enforces the interface contract.

    After PR #25182, all seven providers are subclasses of
    :class:`agent.web_search_provider.WebSearchProvider`. The legacy
    in-tree ABCs at ``tools.web_providers.base`` (separate
    ``WebSearchProvider`` + ``WebExtractProvider``) were deleted in the
    same PR — providers now advertise capabilities via
    ``supports_search() / supports_extract() / supports_crawl()`` flags.
    """

    def test_cannot_instantiate_abc_directly(self):
        from agent.web_search_provider import WebSearchProvider

        with pytest.raises(TypeError):
            WebSearchProvider()  # type: ignore[abstract]

    def test_concrete_search_only_provider_works(self):
        from agent.web_search_provider import WebSearchProvider

        class Dummy(WebSearchProvider):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def display_name(self) -> str:
                return "Dummy Search"

            def is_available(self) -> bool:
                return True

            def supports_search(self) -> bool:
                return True

            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                return {"success": True, "data": {"web": []}}

        d = Dummy()
        assert d.name == "dummy"
        assert d.display_name == "Dummy Search"
        assert d.is_available() is True
        assert d.supports_search() is True
        assert d.supports_extract() is False  # default
        assert d.supports_crawl() is False  # default
        assert d.search("test")["success"] is True

    def test_concrete_multi_capability_provider_works(self):
        from agent.web_search_provider import WebSearchProvider

        class Dummy(WebSearchProvider):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def display_name(self) -> str:
                return "Dummy Multi"

            def is_available(self) -> bool:
                return True

            def supports_search(self) -> bool:
                return True

            def supports_extract(self) -> bool:
                return True

            def supports_crawl(self) -> bool:
                return True

            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                return {"success": True, "data": {"web": []}}

            def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
                return [{"url": urls[0], "content": "x"}]

            def crawl(self, url: str, **kwargs: Any) -> Dict[str, Any]:
                return {"results": [{"url": url, "content": "x"}]}

        d = Dummy()
        assert d.supports_search() is True
        assert d.supports_extract() is True
        assert d.supports_crawl() is True
        assert d.extract(["https://example.com"])[0]["url"] == "https://example.com"
        assert d.crawl("https://example.com")["results"][0]["url"] == "https://example.com"

    def test_search_only_provider_skips_extract_and_crawl(self):
        """Search-only providers don't have to implement extract() / crawl()."""
        from agent.web_search_provider import WebSearchProvider

        class SearchOnly(WebSearchProvider):
            @property
            def name(self) -> str:
                return "search-only"

            @property
            def display_name(self) -> str:
                return "Search Only"

            def is_available(self) -> bool:
                return True

            def supports_search(self) -> bool:
                return True

            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                return {"success": True, "data": {"web": []}}

        # Should instantiate fine — extract/crawl have default
        # supports_*() returning False and aren't required to be
        # overridden when not advertised.
        s = SearchOnly()
        assert s.supports_search() is True
        assert s.supports_extract() is False
        assert s.supports_crawl() is False


# ---------------------------------------------------------------------------
# Per-capability backend selection
# ---------------------------------------------------------------------------


class TestPerCapabilityBackendSelection:
    """_get_search_backend and _get_extract_backend read per-capability config."""

    def test_search_backend_overrides_generic(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "firecrawl",
            "search_backend": "tavily",
        })
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        assert web_tools._get_search_backend() == "tavily"

    def test_extract_backend_overrides_generic(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "tavily",
            "extract_backend": "exa",
        })
        monkeypatch.setenv("EXA_API_KEY", "test-key")
        assert web_tools._get_extract_backend() == "exa"

    def test_falls_back_to_generic_backend_when_search_backend_empty(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "tavily",
            "search_backend": "",
        })
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        assert web_tools._get_search_backend() == "tavily"

    def test_falls_back_to_generic_backend_when_extract_backend_empty(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "parallel",
            "extract_backend": "",
        })
        monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
        assert web_tools._get_extract_backend() == "parallel"

    def test_search_backend_ignored_when_not_available(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "firecrawl",
            "search_backend": "exa",  # set but no EXA_API_KEY
        })
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        # Should fall back to firecrawl since exa isn't configured
        assert web_tools._get_search_backend() == "firecrawl"

    def test_fully_backward_compatible_with_web_backend_only(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "tavily",
        })
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        # No search_backend or extract_backend set — both fall through
        assert web_tools._get_search_backend() == "tavily"
        assert web_tools._get_extract_backend() == "tavily"


# ---------------------------------------------------------------------------
# Config key presence in DEFAULT_CONFIG
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """The web section exists in DEFAULT_CONFIG with per-capability keys."""

    def test_web_section_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert "web" in DEFAULT_CONFIG
        web = DEFAULT_CONFIG["web"]
        assert "backend" in web
        assert "search_backend" in web
        assert "extract_backend" in web
        # All empty string by default (no override)
        assert web["backend"] == ""
        assert web["search_backend"] == ""
        assert web["extract_backend"] == ""


# ---------------------------------------------------------------------------
# web_search_tool uses _get_search_backend
# ---------------------------------------------------------------------------


class TestWebSearchUsesSearchBackend:
    """web_search_tool dispatches through _get_search_backend not _get_backend."""

    def test_search_tool_calls_search_backend(self, monkeypatch):
        from tools import web_tools

        called_with = []
        original_get_search = web_tools._get_search_backend

        def tracking_get_search():
            result = original_get_search()
            called_with.append(("search", result))
            return result

        monkeypatch.setattr(web_tools, "_get_search_backend", tracking_get_search)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "firecrawl"})
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fake")

        # The function will fail at Firecrawl client level but we just
        # need to verify _get_search_backend was called
        try:
            web_tools.web_search_tool("test", 1)
        except Exception:
            pass

        assert len(called_with) > 0
        assert called_with[0][0] == "search"


class TestUnconfiguredErrorEnvelopeParity:
    """Regression tests for PR #25182: the post-migration dispatcher must
    emit the same top-level error envelope as pre-migration main when no
    web backend is configured.

    Plugin-level error wrapping is correct for in-flight errors (per-page
    SDK exceptions, scrape timeouts) but PRE-FLIGHT configuration errors
    must surface at the top level so function-calling models that check
    ``result.get("error")`` detect the failure cleanly.
    """

    def _clear_web_creds(self, monkeypatch):
        for k in (
            "BRAVE_SEARCH_API_KEY",
            "SEARXNG_URL",
            "TAVILY_API_KEY",
            "EXA_API_KEY",
            "PARALLEL_API_KEY",
            "FIRECRAWL_API_KEY",
            "FIRECRAWL_API_URL",
            "FIRECRAWL_GATEWAY_URL",
            "TOOL_GATEWAY_DOMAIN",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_unconfigured_search_emits_top_level_error(self, monkeypatch):
        """``web_search_tool`` with no creds returns ``{"error": "Error searching web: ..."}``
        — matching main's ``tool_error()`` envelope, not a per-result shape.
        """
        import json
        from tools import web_tools

        self._clear_web_creds(monkeypatch)
        # Reset firecrawl client cache so the unconfigured state is re-evaluated
        monkeypatch.setattr(web_tools, "_firecrawl_client", None, raising=False)
        monkeypatch.setattr(web_tools, "_firecrawl_client_config", None, raising=False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        result = json.loads(web_tools.web_search_tool("hello world", limit=3))
        assert "error" in result, f"expected top-level 'error' key, got {result}"
        # ``Error searching web:`` prefix comes from web_tools' top-level except handler
        assert "Error searching web:" in result["error"]
        assert "FIRECRAWL_API_KEY" in result["error"]
        # No per-result burying
        assert "results" not in result

    def test_unconfigured_crawl_emits_top_level_error(self, monkeypatch):
        """``web_crawl_tool`` with no creds returns ``{"success": False, "error": "web_crawl requires Firecrawl..."}``
        — the dispatcher gates on ``provider.is_available()`` BEFORE
        delegating to the plugin so pre-config errors don't get wrapped
        into ``results[]``.
        """
        import asyncio
        import json
        from tools import web_tools

        self._clear_web_creds(monkeypatch)
        monkeypatch.setattr(web_tools, "_firecrawl_client", None, raising=False)
        monkeypatch.setattr(web_tools, "_firecrawl_client_config", None, raising=False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        result = json.loads(asyncio.run(web_tools.web_crawl_tool("https://example.com", use_llm_processing=False)))
        assert result.get("success") is False
        assert "error" in result, f"expected top-level 'error' key, got {result}"
        assert "web_crawl requires Firecrawl" in result["error"]
        # Crucially: no per-page burying
        assert "results" not in result
