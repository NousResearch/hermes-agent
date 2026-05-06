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
    """The ABCs enforce the interface contract."""

    def test_cannot_instantiate_search_provider(self):
        from tools.web_providers.base import WebSearchProvider

        with pytest.raises(TypeError):
            WebSearchProvider()  # type: ignore[abstract]

    def test_cannot_instantiate_extract_provider(self):
        from tools.web_providers.base import WebExtractProvider

        with pytest.raises(TypeError):
            WebExtractProvider()  # type: ignore[abstract]

    def test_concrete_search_provider_works(self):
        from tools.web_providers.base import WebSearchProvider

        class Dummy(WebSearchProvider):
            def provider_name(self) -> str:
                return "dummy"
            def is_configured(self) -> bool:
                return True
            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                return {"success": True, "data": {"web": []}}

        d = Dummy()
        assert d.provider_name() == "dummy"
        assert d.is_configured() is True
        assert d.search("test")["success"] is True

    def test_concrete_extract_provider_works(self):
        from tools.web_providers.base import WebExtractProvider

        class Dummy(WebExtractProvider):
            def provider_name(self) -> str:
                return "dummy"
            def is_configured(self) -> bool:
                return True
            def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]:
                return {"success": True, "data": [{"url": urls[0], "content": "x"}]}

        d = Dummy()
        assert d.provider_name() == "dummy"
        assert d.extract(["https://example.com"])["success"] is True


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
