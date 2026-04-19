import inspect
import importlib

import pytest

from tools.registry import registry

web_tools = importlib.import_module("tools.web_tools")


class TestWebToolSchemas:
    def test_web_search_schema_frames_builtin_grounding_primitive(self):
        description = web_tools.WEB_SEARCH_SCHEMA["description"]

        assert "Hermes built-in web/research grounding primitive" in description
        assert "live web search" in description
        assert "current external information" in description

    def test_web_extract_schema_frames_text_first_boundary_and_browser_fallback(self):
        description = web_tools.WEB_EXTRACT_SCHEMA["description"]

        assert "Hermes built-in web/research grounding primitive" in description
        assert "text-first extraction" in description
        assert "browser-driven navigation and document/PDF intelligence are outside this tool's scope" in description
        assert "fall back to Hermes browser/document tools" in description


class TestWebToolRegistry:
    def test_registry_exposes_only_public_web_search_and_extract_entries(self):
        search_entry = registry.get_entry("web_search")
        extract_entry = registry.get_entry("web_extract")
        crawl_entry = registry.get_entry("web_crawl")

        assert search_entry is not None
        assert search_entry.toolset == "web"
        assert search_entry.schema["name"] == "web_search"

        assert extract_entry is not None
        assert extract_entry.toolset == "web"
        assert extract_entry.schema["name"] == "web_extract"
        assert extract_entry.is_async is True

        assert crawl_entry is None

    def test_web_crawl_docstring_keeps_canonical_defer_truth_explicit(self):
        doc = inspect.getdoc(web_tools.web_crawl_tool)

        assert doc is not None
        assert "lives in ``tools/web_tools.py``" in doc
        assert "not" in doc and "canonical built-in web/research surface today" in doc
        assert "``web_search``" in doc
        assert "``web_extract``" in doc


class TestWebToolAvailabilityChecks:
    @pytest.mark.parametrize("tool_name", ["web_search", "web_extract"])
    def test_registry_check_fn_uses_configured_backend_readiness(self, monkeypatch, tool_name):
        entry = registry.get_entry(tool_name)
        seen = []

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "parallel"})

        def fake_is_backend_available(backend):
            seen.append(backend)
            return backend == "parallel"

        monkeypatch.setattr(web_tools, "_is_backend_available", fake_is_backend_available)

        assert entry is not None
        assert entry.check_fn() is True
        assert seen == ["parallel"]

    @pytest.mark.parametrize("tool_name", ["web_search", "web_extract"])
    def test_registry_check_fn_falls_back_to_any_available_backend(self, monkeypatch, tool_name):
        entry = registry.get_entry(tool_name)
        seen = []

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        def fake_is_backend_available(backend):
            seen.append(backend)
            return backend == "tavily"

        monkeypatch.setattr(web_tools, "_is_backend_available", fake_is_backend_available)

        assert entry is not None
        assert entry.check_fn() is True
        assert seen == ["exa", "parallel", "firecrawl", "tavily"]

    @pytest.mark.parametrize("tool_name", ["web_search", "web_extract"])
    def test_registry_check_fn_reports_unavailable_when_no_backend_is_ready(self, monkeypatch, tool_name):
        entry = registry.get_entry(tool_name)

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "firecrawl"})
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: False)

        assert entry is not None
        assert entry.check_fn() is False
