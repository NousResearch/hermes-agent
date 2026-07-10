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

from tests.tools.conftest import register_all_web_providers


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
    ``supports_search() / supports_extract()`` flags.
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

            def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
                return {"success": True, "data": {"web": []}}

            def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
                return [{"url": urls[0], "content": "x"}]

        d = Dummy()
        assert d.supports_search() is True
        assert d.supports_extract() is True
        assert d.extract(["https://example.com"])[0]["url"] == "https://example.com"

    def test_search_only_provider_skips_extract(self):
        """Search-only providers don't have to implement extract()."""
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

        # Should instantiate fine — extract has default supports_*()
        # returning False and isn't required to be overridden when not
        # advertised.
        s = SearchOnly()
        assert s.supports_search() is True
        assert s.supports_extract() is False


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

    def test_search_backend_honored_even_when_not_available(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "firecrawl",
            "search_backend": "exa",  # set but no EXA_API_KEY
        })
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        # Fork policy: explicit per-capability backend wins so the provider's
        # own setup error surfaces instead of silently rerouting to parallel.
        assert web_tools._get_search_backend() == "exa"

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
        # Upstream defaults are empty strings; this fork may ship explicit
        # cloakbrowser defaults — keys must exist and be strings.
        assert isinstance(web["backend"], str)
        assert isinstance(web["search_backend"], str)
        assert isinstance(web["extract_backend"], str)


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

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

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
        """With no creds and parallel disabled, search returns a top-level error."""
        from tools import web_tools

        self._clear_web_creds(monkeypatch)
        monkeypatch.setattr(web_tools, "_firecrawl_client", None, raising=False)
        monkeypatch.setattr(web_tools, "_firecrawl_client_config", None, raising=False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "firecrawl"})
        monkeypatch.setattr(web_tools, "_get_backend", lambda capability=None: "firecrawl")

        result = json.loads(web_tools.web_search_tool("hello world", limit=3))
        assert "error" in result, f"expected top-level 'error' key, got {result}"
        assert "Error searching web:" in result["error"]
        assert "FIRECRAWL_API_KEY" in result["error"]
        assert "results" not in result


class TestDispatchersTriggerPluginDiscovery:
    """Regression tests for #27580: each web_*_tool dispatcher must
    idempotently call ``_ensure_web_plugins_loaded()`` before consulting
    ``agent.web_search_registry``.

    Without this, a tool call from a context that hasn't already loaded
    plugins (subprocess agent runs, delegate children, standalone scripts,
    test paths that import the registry directly) sees an empty registry
    and returns the misleading "No web extract provider configured" error
    even when the user has both the config key set AND the API key
    exported.

    Mirrors :func:`tools.browser_tool._ensure_browser_plugins_loaded` —
    every other plugin-backed dispatcher (image_gen, video_gen, browser,
    skills) already does this.
    """

    def _clear_registry(self):
        """Reset the web_search registry to empty and return a callback
        that restores the original contents. Used in a try/finally so the
        snapshot is restored even when the dispatcher under test raises."""
        from agent import web_search_registry

        with web_search_registry._lock:
            original = dict(web_search_registry._providers)
            web_search_registry._providers.clear()

        def _restore():
            with web_search_registry._lock:
                web_search_registry._providers.clear()
                web_search_registry._providers.update(original)

        return _restore

    def test_web_extract_tool_runs_discovery_before_registry_lookup(self, monkeypatch):
        """``web_extract_tool`` must invoke ``_ensure_web_plugins_loaded()``
        before looking up the configured backend so the registry is
        populated even from cold-start subprocess contexts.

        Without the fix, ``get_provider('firecrawl')`` returns ``None``
        on a fresh process and the dispatcher emits "No web extract
        provider configured" despite the user having both
        ``web.extract_backend: firecrawl`` and ``FIRECRAWL_API_KEY`` set
        (issue #27580).
        """
        import asyncio
        import json
        from unittest.mock import MagicMock
        from agent.web_search_provider import WebSearchProvider
        from agent import web_search_registry
        from tools import web_tools

        restore = self._clear_registry()
        try:
            class FakeFirecrawl(WebSearchProvider):
                @property
                def name(self) -> str:
                    return "firecrawl"

                @property
                def display_name(self) -> str:
                    return "Fake Firecrawl"

                def is_available(self) -> bool:
                    return True

                def supports_extract(self) -> bool:
                    return True

                async def extract(self, urls, format=None):
                    return [
                        {"url": u, "title": "", "content": "ok",
                         "raw_content": "ok", "metadata": {}}
                        for u in urls
                    ]

            # Simulate "plugin discovery loads the firecrawl plugin": the
            # wrapped helper registers the provider, mirroring what
            # ``plugins/web/firecrawl/__init__.py:register`` does at
            # real-process startup. Wrapping with ``MagicMock`` lets us
            # also assert the dispatcher actually invoked the hook — if
            # a future refactor accidentally drops the call the regression
            # would otherwise hide behind a still-populated registry.
            def _register_fake() -> None:
                if web_search_registry.get_provider("firecrawl") is None:
                    web_search_registry.register_provider(FakeFirecrawl())

            mock_hook = MagicMock(wraps=_register_fake)
            # Patch the helper on ``tools.web_tools`` directly rather than the
            # underlying ``hermes_cli.plugins._ensure_plugins_discovered`` so
            # the test stays valid even if the import inside the helper is
            # later moved to module scope or renamed.
            monkeypatch.setattr(
                web_tools, "_ensure_web_plugins_loaded", mock_hook
            )
            monkeypatch.setattr(
                web_tools, "_load_web_config",
                lambda: {"extract_backend": "firecrawl"},
            )
            # Sanity: registry IS empty before the tool call.
            assert web_search_registry.get_provider("firecrawl") is None

            result = json.loads(asyncio.run(
                web_tools.web_extract_tool(
                    ["https://example.com"],
                )
            ))

            # The hook must have been called BEFORE the registry lookup —
            # that is the invariant under regression test. Without the
            # explicit ``.called`` assertion the test could pass if the
            # registry were populated by some unrelated side effect.
            assert mock_hook.called, (
                "web_extract_tool must call _ensure_web_plugins_loaded() "
                "before resolving the registry"
            )
            assert "No web extract provider configured" not in json.dumps(result)
            assert web_search_registry.get_provider("firecrawl") is not None
        finally:
            restore()

    def test_web_search_tool_runs_discovery_before_registry_lookup(self, monkeypatch):
        """``web_search_tool`` must invoke ``_ensure_web_plugins_loaded()``
        before the registry lookup for the same reason as the extract
        path (issue #27580 root cause applies to all dispatchers).
        """
        import json
        from unittest.mock import MagicMock
        from agent.web_search_provider import WebSearchProvider
        from agent import web_search_registry
        from tools import web_tools

        restore = self._clear_registry()
        try:
            class FakeBrave(WebSearchProvider):
                @property
                def name(self) -> str:
                    return "brave-free"

                @property
                def display_name(self) -> str:
                    return "Fake Brave"

                def is_available(self) -> bool:
                    return True

                def supports_search(self) -> bool:
                    return True

                def search(self, query, limit=5):
                    return {"success": True, "data": {"web": [
                        {"title": "ok", "url": "https://x", "description": "",
                         "position": 0}
                    ]}}

            def _register_fake() -> None:
                if web_search_registry.get_provider("brave-free") is None:
                    web_search_registry.register_provider(FakeBrave())

            mock_hook = MagicMock(wraps=_register_fake)
            monkeypatch.setattr(
                web_tools, "_ensure_web_plugins_loaded", mock_hook
            )
            monkeypatch.setattr(
                web_tools, "_load_web_config",
                lambda: {"search_backend": "brave-free"},
            )
            assert web_search_registry.get_provider("brave-free") is None

            result = json.loads(web_tools.web_search_tool("hello", limit=1))
            assert mock_hook.called, (
                "web_search_tool must call _ensure_web_plugins_loaded() "
                "before resolving the registry"
            )
            assert "No web search provider configured" not in json.dumps(result)
            assert web_search_registry.get_provider("brave-free") is not None
        finally:
            restore()


class TestDisabledPluginDiagnostic:
    """#40190 follow-up: when the configured web backend names a bundled
    web plugin the user put in ``plugins.disabled``, the dispatcher must
    tell the user to re-enable the plugin instead of the misleading
    "No web extract provider configured. Set web.extract_backend to ..."
    (they already set it correctly — the provider just isn't loaded).
    """

    def _clear_registry(self):
        from agent import web_search_registry

        with web_search_registry._lock:
            original = dict(web_search_registry._providers)
            web_search_registry._providers.clear()

        def _restore():
            with web_search_registry._lock:
                web_search_registry._providers.clear()
                web_search_registry._providers.update(original)

        return _restore

    class _FakeLoaded:
        def __init__(self, enabled, error):
            self.enabled = enabled
            self.error = error

    def _patch_manager(self, monkeypatch, plugins_map):
        """Point ``get_plugin_manager()`` at a stub whose ``_plugins``
        dict is ``plugins_map`` so ``_disabled_web_plugin_for`` sees the
        simulated disabled/enabled state without touching real config."""
        import hermes_cli.plugins as plugins_mod

        class _StubMgr:
            _plugins = plugins_map

        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: _StubMgr())

    def test_disabled_web_plugin_for_matches_by_key(self, monkeypatch):
        from agent.web_search_registry import _disabled_web_plugin_for

        self._patch_manager(monkeypatch, {
            "web/firecrawl": self._FakeLoaded(False, "disabled via config"),
            "web/ddgs": self._FakeLoaded(True, None),
        })
        assert _disabled_web_plugin_for("firecrawl") == "web/firecrawl"
        # Enabled plugin is not a match
        assert _disabled_web_plugin_for("ddgs") is None
        # Unknown name is not a match
        assert _disabled_web_plugin_for("nope") is None

    def test_disabled_web_plugin_for_normalizes_hyphens(self, monkeypatch):
        from agent.web_search_registry import _disabled_web_plugin_for

        self._patch_manager(monkeypatch, {
            "web/brave_free": self._FakeLoaded(False, "disabled via config"),
        })
        # config name uses a hyphen; plugin key uses an underscore
        assert _disabled_web_plugin_for("brave-free") == "web/brave_free"

    def test_disabled_web_plugin_for_ignores_non_disabled_errors(self, monkeypatch):
        from agent.web_search_registry import _disabled_web_plugin_for

        self._patch_manager(monkeypatch, {
            # a plugin that failed to import is NOT "disabled via config"
            "web/exa": self._FakeLoaded(False, "ImportError: boom"),
        })
        assert _disabled_web_plugin_for("exa") is None

    def test_extract_tool_reports_disabled_plugin(self, monkeypatch):
        import asyncio

        from tools import web_tools

        restore = self._clear_registry()
        try:
            monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
            monkeypatch.setattr(
                web_tools, "_load_web_config",
                lambda: {"extract_backend": "firecrawl"},
            )
            import agent.web_search_registry as wsr
            monkeypatch.setattr(
                wsr, "_read_config_key",
                lambda *path: "firecrawl" if path == ("web", "extract_backend") else None,
            )
            self._patch_manager(monkeypatch, {
                "web/firecrawl": self._FakeLoaded(False, "disabled via config"),
            })
            result = json.loads(
                asyncio.new_event_loop().run_until_complete(
                    web_tools.web_extract_tool(["https://example.com"])
                )
            )
            err = result["error"]
            assert "disabled" in err
            assert "web/firecrawl" in err
            assert "hermes plugins enable" in err
            # Must NOT tell them to set extract_backend (already set)
            assert "Set web.extract_backend to firecrawl" not in err
        finally:
            restore()

    def test_search_tool_reports_disabled_plugin(self, monkeypatch):
        from tools import web_tools

        restore = self._clear_registry()
        try:
            monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
            monkeypatch.setattr(
                web_tools, "_load_web_config",
                lambda: {"search_backend": "firecrawl"},
            )
            import agent.web_search_registry as wsr
            monkeypatch.setattr(
                wsr, "_read_config_key",
                lambda *path: "firecrawl" if path == ("web", "search_backend") else None,
            )
            self._patch_manager(monkeypatch, {
                "web/firecrawl": self._FakeLoaded(False, "disabled via config"),
            })
            result = json.loads(web_tools.web_search_tool("hello", limit=1))
            err = result["error"]
            assert "disabled" in err
            assert "web/firecrawl" in err
            assert "No web search provider configured" not in err
        finally:
            restore()

