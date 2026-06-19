"""Tests for the web tools provider architecture.

Covers:
- WebSearchProvider / WebExtractProvider ABC enforcement
- Per-capability backend selection (_get_search_backend, _get_extract_backend)
- Backward compatibility (web.backend still works as shared fallback)
- Config keys merge correctly via DEFAULT_CONFIG
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from tests.tools.conftest import register_all_web_providers


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestWebProviderABCs:
    """The unified WebSearchProvider ABC enforces the interface contract.

    After PR #25182, web providers are subclasses of
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

    _register_providers = staticmethod(register_all_web_providers)

    @pytest.fixture(autouse=True)
    def _populate_web_registry(self):
        self._register_providers()
        yield
        from agent.web_search_registry import _reset_for_tests
        _reset_for_tests()

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

    def test_camofox_shared_backend_wins_over_ddgs_when_configured(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "camofox"})
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        assert web_tools._get_search_backend() == "camofox"
        assert web_tools._get_extract_backend() == "camofox"

    def test_camofox_per_capability_backends_win_over_shared_ddgs(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {
            "backend": "ddgs",
            "search_backend": "camofox",
            "extract_backend": "camofox",
        })
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        assert web_tools._get_search_backend() == "camofox"
        assert web_tools._get_extract_backend() == "camofox"

    def test_check_web_api_key_tracks_camofox_url_for_configured_backend(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "camofox"})
        monkeypatch.delenv("CAMOFOX_URL", raising=False)

        assert web_tools.check_web_api_key() is False

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        assert web_tools.check_web_api_key() is True


class TestColdWebProviderPluginDiscovery:
    """Configured plugin providers route even before global plugin discovery."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self):
        from agent.web_search_registry import _reset_for_tests

        _reset_for_tests()
        yield
        _reset_for_tests()

    def test_configured_camofox_backend_forces_registry_population(self, monkeypatch):
        from agent.web_search_registry import get_provider
        from tools import web_tools

        assert get_provider("camofox") is None
        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {
                "backend": "camofox",
                "search_backend": "camofox",
                "extract_backend": "camofox",
            },
        )
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        assert web_tools._get_backend() == "camofox"
        assert web_tools._get_search_backend() == "camofox"
        assert web_tools._get_extract_backend() == "camofox"
        assert web_tools.check_web_api_key() is True
        assert get_provider("camofox") is not None

    def test_web_search_tool_dispatches_configured_camofox_from_cold_registry(
        self, monkeypatch
    ):
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"search_backend": "camofox"},
        )
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        calls = []

        def fake_search(self, query: str, limit: int = 5):
            calls.append((query, limit))
            return {
                "success": True,
                "data": {
                    "web": [
                        {
                            "title": "ok",
                            "url": "https://example.com",
                            "description": "",
                            "position": 1,
                        }
                    ]
                },
            }

        monkeypatch.setattr(CamofoxWebSearchProvider, "search", fake_search)

        result = json.loads(web_tools.web_search_tool("camofox query", limit=1))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "ok"
        assert calls == [("camofox query", 1)]

    @pytest.mark.asyncio
    async def test_web_extract_tool_dispatches_configured_camofox_from_cold_registry(
        self, monkeypatch
    ):
        from plugins.web.camofox.provider import CamofoxWebSearchProvider
        from tools import web_tools

        monkeypatch.setattr(
            web_tools,
            "_load_web_config",
            lambda: {"extract_backend": "camofox"},
        )
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr(web_tools, "check_auxiliary_model", lambda: False)

        calls = []

        def fake_extract(self, urls, **kwargs):
            calls.append((list(urls), kwargs))
            return [
                {
                    "url": urls[0],
                    "title": "ok",
                    "content": "body",
                    "raw_content": "body",
                }
            ]

        monkeypatch.setattr(CamofoxWebSearchProvider, "extract", fake_extract)

        result = json.loads(
            await web_tools.web_extract_tool(
                ["https://example.com"], use_llm_processing=False
            )
        )

        assert result["results"][0]["title"] == "ok"
        assert calls == [(["https://example.com"], {"format": None})]

    def test_temp_hermes_home_cold_import_dispatches_configured_camofox(
        self, tmp_path: Path
    ):
        (tmp_path / "config.yaml").write_text(
            "web:\n"
            "  backend: camofox\n"
            "  search_backend: camofox\n"
            "  extract_backend: camofox\n",
            encoding="utf-8",
        )
        script = textwrap.dedent(
            """
            import asyncio
            import json

            from agent.web_search_registry import get_provider, list_providers

            assert [p.name for p in list_providers()] == []

            from tools import interrupt, web_tools

            assert web_tools._get_backend() == "camofox"
            assert web_tools._get_search_backend() == "camofox"
            assert web_tools._get_extract_backend() == "camofox"
            assert web_tools.check_web_api_key() is True
            assert get_provider("camofox") is not None

            from plugins.web.camofox.provider import CamofoxWebSearchProvider

            interrupt.is_interrupted = lambda: False
            search_calls = []

            def fake_search(self, query, limit=5):
                search_calls.append((self.name, query, limit))
                return {
                    "success": True,
                    "data": {
                        "web": [
                            {
                                "title": "camofox search",
                                "url": "https://example.com/search",
                                "description": "",
                                "position": 1,
                            }
                        ]
                    },
                }

            CamofoxWebSearchProvider.search = fake_search
            search_result = json.loads(web_tools.web_search_tool("cold camofox", limit=2))
            assert search_result["data"]["web"][0]["title"] == "camofox search"
            assert search_calls == [("camofox", "cold camofox", 2)]

            extract_calls = []
            to_thread_calls = []

            def fake_extract(self, urls, **kwargs):
                extract_calls.append((self.name, list(urls), kwargs))
                return [
                    {
                        "url": urls[0],
                        "title": "camofox extract",
                        "content": "body",
                        "raw_content": "body",
                    }
                ]

            async def fake_to_thread(func, *args, **kwargs):
                to_thread_calls.append(getattr(func, "__name__", repr(func)))
                return func(*args, **kwargs)

            CamofoxWebSearchProvider.extract = fake_extract
            web_tools.asyncio.to_thread = fake_to_thread
            extract_result = json.loads(
                asyncio.run(
                    web_tools.web_extract_tool(
                        ["https://example.com"], use_llm_processing=False
                    )
                )
            )
            assert extract_result["results"][0]["title"] == "camofox extract"
            assert extract_calls == [("camofox", ["https://example.com"], {"format": None})]
            assert to_thread_calls[-1:] == ["fake_extract"]

            print(json.dumps({"ok": True}))
            """
        )
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(tmp_path),
                "CAMOFOX_URL": "http://localhost:9377",
            }
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr


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
    """web_search_tool dispatches through registry provider resolution."""

    def test_search_tool_calls_registry_resolution(self, monkeypatch):
        from tools import web_tools

        fake_provider = MagicMock(name="fake_web_provider")
        fake_provider.name = "firecrawl"
        fake_provider.search.return_value = {"success": True, "data": {"web": []}}

        called_with = []

        def tracking_resolve(capability: str):
            called_with.append(capability)
            return fake_provider

        monkeypatch.setattr(web_tools, "_resolve_web_provider", tracking_resolve)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

        result = json.loads(web_tools.web_search_tool("test", 1))

        assert result == {"success": True, "data": {"web": []}}
        assert called_with == ["search"]
        fake_provider.search.assert_called_once_with("test", 1)


class TestUnconfiguredErrorEnvelopeParity:
    """Regression tests for provider-resolution failures.

    The dispatcher must surface a top-level error when registry resolution finds
    no provider instead of burying the setup failure in per-result payloads.
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
            "XAI_API_KEY",
            "CAMOFOX_URL",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_unconfigured_search_emits_top_level_error(self, monkeypatch):
        """``web_search_tool`` with no creds returns ``{"error": "Error searching web: ..."}``
        — matching main's ``tool_error()`` envelope, not a per-result shape.
        """
        from tools import web_tools

        self._clear_web_creds(monkeypatch)
        # Reset firecrawl client cache so the unconfigured state is re-evaluated
        monkeypatch.setattr(web_tools, "_firecrawl_client", None, raising=False)
        monkeypatch.setattr(web_tools, "_firecrawl_client_config", None, raising=False)
        monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {})

        result = json.loads(web_tools.web_search_tool("hello world", limit=3))
        assert "error" in result, f"expected top-level 'error' key, got {result}"
        # ``Error searching web:`` prefix comes from web_tools' top-level except handler
        assert "Error searching web:" in result["error"]
        assert "FIRECRAWL_API_KEY" in result["error"]
        # No per-result burying
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
                    use_llm_processing=False,
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

