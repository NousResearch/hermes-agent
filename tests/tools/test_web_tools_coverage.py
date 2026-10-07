"""Coverage expansion for tools/web_tools.py (issue #36603).

This file targets the untested dispatch/error/fallback paths in web_tools
so statement coverage of the module moves past the 68% baseline WITHOUT
touching production code. It is purely additive: one new test module,
behavioral-contract assertions (success flag / error text / ordering),
no vendor-payload snapshots.

Targets (by source region):
  - _env_value            fallback to process env on config-layer exception
  - _load_web_config      {} on load exception
  - registry helpers      _registered_web_provider / _available / _list
                          (incl. import/exception paths)
  - _get_backend          selection_exists branch, registered-provider walk,
                          keyless free-tier walk
  - _is_backend_available delegation to non-legacy names + xai probe
  - _ddgs_package_importable, _keyless_rescue_enabled, _rescue_eligible,
    _rescue_extract
  - _store_full_text / _truncate_with_footer failure + no-store footer
  - _ensure_web_plugins_loaded warning path
  - web_search_tool       limit coercion, interrupted, selection-error,
                          disabled-plugin, and no-provider responses
  - web_extract_tool      invalid items, secret/credential URL blocks,
                          SSRF block, empty results, search-only error,
                          extract selection errors, cache-hit merge,
                          provider-raise rescue, order reconstruction,
                          char_limit fallback, empty-content skip, outer catch
  - _provider_is_ready and check_web_api_key error/excitement paths

Lines 1571-1647 (the ``if __name__ == "__main__":`` demo block) are
deliberately not covered.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from agent import web_search_registry as wsr
from tools import web_tools as wt


# ---------------------------------------------------------------------------
# Fakes & shared patching helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal WebSearchProvider stand-in for dispatch tests."""

    def __init__(self, name="fake", supports_search=True, supports_extract=True):
        self.name = name
        self.display_name = name.replace("_", " ").title()
        self._supports_search = supports_search
        self._supports_extract = supports_extract

    def supports_search(self):
        return self._supports_search

    def supports_extract(self):
        return self._supports_extract

    def is_available(self):
        return True

    def is_keyless_available(self):
        return True


class _SearchProvider(_FakeProvider):
    def search(self, query, limit, **kwargs):
        return {
            "success": True,
            "data": {
                "web": [
                    {"title": "t", "url": "https://e.com/1", "description": "d",
                     "position": 1}
                ]
            },
        }


class _AsyncExtractProvider(_FakeProvider):
    """Provider whose extract() is a coroutine (covers the async dispatch branch).

    ``results`` may be a fixed list (returned verbatim), a callable
    (urls, kwargs) -> list, or an Exception instance (raised).
    """

    def __init__(self, name="fake", results=None, calls=None):
        super().__init__(name, supports_search=True, supports_extract=True)
        self._results = results
        self.calls = [] if calls is None else calls

    async def extract(self, urls, **kwargs):
        self.calls.extend(urls)
        if isinstance(self._results, Exception):
            raise self._results
        if callable(self._results):
            return self._results(urls, kwargs)
        if self._results is not None:
            return self._results
        return [
            {"url": u, "title": "", "content": "ok", "raw_content": "ok",
             "metadata": {}}
            for u in urls
        ]


def _async_true(*_a, **_k):
    async def _coro(*_a, **_k):
        return True
    return _coro()


def _raising(*_a, **_k):
    raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _silence_debug_and_interrupt(monkeypatch):
    """Keep debug files out of the tree and default to non-interrupted."""
    monkeypatch.setattr(wt._debug, "log_call", MagicMock())
    monkeypatch.setattr(wt._debug, "save", MagicMock())
    import tools.interrupt
    monkeypatch.setattr(tools.interrupt, "is_interrupted", lambda: False)


# ---------------------------------------------------------------------------
# _env_value / _load_web_config
# ---------------------------------------------------------------------------


class TestConfigEnvLayer:
    def test_env_value_falls_back_to_process_env(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "get_env_value", _raising)
        monkeypatch.setenv("SEARXNG_URL", "http://searx.local")
        assert wt._env_value("SEARXNG_URL") == "http://searx.local"

    def test_env_value_honors_config_layer(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "get_env_value", lambda name: "cfg://value")
        assert wt._env_value("SEARXNG_URL") == "cfg://value"

    def test_load_web_config_empty_on_exception(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "load_config", _raising)
        assert wt._load_web_config() == {}

    def test_load_web_config_null_web_section(self, monkeypatch):
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "load_config", lambda: {"web": None})
        assert wt._load_web_config() == {}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


class TestRegisteredProviderHelpers:
    def test_registered_provider_empty_backend(self):
        assert wt._registered_web_provider("") is None

    def test_registered_provider_lookup_failure(self, monkeypatch):
        monkeypatch.setattr(wsr, "get_provider", _raising)
        assert wt._registered_web_provider("x") is None

    def test_registered_available_unregistered(self, monkeypatch):
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        assert wt._registered_web_provider_available("x") is None

    def test_registered_available_true(self, monkeypatch):
        monkeypatch.setattr(wsr, "get_provider", lambda name: _FakeProvider())
        assert wt._registered_web_provider_available("x") is True

    def test_registered_available_raises(self, monkeypatch):
        p = _FakeProvider()
        p.is_available = _raising
        monkeypatch.setattr(wsr, "get_provider", lambda name: p)
        assert wt._registered_web_provider_available("x") is False

    def test_list_registered_providers_empty_on_failure(self, monkeypatch):
        monkeypatch.setattr(wsr, "list_providers", _raising)
        assert wt._list_registered_web_providers() == []


# ---------------------------------------------------------------------------
# _get_backend
# ---------------------------------------------------------------------------


class TestGetBackend:
    def _disable_builtin_ladder(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wt, "_has_env", lambda name: False)
        monkeypatch.setattr(wt, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(wt, "_ddgs_package_importable", lambda: False)
        # Pin selection_exists so walk tests don't depend on conftest
        # sandboxing HERMES_HOME (spec-reviewer hardening note).
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: False)

    def test_selection_exists_branch(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {"backend": ""})
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: True)
        assert wt._get_backend() == "firecrawl"

    def test_registered_provider_walk_returns_available(self, monkeypatch):
        self._disable_builtin_ladder(monkeypatch)
        legacy = _FakeProvider(name="tavily")
        available = _FakeProvider(name="my-custom")
        broken = _FakeProvider(name="my-broken")
        broken.is_available = _raising
        monkeypatch.setattr(wt, "_list_registered_web_providers",
                            lambda: [legacy, broken, available])
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: False)
        assert wt._get_backend() == "my-custom"

    def test_registered_provider_walk_skips_broken(self, monkeypatch):
        self._disable_builtin_ladder(monkeypatch)
        broken = _FakeProvider(name="my-broken")
        broken.is_available = _raising
        monkeypatch.setattr(wt, "_list_registered_web_providers",
                            lambda: [broken])
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: False)
        assert wt._get_backend() == "firecrawl"

    def test_keyless_walk_returns_ring_vendor(self, monkeypatch):
        self._disable_builtin_ladder(monkeypatch)
        exa = _FakeProvider(name="exa")
        monkeypatch.setattr(wt, "_list_registered_web_providers", lambda: [])
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        monkeypatch.setattr(wsr, "_keyless_preference", lambda: ("none", "exa"))
        monkeypatch.setattr(wt, "_registered_web_provider",
                            lambda name: exa if name == "exa" else None)
        assert wt._get_backend() == "exa"

    def test_keyless_walk_skips_broken_provider(self, monkeypatch):
        self._disable_builtin_ladder(monkeypatch)
        exa = _FakeProvider(name="exa")
        exa.is_keyless_available = _raising
        monkeypatch.setattr(wt, "_list_registered_web_providers", lambda: [])
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        monkeypatch.setattr(wsr, "_keyless_preference", lambda: ("exa",))
        monkeypatch.setattr(wt, "_registered_web_provider",
                            lambda name: exa)
        assert wt._get_backend() == "firecrawl"

    def test_keyless_walk_outer_exception(self, monkeypatch):
        self._disable_builtin_ladder(monkeypatch)
        monkeypatch.setattr(wt, "_list_registered_web_providers", lambda: [])
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", _raising)
        assert wt._get_backend() == "firecrawl"


# ---------------------------------------------------------------------------
# _is_backend_available
# ---------------------------------------------------------------------------


class TestIsBackendAvailable:
    def test_delegates_non_legacy_name_true(self, monkeypatch):
        monkeypatch.setattr(wt, "_registered_web_provider_available",
                            lambda name: True)
        assert wt._is_backend_available("custom") is True

    def test_delegates_non_legacy_name_false(self, monkeypatch):
        monkeypatch.setattr(wt, "_registered_web_provider_available",
                            lambda name: False)
        assert wt._is_backend_available("custom") is False

    def test_xai_probe_true(self, monkeypatch):
        monkeypatch.setattr("tools.xai_http.has_xai_credentials",
                            lambda: True)
        assert wt._is_backend_available("xai") is True

    def test_xai_probe_exception_false(self, monkeypatch):
        monkeypatch.setattr("tools.xai_http.has_xai_credentials", _raising)
        assert wt._is_backend_available("xai") is False

    def test_unknown_name_falls_through_to_false(self, monkeypatch):
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        assert wt._is_backend_available("unknown") is False


# ---------------------------------------------------------------------------
# Misc helpers: ddgs, keyless rescue, store, truncate, plugin load
# ---------------------------------------------------------------------------


class TestMiscHelpers:
    def test_ddgs_importable_true(self, monkeypatch):
        import sys
        import types
        # Inject a loadable stub so ``import ddgs`` succeeds and covers the
        # ``return True`` branch of _ddgs_package_importable.
        monkeypatch.setitem(sys.modules, "ddgs", types.ModuleType("ddgs"))
        assert wt._ddgs_package_importable() is True

    def test_keyless_rescue_disabled_by_config(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config",
                            lambda: {"keyless_rescue": False})
        assert wt._keyless_rescue_enabled() is False

    def test_keyless_rescue_tier_check_exception(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", _raising)
        assert wt._keyless_rescue_enabled() is False

    def test_rescue_eligible_none_provider(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        assert wt._rescue_eligible(None) is False

    def test_rescue_eligible_ring_vendor_keyed(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        monkeypatch.setattr("plugins.web.keyless_mcp.use_keyless",
                            lambda name, key: False)
        monkeypatch.setattr("agent.web_search_provider.get_provider_env",
                            lambda key: "k")
        assert wt._rescue_eligible(_FakeProvider(name="exa")) is True

    def test_rescue_eligible_non_ring_vendor(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        assert wt._rescue_eligible(_FakeProvider(name="my-custom")) is True

    def test_rescue_eligible_exception(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wsr, "_keyless_tier_enabled", lambda: True)
        monkeypatch.setattr("agent.web_search_provider.get_provider_env",
                            _raising)
        assert wt._rescue_eligible(_FakeProvider(name="exa")) is False

    def test_rescue_extract_all_policy_blocked(self, monkeypatch):
        results = [
            {"url": "https://a.com", "error": "blocked by website policy"},
            {"url": "https://b.com", "error": "blocked by website policy"},
        ]
        assert wt._rescue_extract("exa", ["https://a.com", "https://b.com"],
                                  results) is results

    def test_rescue_extract_merges_successes(self, monkeypatch):
        urls = ["https://a.com", "https://b.com"]
        failed = [{"url": u, "error": "backend down"} for u in urls]
        rescued = [
            {"url": urls[0], "content": "a", "metadata": {}},
            {"url": urls[1], "content": "b", "metadata": {}},
        ]
        monkeypatch.setattr(
            "plugins.web.keyless_mcp.extract_with_failover",
            lambda name, u: rescued,
        )
        out = wt._rescue_extract("exa", urls, failed)
        assert out is not failed
        assert out[0]["metadata"]["rescued_from"] == "exa"

    def test_rescue_extract_rescue_also_fails(self, monkeypatch):
        urls = ["https://a.com"]
        failed = [{"url": "https://a.com", "error": "backend down"}]
        monkeypatch.setattr(
            "plugins.web.keyless_mcp.extract_with_failover",
            lambda name, u: [{"url": u[0], "error": "ring down"}],
        )
        assert wt._rescue_extract("exa", urls, failed) is failed

    def test_rescue_extract_length_mismatch_returns_rescued(self, monkeypatch):
        # Provider returned fewer results than URLs: the defensive parity
        # branch treats every result as rescueable and returns the rescued
        # list (no in-place merge).
        monkeypatch.setattr(
            "plugins.web.keyless_mcp.extract_with_failover",
            lambda name, u: [{"url": u[0], "content": "r"}],
        )
        out = wt._rescue_extract("exa", ["a", "b"], [{"url": "a", "error": "e"}])
        assert out[0]["content"] == "r"
        assert out[0]["metadata"]["rescued_from"] == "exa"

    def test_policy_blocked_by_flag(self):
        assert wt._policy_blocked_result({"blocked_by_policy": True}) is True

    def test_policy_blocked_by_error_text(self):
        assert wt._policy_blocked_result({"error": "Blocked by website policy"}) is True

    def test_not_policy_blocked(self):
        assert wt._policy_blocked_result({"error": "backend down"}) is False

    def test_store_full_text_returns_none_on_failure(self, monkeypatch):
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_dir", _raising)
        assert wt._store_full_text("https://example.com/x", "content") is None

    def test_truncate_footer_when_store_fails(self, monkeypatch):
        monkeypatch.setattr(wt, "_store_full_text", lambda url, c: None)
        body = "line one\n" + "x" * 20000 + "\nline two\n"
        out, truncated = wt._truncate_with_footer(body, "https://e.com/d", 3000)
        assert truncated is True
        assert "Full text could not be stored" in out
        assert "[TRUNCATED]" in out

    def test_ensure_plugins_warning_path(self, monkeypatch):
        import hermes_cli.plugins
        monkeypatch.setattr(hermes_cli.plugins, "_ensure_plugins_discovered",
                            _raising)
        wt._ensure_web_plugins_loaded()  # must not raise


# ---------------------------------------------------------------------------
# web_search_tool
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    def _happy_path(self, monkeypatch, *, backend="fake"):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_search_backend", lambda: backend)
        monkeypatch.setattr(wsr, "get_provider", lambda name: _SearchProvider())

    def test_limit_coercion_on_bad_value(self, monkeypatch):
        self._happy_path(monkeypatch)
        out = json.loads(wt.web_search_tool("query", limit="abc"))
        assert out["success"] is True
        assert out["data"]["web"][0]["url"] == "https://e.com/1"

    def test_interrupted_returns_error(self, monkeypatch):
        import tools.interrupt
        monkeypatch.setattr(tools.interrupt, "is_interrupted", lambda: True)
        out = json.loads(wt.web_search_tool("query"))
        assert out["success"] is False
        assert "Interrupted" in out["error"]

    def test_configured_unregistered_backend(self, monkeypatch):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_search_backend", lambda: "ghost")
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: True)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_error",
                            lambda scope, name, reason: f"no registered {name}")
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: None)
        out = json.loads(wt.web_search_tool("query"))
        assert out["success"] is False
        assert "no registered" in out["error"]

    def test_configured_disabled_plugin(self, monkeypatch):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_search_backend", lambda: "firecrawl")
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: True)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: "web/firecrawl")
        out = json.loads(wt.web_search_tool("query"))
        assert out["success"] is False
        assert "disabled" in out["error"]
        assert "web/firecrawl" in out["error"]

    def test_no_provider_no_selection(self, monkeypatch):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_search_backend", lambda: "ghost")
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: False)
        monkeypatch.setattr(wsr, "get_active_search_provider", lambda: None)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: None)
        out = json.loads(wt.web_search_tool("query"))
        assert out["success"] is False
        assert "No web search provider configured" in out["error"]

    def test_no_provider_disabled_plugin(self, monkeypatch):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_search_backend", lambda: "brave_free")
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: False)
        monkeypatch.setattr(wsr, "get_active_search_provider", lambda: None)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: "web/brave_free")
        out = json.loads(wt.web_search_tool("query"))
        assert out["success"] is False
        assert "disabled" in out["error"]


# ---------------------------------------------------------------------------
# web_extract_tool
# ---------------------------------------------------------------------------


class TestWebExtractTool:
    def _patch_dispatch(self, monkeypatch, provider, *, backend="fake",
                        safe=None):
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
        monkeypatch.setattr(wt, "_get_extract_backend", lambda: backend)
        monkeypatch.setattr(wt, "async_is_safe_url", _async_true if safe is None
                            else safe)
        monkeypatch.setattr(wsr, "get_provider", lambda name: provider)
        import tools.website_policy
        monkeypatch.setattr(tools.website_policy, "check_website_access",
                            lambda url: None)

    @pytest.mark.asyncio
    async def test_invalid_url_items_produce_per_index_errors(self, monkeypatch):
        provider = _AsyncExtractProvider()
        self._patch_dispatch(monkeypatch, provider)
        out = json.loads(await wt.web_extract_tool(
            ["https://good.com", None, 123, {"nourl": 1}]
        ))
        results = out["results"]
        assert len(results) == 4
        assert results[0]["content"] == "ok"
        assert "Invalid URL item at index 1" in results[1]["error"]
        assert "Invalid URL item at index 3" in results[3]["error"]

    @pytest.mark.asyncio
    async def test_secret_in_url_blocked(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider())
        out = json.loads(await wt.web_extract_tool(
            ["https://example.com/?token=sk-1234567890abcdef"]
        ))
        assert out["success"] is False
        assert "API key or token" in out["error"]

    @pytest.mark.asyncio
    async def test_credential_query_param_blocked(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider())
        out = json.loads(await wt.web_extract_tool(
            ["https://example.com/?api_key=xyz"]
        ))
        assert out["success"] is False
        assert "credential-like query parameter" in out["error"]

    @pytest.mark.asyncio
    async def test_ssrf_blocked_url(self, monkeypatch):
        provider = _AsyncExtractProvider(calls=[])
        self._patch_dispatch(monkeypatch, provider, safe=_async_false())
        out = json.loads(await wt.web_extract_tool(["https://private.example"]))
        assert out["results"][0]["error"].startswith("Blocked:")
        # Provider was never asked to fetch the private URL.
        assert provider.calls == []

    @pytest.mark.asyncio
    async def test_empty_input_results_error(self, monkeypatch):
        # No URLs at all: safe_urls is empty, nothing was invalid or
        # SSRF-blocked, so the empty-results path emits the generic error.
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider())
        out = json.loads(await wt.web_extract_tool([]))
        assert out["error"] == "Content was inaccessible or not found"

    @pytest.mark.asyncio
    async def test_search_only_provider_error(self, monkeypatch):
        provider = _FakeProvider(name="brave-free",
                                 supports_extract=False, supports_search=True)
        self._patch_dispatch(monkeypatch, provider)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["success"] is False
        assert "search-only" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_unregistered_extract_backend(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider(),
                             backend="ghost")
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: True)
        monkeypatch.setattr("tools.tool_backend_helpers.selection_error",
                            lambda scope, name, reason: f"no registered {name}")
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for", lambda **k: None)
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["success"] is False
        assert "no registered" in out["error"]

    @pytest.mark.asyncio
    async def test_extract_disabled_plugin(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider(),
                             backend="firecrawl")
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: True)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: "web/firecrawl")
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["success"] is False
        assert "disabled" in out["error"]

    @pytest.mark.asyncio
    async def test_no_extract_provider(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider(),
                             backend="ghost")
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: False)
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr(wsr, "get_active_extract_provider", lambda: None)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for", lambda **k: None)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["success"] is False
        assert "No web extract provider configured" in out["error"]

    @pytest.mark.asyncio
    async def test_no_extract_provider_disabled_plugin(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider(),
                             backend="ghost")
        monkeypatch.setattr("tools.tool_backend_helpers.selection_exists",
                            lambda scope: False)
        monkeypatch.setattr(wsr, "get_provider", lambda name: None)
        monkeypatch.setattr(wsr, "get_active_extract_provider", lambda: None)
        monkeypatch.setattr(wsr, "_disabled_web_plugin_for",
                            lambda **k: "web/brave_free")
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["success"] is False
        assert "disabled" in out["error"]

    @pytest.mark.asyncio
    async def test_extract_policy_check_raises(self, monkeypatch):
        provider = _AsyncExtractProvider(calls=[])
        self._patch_dispatch(monkeypatch, provider)
        import tools.website_policy
        monkeypatch.setattr(tools.website_policy, "check_website_access",
                            _raising)
        monkeypatch.setattr("tools.web_result_cache.extract_cache_get",
                            lambda url, **k: None)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["results"][0]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_extract_all_cached(self, monkeypatch):
        provider = _AsyncExtractProvider(calls=[])
        self._patch_dispatch(monkeypatch, provider)
        monkeypatch.setattr(
            "tools.web_result_cache.extract_cache_get",
            lambda url, **k: {"url": url, "title": "T", "content": "cached",
                              "cached": True},
        )
        monkeypatch.setattr("tools.web_result_cache.extract_cache_put",
                            MagicMock())
        out = json.loads(await wt.web_extract_tool(
            ["https://a.com", "https://b.com"]
        ))
        assert provider.calls == []
        assert [r["content"] for r in out["results"]] == ["cached", "cached"]

    @pytest.mark.asyncio
    async def test_extract_cache_hit_and_fetch_merge(self, monkeypatch):
        cached = {"url": "https://cached.com", "title": "Cached",
                  "content": "cached body", "cached": True}
        provider = _AsyncExtractProvider(calls=[])
        self._patch_dispatch(monkeypatch, provider)
        monkeypatch.setattr(
            "tools.web_result_cache.extract_cache_get",
            lambda url, **k: cached if url == "https://cached.com" else None,
        )
        put = MagicMock()
        monkeypatch.setattr("tools.web_result_cache.extract_cache_put", put)
        out = json.loads(await wt.web_extract_tool(
            ["https://cached.com", "https://fetched.com"]
        ))
        assert provider.calls == ["https://fetched.com"]
        assert out["results"][0]["content"] == "cached body"
        assert out["results"][1]["content"] == "ok"
        # Only the fetched URL should be put into the cache.
        put.assert_called_once_with("https://fetched.com", "ok", title="",
                                    format=None, provider="fake")

    @pytest.mark.asyncio
    async def test_provider_raise_rescued(self, monkeypatch):
        provider = _AsyncExtractProvider(results=RuntimeError("boom"))
        self._patch_dispatch(monkeypatch, provider)
        monkeypatch.setattr(wt, "_rescue_eligible", lambda p: True)
        monkeypatch.setattr(
            wt, "_rescue_extract",
            lambda name, urls, failed: [
                {"url": u, "content": "rescued", "metadata": {}}
                for u in urls
            ],
        )
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["results"][0]["content"] == "rescued"

    @pytest.mark.asyncio
    async def test_provider_raise_not_rescued(self, monkeypatch):
        provider = _AsyncExtractProvider(results=RuntimeError("boom"))
        self._patch_dispatch(monkeypatch, provider)
        monkeypatch.setattr(wt, "_rescue_eligible", lambda p: False)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["error"].startswith("Error extracting content")

    @pytest.mark.asyncio
    async def test_skip_caching_beyond_fetch_urls(self, monkeypatch):
        # Provider returns one extra entry; loop must break past fetch_urls.
        results = [
            {"url": "https://good.com", "content": "only", "raw_content": "only",
             "metadata": {}},
            {"url": "https://extra.com", "content": "extra",
             "raw_content": "extra", "metadata": {}},
        ]
        provider = _AsyncExtractProvider(results=results)
        self._patch_dispatch(monkeypatch, provider)
        put = MagicMock()
        monkeypatch.setattr("tools.web_result_cache.extract_cache_put", put)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        # The extra entry is returned to the caller, but the cache-put loop
        # must break past fetch_urls so only the genuinely-fetched URL is
        # stored (invariant: exactly one put for the one real fetch).
        assert len(out["results"]) == 2
        put.assert_called_once_with("https://good.com", "only", title="",
                                    format=None, provider="fake")

    @pytest.mark.asyncio
    async def test_order_reconstruction_mixed(self, monkeypatch):
        provider = _AsyncExtractProvider(calls=[])

        async def _safe(url):
            return "private" not in url

        self._patch_dispatch(monkeypatch, provider, safe=_safe)
        out = json.loads(await wt.web_extract_tool(
            ["https://private.example", "https://good.com", None]
        ))
        results = out["results"]
        assert len(results) == 3
        assert results[0]["error"].startswith("Blocked:")
        assert results[1]["content"] == "ok"
        assert "Invalid URL item at index 2" in results[2]["error"]

    @pytest.mark.asyncio
    async def test_char_limit_bad_fallback(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider())
        out = json.loads(await wt.web_extract_tool(
            ["https://good.com"], char_limit="abc"
        ))
        assert out["results"][0]["content"] == "ok"

    @pytest.mark.asyncio
    async def test_empty_content_skipped_truncation(self, monkeypatch):
        provider = _AsyncExtractProvider(
            results=[{"url": "https://good.com", "title": "",
                      "content": "", "raw_content": ""}]
        )
        self._patch_dispatch(monkeypatch, provider)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["results"][0]["content"] == ""

    @pytest.mark.asyncio
    async def test_outer_exception_returns_error(self, monkeypatch):
        self._patch_dispatch(monkeypatch, _AsyncExtractProvider())
        monkeypatch.setattr(wt, "_get_extract_backend", _raising)
        out = json.loads(await wt.web_extract_tool(["https://good.com"]))
        assert out["error"].startswith("Error extracting content")


class TestProviderReady:
    def test_none_provider_not_ready(self):
        assert wt._provider_is_ready(None) is False

    def test_available_provider_ready(self):
        assert wt._provider_is_ready(_FakeProvider()) is True

    def test_broken_provider_not_ready(self, monkeypatch):
        p = _FakeProvider()
        p.is_available = _raising
        assert wt._provider_is_ready(p) is False

    def test_keyless_provider_ready(self, monkeypatch):
        p = _FakeProvider()
        p.is_available = lambda: False
        p.is_keyless_available = lambda: True
        assert wt._provider_is_ready(p) is True


class TestCheckWebApiKey:
    def test_configured_backend_available(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config",
                            lambda: {"backend": "firecrawl"})
        monkeypatch.setattr(wt, "_is_backend_available", lambda b: True)
        assert wt.check_web_api_key() is True

    def test_registry_exception_returns_false(self, monkeypatch):
        monkeypatch.setattr(wt, "_load_web_config", lambda: {})
        monkeypatch.setattr(wt, "_is_backend_available", lambda b: False)
        monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", _raising)
        assert wt.check_web_api_key() is False


def _async_false():
    async def _coro(*_a, **_k):
        return False
    return _coro
