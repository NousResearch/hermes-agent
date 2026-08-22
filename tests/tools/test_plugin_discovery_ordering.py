"""Regression tests for PR #67309: plugin-registered web providers must be
discoverable from backend selection even when no earlier code path has
already triggered plugin discovery.

Bug: _get_backend() and _is_backend_available() read the web_search_registry
directly without first calling _ensure_web_plugins_loaded(). On a cold
registry (subprocess agent runs, delegate children, standalone scripts),
plugin-contributed providers are invisible to selection even though
discovery would find them if triggered.

Fix: _ensure_web_plugins_loaded() added at the top of both chokepoints.

These tests replace test_cold_registry_plugin_provider_selected_before_discovery,
which pre-registered its fake provider via register_provider() directly and
therefore passed identically whether or not the fix was present — it never
exercised the discovery call the fix adds.
"""
import os

import pytest
from unittest.mock import patch

from agent.web_search_provider import WebSearchProvider
from agent.web_search_registry import _reset_for_tests, register_provider, get_provider


# Env vars that could make a built-in backend available and mask the
# plugin-provider path we're testing. Mirrors
# TestNonBuiltinProviderAvailability._WEB_ENV_KEYS.
_WEB_ENV_KEYS = (
    "EXA_API_KEY",
    "PARALLEL_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_GATEWAY_URL",
    "TOOL_GATEWAY_DOMAIN",
    "TOOL_GATEWAY_SCHEME",
    "TOOL_GATEWAY_USER_TOKEN",
    "TAVILY_API_KEY",
    "SEARXNG_URL",
    "BRAVE_SEARCH_API_KEY",
    "XAI_API_KEY",
)


class ColdPluginProvider(WebSearchProvider):
    """A provider real plugin discovery would register. Module-level (not
    nested) — nested class redefinition under pytest hits the Python 3.13
    __bases__ deallocator issue documented in TestNonBuiltinProviderAvailability."""

    @property
    def name(self):
        return "cold-plugin-extract"

    def is_available(self):
        return True

    def supports_search(self):
        return False

    def supports_extract(self):
        return True


@pytest.fixture
def cold_web_env(monkeypatch):
    """Truly empty registry + no plugin-discovered flag + no builtin creds.

    Unlike TestNonBuiltinProviderAvailability.setup_method, this does NOT
    pre-register any provider — that's the entire point of these tests.
    """
    for key in _WEB_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    _reset_for_tests()

    # Reset the plugin-discovery "already ran" flag so _ensure_web_plugins_loaded
    # is forced to do real work (or, when mocked, is forced to be called) rather
    # than short-circuiting because an earlier test in the same process already
    # discovered plugins. This is the module-global re-entrancy guard in
    # hermes_cli.plugins.PluginManager.discover_and_load.
    import hermes_cli.plugins as plugins_mod
    if plugins_mod._plugin_manager is not None:
        monkeypatch.setattr(plugins_mod._plugin_manager, "_discovered", False, raising=False)

    yield

    _reset_for_tests()


def _register_cold_provider():
    """Side effect standing in for what real plugin discovery does:
    import a plugin module, which calls register_provider() at import time."""
    register_provider(ColdPluginProvider())


class TestPluginDiscoveryOrdering:
    """Proves discovery runs BEFORE the registry is consulted, for every
    chokepoint that reads it. Parametrized so a future chokepoint that
    forgets the _ensure_web_plugins_loaded() call is caught the same way.
    """

    @pytest.mark.parametrize(
        "entry_point",
        ["is_backend_available", "get_backend", "get_extract_backend"],
    )
    def test_discovery_precedes_registry_read(self, entry_point, cold_web_env):
        import tools.web_tools as wt

        order = []
        # _get_backend() reads the registry via _registered_web_provider()
        # (web_tools.py:231); _is_backend_available() reads it via
        # _registered_web_provider_available() (web_tools.py:325). Both must
        # be spied — wrapping only one silently blinds this test to the
        # other chokepoint's registry access.
        real_provider = wt._registered_web_provider
        real_provider_available = wt._registered_web_provider_available

        def spy_provider(backend):
            order.append("registry_read")
            return real_provider(backend)

        def spy_provider_available(backend):
            order.append("registry_read")
            return real_provider_available(backend)

        def spy_discovery():
            order.append("discovery")
            _register_cold_provider()

        with patch.object(wt, "_ensure_web_plugins_loaded", side_effect=spy_discovery) as ensure_mock, \
             patch.object(wt, "_registered_web_provider", side_effect=spy_provider), \
             patch.object(wt, "_registered_web_provider_available", side_effect=spy_provider_available), \
             patch.object(wt, "_ddgs_package_importable", return_value=False), \
             patch.object(wt, "_peek_nous_access_token", return_value=None), \
             patch.object(wt, "_load_web_config", return_value={
                 "backend": "cold-plugin-extract",
                 "extract_backend": "cold-plugin-extract",
             }):

            if entry_point == "is_backend_available":
                result = wt._is_backend_available("cold-plugin-extract")
                assert result is True
            elif entry_point == "get_backend":
                result = wt._get_backend()
                assert result == "cold-plugin-extract"
            else:
                result = wt._get_extract_backend()
                assert result == "cold-plugin-extract"

            assert ensure_mock.called, (
                f"{entry_point}() never called _ensure_web_plugins_loaded() — "
                "a cold registry would never be populated"
            )
            assert "registry_read" in order, (
                f"{entry_point}() never consulted the registry at all"
            )
            assert order.index("discovery") < order.index("registry_read"), (
                f"{entry_point}() read the registry before triggering "
                f"discovery: {order}"
            )


class TestColdRegistryProviderSelection:
    """End-to-end: with the fix, a cold registry ends up with the right
    backend selected via the real _get_extract_backend() / _get_backend()
    call chain — not just an ordering check, but the actual outcome the
    bug report cared about.
    """

    def test_capability_backend_selects_plugin_provider_from_cold_registry(self, cold_web_env):
        # Precondition: registry really is empty before the chokepoint runs.
        assert get_provider("cold-plugin-extract") is None

        import tools.web_tools as wt
        with patch.object(wt, "_ensure_web_plugins_loaded", side_effect=_register_cold_provider), \
             patch.object(wt, "_ddgs_package_importable", return_value=False), \
             patch.object(wt, "_peek_nous_access_token", return_value=None), \
             patch.object(wt, "_load_web_config", return_value={"extract_backend": "cold-plugin-extract"}):

            assert wt._get_extract_backend() == "cold-plugin-extract"

    def test_default_backend_path_selects_plugin_provider_from_cold_registry(self, cold_web_env):
        """Pins the SECOND fix line. _get_capability_backend() short-circuits
        through _is_backend_available() when a per-capability override is
        configured (see _get_capability_backend in web_tools.py); to reach
        the plain _get_backend() chokepoint we must configure only
        ``backend``, not ``extract_backend``/``search_backend``, so
        _get_capability_backend falls through to _get_backend() and that
        function's own _ensure_web_plugins_loaded() call is what's exercised.
        """
        assert get_provider("cold-plugin-extract") is None

        import tools.web_tools as wt
        with patch.object(wt, "_ensure_web_plugins_loaded", side_effect=_register_cold_provider) as ensure_mock, \
             patch.object(wt, "_ddgs_package_importable", return_value=False), \
             patch.object(wt, "_peek_nous_access_token", return_value=None), \
             patch.object(wt, "_load_web_config", return_value={"backend": "cold-plugin-extract"}):

            assert wt._get_backend() == "cold-plugin-extract"
            assert ensure_mock.called

    def test_unregistered_backend_still_unavailable_after_discovery(self, cold_web_env):
        """Negative case: discovery running doesn't make an unknown backend
        name pass — guards against the fix accidentally turning the registry
        check into an unconditional True."""
        import tools.web_tools as wt
        with patch.object(wt, "_ensure_web_plugins_loaded", side_effect=_register_cold_provider), \
             patch.object(wt, "_ddgs_package_importable", return_value=False):

            assert wt._is_backend_available("totally-unregistered-backend") is False


class TestDiscoveryReentrancy:
    """A plugin's own import-time code can legitimately call back into
    _is_backend_available() (e.g. to decide whether to self-register). The
    re-entrancy guard in PluginManager.discover_and_load sets _discovered =
    True BEFORE the scan runs specifically to prevent infinite recursion.

    This is a property of _ensure_web_plugins_loaded() / discover_and_load()
    itself, not of the two chokepoint call sites the PR adds — so it's
    exercised directly rather than through _is_backend_available(), and
    stays meaningful regardless of whether the two-line fix is present.
    """

    def test_reentrant_call_during_discovery_does_not_recurse_or_hang(self, cold_web_env):
        import tools.web_tools as wt

        calls = {"count": 0}
        real_ensure = wt._ensure_web_plugins_loaded

        def reentrant_ensure():
            calls["count"] += 1
            if calls["count"] == 1:
                # Simulate a plugin's import-time code asking "am I needed?"
                # before registering itself — a real, supported pattern.
                # Must not recurse back into real discovery.
                real_ensure()
            register_provider(ColdPluginProvider())

        with patch.object(wt, "_ensure_web_plugins_loaded", side_effect=reentrant_ensure) as ensure_mock, \
             patch.object(wt, "_ddgs_package_importable", return_value=False), \
             patch.object(wt, "_peek_nous_access_token", return_value=None):

            # Must terminate (no infinite recursion) and must not raise.
            result = wt._is_backend_available("cold-plugin-extract")

        if not ensure_mock.called:
            pytest.skip(
                "_is_backend_available() does not call _ensure_web_plugins_loaded() "
                "— fix not applied; reentrancy guard has nothing to exercise here"
            )
        assert calls["count"] == 1, "discovery should only run once, not recurse"
        assert result is True


@pytest.mark.integration
class TestRealPluginDiscovery:
    """The tests above mock _ensure_web_plugins_loaded, which proves the
    call-order contract but never proves real discovery actually registers
    anything. This test exercises the genuine discovery path so the fix's
    premise ("discovery, once triggered, populates the registry") is not
    purely assumed.
    """

    def test_real_discovery_registers_bundled_firecrawl_provider(self, cold_web_env, monkeypatch):
        """Cheapest real-discovery smoke test: don't fabricate a plugin at
        all, just prove that letting the *real* _ensure_web_plugins_loaded()
        run (unmocked) populates the registry with a bundled provider that
        ships in-repo (plugins/web/firecrawl). This confirms discovery
        itself works, independent of the ordering fix.
        """
        import tools.web_tools as wt

        assert get_provider("firecrawl") is None  # cold

        wt._ensure_web_plugins_loaded()  # real call, not mocked

        assert get_provider("firecrawl") is not None, (
            "real plugin discovery did not register the bundled firecrawl "
            "provider — either discovery is broken or plugins/web/firecrawl "
            "is not being picked up, independent of the ordering fix"
        )
