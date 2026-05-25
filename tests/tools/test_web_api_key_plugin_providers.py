"""Regression tests for check_web_api_key() plugin-provider awareness (#31873).

Third-party web search plugins register a WebSearchProvider via
PluginContext.register_web_search_provider(). Previously check_web_api_key()
only knew the hardcoded built-in backends, so the web_search/web_extract tools
were gated off whenever ONLY a plugin provider (e.g. Kagi on KAGI_API_KEY) was
configured. The check now also consults agent.web_search_registry.
"""
from unittest.mock import MagicMock, patch

import tools.web_tools as wt


def _no_builtins():
    """All built-in backends unavailable; no backend configured."""
    return (
        patch.object(wt, "_load_web_config", return_value={}),
        patch.object(wt, "_is_backend_available", return_value=False),
    )


def test_available_plugin_provider_enables_web_tools():
    cfg, avail = _no_builtins()
    provider = MagicMock()
    provider.is_available.return_value = True
    with cfg, avail, patch(
        "agent.web_search_registry.get_active_search_provider",
        return_value=provider,
    ):
        assert wt.check_web_api_key() is True


def test_no_builtin_and_no_plugin_returns_false():
    cfg, avail = _no_builtins()
    with cfg, avail, patch(
        "agent.web_search_registry.get_active_search_provider",
        return_value=None,
    ):
        assert wt.check_web_api_key() is False


def test_plugin_present_but_unavailable_returns_false():
    cfg, avail = _no_builtins()
    provider = MagicMock()
    provider.is_available.return_value = False
    with cfg, avail, patch(
        "agent.web_search_registry.get_active_search_provider",
        return_value=provider,
    ):
        assert wt.check_web_api_key() is False


def test_registry_failure_does_not_crash_the_gate():
    cfg, avail = _no_builtins()
    with cfg, avail, patch(
        "agent.web_search_registry.get_active_search_provider",
        side_effect=RuntimeError("registry boom"),
    ):
        assert wt.check_web_api_key() is False


def test_builtin_backend_still_wins_without_touching_registry():
    """An available built-in backend short-circuits to True as before."""
    with patch.object(wt, "_load_web_config", return_value={"backend": "exa"}), \
         patch.object(wt, "_is_backend_available", return_value=True):
        assert wt.check_web_api_key() is True
