"""Autodetect must skip server-side-only web providers (#137475).

``openai-native`` declares OpenAI's server-side Responses ``web_search`` tool: it only
works when the Codex Responses transport swaps the client-side function out. When it was
probed like any other plugin provider (``is_available()`` == "has Codex credentials"), a
never-configured install with a Codex login autodetected ``openai-native`` as the shared
client-side backend and every ``web_search`` / ``web_extract`` call failed. These tests
pin the contract: server-side-only providers are unreachable by autodetect, while an
explicit ``web.backend`` pin and ordinary client-side plugin providers keep working.
"""

import os
from unittest.mock import patch

import pytest


class _ServerSideOnlyProvider:
    """Stand-in for openai-native: credentialed but unable to execute client-side calls."""

    name = "openai-native"

    def is_available(self) -> bool:
        return True

    def supports_client_side(self) -> bool:
        return False


class _ClientSidePluginProvider:
    """An ordinary plugin-contributed provider (any name outside the built-in ladder)."""

    name = "acme-web"

    def is_available(self) -> bool:
        return True

    def supports_client_side(self) -> bool:
        return True


def _no_keys():
    """Context manager stack: a never-configured install with zero web credentials."""
    env = {k: "" for k in (
        "TAVILY_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY", "PARALLEL_API_KEY",
        "KEENABLE_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "SEARXNG_URL",
        "BRAVE_SEARCH_API_KEY",
    )}
    return patch.dict(os.environ, env)


class TestServerSideOnlyAutodetect:
    def test_serverside_only_provider_not_autodetected(self):
        from tools.web_tools import _autodetect_backend
        with _no_keys(), \
             patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._managed_web_search", return_value=False), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._list_registered_web_providers",
                   return_value=[_ServerSideOnlyProvider()]):
            assert _autodetect_backend() is None

    def test_get_backend_falls_through_instead_of_openai_native(self):
        from tools.web_tools import _get_backend
        with _no_keys(), \
             patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._managed_web_search", return_value=False), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._list_registered_web_providers",
                   return_value=[_ServerSideOnlyProvider()]), \
             patch("tools.web_tools._keyless_backend", return_value=None):
            assert _get_backend() == "firecrawl"

    def test_client_side_plugin_provider_still_autodetected(self):
        from tools.web_tools import _autodetect_backend
        with _no_keys(), \
             patch("tools.web_tools._load_web_config", return_value={}), \
             patch("tools.web_tools._managed_web_search", return_value=False), \
             patch("tools.web_tools._is_tool_gateway_ready", return_value=False), \
             patch("tools.web_tools._ddgs_package_importable", return_value=False), \
             patch("tools.web_tools._list_registered_web_providers",
                   return_value=[_ClientSidePluginProvider()]):
            assert _autodetect_backend() == "acme-web"

    def test_explicit_pin_still_reaches_serverside_provider(self):
        """Strict selection: an explicit ``web.backend`` is returned as-is — the fix only
        removes autodetect eligibility, never the user's deliberate configuration."""
        from tools.web_tools import _get_backend
        with _no_keys(), \
             patch("tools.web_tools._load_web_config",
                   return_value={"backend": "openai-native"}):
            assert _get_backend() == "openai-native"


class TestOpenAINativeDeclaration:
    def test_openai_native_declares_serverside_only(self):
        from plugins.web.openai_native.provider import OpenAINativeWebSearchProvider
        assert OpenAINativeWebSearchProvider().supports_client_side() is False

    def test_base_provider_defaults_to_client_side(self):
        from agent.web_search_provider import WebSearchProvider
        assert WebSearchProvider.supports_client_side(None) is True
