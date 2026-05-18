"""Regression tests for issue #27683.

`tools/web_tools.py` previously dispatched to providers in
`agent.web_search_registry` without first triggering plugin discovery.
On a fresh install (or before the gateway / CLI had imported plugins for
any other reason) the registry was empty and every provider lookup
returned ``None`` — search, extract, and crawl all silently fell through
to "no provider configured" even though the user had a backend installed
and configured.

These tests pin the fix: each of the three dispatch sites in
``tools/web_tools.py`` must call
``hermes_cli.plugins._ensure_plugins_discovered`` before consulting the
registry. We patch the discovery hook and assert it was invoked.

We don't care what happens AFTER discovery — the tools may legitimately
return a "no provider configured" envelope in the test environment.
We only care that the discovery side-effect happens.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest


def _called(mock: MagicMock) -> bool:
    """Return True if the discovery hook was invoked at least once."""
    return mock.call_count >= 1


class TestPluginDiscoveryFiredBeforeDispatch:
    """Each of the three dispatchers must trigger plugin discovery."""

    def test_web_search_triggers_plugin_discovery(self):
        from tools import web_tools

        with patch("hermes_cli.plugins._ensure_plugins_discovered") as ensure:
            # Force no provider to be returned so the tool short-circuits
            # quickly after the discovery call.
            with patch(
                "agent.web_search_registry.get_provider", return_value=None
            ), patch(
                "agent.web_search_registry.get_active_search_provider",
                return_value=None,
            ):
                # We don't care about the return value — only that
                # discovery was invoked before the registry was consulted.
                try:
                    web_tools.web_search_tool("hello world", limit=1)
                except Exception:
                    # Any downstream raise is fine; the assertion is on
                    # the discovery call.
                    pass

        assert _called(ensure), (
            "web_search_tool must call _ensure_plugins_discovered() "
            "before consulting agent.web_search_registry (issue #27683)"
        )

    def test_web_extract_triggers_plugin_discovery(self):
        from tools import web_tools

        with patch("hermes_cli.plugins._ensure_plugins_discovered") as ensure:
            with patch(
                "agent.web_search_registry.get_provider", return_value=None
            ), patch(
                "agent.web_search_registry.get_active_extract_provider",
                return_value=None,
            ):
                try:
                    asyncio.run(
                        web_tools.web_extract_tool(
                            urls=["https://example.com"],
                            use_llm_processing=False,
                        )
                    )
                except Exception:
                    pass

        assert _called(ensure), (
            "web_extract_tool must call _ensure_plugins_discovered() "
            "before consulting agent.web_search_registry (issue #27683)"
        )

    def test_web_crawl_triggers_plugin_discovery(self):
        from tools import web_tools

        with patch("hermes_cli.plugins._ensure_plugins_discovered") as ensure:
            with patch(
                "agent.web_search_registry.get_provider", return_value=None
            ), patch(
                "agent.web_search_registry.get_active_crawl_provider",
                return_value=None,
            ):
                try:
                    asyncio.run(
                        web_tools.web_crawl_tool(
                            url="https://example.com",
                            use_llm_processing=False,
                        )
                    )
                except Exception:
                    pass

        assert _called(ensure), (
            "web_crawl_tool must call _ensure_plugins_discovered() "
            "before consulting agent.web_search_registry (issue #27683)"
        )
