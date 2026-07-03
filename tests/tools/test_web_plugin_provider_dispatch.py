"""Regression tests for #57581 — plugin-registered web providers in config.

When ``web.extract_backend`` / ``web.search_backend`` names a provider
registered by a plugin (not in the legacy hardcoded allow-list), dispatch
must honor that name via the registry instead of silently auto-detecting a
different built-in backend.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from agent.web_search_provider import WebSearchProvider
from agent.web_search_registry import _reset_for_tests, register_provider


class _PagepruneProvider(WebSearchProvider):
    """Extract-only plugin provider used to mimic custom plugin backends."""

    @property
    def name(self) -> str:
        return "pageprune"

    @property
    def display_name(self) -> str:
        return "PagePrune"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {
                "url": urls[0],
                "title": "PagePrune title",
                "content": "from-pageprune",
                "raw_content": "from-pageprune",
            }
        ]


class _CustomSearchProvider(WebSearchProvider):
    """Search-only plugin provider."""

    @property
    def name(self) -> str:
        return "customsearch"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "custom",
                        "url": "https://example.com",
                        "description": query,
                        "position": 1,
                    }
                ]
            },
        }


@pytest.fixture(autouse=True)
def _reset_registry():
    yield
    _reset_for_tests()


@pytest.fixture
def _allow_ssrf(monkeypatch):
    async def _allow(_url: str) -> bool:
        return True

    from tools import web_tools

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)


def test_web_extract_honors_plugin_extract_backend(monkeypatch, _allow_ssrf):
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    register_all_web_providers()
    register_provider(_PagepruneProvider())

    monkeypatch.setattr(
        web_tools,
        "_load_web_config",
        lambda: {
            "backend": "",
            "search_backend": "exa",
            "extract_backend": "pageprune",
        },
    )
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-should-not-win")
    monkeypatch.setenv("EXA_API_KEY", "exa-key")

    result = json.loads(
        asyncio.get_event_loop().run_until_complete(
            web_tools.web_extract_tool(["https://example.com"])
        )
    )
    assert result["results"][0]["content"] == "from-pageprune"


def test_web_search_honors_plugin_search_backend(monkeypatch, _allow_ssrf):
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    register_all_web_providers()
    register_provider(_CustomSearchProvider())

    monkeypatch.setattr(
        web_tools,
        "_load_web_config",
        lambda: {
            "backend": "tavily",
            "search_backend": "customsearch",
            "extract_backend": "tavily",
        },
    )
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-should-not-win")

    result = json.loads(web_tools.web_search_tool("plugin query", limit=1))
    assert result["success"] is True
    assert result["data"]["web"][0]["description"] == "plugin query"
