"""Regression coverage for quota-aware web backend fallback chains."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent.web_search_provider import WebSearchProvider
from agent import web_search_registry


class FakeWebProvider(WebSearchProvider):
    def __init__(
        self,
        name: str,
        *,
        search_response: dict[str, Any] | Exception | None = None,
        extract_response: list[dict[str, Any]] | Exception | None = None,
        available: bool = True,
    ) -> None:
        self._name = name
        self.search_response = search_response
        self.extract_response = extract_response
        self.available = available
        self.search_calls: list[tuple[str, int]] = []
        self.extract_calls: list[tuple[list[str], dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self.available

    def supports_search(self) -> bool:
        return self.search_response is not None

    def supports_extract(self) -> bool:
        return self.extract_response is not None

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        self.search_calls.append((query, limit))
        if isinstance(self.search_response, Exception):
            raise self.search_response
        assert self.search_response is not None
        return self.search_response

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        self.extract_calls.append((urls, kwargs))
        if isinstance(self.extract_response, Exception):
            raise self.extract_response
        assert self.extract_response is not None
        return self.extract_response


@pytest.fixture(autouse=True)
def reset_registry():
    web_search_registry._reset_for_tests()
    yield
    web_search_registry._reset_for_tests()


def test_parse_backend_chain_accepts_lists_and_comma_strings():
    import tools.web_tools as wt

    assert wt._parse_backend_chain(["Brave-Free", "exa", "unknown", "exa"]) == [
        "brave-free",
        "exa",
    ]
    assert wt._parse_backend_chain("brave-free, exa parallel  tavily") == [
        "brave-free",
        "exa",
        "parallel",
        "tavily",
    ]


def test_web_search_falls_back_to_next_provider_on_error(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider(
        "brave-free",
        search_response={"success": False, "error": "Brave Search returned HTTP 429"},
    )
    exa = FakeWebProvider(
        "exa",
        search_response={
            "success": True,
            "data": {"web": [{"title": "Exa hit", "url": "https://example.com", "description": "ok"}]},
        },
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["brave-free", "exa"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("agent search", limit=3))

    assert result["success"] is True
    assert result["data"]["web"][0]["title"] == "Exa hit"
    assert brave.search_calls == [("agent search", 3)]
    assert exa.search_calls == [("agent search", 3)]


def test_web_search_does_not_spend_fallback_on_clean_empty_by_default(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider("brave-free", search_response={"success": True, "data": {"web": []}})
    exa = FakeWebProvider(
        "exa",
        search_response={"success": True, "data": {"web": [{"title": "would cost", "url": "https://example.com"}]}},
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["brave-free", "exa"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("definitely empty", limit=2))

    assert result == {"success": True, "data": {"web": []}}
    assert brave.search_calls == [("definitely empty", 2)]
    assert exa.search_calls == []


def test_web_extract_skips_search_only_and_falls_back_to_exa(monkeypatch):
    import tools.web_tools as wt

    brave = FakeWebProvider("brave-free", search_response={"success": True, "data": {"web": []}})
    firecrawl = FakeWebProvider(
        "firecrawl",
        extract_response=[{"url": "https://example.com", "title": "", "content": "", "error": "quota exhausted"}],
    )
    exa = FakeWebProvider(
        "exa",
        extract_response=[{"url": "https://example.com", "title": "Example", "content": "clean markdown"}],
    )
    web_search_registry.register_provider(brave)
    web_search_registry.register_provider(firecrawl)
    web_search_registry.register_provider(exa)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        wt,
        "_load_web_config",
        lambda: {"extract_backends": ["brave-free", "firecrawl", "exa"]},
    )
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)
    monkeypatch.setattr(wt, "async_is_safe_url", lambda url: asyncio.sleep(0, result=True))

    result = json.loads(asyncio.run(wt.web_extract_tool(["https://example.com"])))

    assert result["results"][0]["title"] == "Example"
    assert result["results"][0]["content"] == "clean markdown"
    assert brave.extract_calls == []
    assert firecrawl.extract_calls
    assert exa.extract_calls


def test_xai_can_remain_first_in_search_chain(monkeypatch):
    import tools.web_tools as wt

    xai = FakeWebProvider(
        "xai",
        search_response={"success": True, "data": {"web": [{"title": "Grok hit", "url": "https://x.ai"}]}},
    )
    brave = FakeWebProvider(
        "brave-free",
        search_response={"success": True, "data": {"web": [{"title": "Brave hit", "url": "https://brave.com"}]}},
    )
    web_search_registry.register_provider(xai)
    web_search_registry.register_provider(brave)

    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_load_web_config", lambda: {"search_backends": ["xai", "brave-free"]})
    monkeypatch.setattr(wt, "_provider_available_for_chain", lambda name: True)

    result = json.loads(wt.web_search_tool("X post context", limit=1))

    assert result["data"]["web"][0]["title"] == "Grok hit"
    assert xai.search_calls == [("X post context", 1)]
    assert brave.search_calls == []
