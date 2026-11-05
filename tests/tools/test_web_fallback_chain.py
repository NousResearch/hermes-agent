"""Opt-in ordered fallback chain (web.search_fallback_chain / web.extract_fallback_chain).

Covers:
- search: walks the chain in order after the selected backend fails (error result or exception),
  stops at the first success, skips backends without credentials, annotates the result
- an empty result list is a success (no failover)
- an exhausted chain hands the call to the keyless rescue, or fails with every backend's error
- never sticky: a chain-served result is not cached, the next call tries the selection first
- a stored selection that resolves to no provider is served by the chain
- extract: only the URLs the selection failed are re-sent; partial success and origin HTTP errors
  stay; fallback pages carry served_by and are not cached under the selection's key
- no chain configured: behaviour unchanged
"""

import copy
import json
from unittest.mock import patch

import pytest

from plugins.web import keyless_mcp
from plugins.web._common import origin_http_error
from tools import web_tools


class _Provider:
    """Web provider double: canned search response or exception, extract via a callable."""

    def __init__(self, name, search=None, raises=None, extract=None, available=True, can_extract=True):
        self.name = name
        self.display_name = name.title()
        self._search, self._raises, self._extract = search, raises, extract
        self._available, self._can_extract = available, can_extract
        self.calls = []

    def supports_search(self):
        return True

    def supports_extract(self):
        return self._can_extract

    def is_available(self):
        return self._available

    def is_keyless_available(self):
        return False

    def search(self, query, limit=5):
        self.calls.append(query)
        if self._raises:
            raise RuntimeError(self._raises)
        return copy.deepcopy(self._search)

    def extract(self, urls, **kwargs):
        self.calls.append(list(urls))
        if self._raises:
            raise RuntimeError(self._raises)
        return self._extract(urls)


def _ok(vendor):
    return {"success": True, "data": {"web": [{"url": f"https://{vendor}.example", "title": vendor}]}}


def _fail(error="HTTP 429 Too Many Requests"):
    return {"success": False, "error": error}


def _page(url, text):
    return {"url": url, "title": url, "content": text, "raw_content": text, "metadata": {"sourceURL": url}}


def _failed(url, error="HTTP 500 upstream exploded"):
    return {"url": url, "title": "", "content": "", "error": error}


@pytest.fixture
def setup(monkeypatch):
    """Wire a provider map and a web config; returns a configurator."""
    state = {"providers": {}, "config": {}}
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: state["config"])
    monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: state["providers"].get(name))
    monkeypatch.setattr("agent.web_search_registry._keyless_tier_enabled", lambda: True)

    def configure(providers, **config):
        state["providers"] = {p.name: p for p in providers}
        state["config"] = config
    return configure


def _search(query="q"):
    return json.loads(web_tools.web_search_tool(query, limit=3))


class TestSearchChain:
    def test_walks_in_order_and_annotates_the_serving_backend(self, setup):
        primary, exa, tavily = _Provider("searxng", _fail()), _Provider("exa", _fail("HTTP 401")), _Provider("tavily", _ok("tavily"))
        setup([primary, exa, tavily], search_backend="searxng", search_fallback_chain=["exa", "tavily"])
        with patch.object(keyless_mcp, "search_with_failover") as ring:
            out = _search()
        assert out["success"] is True
        data = out["data"]
        assert data["served_by"] == "tavily" and data["fallback_from"] == "searxng"
        assert [e["backend"] for e in data["chain_errors"]] == ["searxng", "exa"]
        assert "HTTP 429" in data["backend_error"] and "next call" in data["backend_error"].lower()
        assert len(exa.calls) == 1 and len(tavily.calls) == 1
        ring.assert_not_called()

    def test_first_success_stops_the_walk(self, setup):
        primary, exa, tavily = _Provider("searxng", _fail()), _Provider("exa", _ok("exa")), _Provider("tavily", _ok("tavily"))
        setup([primary, exa, tavily], search_backend="searxng", search_fallback_chain=["exa", "tavily"])
        assert _search()["data"]["served_by"] == "exa"
        assert tavily.calls == []

    def test_empty_results_are_a_success_not_a_failure(self, setup):
        primary = _Provider("searxng", {"success": True, "data": {"web": []}})
        exa = _Provider("exa", _ok("exa"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain=["exa"])
        out = _search()
        assert out["success"] is True and "served_by" not in out["data"]
        assert exa.calls == []

    def test_raised_exception_fails_over(self, setup):
        primary, exa = _Provider("searxng", raises="connection refused"), _Provider("exa", _ok("exa"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain=["exa"])
        out = _search()
        assert out["data"]["served_by"] == "exa"
        assert "connection refused" in out["data"]["chain_errors"][0]["error"]

    def test_backend_without_credentials_is_skipped_and_named(self, setup):
        primary = _Provider("searxng", _fail())
        exa = _Provider("exa", _ok("exa"), available=False)
        tavily = _Provider("tavily", _ok("tavily"))
        setup([primary, exa, tavily], search_backend="searxng", search_fallback_chain=["exa", "tavily"])
        out = _search()
        assert out["data"]["served_by"] == "tavily" and exa.calls == []
        assert "not configured" in out["data"]["chain_errors"][1]["error"]

    def test_exhausted_chain_then_keyless_rescue(self, setup):
        primary, exa = _Provider("searxng", _fail()), _Provider("exa", _fail("HTTP 402 credits"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain=["exa"])
        with patch.object(keyless_mcp, "search_with_failover", return_value=_ok("ring")) as ring:
            out = _search()
        assert out["success"] is True and out["data"]["rescued_from"] == "searxng"
        assert "fallback chain also failed: exa: HTTP 402 credits" in out["data"]["backend_error"]
        ring.assert_called_once()

    def test_exhausted_chain_without_rescue_reports_every_backend(self, setup):
        primary, exa = _Provider("searxng", _fail()), _Provider("exa", _fail("HTTP 402 credits"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain=["exa"], keyless_rescue=False)
        out = _search()
        assert out["success"] is False
        assert "HTTP 429" in out["error"] and "exa: HTTP 402 credits" in out["error"]

    def test_never_sticky_next_call_tries_the_selection_first(self, setup):
        primary, exa = _Provider("searxng", _fail()), _Provider("exa", _ok("exa"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain=["exa"])
        _search("same query")
        _search("same query")
        assert len(primary.calls) == 2 and len(exa.calls) == 2

    def test_unresolvable_selection_is_served_by_the_chain(self, setup, monkeypatch):
        monkeypatch.setattr(web_tools, "selection_exists", lambda kind: True)
        exa = _Provider("exa", _ok("exa"))
        setup([exa], search_backend="gone", search_fallback_chain=["exa"])
        out = _search()
        assert out["success"] is True
        assert out["data"]["served_by"] == "exa" and out["data"]["fallback_from"] == "gone"

    def test_no_chain_keeps_the_selection_error(self, setup):
        primary, exa = _Provider("searxng", _fail()), _Provider("exa", _ok("exa"))
        setup([primary, exa], search_backend="searxng", keyless_rescue=False)
        out = _search()
        assert out == {"success": False, "error": "HTTP 429 Too Many Requests"}
        assert exa.calls == []

    def test_a_bare_string_is_a_one_element_chain(self, setup):
        primary, exa = _Provider("searxng", _fail()), _Provider("exa", _ok("exa"))
        setup([primary, exa], search_backend="searxng", search_fallback_chain="exa")
        assert _search()["data"]["served_by"] == "exa"


class TestExtractChain:
    async def _extract(self, monkeypatch, urls):
        async def _allow(url, **kwargs):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", _allow)
        monkeypatch.setattr("tools.web_result_cache.extract_cache_get", lambda *a, **kw: None)
        stored = []
        monkeypatch.setattr("tools.web_result_cache.extract_cache_put",
                            lambda url, *a, **kw: stored.append(url) or True)
        raw = await web_tools.web_extract_tool(list(urls))
        return json.loads(raw)["results"], stored

    @pytest.mark.asyncio
    async def test_only_failed_urls_are_resent_and_partial_success_stays(self, setup, monkeypatch):
        primary = _Provider("tavily", extract=lambda urls: [_page("https://a", "A" * 60), _failed("https://b")])
        exa = _Provider("exa", extract=lambda urls: [_page(u, "B" * 60) for u in urls])
        setup([primary, exa], extract_backend="tavily", extract_fallback_chain=["exa"])
        results, stored = await self._extract(monkeypatch, ["https://a", "https://b"])
        assert exa.calls == [["https://b"]]
        assert results[0]["content"].startswith("A") and "served_by" not in results[0]
        assert results[1]["content"].startswith("B") and results[1]["served_by"] == "exa"
        assert stored == ["https://a"]  # the fallback page is not cached under tavily's key

    @pytest.mark.asyncio
    async def test_origin_http_error_is_the_pages_answer_not_resent(self, setup, monkeypatch):
        primary = _Provider("tavily", extract=lambda urls: [
            _page("https://a", "A" * 60), _failed("https://b", origin_http_error(404))])
        exa = _Provider("exa", extract=lambda urls: [_page(u, "B" * 60) for u in urls])
        setup([primary, exa], extract_backend="tavily", extract_fallback_chain=["exa"])
        results, _ = await self._extract(monkeypatch, ["https://a", "https://b"])
        assert exa.calls == [] and results[1]["error"] == origin_http_error(404)

    @pytest.mark.asyncio
    async def test_raised_exception_without_rescue_is_served_by_the_chain(self, setup, monkeypatch):
        primary = _Provider("tavily", raises="403 Forbidden", extract=lambda urls: [])
        exa = _Provider("exa", extract=lambda urls: [_page(u, "E" * 60) for u in urls])
        setup([primary, exa], extract_backend="tavily", extract_fallback_chain=["exa"], keyless_rescue=False)
        results, _ = await self._extract(monkeypatch, ["https://a"])
        assert results[0]["served_by"] == "exa" and not results[0]["error"]

    @pytest.mark.asyncio
    async def test_exhausted_chain_names_every_backend_in_the_error(self, setup, monkeypatch):
        primary = _Provider("tavily", extract=lambda urls: [_failed(u) for u in urls])
        exa = _Provider("exa", extract=lambda urls: [_failed(u, "HTTP 402 credits") for u in urls])
        setup([primary, exa], extract_backend="tavily", extract_fallback_chain=["exa"], keyless_rescue=False)
        results, stored = await self._extract(monkeypatch, ["https://a"])
        assert "HTTP 500 upstream exploded" in results[0]["error"]
        assert "fallback chain also failed: exa: HTTP 402 credits" in results[0]["error"]
        assert stored == []

    @pytest.mark.asyncio
    async def test_search_only_fallback_is_skipped(self, setup, monkeypatch):
        primary = _Provider("tavily", extract=lambda urls: [_failed(u) for u in urls])
        searx = _Provider("searxng", can_extract=False)
        exa = _Provider("exa", extract=lambda urls: [_page(u, "E" * 60) for u in urls])
        setup([primary, searx, exa], extract_backend="tavily", extract_fallback_chain=["searxng", "exa"],
              keyless_rescue=False)
        results, _ = await self._extract(monkeypatch, ["https://a"])
        assert results[0]["served_by"] == "exa" and searx.calls == []
