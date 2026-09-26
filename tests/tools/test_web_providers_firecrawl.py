"""Tests for the Firecrawl web search + extract provider (``plugins/web/firecrawl/``).

Focused behavior contracts (#36514): lazy SDK proxy construction, keyless-ring
routing (direct credentials and a stored managed selection keep the ring off; a
raising probe never blocks it), client resolution (managed = gateway only, never
a silent direct fallback; explicit selection without credentials = keyless
cloud; resolved config cached by value), the keyless REST client sending no
``Authorization`` header, response-shape normalization, ``_scrape_one`` policy /
timeout / redirect guards, and the ``search`` / ``extract`` failure shapes.
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

import plugins.web.firecrawl.provider as fc
from plugins.web.firecrawl.provider import (
    FirecrawlWebSearchProvider,
    _KeylessFirecrawlClient,
    _error_entry,
    _extract_scrape_payload,
    _extract_web_search_results,
    _scrape_one,
    _to_plain_object,
)

MANAGED = "nous"


def _env_from(**values: str):
    """Stand-in for ``provider._env`` so one test owns the credential view."""
    return lambda name: values.get(name, "")


def _sdk_ctor(calls: List[Dict[str, Any]]):
    """Stand-in for the lazily imported SDK class; records constructor kwargs."""
    class _SdkClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self.kwargs = kwargs
            self.search = MagicMock()
            self.scrape = MagicMock()

    return _SdkClient


class _Metadata:
    """Metadata object whose ``model_dump`` is the only readable shape (SDK-style)."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> Dict[str, Any]:
        return self._payload


class _BrokenDump:
    def __init__(self) -> None:
        self.public = "visible"
        self._private = "hidden"

    def model_dump(self) -> Dict[str, Any]:
        raise RuntimeError("model_dump exploded")


class _SlotsObject:
    """No ``model_dump`` and no ``__dict__`` — normalization must hand it back untouched."""

    __slots__ = ("web",)

    def __init__(self, web: List[Dict[str, Any]]) -> None:
        self.web = web


class _FakeScrapeClient:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.scrape_calls: List[Dict[str, Any]] = []
        self._result = result
        self._error = error

    def scrape(self, *, url: str, formats: List[str]) -> Any:
        self.scrape_calls.append({"url": url, "formats": formats})
        if self._error is not None:
            raise self._error
        return self._result


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch):
    """No ambient FIRECRAWL_* credential may decide a routing test."""
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)


@pytest.fixture
def client_slots(monkeypatch):
    """The client cache slots ``_get_firecrawl_client`` reads/writes live on tools.web_tools."""
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_firecrawl_client", None, raising=False)
    monkeypatch.setattr(wt, "_firecrawl_client_config", None, raising=False)
    return wt


# ---------------------------------------------------------------------------
# Lazy SDK proxy
# ---------------------------------------------------------------------------


class TestLazyFirecrawlProxy:
    def test_loads_the_sdk_class_once_and_caches_it(self, monkeypatch):
        fake_module = types.ModuleType("firecrawl")
        fake_module.Firecrawl = _sdk_ctor([])
        monkeypatch.setitem(sys.modules, "firecrawl", fake_module)
        monkeypatch.setattr(fc, "lazy_ensure", MagicMock())
        monkeypatch.setattr(fc, "_FIRECRAWL_CLS_CACHE", None)

        loaded = fc._load_firecrawl_cls()

        assert loaded is fake_module.Firecrawl
        assert fc._FIRECRAWL_CLS_CACHE is loaded
        assert fc.lazy_ensure.call_count == 1
        # The cache is the contract: a second resolve does not re-run the dep check.
        assert fc._load_firecrawl_cls() is loaded
        assert fc.lazy_ensure.call_count == 1

    def test_call_builds_the_loaded_class_with_the_given_kwargs(self, monkeypatch):
        calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(fc, "_FIRECRAWL_CLS_CACHE", _sdk_ctor(calls))

        client = fc.Firecrawl(api_key="fc-key", api_url="https://fc.example")

        assert calls == [{"api_key": "fc-key", "api_url": "https://fc.example"}]
        assert isinstance(client, fc._FIRECRAWL_CLS_CACHE)

    def test_instancecheck_delegates_to_the_loaded_class(self, monkeypatch):
        fake_cls = _sdk_ctor([])
        monkeypatch.setattr(fc, "_FIRECRAWL_CLS_CACHE", fake_cls)

        assert isinstance(fake_cls(), fc.Firecrawl) is True
        assert isinstance(object(), fc.Firecrawl) is False


# ---------------------------------------------------------------------------
# Routing / availability
# ---------------------------------------------------------------------------


class TestUseKeylessRing:
    def test_direct_credentials_keep_the_ring_off(self, monkeypatch):
        from plugins.web import keyless_mcp

        ring = MagicMock(return_value=True)
        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="fc-key"))
        monkeypatch.setattr(keyless_mcp, "use_keyless", ring)

        assert fc._use_keyless_ring() is False
        ring.assert_not_called()

    def test_managed_selection_or_ready_gateway_keep_the_ring_off(self, monkeypatch):
        from plugins.web import keyless_mcp
        from tools import tool_backend_helpers as tbh

        ring = MagicMock(return_value=True)
        monkeypatch.setattr(fc, "_env", _env_from())
        monkeypatch.setattr(keyless_mcp, "use_keyless", ring)

        monkeypatch.setattr(tbh, "read_selection", lambda section: MANAGED)
        assert fc._use_keyless_ring() is False

        monkeypatch.setattr(tbh, "read_selection", lambda section: "some-backend")
        monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: True)
        monkeypatch.setattr(keyless_mcp, "_web_config_selects", lambda name: False)
        assert fc._use_keyless_ring() is False
        ring.assert_not_called()

    def test_ring_is_consulted_and_a_raising_probe_does_not_block_it(self, monkeypatch):
        from plugins.web import keyless_mcp
        from tools import tool_backend_helpers as tbh

        ring = MagicMock(return_value=True)
        monkeypatch.setattr(fc, "_env", _env_from())
        monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: False)
        monkeypatch.setattr(keyless_mcp, "use_keyless", ring)

        monkeypatch.setattr(tbh, "read_selection", lambda section: None)
        assert fc._use_keyless_ring() is True
        assert ring.call_args.args[0] == FirecrawlWebSearchProvider.NAME

        def _explode(section: str) -> str:
            raise RuntimeError("selection store unreadable")

        monkeypatch.setattr(tbh, "read_selection", _explode)
        assert fc._use_keyless_ring() is True


class TestIsToolGatewayReady:
    def test_reports_a_resolved_gateway_without_refreshing_the_token(self, monkeypatch):
        from tools import managed_tool_gateway as gateway

        seen: Dict[str, Any] = {}

        def _resolve(vendor: str, **kwargs: Any):
            seen["vendor"] = vendor
            seen["token_reader"] = kwargs.get("token_reader")
            return types.SimpleNamespace(gateway_origin="https://gw.example", nous_user_token="token")

        monkeypatch.setattr(gateway, "resolve_managed_tool_gateway", _resolve)

        assert fc._is_tool_gateway_ready() is True
        assert seen["vendor"] == FirecrawlWebSearchProvider.NAME
        assert seen["token_reader"] is gateway.peek_nous_access_token

        monkeypatch.setattr(gateway, "resolve_managed_tool_gateway", lambda *a, **kw: None)
        assert fc._is_tool_gateway_ready() is False


class TestCheckFirecrawlApiKey:
    def test_managed_selection_reports_gateway_readiness_without_direct_fallback(self, monkeypatch):
        from tools import tool_backend_helpers as tbh

        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="fc-key"))
        monkeypatch.setattr(tbh, "NOUS_MANAGED_PROVIDER", "managed-marker")
        monkeypatch.setattr(tbh, "read_selection", lambda section: "managed-marker")

        monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: True)
        assert fc.check_firecrawl_api_key() is True

        monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: False)
        assert fc.check_firecrawl_api_key() is False

    def test_unconfigured_install_is_available_with_direct_credentials(self, monkeypatch):
        from tools import tool_backend_helpers as tbh

        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="fc-key"))
        monkeypatch.setattr(tbh, "read_selection", lambda section: None)
        monkeypatch.setattr(fc, "_is_tool_gateway_ready", lambda: False)

        assert fc.check_firecrawl_api_key() is True


class TestBackendHelpSuffix:
    def test_mentions_the_tool_gateway_only_when_managed_tools_are_enabled(self, monkeypatch):
        from tools import tool_backend_helpers as tbh

        monkeypatch.setattr(tbh, "managed_nous_tools_enabled", lambda **kw: True)
        assert "Tool Gateway" in fc._firecrawl_backend_help_suffix()

        monkeypatch.setattr(tbh, "managed_nous_tools_enabled", lambda **kw: False)
        assert fc._firecrawl_backend_help_suffix() == ""


# ---------------------------------------------------------------------------
# Client resolution
# ---------------------------------------------------------------------------


class TestGetFirecrawlClient:
    def test_managed_selection_without_gateway_raises_and_never_uses_direct(
        self, monkeypatch, client_slots
    ):
        from tools import managed_tool_gateway as gateway
        from tools import tool_backend_helpers as tbh

        calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(fc, "Firecrawl", _sdk_ctor(calls))
        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="direct-key"))
        monkeypatch.setattr(tbh, "NOUS_MANAGED_PROVIDER", MANAGED)
        monkeypatch.setattr(tbh, "read_selection", lambda section: MANAGED)
        monkeypatch.setattr(
            tbh, "selection_error", lambda section, name, failure: f"{section}:{name}:{failure}"
        )
        monkeypatch.setattr(gateway, "resolve_managed_tool_gateway", lambda *a, **kw: None)

        with pytest.raises(ValueError, match="Nous Tool Gateway is not available"):
            fc._get_firecrawl_client()

        assert calls == []  # a stored managed selection never degrades to the local key

    def test_direct_selection_builds_the_sdk_client(self, monkeypatch, client_slots):
        from tools import tool_backend_helpers as tbh

        calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(fc, "Firecrawl", _sdk_ctor(calls))
        monkeypatch.setattr(tbh, "read_selection", lambda section: FirecrawlWebSearchProvider.NAME)

        # Real env path (also exercises the ``_env`` read+strip helper behaviorally).
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        client = fc._get_firecrawl_client()
        assert calls == [{"api_key": "fc-key"}]
        assert isinstance(client, fc.Firecrawl)
        assert client_slots._firecrawl_client is client

        monkeypatch.delenv("FIRECRAWL_API_KEY")
        monkeypatch.setenv("FIRECRAWL_API_URL", "https://fc.self/")
        monkeypatch.setattr(client_slots, "_firecrawl_client", None)
        monkeypatch.setattr(client_slots, "_firecrawl_client_config", None)
        fc._get_firecrawl_client()
        assert calls[-1] == {"api_url": "https://fc.self"}

    def test_explicit_selection_without_credentials_uses_the_keyless_client(
        self, monkeypatch, client_slots
    ):
        from plugins.web import keyless_mcp
        from tools import tool_backend_helpers as tbh

        monkeypatch.setattr(fc, "_env", _env_from())
        monkeypatch.setattr(tbh, "read_selection", lambda section: FirecrawlWebSearchProvider.NAME)
        monkeypatch.setattr(keyless_mcp, "_web_config_selects", lambda name: True)

        client = fc._get_firecrawl_client()

        assert isinstance(client, _KeylessFirecrawlClient)
        assert client.api_url == fc._FIRECRAWL_CLOUD_API_URL

    def test_unconfigured_install_prefers_direct_then_gateway(self, monkeypatch, client_slots):
        from tools import managed_tool_gateway as gateway
        from tools import tool_backend_helpers as tbh

        calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(fc, "Firecrawl", _sdk_ctor(calls))
        monkeypatch.setattr(tbh, "read_selection", lambda section: None)

        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="fc-key"))
        assert isinstance(fc._get_firecrawl_client(), fc.Firecrawl)
        assert calls == [{"api_key": "fc-key"}]

        monkeypatch.setattr(fc, "_env", _env_from())
        monkeypatch.setattr(client_slots, "_firecrawl_client", None)
        monkeypatch.setattr(client_slots, "_firecrawl_client_config", None)
        monkeypatch.setattr(gateway, "resolve_managed_tool_gateway", lambda *a, **kw: types.SimpleNamespace(
            gateway_origin="https://gateway.example/firecrawl", nous_user_token="nous-token"
        ))
        fc._get_firecrawl_client()
        assert calls[-1] == {"api_key": "nous-token", "api_url": "https://gateway.example/firecrawl"}

    def test_unconfigured_install_without_any_route_raises_honestly(self, monkeypatch, client_slots):
        from plugins.web import keyless_mcp
        from tools import managed_tool_gateway as gateway
        from tools import tool_backend_helpers as tbh

        monkeypatch.setattr(fc, "_env", _env_from())
        monkeypatch.setattr(tbh, "read_selection", lambda section: None)
        monkeypatch.setattr(keyless_mcp, "_web_config_selects", lambda name: False)
        monkeypatch.setattr(gateway, "resolve_managed_tool_gateway", lambda *a, **kw: None)
        monkeypatch.setattr(tbh, "nous_tool_gateway_unavailable_message", lambda capability: "GATEWAY-UNSET")

        monkeypatch.setattr(tbh, "managed_nous_tools_enabled", lambda **kw: False)
        with pytest.raises(ValueError, match="GATEWAY-UNSET"):
            fc._get_firecrawl_client()

        monkeypatch.setattr(tbh, "managed_nous_tools_enabled", lambda **kw: True)
        with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
            fc._get_firecrawl_client()

    def test_resolved_config_is_cached_by_value(self, monkeypatch, client_slots):
        from tools import tool_backend_helpers as tbh

        calls: List[Dict[str, Any]] = []
        monkeypatch.setattr(fc, "Firecrawl", _sdk_ctor(calls))
        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="key-1"))
        monkeypatch.setattr(tbh, "read_selection", lambda section: FirecrawlWebSearchProvider.NAME)

        first = fc._get_firecrawl_client()
        assert fc._get_firecrawl_client() is first
        assert len(calls) == 1

        monkeypatch.setattr(fc, "_env", _env_from(FIRECRAWL_API_KEY="key-2"))
        second = fc._get_firecrawl_client()

        assert second is not first
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Keyless cloud REST client
# ---------------------------------------------------------------------------


class TestKeylessFirecrawlClient:
    def test_posts_plain_json_without_an_authorization_header(self, monkeypatch):
        posted: List[Dict[str, Any]] = []

        class _Response:
            def raise_for_status(self) -> None:  # pragma: no cover - trivial passthrough
                pass

            def json(self) -> Dict[str, Any]:
                return {"ok": True}

        def _post(url: str, **kwargs: Any):
            posted.append({"url": url, **kwargs})
            return _Response()

        monkeypatch.setattr(fc.httpx, "post", _post)
        client = _KeylessFirecrawlClient(api_url="https://keyless.example/")

        assert client.search(query="q", limit=3) == {"ok": True}
        assert posted[-1]["url"] == "https://keyless.example/v2/search"
        assert posted[-1]["json"] == {"query": "q", "limit": 3}
        assert "Authorization" not in posted[-1]["headers"]

        assert client.scrape(url="https://target.example", formats=["markdown"]) == {"ok": True}
        assert posted[-1]["url"] == "https://keyless.example/v2/scrape"
        assert posted[-1]["json"] == {"url": "https://target.example", "formats": ["markdown"]}


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


class TestToPlainObject:
    def test_sdk_shapes_normalize_and_scalars_pass_through(self):
        dumped = {"markdown": "body"}
        payload = {"title": "A"}
        assert _to_plain_object(payload) is payload
        assert _to_plain_object(_Metadata(dumped)) == dumped
        assert _to_plain_object("text") == "text"

    def test_unreadable_shapes_fall_back_or_pass_through(self):
        assert _to_plain_object(_BrokenDump()) == {"public": "visible"}
        obj = _SlotsObject([{"title": "A"}])
        assert _to_plain_object(obj) is obj


class TestExtractWebSearchResults:
    _ROWS = [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]

    @pytest.mark.parametrize(
        "response",
        [
            {"data": [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]},
            {"data": {"web": [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]}},
            {"data": {"results": [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]}},
            {"web": [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]},
        ],
    )
    def test_response_shapes_normalize_to_the_same_rows(self, response):
        assert _extract_web_search_results(response) == self._ROWS

    def test_object_with_a_web_attribute_and_junk(self):
        assert _extract_web_search_results(_SlotsObject(self._ROWS)) == self._ROWS
        assert _extract_web_search_results("not a response") == []
        assert _extract_web_search_results(None) == []
        assert _extract_web_search_results({"data": {"web": "not-a-list"}}) == []

    def test_non_dict_rows_are_dropped_but_order_is_kept(self):
        rows = _extract_web_search_results({"data": [self._ROWS[0], "junk", self._ROWS[1]]})
        assert rows == self._ROWS


class TestExtractScrapePayload:
    def test_unwrap_and_ignore(self):
        assert _extract_scrape_payload("junk") == {}
        assert _extract_scrape_payload({"data": {"markdown": "body"}}) == {"markdown": "body"}
        payload = {"data": "junk", "markdown": "body"}
        assert _extract_scrape_payload(payload) == payload
        assert _extract_scrape_payload(_Metadata({"markdown": "body"})) == {"markdown": "body"}


class TestErrorEntry:
    def test_raw_and_policy_fields(self):
        blocked = {"host": "x.example", "rule": "x.example", "source": "config.yaml", "message": "nope"}
        assert _error_entry("https://x.example", "boom") == {
            "url": "https://x.example",
            "title": "",
            "content": "",
            "error": "boom",
        }
        assert _error_entry("https://x.example", "boom", raw=True)["raw_content"] == ""
        entry = _error_entry("https://x.example", blocked["message"], blocked=blocked)
        assert entry["blocked_by_policy"] == {"host": "x.example", "rule": "x.example", "source": "config.yaml"}


# ---------------------------------------------------------------------------
# _scrape_one
# ---------------------------------------------------------------------------


class TestScrapeOne:
    _URL = "https://target.example/page"
    _PAYLOAD = {"data": {"markdown": "body", "html": "<p>body</p>", "metadata": {"title": "Page"}}}

    @pytest.fixture(autouse=True)
    def _allow_everything(self, monkeypatch):
        monkeypatch.setattr(fc, "check_website_access", lambda url: None)
        monkeypatch.setattr(fc, "is_safe_url", lambda url: True)

    @staticmethod
    def _use_client(monkeypatch, client: _FakeScrapeClient) -> _FakeScrapeClient:
        monkeypatch.setattr(fc, "_get_firecrawl_client", lambda: client)
        return client

    @pytest.mark.asyncio
    async def test_policy_blocked_url_returns_an_entry_without_scraping(self, monkeypatch):
        blocked = {"host": "target.example", "rule": "target.example", "source": "config.yaml", "message": "Blocked by website policy"}
        monkeypatch.setattr(fc, "check_website_access", lambda url: blocked)
        client = self._use_client(monkeypatch, _FakeScrapeClient(result=self._PAYLOAD))

        entry = await _scrape_one(self._URL, ["markdown", "html"], None)

        assert client.scrape_calls == []
        assert entry["error"] == blocked["message"]
        assert entry["blocked_by_policy"] == {"host": "target.example", "rule": "target.example", "source": "config.yaml"}
        assert "raw_content" not in entry

    @pytest.mark.asyncio
    async def test_timeout_returns_the_60s_message(self, monkeypatch):
        client = self._use_client(monkeypatch, _FakeScrapeClient(result=self._PAYLOAD))

        async def _timeout(awaitable, timeout=None):
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", _timeout)

        entry = await _scrape_one(self._URL, ["markdown", "html"], None)

        assert "60s" in entry["error"]
        assert client.scrape_calls == []  # the thread body never ran

    @pytest.mark.asyncio
    async def test_unsafe_redirect_target_is_blocked_after_the_scrape(self, monkeypatch):
        final_url = "http://10.0.0.1/secret"
        payload = {"data": {"markdown": "body", "metadata": {"title": "Page", "sourceURL": final_url}}}
        self._use_client(monkeypatch, _FakeScrapeClient(result=payload))
        checked: List[str] = []

        def _looks_unsafe(url: str) -> bool:
            checked.append(url)
            return False

        monkeypatch.setattr(fc, "is_safe_url", _looks_unsafe)

        entry = await _scrape_one(self._URL, ["markdown", "html"], None)

        assert checked == [final_url]
        assert entry["url"] == final_url
        assert entry["title"] == "Page"
        assert entry["error"] == fc._UNSAFE_REDIRECT_MSG
        assert entry["raw_content"] == ""

    @pytest.mark.asyncio
    async def test_policy_blocked_redirect_keeps_the_scraped_title(self, monkeypatch):
        final_url = "https://blocked.example/page"
        payload = {"data": {"markdown": "body", "metadata": _Metadata({"title": "Page", "sourceURL": final_url})}}
        self._use_client(monkeypatch, _FakeScrapeClient(result=payload))
        blocked = {"host": "blocked.example", "rule": "blocked.example", "source": "defaults", "message": "Blocked"}
        monkeypatch.setattr(fc, "check_website_access", lambda url: blocked if url == final_url else None)

        entry = await _scrape_one(self._URL, ["markdown", "html"], None)

        assert entry["title"] == "Page"
        assert entry["error"] == "Blocked"
        assert entry["raw_content"] == ""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "requested,payload,expected",
        [
            ("markdown", {"markdown": "md-body", "html": "<p>html-body</p>"}, "md-body"),
            ("html", {"markdown": "md-body", "html": "<p>html-body</p>"}, "<p>html-body</p>"),
            (None, {"markdown": "md-body", "html": "<p>html-body</p>"}, "md-body"),
        ],
    )
    async def test_content_follows_the_requested_format(self, monkeypatch, requested, payload, expected):
        self._use_client(monkeypatch, _FakeScrapeClient(result={"data": {"metadata": {"title": "Page"}, **payload}}))

        entry = await _scrape_one(self._URL, ["markdown", "html"], requested)

        assert entry["content"] == expected
        assert entry["raw_content"] == expected

    @pytest.mark.asyncio
    async def test_scrape_errors_become_entries_instead_of_raising(self, monkeypatch):
        self._use_client(monkeypatch, _FakeScrapeClient(error=RuntimeError("firecrawl exploded")))

        entry = await _scrape_one(self._URL, ["markdown", "html"], None)

        assert "firecrawl exploded" in entry["error"]
        assert entry["url"] == self._URL
        assert entry["raw_content"] == ""


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------


class TestSearch:
    def test_interrupt_short_circuits_before_routing(self, monkeypatch):
        import tools.interrupt as interrupt

        monkeypatch.setattr(interrupt, "is_interrupted", lambda: True)
        monkeypatch.setattr(fc, "_use_keyless_ring", MagicMock(side_effect=AssertionError("routing consulted")))

        result = FirecrawlWebSearchProvider().search("query")

        assert result == {"success": False, "error": "Interrupted"}

    def test_keyless_hand_off_forwards_the_query(self, monkeypatch):
        import tools.interrupt as interrupt

        seen: List[Any] = []

        def _keyless_search(display: str, name: str, query: str, limit: int, logger: Any):
            seen.append((display, name, query, limit))
            return {"success": True, "data": {"web": []}}

        monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)
        monkeypatch.setattr(fc, "_use_keyless_ring", lambda: True)
        monkeypatch.setattr(fc, "keyless_search", _keyless_search)

        assert FirecrawlWebSearchProvider().search("query", limit=7) == {"success": True, "data": {"web": []}}
        assert seen == [(seen[0][0], seen[0][1], "query", 7)]

    def test_direct_path_normalizes_results_and_reports_failures(self, monkeypatch):
        import tools.interrupt as interrupt

        raw_rows = [{"title": "A", "url": "https://a.example"}, {"title": "B", "url": "https://b.example"}]
        client = MagicMock()
        client.search.return_value = {"data": raw_rows}
        monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)
        monkeypatch.setattr(fc, "_use_keyless_ring", lambda: False)
        monkeypatch.setattr(fc, "_get_firecrawl_client", lambda: client)

        result = FirecrawlWebSearchProvider().search("query", limit=2)

        assert client.search.call_args.kwargs == {"query": "query", "limit": 2}
        assert result["success"] is True
        assert [row["title"] for row in result["data"]["web"]] == ["A", "B"]

        client.search.side_effect = ValueError("HTTP 500 from Firecrawl")
        result = FirecrawlWebSearchProvider().search("query")
        assert result["success"] is False
        assert "HTTP 500 from Firecrawl" in result["error"]


class TestExtract:
    @pytest.fixture(autouse=True)
    def _scrape_recorder(self, monkeypatch):
        self.scraped: List[Dict[str, Any]] = []

        async def _scrape(url: str, formats: List[str], format: Any) -> Dict[str, Any]:
            self.scraped.append({"url": url, "formats": formats, "format": format})
            return {"url": url, "title": "", "content": url, "raw_content": url}

        monkeypatch.setattr(fc, "_scrape_one", _scrape)
        return self.scraped

    @pytest.mark.asyncio
    async def test_interrupted_input_yields_one_item_per_url(self, monkeypatch):
        import tools.interrupt as interrupt

        monkeypatch.setattr(interrupt, "is_interrupted", lambda: True)
        monkeypatch.setattr(fc, "_use_keyless_ring", lambda: True)

        result = await FirecrawlWebSearchProvider().extract(["https://a.example", "https://b.example"])

        assert [item["url"] for item in result] == ["https://a.example", "https://b.example"]
        assert {item["error"] for item in result} == {"Interrupted"}
        assert self.scraped == []

    @pytest.mark.asyncio
    async def test_keyless_hand_off_and_format_selection(self, monkeypatch):
        import tools.interrupt as interrupt

        seen: List[List[str]] = []

        def _keyless_extract(display: str, name: str, urls: List[str], logger: Any):
            seen.append(list(urls))
            return [{"url": url, "title": "", "content": "keyless", "raw_content": "keyless"} for url in urls]

        monkeypatch.setattr(interrupt, "is_interrupted", lambda: False)
        monkeypatch.setattr(fc, "_use_keyless_ring", lambda: True)
        monkeypatch.setattr(fc, "keyless_extract", _keyless_extract)

        result = await FirecrawlWebSearchProvider().extract(["https://a.example"])

        assert seen == [["https://a.example"]]
        assert result[0]["url"] == "https://a.example"
        assert self.scraped == []

        monkeypatch.setattr(fc, "_use_keyless_ring", lambda: False)
        await FirecrawlWebSearchProvider().extract(["https://a.example"], format="markdown")
        await FirecrawlWebSearchProvider().extract(["https://a.example"])
        assert [call["formats"] for call in self.scraped] == [["markdown"], ["markdown", "html"]]


class TestProviderAvailability:
    def test_is_available_mirrors_the_key_check(self, monkeypatch):
        monkeypatch.setattr(fc, "check_firecrawl_api_key", MagicMock(return_value=True))
        assert FirecrawlWebSearchProvider().is_available() is True

        fc.check_firecrawl_api_key.return_value = False
        assert FirecrawlWebSearchProvider().is_available() is False


class TestSetupSchema:
    def test_picker_entry_names_the_provider_and_its_env_var(self):
        schema = FirecrawlWebSearchProvider().get_setup_schema()

        assert schema["name"] == FirecrawlWebSearchProvider.DISPLAY_NAME
        assert [entry["key"] for entry in schema["env_vars"]] == ["FIRECRAWL_API_KEY"]
        assert schema["env_vars"][0]["prompt"]
        assert schema["env_vars"][0]["url"].startswith("https://")
