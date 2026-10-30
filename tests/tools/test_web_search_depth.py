"""``web_search(depth=...)`` across every bundled web provider.

Each test drives the real ``web_search_tool`` dispatcher with the backend pinned to one provider and
records the request that provider puts on the wire (HTTP call, SDK call or subprocess hand-off), so
the assertions see exactly what the vendor would receive.
"""
from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.tools.conftest import register_all_web_providers

ALL_PROVIDERS = (
    "brave-free", "ddgs", "exa", "firecrawl", "keenable", "openai-native", "parallel", "perplexity",
    "searxng", "tavily", "xai",
)


def _http_ok(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    resp.raise_for_status.return_value = None
    return resp


class _Wire:
    """Every request a provider sent, in order, as ``(kind, target, body, headers)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any, Any]] = []

    def last(self) -> tuple[str, Any, Any, Any]:
        assert self.calls, "provider sent no request"
        return self.calls[-1]


def _hit() -> dict:
    return {"title": "t", "url": "https://example.com", "description": "d", "content": "c", "snippet": "s"}


def _http(wire: _Wire, kind: str, payload: dict):
    """A recording stand-in for ``httpx.get`` / ``httpx.post`` / ``requests.post``."""
    def _send(url, *args, **kwargs):
        wire.calls.append((kind, url, kwargs.get("json", kwargs.get("params")), kwargs.get("headers")))
        return _http_ok(payload)
    return _send


def _wire_brave(wire, mp):
    mp.setenv("BRAVE_SEARCH_API_KEY", "brave-key")
    mp.setattr("plugins.web._common.httpx.get", _http(wire, "GET", {"web": {"results": [_hit()]}}))


def _wire_searxng(wire, mp):
    mp.setenv("SEARXNG_URL", "http://searx.test")
    mp.setattr("plugins.web._common.httpx.get", _http(wire, "GET", {"results": [_hit()]}))


def _wire_ddgs(wire, mp):
    from plugins.web._common import title_hit

    def _bounded(query, limit):
        wire.calls.append(("ddgs", None, (query, limit), None))
        return [title_hit("t", "https://example.com", "d", 1)]
    mp.setattr("plugins.web.ddgs.provider.DDGSWebSearchProvider.is_available", lambda self: True)
    mp.setattr("plugins.web.ddgs.provider._run_ddgs_search_bounded", _bounded)


def _wire_firecrawl(wire, mp):
    class _Firecrawl:
        def search(self, **kwargs):
            wire.calls.append(("sdk", "search", kwargs, None))
            return {"data": {"web": [_hit()]}}
    mp.setenv("FIRECRAWL_API_KEY", "fc-key")
    mp.setattr("plugins.web.firecrawl.provider._get_firecrawl_client", lambda *a: _Firecrawl())


def _wire_exa(wire, mp):
    from tools import web_tools

    class _Exa:
        def search(self, *args, **kwargs):
            wire.calls.append(("sdk", args, kwargs, None))
            hit = types.SimpleNamespace(url="https://example.com", title="t", highlights=["h"])
            return types.SimpleNamespace(results=[hit])
    mp.setenv("EXA_API_KEY", "exa-key")
    mp.setattr(web_tools, "_exa_client", ("exa-key", _Exa()), raising=False)


def _wire_parallel(wire, mp):
    from tools import web_tools

    def _search(**kwargs):
        wire.calls.append(("sdk", "beta.search", kwargs, None))
        hit = types.SimpleNamespace(url="https://example.com", title="t", excerpts=["e"])
        return types.SimpleNamespace(results=[hit])
    mp.setenv("PARALLEL_API_KEY", "par-key")
    client = types.SimpleNamespace(beta=types.SimpleNamespace(search=_search))
    mp.setattr(web_tools, "_parallel_client", ("par-key", client), raising=False)


def _wire_keenable(wire, mp):
    mp.setenv("KEENABLE_API_KEY", "keen-key")
    mp.setattr("requests.post", _http(wire, "POST", {"results": [_hit()]}))


def _wire_tavily(wire, mp):
    mp.setenv("TAVILY_API_KEY", "tvly-key")
    mp.setattr("plugins.web.tavily.provider.httpx.post", _http(wire, "POST", {"results": [_hit()]}))


def _wire_perplexity(wire, mp):
    mp.setenv("PERPLEXITY_API_KEY", "pplx-key")
    mp.setattr("plugins.web.perplexity.provider.httpx.post", _http(wire, "POST", {"results": [_hit()]}))


def _wire_xai(wire, mp):
    creds = {"api_key": "xai-key", "base_url": "https://api.x.ai/v1", "provider": "xai"}
    text = json.dumps({"results": [{"title": "t", "url": "https://example.com", "description": "d"}]})
    reply = {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
    mp.setattr("plugins.web.xai.provider.resolve_xai_http_credentials", lambda **kw: creds)
    mp.setattr("plugins.web.xai.provider._load_xai_web_config", dict)
    mp.setattr("httpx.post", _http(wire, "POST", reply))


_WIRES = {
    "brave-free": _wire_brave, "searxng": _wire_searxng, "ddgs": _wire_ddgs, "firecrawl": _wire_firecrawl,
    "exa": _wire_exa, "parallel": _wire_parallel, "keenable": _wire_keenable, "tavily": _wire_tavily,
    "perplexity": _wire_perplexity, "xai": _wire_xai,
    "openai-native": lambda wire, mp: None,  # never sends a client-side request
}


def _setup_wire(name: str, monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Keyed credentials plus a recording transport for provider *name*."""
    wire = _Wire()
    _WIRES[name](wire, monkeypatch)
    return wire


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch):
    """Dispatcher pinned to one provider; ``web.config`` is the ``web:`` config section."""
    from agent.web_search_registry import register_provider
    from plugins.web.openai_native.provider import OpenAINativeWebSearchProvider
    from tools import web_tools

    for var in ("BRAVE_SEARCH_API_KEY", "SEARXNG_URL", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "EXA_API_KEY",
                "PARALLEL_API_KEY", "PARALLEL_SEARCH_MODE", "KEENABLE_API_KEY", "TAVILY_API_KEY", "TAVILY_BASE_URL",
                "PERPLEXITY_API_KEY", "PERPLEXITY_BASE_URL", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    register_all_web_providers()
    register_provider(OpenAINativeWebSearchProvider())
    state = types.SimpleNamespace(config={}, backend="")
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_load_web_config", lambda: state.config)
    monkeypatch.setattr(web_tools, "_get_search_backend", lambda: state.backend)
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda provider: False)
    monkeypatch.setattr("plugins.web.keyless_mcp.keyless_enabled", lambda: False)
    monkeypatch.setattr("plugins.web.firecrawl.provider._use_keyless_ring", lambda *a: False)
    return state


def _run(web, name: str, monkeypatch, **kwargs) -> tuple[dict, _Wire]:
    from tools import web_tools

    web.backend = name
    wire = _setup_wire(name, monkeypatch)
    out = json.loads(web_tools.web_search_tool("q", limit=3, **kwargs))
    return out, wire


# Today's request for "q" (limit 3 -> the dispatcher's fetch bucket of 10), as each vendor receives it.
_TODAY = {
    "brave-free": ("GET", "https://api.search.brave.com/res/v1/web/search", {"q": "q", "count": 10},
                   {"X-Subscription-Token": "brave-key", "Accept": "application/json"}),
    "searxng": ("GET", "http://searx.test/search", {"q": "q", "format": "json", "pageno": 1},
                {"Accept": "application/json"}),
    "ddgs": ("ddgs", None, ("q", 10), None),
    "firecrawl": ("sdk", "search", {"query": "q", "limit": 10}, None),
    "exa": ("sdk", ("q",), {"num_results": 10, "contents": {"highlights": True}}, None),
    "parallel": ("sdk", "beta.search", {"search_queries": ["q"], "objective": "q", "mode": "agentic", "max_results": 10}, None),
    "keenable": ("POST", "https://api.keenable.ai/v1/search", {"query": "q", "max_results": 10},
                 {"X-Keenable-Title": "hermes-agent", "Authorization": "Bearer keen-key"}),
    "tavily": ("POST", "https://api.tavily.com/search",
               {"query": "q", "max_results": 10, "include_raw_content": False, "include_images": False},
               {"X-Client-Name": "hermes-agent", "Authorization": "Bearer tvly-key"}),
    "perplexity": ("POST", "https://api.perplexity.ai/search", {"query": "q", "max_results": 10, "search_context_size": "low"}, None),
}


def _canonical(call: tuple) -> str:
    """Order-sensitive serialisation: key order reaches the vendor as JSON, so it is compared too."""
    return json.dumps(call, default=repr)


def _request(wire: _Wire, name: str) -> tuple:
    kind, target, body, headers = wire.last()
    if name == "perplexity":
        headers = None  # identity headers are asserted in test_web_tools_perplexity.py
    return kind, target, body, headers


def _assert_today(out: dict, wire: _Wire, name: str) -> None:
    """*name* sent exactly today's request (openai-native: no request, today's error)."""
    if name == "openai-native":
        assert out["success"] is False and "Codex Responses transport" in out["error"] and not wire.calls
        return
    assert out["success"] is True, out
    assert len(wire.calls) == 1
    kind, target, body, headers = _request(wire, name)
    if name == "xai":
        assert target == "https://api.x.ai/v1/responses"
        assert list(body) == ["model", "input", "tools", "include"]
        assert body["tools"] == [{"type": "web_search"}] and "at most 10 results" in body["input"][0]["content"]
        return
    assert _canonical((kind, target, body, headers)) == _canonical(_TODAY[name])


@pytest.mark.parametrize("depth", [None, "standard"])
@pytest.mark.parametrize("name", ALL_PROVIDERS)
def test_omitted_or_standard_depth_sends_todays_request(web, monkeypatch, name, depth):
    web.config = {"max_depth": "deep"}
    out, wire = _run(web, name, monkeypatch, **({} if depth is None else {"depth": depth}))
    _assert_today(out, wire, name)
    assert "depth_note" not in out.get("data", {})


def _with(name: str, **extra) -> str:
    kind, target, body, headers = _TODAY[name]
    return _canonical((kind, target, {**body, **extra}, headers))


# What quick / deep send for providers with a native depth option (None = today's request).
_MAPPED = {
    "exa": {"quick": {"type": "lite"}, "deep": {"type": "deep"}},
    "tavily": {"quick": {"search_depth": "fast"}, "deep": {"search_depth": "advanced"}},
    "parallel": {"quick": {"mode": "fast"}, "deep": {"mode": "agentic"}},
    "keenable": {"quick": {"mode": "realtime"}, "deep": None},
    "perplexity": {"quick": {"search_type": "fast"}, "deep": None},
}
_NO_DEPTH = tuple(n for n in ALL_PROVIDERS if n not in _MAPPED)


@pytest.mark.parametrize("depth", ["quick", "deep"])
@pytest.mark.parametrize("name", sorted(_MAPPED))
def test_mapped_provider_sends_its_native_option(web, monkeypatch, name, depth):
    web.config = {"max_depth": "deep"}
    out, wire = _run(web, name, monkeypatch, depth=depth)
    assert out["success"] is True, out
    extra = _MAPPED[name][depth]
    assert _canonical(_request(wire, name)) == (_with(name, **extra) if extra else _canonical(_TODAY[name]))
    assert "depth_note" not in out["data"]


def test_parallel_deep_overrides_a_configured_mode(web, monkeypatch):
    web.config = {"max_depth": "deep"}
    monkeypatch.setenv("PARALLEL_SEARCH_MODE", "one-shot")
    _, wire = _run(web, "parallel", monkeypatch)
    assert wire.last()[2]["mode"] == "one-shot"
    _, wire = _run(web, "parallel", monkeypatch, depth="deep")
    assert wire.last()[2]["mode"] == "agentic"


@pytest.mark.parametrize("depth", ["quick", "deep"])
@pytest.mark.parametrize("name", _NO_DEPTH)
def test_provider_without_depth_ignores_it_and_says_so(web, monkeypatch, name, depth):
    from tools.web_result_cache import search_memo

    web.config = {"max_depth": "deep"}
    out, wire = _run(web, name, monkeypatch, depth=depth)
    _assert_today(out, wire, name)
    if name != "openai-native":
        assert out["data"]["depth_note"] == f"{name} search has no depth setting; ran its standard search"
    search_memo.clear()


_KEY_ENV = {"exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY", "keenable": "KEENABLE_API_KEY", "firecrawl": "FIRECRAWL_API_KEY"}


@pytest.mark.parametrize("name", sorted(_KEY_ENV))
def test_keyless_route_ignores_depth(web, monkeypatch, name):
    """The keyless ring has no depth option: the ring call is today's, and the note says so."""
    from tools import web_tools

    calls = []

    def _ring(vendor, query, limit):
        calls.append((vendor, query, limit))
        return {"success": True, "data": {"web": [{"title": "t", "url": "https://example.com", "description": "d", "position": 1}]}}

    web.config, web.backend = {"max_depth": "deep"}, name
    wire = _setup_wire(name, monkeypatch)
    monkeypatch.delenv(_KEY_ENV[name])
    monkeypatch.setattr("plugins.web.keyless_mcp.keyless_enabled", lambda: True)
    monkeypatch.setattr("plugins.web.firecrawl.provider._use_keyless_ring", lambda *a: True)
    monkeypatch.setattr("plugins.web.keyless_mcp.search_with_failover", _ring)
    out = json.loads(web_tools.web_search_tool("q", limit=3, depth="deep"))
    assert calls == [(name, "q", 10)] and not wire.calls
    assert out["data"]["depth_note"] == f"{name} keyless search has no depth setting; ran its standard search"


def test_managed_perplexity_ignores_depth(web, monkeypatch):
    """Without a direct key Perplexity serves through the managed route, which is always fast."""
    import types as _t

    web.config, web.backend = {"max_depth": "deep"}, "perplexity"
    wire = _setup_wire("perplexity", monkeypatch)
    monkeypatch.delenv("PERPLEXITY_API_KEY")
    gateway = _t.SimpleNamespace(gateway_origin="https://gw.test", nous_user_token="nous-token")
    monkeypatch.setattr("tools.web_tools._managed_web_search", lambda: True)
    monkeypatch.setattr("tools.managed_tool_gateway.resolve_free_search_gateway", lambda **kw: gateway)
    from tools import web_tools

    out = json.loads(web_tools.web_search_tool("q", limit=3, depth="deep"))
    assert wire.last()[1] == "https://gw.test/search" and wire.last()[2]["search_type"] == "fast"
    assert out["data"]["depth_note"] == "perplexity managed search has no depth setting; ran its standard search"


def test_rescued_call_says_depth_was_not_applied(web, monkeypatch):
    """A keyed depth call that fails and is served by the keyless rescue ran without depth."""
    from tools import web_tools

    rescued = {"success": True, "data": {"web": [], "rescued_from": "tavily", "backend_error": "boom"}}
    web.config, web.backend = {"max_depth": "deep"}, "tavily"
    _setup_wire("tavily", monkeypatch)
    failing = MagicMock(status_code=500, text="boom")
    monkeypatch.setattr("plugins.web.tavily.provider.httpx.post", lambda *a, **kw: failing)
    monkeypatch.setattr(web_tools, "_rescue_eligible", lambda provider: True)
    monkeypatch.setattr(web_tools, "_rescue_search", lambda *a: json.loads(json.dumps(rescued)))
    out = json.loads(web_tools.web_search_tool("q", limit=3, depth="deep"))
    assert out["data"]["depth_note"] == "keyless rescue has no depth setting; ran its standard search"
    out = json.loads(web_tools.web_search_tool("q", limit=3))
    assert "depth_note" not in out["data"]


def test_tavily_keyless_still_sends_depth(web, monkeypatch):
    """Tavily's keyless mode is the same /search endpoint, which takes search_depth."""
    web.config = {"max_depth": "deep"}
    monkeypatch.setattr("plugins.web.tavily.provider.use_keyless", lambda name, key: True)
    out, wire = _run(web, "tavily", monkeypatch, depth="deep")
    assert out["success"] is True
    _, _, body, headers = wire.last()
    assert headers.get("X-Tavily-Access-Mode") == "keyless" and body["search_depth"] == "advanced"


# ── cost cap ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("config", [{}, {"max_depth": "standard"}, {"max_depth": "bogus"}])
def test_cap_lowers_deep_to_standard_by_default(web, monkeypatch, config):
    web.config = config
    out, wire = _run(web, "tavily", monkeypatch, depth="deep")
    assert _canonical(_request(wire, "tavily")) == _canonical(_TODAY["tavily"])
    assert out["data"]["depth_note"] == "depth lowered from deep to standard by web.max_depth"


def test_cap_allows_quick_by_default(web, monkeypatch):
    out, wire = _run(web, "tavily", monkeypatch, depth="quick")
    assert _canonical(_request(wire, "tavily")) == _with("tavily", search_depth="fast")
    assert "depth_note" not in out["data"]


def test_quick_cap_lowers_an_omitted_depth_and_says_so(web, monkeypatch):
    web.config = {"max_depth": "quick"}
    out, wire = _run(web, "tavily", monkeypatch)
    assert _canonical(_request(wire, "tavily")) == _with("tavily", search_depth="fast")
    assert out["data"]["depth_note"] == "depth lowered from standard to quick by web.max_depth"
    out, _ = _run(web, "exa", monkeypatch, depth="deep")
    assert out["data"]["depth_note"] == "depth lowered from deep to quick by web.max_depth"


def test_unknown_depth_runs_standard(web, monkeypatch):
    web.config = {"max_depth": "deep"}
    out, wire = _run(web, "tavily", monkeypatch, depth="extreme")
    assert _canonical(_request(wire, "tavily")) == _canonical(_TODAY["tavily"])
    assert out["data"]["depth_note"] == "unknown depth 'extreme'; ran a standard search"


def test_max_depth_is_read_from_config_yaml(web, monkeypatch, tmp_path):
    """The cap reaches the dispatcher through the real config loader."""
    from hermes_cli.config import atomic_config_write, load_config
    from tools import web_tools

    monkeypatch.setattr(web_tools, "_load_web_config", lambda: load_config().get("web") or {})
    for home, web_cfg in (("capped", {}), ("open", {"max_depth": "deep"})):
        (tmp_path / home).mkdir()
        atomic_config_write(tmp_path / home / "config.yaml", {"web": web_cfg})

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "open"))
    out, wire = _run(web, "tavily", monkeypatch, depth="deep")
    assert wire.last()[2]["search_depth"] == "advanced" and "depth_note" not in out["data"]

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "capped"))  # default cap: standard
    out, wire = _run(web, "tavily", monkeypatch, depth="deep")
    assert "search_depth" not in wire.last()[2]
    assert out["data"]["depth_note"] == "depth lowered from deep to standard by web.max_depth"


# ── result cache ──────────────────────────────────────────────────────────

def test_cache_keeps_depths_apart(web, monkeypatch):
    web.config = {"max_depth": "deep"}
    web.backend = "tavily"
    wire = _setup_wire("tavily", monkeypatch)
    from tools import web_tools

    for depth in ("standard", "deep", "quick", "deep", "standard", "quick"):
        web_tools.web_search_tool("q", limit=3, depth=depth)
    sent = [body.get("search_depth") for _, _, body, _ in wire.calls]
    assert sent == [None, "advanced", "fast"]  # one paid call per depth; repeats are cache hits


def test_cache_shares_one_entry_when_the_provider_has_no_depth(web, monkeypatch):
    web.config = {"max_depth": "deep"}
    web.backend = "brave-free"
    wire = _setup_wire("brave-free", monkeypatch)
    from tools import web_tools

    first = json.loads(web_tools.web_search_tool("q", limit=3))
    second = json.loads(web_tools.web_search_tool("q", limit=3, depth="deep"))
    assert len(wire.calls) == 1  # same request either way, so one cache entry
    assert "depth_note" not in first["data"] and second["data"]["depth_note"]
    third = json.loads(web_tools.web_search_tool("q", limit=3))
    assert "depth_note" not in third["data"]  # the note is never cached


# ── third-party providers ─────────────────────────────────────────────────

def test_provider_without_the_hook_is_called_as_before(web, monkeypatch):
    """A plugin provider written before depth existed (``search(query, limit)`` only) still works."""
    from agent.web_search_provider import WebSearchProvider
    from agent.web_search_registry import register_provider
    from tools import web_tools

    calls = []

    class _Legacy(WebSearchProvider):
        name = "legacy-plugin"

        def is_available(self):
            return True

        def search(self, query, limit=5):
            calls.append((query, limit))
            return {"success": True, "data": {"web": []}}

    register_provider(_Legacy())
    web.backend, web.config = "legacy-plugin", {"max_depth": "deep"}
    out = json.loads(web_tools.web_search_tool("q", limit=3, depth="deep"))
    assert calls == [("q", 10)] and out["success"] is True
    assert out["data"]["depth_note"] == "legacy-plugin search has no depth setting; ran its standard search"


# ── tool schema ───────────────────────────────────────────────────────────

def test_schema_declares_depth_and_validates():
    from jsonschema.validators import Draft202012Validator

    from agent.web_search_provider import SEARCH_DEPTHS
    from tools.web_tools import WEB_SEARCH_SCHEMA

    params = WEB_SEARCH_SCHEMA["parameters"]
    Draft202012Validator.check_schema(params)
    assert params["properties"]["depth"]["enum"] == list(SEARCH_DEPTHS)
    assert params["required"] == ["query"]
    valid = Draft202012Validator(params).is_valid
    assert valid({"query": "q"}) and valid({"query": "q", "depth": "deep"})
    assert not valid({"query": "q", "depth": "max"})


def test_schema_survives_provider_sanitizers():
    from agent.gemini_schema import sanitize_gemini_tool_parameters
    from agent.moonshot_schema import sanitize_moonshot_tool_parameters
    from tools.schema_sanitizer import sanitize_tool_schemas
    from tools.web_tools import WEB_SEARCH_SCHEMA

    params = WEB_SEARCH_SCHEMA["parameters"]
    for cleaned in (sanitize_gemini_tool_parameters(params), sanitize_moonshot_tool_parameters(params)):
        assert cleaned["properties"]["depth"]["enum"] == ["quick", "standard", "deep"]
    tool = sanitize_tool_schemas([{"type": "function", "function": WEB_SEARCH_SCHEMA}])[0]
    assert tool["function"]["parameters"]["properties"]["depth"]["enum"] == ["quick", "standard", "deep"]


def test_registry_handler_passes_depth(web, monkeypatch):
    from tools.registry import registry

    web.config = {"max_depth": "deep"}
    web.backend = "tavily"
    wire = _setup_wire("tavily", monkeypatch)
    out = json.loads(registry.dispatch("web_search", {"query": "q", "limit": 3, "depth": "quick"}))
    assert out["success"] is True and wire.last()[2]["search_depth"] == "fast"
