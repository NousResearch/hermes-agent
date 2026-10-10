"""Programmatic acquisition and model tools use the same native search pipeline."""

import json

import pytest

from agent.web_acquisition import WebSearchRequest
from agent.web_acquisition_errors import WebCancelledError, WebServiceUnavailableError
from agent.web_search_provider import WebSearchProvider
from tools.web_acquisition import web_search


class FixtureProvider(WebSearchProvider):
    name = "fixture"
    display_name = "Fixture"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def configured_search(tmp_path, monkeypatch):
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from tools import web_result_cache, web_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "fixture", "keyless_rescue": False}},
    )
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    from tools.interrupt import set_interrupt

    set_interrupt(False)
    web_search_registry._registry.reset_for_tests()
    web_result_cache.search_memo.clear()

    def configure(response):
        provider = FixtureProvider(response)
        web_search_registry._registry.register(provider)
        return provider

    yield configure
    set_interrupt(False)
    web_search_registry._registry.reset_for_tests()
    web_result_cache.search_memo.clear()


@pytest.mark.parametrize(
    "mode", ["cache", "legacy", "malformed", "cancelled", "rescue", "failed-rescue"]
)
def test_shared_search_preserves_safe_results_and_actual_attempts(
    configured_search, monkeypatch, mode
):
    from plugins.web import keyless_mcp
    from tools import web_tools

    hit = {"url": "https://example.test/news", "title": "News", "description": "Today"}
    response = {
        "success": True,
        "data": {"web": [hit], "backend_error": "Bearer synthetic-secret"},
    }
    failure = None
    if mode == "legacy":
        response = {"success": False, "error": "HTTP 429 Bearer synthetic-secret"}
        failure = {"kind": "unclassified", "retry": "unknown", "scope": "provider"}
    if mode == "malformed":
        response = {"success": True, "data": {"web": [{**hit, "description": 42}]}}
        failure = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
    if mode == "cancelled":
        response = WebCancelledError()
        failure = response.to_failure()
    if mode in {"rescue", "failed-rescue"}:
        response = WebServiceUnavailableError(
            503, cause=RuntimeError("Bearer synthetic-secret")
        )
        failure = response.to_failure()
    provider = configured_search(response)
    ring_calls = []

    def rescue(name, query, limit):
        ring_calls.append((name, query, limit))
        if mode == "failed-rescue":
            return {
                "success": False,
                "error": "Bearer synthetic-secret",
                "failure": {
                    "kind": "authentication",
                    "retry": "never",
                    "scope": "provider",
                    "status": 401,
                },
            }
        return {"success": True, "data": {"web": [hit], "served_by": "exa"}}

    if mode in {"rescue", "failed-rescue", "cancelled"}:
        monkeypatch.setattr(
            "tools.web_tools_rescue._rescue_eligible",
            lambda provider, capability=None: True,
        )
        monkeypatch.setattr(keyless_mcp, "search_with_failover", rescue)

    request = WebSearchRequest(query="today", limit=2)
    result = web_search(request).model_dump(mode="json", exclude_none=True)
    attempts = []
    primary = {
        "provider": "fixture",
        "route": "selected",
        "status": "failed" if failure else "succeeded",
    }
    if failure:
        primary["failure"] = failure
    attempts.append(primary)
    if mode in {"rescue", "failed-rescue"}:
        alternative = {
            "route": "keyless-rescue",
            "status": "failed" if mode == "failed-rescue" else "succeeded",
        }
        if mode == "rescue":
            alternative["provider"] = "exa"
        else:
            alternative["failure"] = {
                "kind": "authentication",
                "retry": "never",
                "scope": "provider",
                "status": 401,
            }
        attempts.append(alternative)
    if failure and mode != "rescue":
        expected = {
            "status": "failed",
            "selected_provider": "fixture",
            "failure": failure,
            "attempts": attempts,
        }
    else:
        expected = {
            "status": "ok",
            "selected_provider": "fixture",
            "served_provider": "exa" if mode == "rescue" else "fixture",
            "cache": "miss",
            "hits": [hit],
            "attempts": attempts,
        }
    assert result == expected
    assert "synthetic-secret" not in json.dumps(result)
    tool_result = json.loads(web_tools.web_search_tool("today", limit=2))
    assert tool_result.get("success", False) == (mode in {"cache", "rescue"})
    if mode == "cache":
        cached = web_search(request).model_dump(mode="json", exclude_none=True)
        assert cached == {**expected, "cache": "hit", "attempts": []}
        assert provider.calls == [("today", 10)]
    else:
        assert provider.calls == [("today", 10), ("today", 10)]
    assert ring_calls == (
        [("fixture", "today", 10)] * 2 if mode in {"rescue", "failed-rescue"} else []
    )


@pytest.mark.parametrize("malformed", [True, "truthy", {"status": "bad"}])
def test_malformed_success_never_enters_the_cache(configured_search, malformed):
    from tools.web_result_cache import search_memo

    provider = configured_search({"success": malformed, "data": {"web": []}})
    if malformed is True:
        provider.response["data"]["web"] = [None]
    result = web_search(WebSearchRequest(query="today", limit=2)).model_dump(
        mode="json", exclude_none=True
    )
    failure = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
    assert result == {
        "status": "failed",
        "selected_provider": "fixture",
        "failure": failure,
        "attempts": [
            {
                "provider": "fixture",
                "route": "selected",
                "status": "failed",
                "failure": failure,
            },
        ],
    }
    assert search_memo.lookup("fixture", "today", 2) is None


@pytest.mark.parametrize("boundary", ["lookup", "single-flight"])
def test_interruption_at_cache_boundaries_prevents_dispatch(
    configured_search, monkeypatch, boundary
):
    from contextlib import contextmanager
    from tools.interrupt import set_interrupt
    from tools.web_result_cache import search_memo

    provider = configured_search({"success": True, "data": {"web": []}})

    if boundary == "lookup":

        def lookup(*args):
            set_interrupt(True)
            return {"success": True, "data": {"web": []}}

        monkeypatch.setattr(search_memo, "lookup", lookup)
    else:

        @contextmanager
        def flight_lock(*args):
            set_interrupt(True)
            yield

        monkeypatch.setattr(search_memo, "flight_lock", flight_lock)

    result = web_search(WebSearchRequest(query="today", limit=2)).model_dump(
        mode="json", exclude_none=True
    )
    assert result == {
        "status": "failed",
        "selected_provider": "fixture",
        "failure": {
            "kind": "cancelled",
            "retry": "never",
            "scope": "provider",
        },
        "attempts": [],
    }
    assert provider.calls == []


def test_search_cache_is_scoped_to_the_served_profile(
    configured_search, tmp_path, monkeypatch
):
    import requests

    pytest.importorskip("exa_py")
    from agent import secret_scope, web_search_registry
    from hermes_cli.config import atomic_config_replace
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.web.exa.provider import ExaWebSearchProvider
    from tools import web_tools

    homes = {}
    for label in ("a", "b"):
        home = tmp_path / label
        home.mkdir()
        (home / ".env").write_text(f"EXA_API_KEY=fixture-{label}\n", encoding="utf-8")
        atomic_config_replace(
            home / "config.yaml", {"web": {"backend": "exa", "keyless_rescue": False}}
        )
        homes[label] = home
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr("plugins.web._common.lazy_ensure", lambda feature: None)
    monkeypatch.setattr(web_tools, "_exa_client", None)
    sent = []

    def send(session, request, **kwargs):
        label = request.headers["x-api-key"].removeprefix("fixture-")
        sent.append(label)
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps({
            "results": [
                {
                    "url": f"https://example.test/{label}",
                    "title": label,
                    "highlights": [label],
                }
            ]
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    web_search_registry._registry.register(ExaWebSearchProvider())
    observed = []
    for label in ("a", "b", "a"):
        home_token = set_hermes_home_override(homes[label])
        secret_token = secret_scope.set_secret_scope(
            secret_scope.build_profile_secret_scope(homes[label]),
            profile_home=str(homes[label]),
        )
        try:
            observed.append(
                web_search(
                    WebSearchRequest(query="identical query", limit=2)
                ).model_dump(mode="json", exclude_none=True)
            )
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    expected = []
    for label, cache in (("a", "miss"), ("b", "miss"), ("a", "hit")):
        expected.append({
            "status": "ok",
            "selected_provider": "exa",
            "served_provider": "exa",
            "cache": cache,
            "hits": [
                {
                    "url": f"https://example.test/{label}",
                    "title": label,
                    "description": label,
                    "position": 1,
                }
            ],
            "attempts": []
            if cache == "hit"
            else [{"provider": "exa", "route": "selected", "status": "succeeded"}],
        })
    assert observed == expected
    assert sent == ["a", "b"]


def test_concurrent_firecrawl_profiles_keep_their_sdk_clients(
    configured_search, tmp_path, monkeypatch
):
    import threading
    import types
    import requests

    pytest.importorskip("firecrawl")
    from agent import secret_scope, web_search_registry
    from hermes_cli.config import atomic_config_replace
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
    from tools import web_tools

    homes = {}
    for label in ("a", "b"):
        home = tmp_path / label
        home.mkdir()
        atomic_config_replace(
            home / "config.yaml",
            {
                "web": {
                    "backend": "firecrawl",
                    "keyless_rescue": False,
                    "provider_tier": {"firecrawl": "paid"},
                }
            },
        )
        homes[label] = home
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr("pm.ensure_import", lambda feature: None)
    web_search_registry._registry.register(FirecrawlWebSearchProvider())
    monkeypatch.setattr(web_tools, "_firecrawl_client", None)
    monkeypatch.setattr(web_tools, "_firecrawl_client_config", None)
    built_a = threading.Event()
    finished_b = threading.Event()

    class SlotRecorder(types.ModuleType):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if (
                name == "_firecrawl_client"
                and threading.current_thread().name == "profile-a"
            ):
                built_a.set()
                finished_b.wait()

    monkeypatch.setattr(web_tools, "__class__", SlotRecorder)
    sent = []
    observed = {}

    def send(session, request, **kwargs):
        label = request.headers["Authorization"].removeprefix("Bearer synthetic-")
        sent.append((threading.current_thread().name, label))
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps({
            "success": True,
            "data": {
                "web": [
                    {
                        "url": f"https://example.test/{label}",
                        "title": label,
                        "description": label,
                    }
                ]
            },
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)

    def run(label):
        home_token = set_hermes_home_override(homes[label])
        secret_token = secret_scope.set_secret_scope(
            {"FIRECRAWL_API_KEY": f"synthetic-{label}"}, profile_home=str(homes[label])
        )
        try:
            observed[label] = web_search(
                WebSearchRequest(query="identical query")
            ).model_dump(mode="json", exclude_none=True)
        finally:
            secret_scope.reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
            if label == "b":
                finished_b.set()

    a = threading.Thread(target=run, args=("a",), name="profile-a")
    b = threading.Thread(target=run, args=("b",), name="profile-b")
    a.start()
    built_a.wait()
    b.start()
    b.join()
    a.join()
    assert sorted(sent) == [("profile-a", "a"), ("profile-b", "b")]
    assert observed == {
        label: {
            "status": "ok",
            "selected_provider": "firecrawl",
            "served_provider": "firecrawl",
            "cache": "miss",
            "hits": [
                {
                    "url": f"https://example.test/{label}",
                    "title": label,
                    "description": label,
                }
            ],
            "attempts": [
                {"status": "succeeded", "provider": "firecrawl", "route": "selected"}
            ],
        }
        for label in ("a", "b")
    }


def test_actual_keyless_ring_reports_each_attempt_and_serving_provider(
    configured_search, tmp_path, monkeypatch
):
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web import keyless_mcp
    from plugins.web.exa.provider import ExaWebSearchProvider

    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "exa", "provider_tier": {"exa": "free"}}},
    )
    web_search_registry._registry.register(ExaWebSearchProvider())
    sent = []

    def send(session, request, **kwargs):
        sent.append(request.url)
        response = requests.Response()
        response.request = request
        response.headers["content-type"] = "application/json"
        if request.url == keyless_mcp.EXA_MCP_URL:
            response.status_code = 429
            response.headers["retry-after"] = "3"
            response._content = b"Bearer synthetic-secret"
            return response
        assert request.url == keyless_mcp.PARALLEL_MCP_URL
        response.status_code = 200
        payload = {
            "results": [
                {
                    "url": "https://example.test/news",
                    "title": "News",
                    "excerpts": ["Today"],
                }
            ]
        }
        response._content = json.dumps({
            "result": {"content": [{"type": "text", "text": json.dumps(payload)}]}
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    request = WebSearchRequest(query="today", limit=2)
    result = web_search(request).model_dump(mode="json", exclude_none=True)
    assert result == {
        "status": "ok",
        "selected_provider": "exa",
        "served_provider": "parallel",
        "cache": "miss",
        "hits": [
            {
                "url": "https://example.test/news",
                "title": "News",
                "description": "Today",
                "position": 1,
            }
        ],
        "attempts": [
            {
                "provider": "exa",
                "route": "keyless",
                "status": "failed",
                "failure": {
                    "kind": "rate-limited",
                    "retry": "transient",
                    "scope": "provider",
                    "status": 429,
                    "retry_after_ms": 3000,
                },
            },
            {"provider": "parallel", "route": "keyless", "status": "succeeded"},
        ],
    }
    cached = web_search(request).model_dump(mode="json", exclude_none=True)
    assert cached == {**result, "cache": "hit", "attempts": []}
    assert sent == [keyless_mcp.EXA_MCP_URL, keyless_mcp.PARALLEL_MCP_URL]


@pytest.mark.parametrize(
    "vendor", ["tavily", "exa", "parallel", "perplexity", "keenable"]
)
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"error": "upstream failed"},
        {"results": None},
        {"results": [None]},
        {"results": []},
    ],
)
def test_native_search_distinguishes_malformed_payloads_from_empty_results(
    configured_search, tmp_path, monkeypatch, vendor, payload
):
    import httpx
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.exa.provider import ExaWebSearchProvider
    from plugins.web.parallel.provider import ParallelWebSearchProvider
    from plugins.web.perplexity.provider import PerplexityWebSearchProvider
    from plugins.web.tavily.provider import TavilyWebSearchProvider
    from plugins.web.keenable.provider import KeenableWebSearchProvider
    from tools import web_result_cache, web_tools

    if vendor == "exa":
        pytest.importorskip("exa_py")
    if vendor == "parallel":
        pytest.importorskip("parallel")
    atomic_config_replace(
        tmp_path / "config.yaml",
        {
            "web": {
                "backend": vendor,
                "keyless_rescue": False,
                "provider_tier": {vendor: "paid"},
            }
        },
    )
    monkeypatch.setenv(f"{vendor.upper()}_API_KEY", "synthetic-key")
    monkeypatch.setattr("plugins.web._common.lazy_ensure", lambda feature: None)
    monkeypatch.setattr("pm.ensure_import", lambda feature: None)
    monkeypatch.setattr(web_tools, "_exa_client", None)
    monkeypatch.setattr(web_tools, "_parallel_client", None)
    providers = {
        "exa": ExaWebSearchProvider,
        "tavily": TavilyWebSearchProvider,
        "parallel": ParallelWebSearchProvider,
        "perplexity": PerplexityWebSearchProvider,
        "keenable": KeenableWebSearchProvider,
    }
    web_search_registry._registry.register(providers[vendor]())
    body = (
        {**payload, "search_id": "fixture-search"} if vendor == "parallel" else payload
    )

    def send(session, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("POST", url), json=body
        ),
    )
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda client, request, **kwargs: httpx.Response(
            200, request=request, json=body
        ),
    )
    result = web_search(WebSearchRequest(query="today", limit=2)).model_dump(
        mode="json", exclude_none=True
    )
    if payload == {"results": []}:
        assert result == {
            "status": "ok",
            "selected_provider": vendor,
            "served_provider": vendor,
            "cache": "miss",
            "hits": [],
            "attempts": [
                {"provider": vendor, "route": "selected", "status": "succeeded"},
            ],
        }
        return
    failure = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
    assert result == {
        "status": "failed",
        "selected_provider": vendor,
        "failure": failure,
        "attempts": [
            {
                "provider": vendor,
                "route": "selected",
                "status": "failed",
                "failure": failure,
            },
        ],
    }
    assert web_result_cache.search_memo.lookup(vendor, "today", 2) is None


@pytest.mark.parametrize(
    "body",
    [
        "garbage",
        "{}",
        '{"result":{}}',
        '{"result":42}',
        '{"error":"bad envelope"}',
        '{"result":{"content":42}}',
        '{"result":{"content":[{"type":"text","text":123}]}}',
    ],
)
def test_malformed_mcp_envelope_preserves_invalid_response(
    configured_search, tmp_path, monkeypatch, body
):
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.exa.provider import ExaWebSearchProvider

    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "exa", "provider_tier": {"exa": "free"}}},
    )
    web_search_registry._registry.register(ExaWebSearchProvider())

    def send(session, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = body.encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    failure = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
    result = web_search(WebSearchRequest(query="today")).model_dump(
        mode="json", exclude_none=True
    )
    assert result == {
        "status": "failed",
        "selected_provider": "exa",
        "failure": failure,
        "attempts": [
            {
                "status": "failed",
                "provider": "exa",
                "route": "keyless",
                "failure": failure,
            },
        ],
    }


@pytest.mark.parametrize("tier", ["paid", "free"])
@pytest.mark.parametrize("rows", [[None], [{"url": "https://example.test"}]])
def test_firecrawl_sdk_and_keyless_search_share_nullable_metadata(
    configured_search, tmp_path, monkeypatch, tier, rows
):
    import httpx
    import requests

    pytest.importorskip("firecrawl")
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
    from tools import web_tools

    atomic_config_replace(
        tmp_path / "config.yaml",
        {
            "web": {
                "backend": "firecrawl",
                "keyless_rescue": False,
                "provider_tier": {"firecrawl": tier},
            }
        },
    )
    monkeypatch.setenv("FIRECRAWL_API_KEY", "synthetic-key" if tier == "paid" else "")
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    monkeypatch.setattr("pm.ensure_import", lambda feature: None)
    monkeypatch.setattr(web_tools, "_firecrawl_client", None)
    web_search_registry._registry.register(FirecrawlWebSearchProvider())
    body = {"success": True, "data": {"web": rows}}

    def send(session, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda client, request, **kwargs: httpx.Response(
            200, request=request, json=body
        ),
    )
    result = web_search(WebSearchRequest(query="today")).model_dump(
        mode="json", exclude_none=True
    )
    route = "selected" if tier == "paid" else "keyless"
    if rows == [None]:
        failure = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
        assert result == {
            "status": "failed",
            "selected_provider": "firecrawl",
            "failure": failure,
            "attempts": [
                {
                    "status": "failed",
                    "provider": "firecrawl",
                    "route": route,
                    "failure": failure,
                },
            ],
        }
        return
    assert result == {
        "status": "ok",
        "selected_provider": "firecrawl",
        "served_provider": "firecrawl",
        "cache": "miss",
        "hits": [{"url": "https://example.test", "title": "", "description": ""}],
        "attempts": [
            {"status": "succeeded", "provider": "firecrawl", "route": route},
        ],
    }


def test_keyless_firecrawl_preserves_explicit_rejection(
    configured_search, tmp_path, monkeypatch
):
    import httpx
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider
    from tools.web_result_cache import search_memo

    atomic_config_replace(
        tmp_path / "config.yaml",
        {
            "web": {
                "backend": "firecrawl",
                "keyless_rescue": False,
                "provider_tier": {"firecrawl": "free"},
            }
        },
    )
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    web_search_registry._registry.register(FirecrawlWebSearchProvider())
    body = {"success": False, "error": "Synthetic upstream rejection"}
    monkeypatch.setattr(
        httpx.Client,
        "send",
        lambda client, request, **kwargs: httpx.Response(
            200, request=request, json=body
        ),
    )
    result = web_search(WebSearchRequest(query="today")).model_dump(
        mode="json", exclude_none=True
    )
    failure = {"kind": "unclassified", "retry": "unknown", "scope": "provider"}
    assert result == {
        "status": "failed",
        "selected_provider": "firecrawl",
        "failure": failure,
        "attempts": [
            {
                "status": "failed",
                "provider": "firecrawl",
                "route": "keyless",
                "failure": failure,
            },
        ],
    }
    assert search_memo.lookup("firecrawl", "today", 5) is None


def test_interrupt_during_primary_prevents_rescue(
    configured_search, tmp_path, monkeypatch
):
    import httpx
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.tavily.provider import TavilyWebSearchProvider
    from plugins.web.keenable.provider import KeenableWebSearchProvider
    from tools.interrupt import set_interrupt

    atomic_config_replace(
        tmp_path / "config.yaml", {"web": {"backend": "tavily", "keyless_rescue": True}}
    )
    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-key")
    web_search_registry._registry.register(TavilyWebSearchProvider())

    def primary(url, **kwargs):
        set_interrupt(True)
        return httpx.Response(
            503, request=httpx.Request("POST", url), json={"error": "upstream failed"}
        )

    sent = []

    def rescue(session, request, **kwargs):
        sent.append(request.url)
        raise AssertionError("A cancelled acquisition must not start rescue")

    monkeypatch.setattr(httpx, "post", primary)
    monkeypatch.setattr(requests.Session, "send", rescue)
    result = web_search(WebSearchRequest(query="today", limit=2)).model_dump(
        mode="json", exclude_none=True
    )
    assert result == {
        "status": "failed",
        "selected_provider": "tavily",
        "failure": {"kind": "cancelled", "retry": "never", "scope": "provider"},
        "attempts": [
            {
                "provider": "tavily",
                "route": "selected",
                "status": "failed",
                "failure": {
                    "kind": "unavailable",
                    "retry": "transient",
                    "scope": "provider",
                    "status": 503,
                },
            },
        ],
    }
    assert sent == []


def test_exhausted_native_ring_preserves_the_first_failure(
    configured_search, tmp_path, monkeypatch
):
    import httpx
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web import keyless_mcp
    from plugins.web.exa.provider import ExaWebSearchProvider

    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "exa", "provider_tier": {"exa": "free"}}},
    )
    web_search_registry._registry.register(ExaWebSearchProvider())
    sent = []

    def send(session, request, **kwargs):
        sent.append(request.url)
        response = requests.Response()
        response.request = request
        response.status_code = 429
        response.headers["retry-after"] = str(
            {keyless_mcp.EXA_MCP_URL: 1, keyless_mcp.PARALLEL_MCP_URL: 2}.get(
                request.url, 4
            )
        )
        response._content = b"rate limit"
        return response

    def post(url, **kwargs):
        sent.append(url)
        return httpx.Response(
            429,
            request=httpx.Request("POST", url),
            headers={"retry-after": "3"},
            text="rate limit",
        )

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(httpx, "post", post)
    result = web_search(WebSearchRequest(query="today", limit=2)).model_dump(
        mode="json", exclude_none=True
    )
    attempts = [
        {
            "provider": vendor,
            "route": "keyless",
            "status": "failed",
            "failure": {
                "kind": "rate-limited",
                "retry": "transient",
                "scope": "provider",
                "status": 429,
                "retry_after_ms": index * 1000,
            },
        }
        for index, vendor in enumerate(("exa", "parallel", "firecrawl", "keenable"), 1)
    ]
    assert result == {
        "status": "failed",
        "selected_provider": "exa",
        "failure": attempts[0]["failure"],
        "attempts": attempts,
    }
    assert sent == [
        keyless_mcp.EXA_MCP_URL,
        keyless_mcp.PARALLEL_MCP_URL,
        "https://api.firecrawl.dev/v2/search",
        "https://api.keenable.ai/v1/search/public",
    ]
