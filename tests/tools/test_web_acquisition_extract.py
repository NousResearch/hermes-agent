"""Native extraction preserves page identity, coverage, controls and provenance."""

import pytest

from agent.web_acquisition import WebExtractCapabilities, WebExtractRequest
from agent.web_acquisition_errors import WebServiceUnavailableError
from agent.web_search_provider import WebSearchProvider
from tools.web_acquisition import web_extract


class ExtractProvider(WebSearchProvider):
    name = "fixture"
    display_name = "Fixture"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def supports_extract(self):
        return True

    def extract_capabilities(self):
        return WebExtractCapabilities(formats=("markdown", "html"), max_characters=True)

    def extract(self, urls, **kwargs):
        self.calls.append((urls, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def configured_extract(tmp_path, monkeypatch):
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from tools import web_tools
    from tools.interrupt import set_interrupt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "fixture", "keyless_rescue": False}},
    )
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)

    async def safe(url):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", safe)
    set_interrupt(False)
    web_search_registry._registry.reset_for_tests()

    def configure(response):
        provider = ExtractProvider(response)
        web_search_registry._registry.register(provider)
        return provider

    yield configure
    set_interrupt(False)
    web_search_registry._registry.reset_for_tests()


def page(url, content="body", **fields):
    return {
        "url": url,
        "title": "Title",
        "content": content,
        "coverage": "full",
        **fields,
    }


def success(
    requested,
    resolved=None,
    content="body",
    coverage="full",
    cache="miss",
    attempts=None,
):
    return {
        "status": "ok",
        "requested_url": requested,
        "resolved_url": resolved or requested,
        "title": "Title",
        "content": content,
        "coverage": coverage,
        "served_provider": "fixture",
        "cache": cache,
        "attempts": attempts
        if attempts is not None
        else [{"status": "succeeded", "provider": "fixture", "route": "selected"}],
    }


def failure(url, kind, scope="page", **facts):
    cause = {"kind": kind, "retry": "never", "scope": scope, **facts}
    return {
        "status": "failed",
        "requested_url": url,
        "failure": cause,
        "attempts": [
            {
                "status": "failed",
                "provider": "fixture",
                "route": "selected",
                "failure": cause,
            }
        ],
    }


@pytest.mark.asyncio
async def test_batch_results_use_identity_and_restore_duplicates(configured_extract):
    a, b, c = (f"https://example.test/{value}" for value in ("a", "b", "c"))
    resolved = "https://example.test/redirected"
    provider = configured_extract([page(resolved, "B", requested_url=b), page(a, "A")])
    result = await web_extract(
        WebExtractRequest(urls=(a, b, a, c), required_coverage="full")
    )
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            success(a, content="A"),
            success(b, resolved, "B"),
            success(a, content="A"),
            failure(c, "result-missing"),
        ],
    }
    assert provider.calls == [([a, b, c], {"format": None})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        page("https://foreign.test/"),
        page("https://example.test/a", requested_url="https://foreign.test/"),
        page(
            "https://example.test/a", metadata={"sourceURL": "https://example.test/b"}
        ),
    ],
)
async def test_unattributable_batches_are_not_cached(configured_extract, row):
    from tools.web_result_cache import extract_cache_get

    a, b = "https://example.test/a", "https://example.test/b"
    configured_extract([row])
    result = await web_extract(WebExtractRequest(urls=(a, b)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            failure(a, "attribution-invalid"),
            failure(b, "attribution-invalid"),
        ],
    }
    assert [extract_cache_get(url, provider="fixture") for url in (a, b)] == [
        None,
        None,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("coverage", ["full", "partial", "unknown"])
async def test_full_requests_require_explicit_coverage(configured_extract, coverage):
    url = "https://example.test/a"
    configured_extract([page(url, raw_content="body", coverage=coverage)])
    result = await web_extract(WebExtractRequest(urls=(url,), required_coverage="full"))
    expected = (
        success(url)
        if coverage == "full"
        else failure(url, "content-coverage", coverage=coverage)
    )
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [expected],
    }


@pytest.mark.asyncio
async def test_guards_apply_to_requests_and_cached_redirects(
    configured_extract, monkeypatch
):
    from tools import web_tools
    from tools.web_result_cache import extract_cache_put

    denied, cached, redirect = (
        "https://denied.test/",
        "https://example.test/a",
        "https://private.test/",
    )
    provider = configured_extract([])
    extract_cache_put(
        cached,
        "body",
        "Title",
        provider="fixture",
        resolved_url=redirect,
        coverage="full",
        served_provider="fixture",
    )
    monkeypatch.setattr(
        "tools.website_policy.check_website_access",
        lambda url: (
            {
                "message": "blocked",
                "host": "denied.test",
                "rule": "deny",
                "source": "test",
            }
            if url == denied
            else None
        ),
    )

    async def safe(url):
        return url != redirect

    monkeypatch.setattr(web_tools, "async_is_safe_url", safe)
    result = await web_extract(WebExtractRequest(urls=(denied, cached)))
    expected = []
    for url, kind in ((denied, "website-policy"), (cached, "private-address")):
        expected.append({
            "status": "failed",
            "requested_url": url,
            "failure": {"kind": kind, "retry": "never", "scope": "policy"},
            "attempts": [],
        })
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": expected,
    }
    assert provider.calls == []


@pytest.fixture
def paid_exa(configured_extract, tmp_path, monkeypatch):
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.exa.provider import ExaWebSearchProvider
    from tools import web_tools

    atomic_config_replace(
        tmp_path / "config.yaml",
        {
            "web": {
                "backend": "exa",
                "keyless_rescue": False,
                "provider_tier": {"exa": "paid"},
            }
        },
    )
    monkeypatch.setenv("EXA_API_KEY", "synthetic-key")
    monkeypatch.setattr("plugins.web._common.lazy_ensure", lambda feature: None)
    monkeypatch.setattr(web_tools, "_exa_client", None)
    web_search_registry._registry.register(ExaWebSearchProvider())
    return tmp_path


def exa_response(request, payload, status=200):
    import json
    import requests

    response = requests.Response()
    response.status_code = status
    response.request = request
    response.headers["retry-after"] = "3"
    response._content = json.dumps(payload).encode()
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,kind,retry",
    [
        (401, "authentication", "never"),
        (429, "rate-limited", "transient"),
        (503, "unavailable", "transient"),
    ],
)
async def test_exa_endpoint_preserves_http_facts(
    paid_exa, monkeypatch, status, kind, retry
):
    import requests

    url = "https://example.test/a"
    monkeypatch.setattr(
        requests.Session,
        "send",
        lambda session, request, **kwargs: exa_response(
            request, {"error": "synthetic-secret"}, status
        ),
    )
    cause = {"kind": kind, "retry": retry, "scope": "provider", "status": status}
    if status != 401:
        cause["retry_after_ms"] = 3000
    result = await web_extract(WebExtractRequest(urls=(url,)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "exa",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": cause,
                "attempts": [
                    {
                        "status": "failed",
                        "provider": "exa",
                        "route": "selected",
                        "failure": cause,
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,kind,retry",
    [
        (404, "page-not-found", "never"),
        (410, "page-not-found", "never"),
        (403, "page-fetch", "never"),
        (429, "page-fetch", "transient"),
        (503, "page-fetch", "transient"),
        (None, "page-fetch", "unknown"),
    ],
)
async def test_exa_target_page_failure_preserves_scope(
    paid_exa, monkeypatch, status, kind, retry
):
    import requests

    url = "https://example.test/a"
    error = {"tag": "CRAWL_ERROR"}
    if status is not None:
        error["httpStatusCode"] = status
    payload = {
        "results": [],
        "statuses": [
            {"id": url, "status": "error", "source": "livecrawl", "error": error}
        ],
    }
    monkeypatch.setattr(
        requests.Session,
        "send",
        lambda session, request, **kwargs: exa_response(request, payload),
    )
    cause = {"kind": kind, "retry": retry, "scope": "page"}
    if status is not None:
        cause["status"] = status
    result = await web_extract(WebExtractRequest(urls=(url,)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "exa",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": cause,
                "attempts": [
                    {
                        "status": "failed",
                        "provider": "exa",
                        "route": "selected",
                        "failure": cause,
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_incompatible_rescue_has_no_remote_attempt(paid_exa, monkeypatch):
    import requests
    from hermes_cli.config import atomic_config_replace

    atomic_config_replace(
        paid_exa / "config.yaml",
        {
            "web": {
                "backend": "exa",
                "keyless_rescue": True,
                "provider_tier": {"exa": "paid"},
            }
        },
    )
    url = "https://example.test/a"
    sent = []

    def send(session, request, **kwargs):
        sent.append(request.url)
        return exa_response(request, {}, 503)

    monkeypatch.setattr(requests.Session, "send", send)
    result = await web_extract(WebExtractRequest(urls=(url,), max_characters=50))
    cause = {
        "kind": "unavailable",
        "retry": "transient",
        "scope": "provider",
        "status": 503,
        "retry_after_ms": 3000,
    }
    assert (sent, result.model_dump(mode="json", exclude_none=True)) == (
        ["https://api.exa.ai/contents"],
        {
            "status": "ok",
            "selected_provider": "exa",
            "results": [
                {
                    "status": "failed",
                    "requested_url": url,
                    "failure": cause,
                    "attempts": [
                        {
                            "status": "failed",
                            "provider": "exa",
                            "route": "selected",
                            "failure": cause,
                        }
                    ],
                },
            ],
        },
    )


@pytest.mark.asyncio
async def test_full_cache_preserves_redirect_and_options(configured_extract):
    url, resolved = "https://example.test/a", "https://example.test/redirected"
    provider = configured_extract([page(resolved, requested_url=url)])
    request = WebExtractRequest(
        urls=(url,), required_coverage="full", max_characters=200
    )
    fresh = await web_extract(request)
    cached = await web_extract(request)
    await web_extract(request.model_copy(update={"max_characters": 300}))
    assert [
        result.model_dump(mode="json", exclude_none=True) for result in (fresh, cached)
    ] == [
        {
            "status": "ok",
            "selected_provider": "fixture",
            "results": [success(url, resolved)],
        },
        {
            "status": "ok",
            "selected_provider": "fixture",
            "results": [success(url, resolved, cache="hit", attempts=[])],
        },
    ]
    assert provider.calls == [
        ([url], {"format": None, "max_chars": 200}),
        ([url], {"format": None, "max_chars": 300}),
    ]


@pytest.mark.asyncio
async def test_legacy_cache_cannot_satisfy_full_coverage(configured_extract):
    from tools.web_result_cache import extract_cache_put

    url = "https://example.test/a"
    provider = configured_extract([page(url)])
    extract_cache_put(url, "legacy", "Old", provider="fixture")
    result = await web_extract(WebExtractRequest(urls=(url,), required_coverage="full"))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [success(url)],
    }
    assert provider.calls == [([url], {"format": None})]


@pytest.mark.asyncio
async def test_failed_rescue_preserves_primary_and_alternative(
    configured_extract, monkeypatch
):
    from plugins.web import keyless_mcp

    url = "https://example.test/a"
    primary = WebServiceUnavailableError(503)
    provider = configured_extract(primary)
    secondary = {
        "kind": "authentication",
        "retry": "never",
        "scope": "provider",
        "status": 401,
    }
    monkeypatch.setattr(
        "tools.web_tools_extract._rescue_eligible",
        lambda provider, capability=None: True,
    )
    monkeypatch.setattr(
        keyless_mcp,
        "extract_with_failover",
        lambda name, urls, **kwargs: [
            {"url": url, "error": "Synthetic secret", "failure": secondary}
        ],
    )
    result = await web_extract(WebExtractRequest(urls=(url,)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": primary.to_failure(),
                "attempts": [
                    {
                        "status": "failed",
                        "provider": "fixture",
                        "route": "selected",
                        "failure": primary.to_failure(),
                    },
                    {
                        "status": "failed",
                        "route": "keyless-rescue",
                        "failure": secondary,
                    },
                ],
            },
        ],
    }
    assert provider.calls == [([url], {"format": None})]


@pytest.mark.asyncio
async def test_secret_url_refuses_whole_call_without_echo(configured_extract):
    provider = configured_extract([])
    result = await web_extract(
        WebExtractRequest(urls=("https://example.test/?key=sk-" + "a" * 48,))
    )
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "failed",
        "failure": {
            "kind": "secret-url",
            "retry": "never",
            "scope": "input",
        },
    }
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars,coverage", [(None, "full"), (300, "partial")])
async def test_exa_sdk_preserves_request_identity_and_options(
    configured_extract, tmp_path, monkeypatch, max_chars, coverage
):
    import json
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.exa.provider import ExaWebSearchProvider
    from tools import web_tools

    atomic_config_replace(
        tmp_path / "config.yaml",
        {
            "web": {
                "backend": "exa",
                "keyless_rescue": False,
                "provider_tier": {"exa": "paid"},
            }
        },
    )
    monkeypatch.setenv("EXA_API_KEY", "synthetic-key")
    monkeypatch.setattr("plugins.web._common.lazy_ensure", lambda feature: None)
    monkeypatch.setattr(web_tools, "_exa_client", None)
    web_search_registry._registry.register(ExaWebSearchProvider())
    url, resolved = "https://example.test/a", "https://example.test/redirected"
    sent = []

    def send(session, request, **kwargs):
        sent.append((request.url, json.loads(request.body)))
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps({
            "results": [{"url": resolved, "id": url, "title": "Title", "text": "body"}]
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = await web_extract(WebExtractRequest(urls=(url,), max_characters=max_chars))
    expected = success(url, resolved, coverage=coverage)
    expected["served_provider"] = "exa"
    expected["attempts"][0]["provider"] = "exa"
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "exa",
        "results": [expected],
    }
    assert sent == [
        (
            "https://api.exa.ai/contents",
            {
                "urls": [url],
                "text": True if max_chars is None else {"maxCharacters": max_chars},
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_keyless_extract_keeps_actual_attempts_and_cancellation(
    configured_extract, tmp_path, monkeypatch, cancel
):
    import json
    import requests
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web import keyless_mcp
    from plugins.web.exa.provider import ExaWebSearchProvider
    from tools.interrupt import set_interrupt
    import threading

    caller = threading.get_ident()

    atomic_config_replace(
        tmp_path / "config.yaml",
        {"web": {"backend": "exa", "provider_tier": {"exa": "free"}}},
    )
    web_search_registry._registry.register(ExaWebSearchProvider())
    url = "https://example.test/a"
    sent = []

    def send(session, request, **kwargs):
        sent.append(request.url)
        response = requests.Response()
        response.request = request
        if request.url == keyless_mcp.EXA_MCP_URL:
            response.status_code = 429
            response._content = b"Bearer synthetic-secret"
            if cancel:
                set_interrupt(True, caller)
            return response
        assert request.url == keyless_mcp.PARALLEL_MCP_URL
        response.status_code = 200
        response._content = json.dumps({
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "results": [
                                {"url": url, "title": "Title", "full_content": "body"}
                            ]
                        }),
                    }
                ]
            }
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = await web_extract(WebExtractRequest(urls=(url,), required_coverage="full"))
    attempts = [
        {
            "status": "failed",
            "provider": "exa",
            "route": "keyless",
            "failure": {
                "kind": "rate-limited",
                "retry": "transient",
                "scope": "provider",
                "status": 429,
            },
        }
    ]
    if cancel:
        expected = {
            "status": "failed",
            "requested_url": url,
            "failure": {"kind": "cancelled", "retry": "never", "scope": "provider"},
            "attempts": attempts,
        }
    else:
        attempts.append({
            "status": "succeeded",
            "provider": "parallel",
            "route": "keyless",
        })
        expected = {**success(url, attempts=attempts), "served_provider": "parallel"}
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "exa",
        "results": [expected],
    }
    assert sent == [keyless_mcp.EXA_MCP_URL] + (
        [] if cancel else [keyless_mcp.PARALLEL_MCP_URL]
    )


@pytest.mark.asyncio
async def test_keyless_exa_receives_native_character_option(
    configured_extract, tmp_path, monkeypatch
):
    import json
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
    url = "https://example.test/a"
    sent = []

    def send(session, request, **kwargs):
        sent.append(json.loads(request.body)["params"])
        response = requests.Response()
        response.request = request
        response.status_code = 200
        response._content = json.dumps({
            "result": {"content": [{"type": "text", "text": "# Title\nbody"}]}
        }).encode()
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    result = await web_extract(WebExtractRequest(urls=(url,), max_characters=300))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "exa",
        "results": [
            {
                **success(
                    url,
                    content="# Title\nbody",
                    coverage="unknown",
                    attempts=[
                        {"status": "succeeded", "provider": "exa", "route": "keyless"}
                    ],
                ),
                "served_provider": "exa",
            },
        ],
    }
    assert sent == [
        {"name": "web_fetch_exa", "arguments": {"urls": [url], "maxCharacters": 300}}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("vendor", ["tavily", "perplexity"])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": None},
        {"results": [None]},
        {"results": [{"url": "https://example.test/a", "raw_content": 42, "text": 42}]},
    ],
)
async def test_http_extractors_reject_malformed_payloads(
    configured_extract, tmp_path, monkeypatch, vendor, payload
):
    import httpx
    from agent import web_search_registry
    from hermes_cli.config import atomic_config_replace
    from plugins.web.perplexity.provider import PerplexityWebSearchProvider
    from plugins.web.tavily.provider import TavilyWebSearchProvider

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
    web_search_registry._registry.register(
        {"tavily": TavilyWebSearchProvider, "perplexity": PerplexityWebSearchProvider}[
            vendor
        ]()
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: httpx.Response(
            200, request=httpx.Request("POST", url), json=payload
        ),
    )
    url = "https://example.test/a"
    result = await web_extract(WebExtractRequest(urls=(url,)))
    cause = {"kind": "invalid-response", "retry": "never", "scope": "provider"}
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": vendor,
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": cause,
                "attempts": [
                    {
                        "status": "failed",
                        "provider": vendor,
                        "route": "selected",
                        "failure": cause,
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   "])
async def test_empty_full_content_is_a_page_failure(configured_extract, content):
    url = "https://example.test/a"
    configured_extract([page(url, content)])
    result = await web_extract(WebExtractRequest(urls=(url,), required_coverage="full"))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [failure(url, "result-missing")],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,capability",
    [
        ({"format": "html"}, "extract-format-html"),
        ({"max_characters": 5}, "extract-max-characters"),
    ],
)
async def test_explicit_options_need_declared_provider_support(
    configured_extract, options, capability
):
    from agent.web_search_provider import WebSearchProvider

    url = "https://example.test/a"
    provider = configured_extract([page(url)])
    provider.extract_capabilities = WebSearchProvider.extract_capabilities.__get__(
        provider
    )
    result = await web_extract(WebExtractRequest(urls=(url,), **options))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": {
                    "kind": "unsupported-capability",
                    "retry": "never",
                    "scope": "provider",
                    "capability": capability,
                },
                "attempts": [],
            },
        ],
    }
    assert provider.calls == []


@pytest.mark.asyncio
async def test_explicit_identity_allows_redirect_to_another_requested_url(
    configured_extract,
):
    a, b = "https://example.test/a", "https://example.test/b"
    configured_extract([
        page(b, "A", requested_url=a, metadata={"sourceURL": b}),
        page(b, "B", requested_url=b),
    ])
    result = await web_extract(WebExtractRequest(urls=(a, b)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            success(a, b, "A"),
            success(b, b, "B"),
        ],
    }


@pytest.mark.asyncio
async def test_legacy_cache_does_not_invent_serving_provider(configured_extract):
    from tools.web_result_cache import extract_cache_put

    url = "https://example.test/a"
    provider = configured_extract([])
    extract_cache_put(url, "legacy", "Old", provider="fixture")
    result = await web_extract(WebExtractRequest(urls=(url,)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            {
                "status": "ok",
                "requested_url": url,
                "resolved_url": url,
                "title": "Old",
                "content": "legacy",
                "coverage": "unknown",
                "cache": "hit",
                "attempts": [],
            },
        ],
    }
    assert provider.calls == []


@pytest.mark.asyncio
async def test_cache_hit_cannot_bypass_capability_validation(configured_extract):
    from tools.web_result_cache import extract_cache_put
    from agent.web_search_provider import WebSearchProvider

    url = "https://example.test/a"
    provider = configured_extract([])
    provider.extract_capabilities = WebSearchProvider.extract_capabilities.__get__(
        provider
    )
    extract_cache_put(
        url,
        "body",
        "Title",
        provider="fixture",
        coverage="full",
        served_provider="fixture",
    )
    result = await web_extract(WebExtractRequest(urls=(url,), format="markdown"))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": {
                    "kind": "unsupported-capability",
                    "retry": "never",
                    "scope": "provider",
                    "capability": "extract-format-markdown",
                },
                "attempts": [],
            },
        ],
    }
    assert provider.calls == []


@pytest.mark.asyncio
async def test_provider_timeout_does_not_invent_dispatch_duration(configured_extract):
    url = "https://example.test/a"
    configured_extract(TimeoutError("Synthetic upstream timeout"))
    cause = {"kind": "timeout", "retry": "transient", "scope": "provider"}
    result = await web_extract(WebExtractRequest(urls=(url,)))
    assert result.model_dump(mode="json", exclude_none=True) == {
        "status": "ok",
        "selected_provider": "fixture",
        "results": [
            {
                "status": "failed",
                "requested_url": url,
                "failure": cause,
                "attempts": [
                    {
                        "status": "failed",
                        "provider": "fixture",
                        "route": "selected",
                        "failure": cause,
                    }
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_default_format_cache_does_not_satisfy_explicit_markdown(
    configured_extract,
):
    url = "https://example.test/a"
    provider = configured_extract([page(url)])
    await web_extract(WebExtractRequest(urls=(url,)))
    result = await web_extract(WebExtractRequest(urls=(url,), format="markdown"))
    assert (provider.calls, result.model_dump(mode="json", exclude_none=True)) == (
        [([url], {"format": None}), ([url], {"format": "markdown"})],
        {"status": "ok", "selected_provider": "fixture", "results": [success(url)]},
    )
