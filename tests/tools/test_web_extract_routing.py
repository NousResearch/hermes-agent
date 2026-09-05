"""Regression coverage for direct web_extract routes."""

import asyncio
import json
import threading

import httpx
import pytest

from tools import web_tools


class _StreamResponse:
    def __init__(
        self,
        url: str,
        body: str = "",
        *,
        headers: dict[str, str] | None = None,
        is_redirect: bool = False,
        status_code: int = 200,
    ) -> None:
        self.url = httpx.URL(url)
        self.headers = {"content-type": "text/plain; charset=utf-8", **(headers or {})}
        self.encoding = "utf-8"
        self.is_redirect = is_redirect
        self.status_code = status_code
        self._body = body.encode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", self.url),
                response=httpx.Response(self.status_code),
            )

    def iter_bytes(self):
        yield self._body


class _SafeClient:
    def __init__(
        self, responses: list[_StreamResponse], seen: list[tuple[str, str, dict]]
    ) -> None:
        self.responses = responses
        self.seen = seen

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method: str, url: str, **kwargs):
        self.seen.append((method, url, kwargs))
        return _StreamContext(self.responses.pop(0))


class _StreamContext:
    def __init__(self, response: _StreamResponse) -> None:
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc, traceback):
        return False


def _dispatch_extract(urls: list[str]) -> dict:
    raw = web_tools.registry.dispatch("web_extract", {"urls": urls})
    assert isinstance(raw, str)
    return json.loads(raw)


def test_web_extract_registry_routes_github_blob_without_provider(monkeypatch):
    """A public GitHub file uses raw.githubusercontent.com without a web backend."""
    requested = "https://github.com/octo/repo/blob/main/README.md"
    raw_url = "https://raw.githubusercontent.com/octo/repo/main/README.md"
    seen = []

    page = _StreamResponse(
        requested,
        '<a id="raw-url" href="/octo/repo/raw/main/README.md">Raw</a>',
    )
    response = _StreamResponse(raw_url, "# Raw README\n")
    responses = [page, response]

    def fake_safe_client(**_kwargs):
        return _SafeClient(responses, seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail("direct GitHub routing must not resolve a web provider"),
    )

    result = _dispatch_extract([requested])

    assert [call[1] for call in seen] == [requested, raw_url]
    assert result["results"] == [
        {
            "url": raw_url,
            "title": "README.md",
            "content": "# Raw README\n",
            "error": None,
        }
    ]


def test_web_extract_routes_github_raw_path_without_reconstructing_the_ref(monkeypatch):
    """A GitHub raw URL is fetched as supplied, so slash refs stay unambiguous."""
    requested = "https://github.com/octo/repo/raw/feature/foo/README.md"
    seen = []

    def fake_safe_client(**_kwargs):
        return _SafeClient([_StreamResponse(requested, "# Raw README\n")], seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail("direct GitHub routing must not resolve a web provider"),
    )

    result = _dispatch_extract([requested])

    assert [call[1] for call in seen] == [requested]
    assert result["results"][0]["content"] == "# Raw README\n"


def test_web_extract_blocks_github_source_matched_by_website_policy(monkeypatch):
    """A source-host block stops routing before GitHub's raw host is requested."""
    requested = "https://github.com/octo/repo/blob/main/README.md"
    seen = []

    def website_policy(url):
        if url == requested:
            return {
                "host": "github.com",
                "rule": "github.com",
                "source": "config",
                "message": "Blocked by website policy: 'github.com' matched rule 'github.com' from config",
            }
        return None

    def fake_safe_client(**_kwargs):
        return _SafeClient([], seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.website_policy.check_website_access", website_policy)
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail(
            "a policy-blocked source URL must not fall through to a provider"
        ),
    )

    result = _dispatch_extract([requested])

    assert seen == []
    assert result["results"] == [
        {
            "url": requested,
            "title": "",
            "content": "",
            "error": "Blocked by website policy: 'github.com' matched rule 'github.com' from config",
        }
    ]


def test_web_extract_blocks_github_redirect_matched_by_website_policy(monkeypatch):
    """Direct routing must not let a GitHub redirect bypass the website blocklist."""
    requested = "https://github.com/octo/repo/blob/main/README.md"
    blocked_url = "https://blocked.example/private"
    seen = []

    response = _StreamResponse(
        requested, headers={"location": blocked_url}, is_redirect=True
    )

    def fake_safe_client(**_kwargs):
        return _SafeClient([response], seen)

    def website_policy(url):
        if url == blocked_url:
            return {
                "host": "blocked.example",
                "rule": "blocked.example",
                "source": "config",
                "message": "Blocked by website policy: 'blocked.example' matched rule 'blocked.example' from config",
            }
        return None

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr("tools.website_policy.check_website_access", website_policy)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail(
            "blocked direct routing must not fall through to a provider"
        ),
    )

    result = _dispatch_extract([requested])

    assert seen[0][0:2] == ("GET", requested)
    assert result["results"] == [
        {
            "url": blocked_url,
            "title": "",
            "content": "",
            "error": "Blocked by website policy: 'blocked.example' matched rule 'blocked.example' from config",
        }
    ]


def test_web_extract_blocks_github_redirect_to_ssrf_target(monkeypatch):
    """A redirect into an internal address is blocked instead of reaching the provider."""
    from tools.url_safety import SSRFConnectionBlocked

    requested = "https://github.com/octo/repo/blob/main/README.md"
    blocked_url = "http://127.0.0.1/private"
    seen = []

    class RedirectingClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def stream(self, method, url, **_kwargs):
            seen.append((method, url))
            if url == blocked_url:
                raise SSRFConnectionBlocked("blocked at connect time")
            return _StreamContext(
                _StreamResponse(
                    requested, headers={"location": blocked_url}, is_redirect=True
                )
            )

    async def safe_url(_url):
        return True

    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        lambda **_kwargs: RedirectingClient(),
    )
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail(
            "an SSRF-blocked redirect must not fall through to a provider"
        ),
    )

    result = _dispatch_extract([requested])

    assert seen == [("GET", requested), ("GET", blocked_url)]
    assert result["results"] == [
        {
            "url": blocked_url,
            "title": "",
            "content": "",
            "error": "Blocked: URL targets a private or internal network address",
        }
    ]


def test_web_extract_registry_routes_x_status_without_provider(monkeypatch):
    """A public X status uses its public HTML instead of requiring an extract backend."""
    requested = "https://x.com/alice/status/1234567890"
    seen = []
    document = r"""
    <html><head>
      <meta property="og:title" content="Alice (@alice)">
      <meta property="og:description" content="fallback description">
    </head><body>
      <script>window.__DATA__ = {"full_text":"hello from x","expanded_url":"https:\/\/x.com\/i\/article\/42"};</script>
    </body></html>
    """

    def fake_safe_client(**_kwargs):
        return _SafeClient([_StreamResponse(requested, document)], seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail("direct X routing must not resolve a web provider"),
    )

    result = _dispatch_extract([requested])

    assert seen[0][0:2] == ("GET", requested)
    entry = result["results"][0]
    assert entry["url"] == requested
    assert entry["title"] == "Alice (@alice)"
    assert "Author: @alice" in entry["content"]
    assert "Tweet ID: 1234567890" in entry["content"]
    assert "Text: hello from x" in entry["content"]
    assert "Linked article: https://x.com/i/article/42" in entry["content"]


def test_web_extract_falls_back_to_provider_when_x_direct_fetch_fails(monkeypatch):
    """A transient X route failure keeps the URL eligible for provider extraction."""
    requested = "https://x.com/alice/status/1234567890"
    seen = []

    class Provider:
        name = "fake"

        async def extract(self, urls, **_kwargs):
            assert urls == [requested]
            return [
                {"url": requested, "title": "Fallback", "content": "provider content"}
            ]

    def fake_safe_client(**_kwargs):
        return _SafeClient([_StreamResponse(requested, status_code=503)], seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        web_tools, "_resolve_extract_provider", lambda _backend: (Provider(), None)
    )

    result = _dispatch_extract([requested])

    assert [call[1] for call in seen] == [requested]
    assert result["results"][0]["content"] == "provider content"


def test_web_extract_uses_github_raw_link_for_refs_with_slashes(monkeypatch):
    """GitHub's raw href disambiguates a branch name containing slashes."""
    requested = "https://github.com/octo/repo/blob/feature/foo/README.md"
    guessed_raw = "https://raw.githubusercontent.com/octo/repo/feature/foo/README.md"
    page_url = requested
    resolved_raw = (
        "https://raw.githubusercontent.com/octo/repo/refs/heads/feature/foo/README.md"
    )
    seen = []
    responses = [
        _StreamResponse(
            page_url,
            '<a id="raw-url" href="/octo/repo/raw/refs/heads/feature/foo/README.md">Raw</a>',
        ),
        _StreamResponse(resolved_raw, "# Feature branch README\n"),
    ]

    def fake_safe_client(**_kwargs):
        return _SafeClient(responses, seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail("direct GitHub routing must not resolve a web provider"),
    )

    result = _dispatch_extract([requested])

    assert [call[1] for call in seen] == [page_url, resolved_raw]
    assert guessed_raw not in [call[1] for call in seen]
    assert result["results"][0]["url"] == resolved_raw
    assert result["results"][0]["content"] == "# Feature branch README\n"


def test_web_extract_falls_back_to_provider_when_github_direct_fetch_fails(monkeypatch):
    """A transient direct-route failure preserves the normal extract backend fallback."""
    requested = "https://github.com/octo/repo/blob/main/README.md"
    seen = []

    class Provider:
        name = "fake"

        async def extract(self, urls, **_kwargs):
            assert urls == [requested]
            return [
                {"url": requested, "title": "Fallback", "content": "provider content"}
            ]

    def fake_safe_client(**_kwargs):
        return _SafeClient([_StreamResponse(requested, status_code=503)], seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        web_tools, "_resolve_extract_provider", lambda _backend: (Provider(), None)
    )

    result = _dispatch_extract([requested])

    assert [call[1] for call in seen] == [requested]
    assert result["results"][0]["content"] == "provider content"


def test_web_extract_preserves_input_order_between_direct_and_provider_routes(
    monkeypatch,
):
    """Provider output and direct-route output are merged at their original input positions."""
    direct_url = "https://github.com/octo/repo/blob/main/README.md"
    raw_url = "https://raw.githubusercontent.com/octo/repo/main/README.md"
    generic_url = "https://example.com/page"
    seen = []

    class Provider:
        name = "fake"

        async def extract(self, urls, **_kwargs):
            assert urls == [generic_url]
            return [
                {"url": generic_url, "title": "Generic", "content": "provider content"}
            ]

    responses = [
        _StreamResponse(
            direct_url,
            '<a id="raw-url" href="/octo/repo/raw/main/README.md">Raw</a>',
        ),
        _StreamResponse(raw_url, "# Raw README\n"),
    ]

    def fake_safe_client(**_kwargs):
        return _SafeClient(responses, seen)

    async def safe_url(_url):
        return True

    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_client", fake_safe_client)
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        web_tools, "_resolve_extract_provider", lambda _backend: (Provider(), None)
    )

    result = _dispatch_extract([direct_url, generic_url])

    assert [call[1] for call in seen] == [direct_url, raw_url]
    assert [entry["url"] for entry in result["results"]] == [raw_url, generic_url]
    assert [entry["content"] for entry in result["results"]] == [
        "# Raw README\n",
        "provider content",
    ]


def test_web_extract_blocks_direct_route_when_safe_client_rejects_target(monkeypatch):
    """A connection-time SSRF rejection is returned as blocked, never retried through a provider."""
    from tools.url_safety import SSRFConnectionBlocked

    requested = "https://github.com/octo/repo/blob/main/README.md"
    seen = []

    class BlockedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def stream(self, method, url, **_kwargs):
            seen.append((method, url))
            raise SSRFConnectionBlocked("blocked at connect time")

    async def safe_url(_url):
        return True

    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client", lambda **_kwargs: BlockedClient()
    )
    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(
        web_tools,
        "_get_extract_backend",
        lambda: pytest.fail(
            "SSRF-blocked direct routing must not fall through to a provider"
        ),
    )

    result = _dispatch_extract([requested])

    assert seen == [("GET", requested)]
    assert result["results"] == [
        {
            "url": requested,
            "title": "",
            "content": "",
            "error": "Blocked: URL targets a private or internal network address",
        }
    ]


@pytest.mark.asyncio
async def test_web_extract_special_routing_does_not_block_the_event_loop(monkeypatch):
    """Direct fetch runs in a worker so the event loop can release a blocked request."""
    routed_url = "https://raw.githubusercontent.com/octo/repo/main/README.md"
    thread_started = threading.Event()
    release_direct_route = threading.Event()

    async def safe_url(_url):
        return True

    def blocking_direct_route(_url, *, format=None):
        thread_started.set()
        assert release_direct_route.wait(timeout=2)
        return {
            "url": routed_url,
            "title": "README.md",
            "content": "# Raw README\n",
            "raw_content": "# Raw README\n",
            "error": None,
        }

    monkeypatch.setattr(web_tools, "async_is_safe_url", safe_url)
    monkeypatch.setattr(web_tools, "extract_special_url", blocking_direct_route)

    task = asyncio.create_task(web_tools.web_extract_tool([routed_url]))
    assert await asyncio.to_thread(thread_started.wait, 2)
    assert not task.done()
    release_direct_route.set()

    result = json.loads(await task)
    assert result["results"][0]["content"] == "# Raw README\n"


def test_web_extract_registry_gate_keeps_direct_routes_available_without_provider(
    monkeypatch,
):
    """Direct GitHub/X extraction stays discoverable even when no vendor backend is configured."""
    monkeypatch.setattr(web_tools, "check_web_api_key", lambda: False)

    assert web_tools.check_web_extract_available()
    entry = web_tools.registry.get_entry("web_extract")
    assert entry is not None
    assert entry.check_fn is web_tools.check_web_extract_available
    assert [
        item["function"]["name"]
        for item in web_tools.registry.get_definitions({"web_extract"})
    ] == ["web_extract"]
