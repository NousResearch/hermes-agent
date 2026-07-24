import json

import httpx
import pytest


async def _always_safe_url(_url: str) -> bool:
    return True


ROUTING_PATCHES = {
    "_get_extract_backend": lambda: "fake",
    "check_website_access": lambda url: None,
    "_ensure_web_plugins_loaded": lambda: None,
}


def patch_web_extract_dependencies(monkeypatch, web_tools, provider=None):
    for name, value in ROUTING_PATCHES.items():
        monkeypatch.setattr(web_tools, name, value)
    monkeypatch.setattr(web_tools, "async_is_safe_url", _always_safe_url)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
    if provider is not None:
        monkeypatch.setattr("agent.web_search_registry.get_provider", lambda name: provider)
        monkeypatch.setattr(
            "agent.web_search_registry.get_active_extract_provider",
            lambda: provider,
        )


def patch_direct_fetch(
    monkeypatch,
    web_tools,
    body,
    content_type="text/plain; charset=utf-8",
):
    def fake_direct_fetch(url, timeout=20):
        return body, content_type, url

    monkeypatch.setattr(web_tools, "_direct_fetch_text", fake_direct_fetch)


class FakeStreamResponse:
    def __init__(
        self,
        url,
        body="",
        status_code=200,
        headers=None,
        content_type="text/plain; charset=utf-8",
    ):
        self.url = httpx.URL(url)
        self.status_code = status_code
        self.headers = {"content-type": content_type, **(headers or {})}
        self.encoding = "utf-8"
        self._body = body.encode()
        self.is_redirect = status_code in {301, 302, 303, 307, 308}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", self.url),
                response=httpx.Response(
                    self.status_code,
                    request=httpx.Request("GET", self.url),
                ),
            )

    def iter_bytes(self):
        yield self._body


def patch_stream_sequence(monkeypatch, web_tools, responses):
    calls = []
    queue = list(responses)

    def fake_stream(method, url, **kwargs):
        calls.append((url, kwargs))
        if not queue:
            pytest.fail(f"unexpected stream call for {url}")
        return queue.pop(0)

    monkeypatch.setattr(web_tools.httpx, "stream", fake_stream)
    return calls


class FakeExtractProvider:
    name = "fake"
    display_name = "Fake"

    def __init__(self, results=None):
        self.calls = []
        self.results = results or []

    def supports_extract(self):
        return True

    def extract(self, urls, **kwargs):
        self.calls.append((urls, kwargs))
        return self.results


@pytest.mark.asyncio
async def test_web_extract_routes_github_blob_to_raw(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider()
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)

    calls = []

    def fake_direct_fetch(url, timeout=20):
        calls.append(url)
        return "# Hello from raw github\n", "text/plain; charset=utf-8", url

    monkeypatch.setattr(web_tools, "_direct_fetch_text", fake_direct_fetch)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://github.com/octo/repo/blob/main/README.md"],
        )
    )

    assert calls == ["https://raw.githubusercontent.com/octo/repo/main/README.md"]
    entry = result["results"][0]
    assert entry["content"] == "# Hello from raw github\n"
    assert entry["url"] == "https://raw.githubusercontent.com/octo/repo/main/README.md"
    assert entry["title"] == "README.md"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_web_extract_uses_github_raw_link_when_ref_contains_slashes(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider()
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)

    calls = []

    def fake_direct_fetch(url, timeout=20):
        calls.append(url)
        if url == "https://raw.githubusercontent.com/octo/repo/feature/foo/README.md":
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", url),
                response=httpx.Response(404),
            )
        if url == "https://github.com/octo/repo/blob/feature/foo/README.md":
            return (
                '<a id="raw-url" href="/octo/repo/raw/refs/heads/feature/foo/README.md">Raw</a>',
                "text/html; charset=utf-8",
                url,
            )
        return "# Feature branch README\n", "text/plain; charset=utf-8", url

    monkeypatch.setattr(web_tools, "_direct_fetch_text", fake_direct_fetch)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://github.com/octo/repo/blob/feature/foo/README.md"],
        )
    )

    assert calls == [
        "https://raw.githubusercontent.com/octo/repo/feature/foo/README.md",
        "https://github.com/octo/repo/blob/feature/foo/README.md",
        "https://raw.githubusercontent.com/octo/repo/refs/heads/feature/foo/README.md",
    ]
    assert result["results"][0]["content"] == "# Feature branch README\n"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_web_extract_routed_only_urls_skip_provider_dispatch(monkeypatch):
    from tools import web_tools

    patch_web_extract_dependencies(monkeypatch, web_tools)
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider",
        lambda name: pytest.fail("provider lookup should not run for routed-only URLs"),
    )

    patch_direct_fetch(monkeypatch, web_tools, "# Hello from raw github\n")

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://github.com/octo/repo/blob/main/README.md"],
        )
    )

    assert result["results"][0]["url"] == (
        "https://raw.githubusercontent.com/octo/repo/main/README.md"
    )


@pytest.mark.asyncio
async def test_web_extract_routes_x_status_to_raw_html_fallback(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider()
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)

    html_doc = r'''
    <html>
      <head>
        <meta property="og:title" content="helicerat (@helicerat0x)">
        <meta property="og:description" content="linked article preview text">
      </head>
      <body>
        <script>
          window.__DATA__ = {"full_text":"https://t.co/rwV6oYg5Rd","expanded_url":"https:\/\/x.com\/i\/article\/2053497254510460928"};
        </script>
      </body>
    </html>
    '''

    patch_direct_fetch(monkeypatch, web_tools, html_doc, "text/html; charset=utf-8")

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://x.com/helicerat0x/status/2054223640573493523"],
        )
    )

    entry = result["results"][0]
    assert entry["title"] == "helicerat (@helicerat0x)"
    assert "Author: @helicerat0x" in entry["content"]
    assert "Tweet ID: 2054223640573493523" in entry["content"]
    assert "Linked article: https://x.com/i/article/2053497254510460928" in entry["content"]
    assert provider.calls == []


@pytest.mark.asyncio
async def test_web_extract_uses_provider_for_general_urls(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider([
        {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "Example Domain body",
            "raw_content": "Example Domain body",
        }
    ])
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)
    monkeypatch.setattr(
        web_tools,
        "_direct_fetch_text",
        lambda *a, **k: pytest.fail("direct fetch should not run for general URLs"),
    )

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com"],
        )
    )

    entry = result["results"][0]
    assert entry["url"] == "https://example.com"
    assert entry["title"] == "Example Domain"
    assert entry["content"] == "Example Domain body"
    assert provider.calls == [(["https://example.com"], {"format": None})]


@pytest.mark.asyncio
async def test_web_extract_preserves_input_order_with_routed_and_generic_urls(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider([
        {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "Example Domain body",
            "raw_content": "Example Domain body",
        }
    ])
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)

    patch_direct_fetch(monkeypatch, web_tools, "# Hello from raw github\n")

    result = json.loads(
        await web_tools.web_extract_tool(
            [
                "https://example.com",
                "https://github.com/octo/repo/blob/main/README.md",
            ],
        )
    )

    urls = [entry["url"] for entry in result["results"]]
    assert urls == [
        "https://example.com",
        "https://raw.githubusercontent.com/octo/repo/main/README.md",
    ]
    assert provider.calls == [(["https://example.com"], {"format": None})]


@pytest.mark.asyncio
async def test_web_extract_drops_extra_provider_results(monkeypatch):
    from tools import web_tools

    provider = FakeExtractProvider([
        {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "Example Domain body",
            "raw_content": "Example Domain body",
        },
        {
            "url": "https://extra.example.com",
            "title": "Unexpected",
            "content": "Unexpected extra result",
            "raw_content": "Unexpected extra result",
        },
    ])
    patch_web_extract_dependencies(monkeypatch, web_tools, provider)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com"],
        )
    )

    assert [entry["url"] for entry in result["results"]] == ["https://example.com"]


@pytest.mark.asyncio
async def test_direct_fetch_validates_redirect_targets(monkeypatch):
    from tools import web_tools

    patch_web_extract_dependencies(monkeypatch, web_tools)
    monkeypatch.setattr(
        web_tools,
        "is_safe_url",
        lambda url: not url.startswith("http://127.0.0.1"),
    )
    calls = patch_stream_sequence(
        monkeypatch,
        web_tools,
        [
            FakeStreamResponse(
                "https://example.com/README.md",
                status_code=302,
                headers={"location": "http://127.0.0.1/private"},
            ),
        ],
    )

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com/README.md"],
        )
    )

    assert len(calls) == 1
    assert result["results"][0]["url"] == "http://127.0.0.1/private"
    assert "private or internal" in result["results"][0]["error"]
