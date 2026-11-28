"""URLs Hermes never caches must not come from an extract backend's page cache either.

``web.cache_exempt_hosts`` (and local dev URLs) promise "always fetched live", and Hermes' own extract
cache honours that. Firecrawl, Keenable, Exa and Parallel keep page caches of their own and answer
from them unless asked otherwise, so an exempt staging page could still come back hours or weeks old.
Each of those backends is now asked for its live fetch for these URLs, and only for these URLs.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.web import keyless_mcp
from plugins.web.firecrawl import provider as firecrawl_provider

LIVE = "https://app.preview.example/pricing"
STORED = "https://docs.example.org/guide"
WEB = {"cache_exempt_hosts": ["preview.example"]}


@pytest.fixture
def web_config(monkeypatch):
    def _set(config: dict) -> None:
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: config)

    _set(WEB)
    return _set


def test_must_fetch_live_is_the_set_hermes_never_caches(web_config):
    from tools.web_result_cache import _cacheable, must_fetch_live

    assert must_fetch_live(LIVE) is True
    assert must_fetch_live("http://localhost:5173/") is True
    assert must_fetch_live(STORED) is False
    assert _cacheable(LIVE) is False and _cacheable(STORED) is True


class _ScrapeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def scrape(self, **kwargs):
        self.calls.append(kwargs)
        url = kwargs["url"]
        return {"success": True, "data": {"markdown": "# page", "metadata": {"title": "P", "url": url, "sourceURL": url, "statusCode": 200}}}


@pytest.mark.parametrize(("url", "max_age"), [(LIVE, 0), (STORED, 21_600_000)])
def test_firecrawl_scrape_goes_live_only_for_live_urls(monkeypatch, web_config, url, max_age):
    web_config({**WEB, "firecrawl_max_age_hours": 6})
    client = _ScrapeClient()
    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda _capability=None: client)
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)
    monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda url: True)
    entry = asyncio.run(firecrawl_provider._scrape_one(url, ["markdown"], "markdown"))
    assert entry["content"] == "# page"
    assert client.calls[0]["max_age"] == max_age


def test_firecrawl_keyless_ring_goes_live_for_live_urls(monkeypatch, web_config):
    sent = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True, "data": {"markdown": "# page", "metadata": {"url": LIVE}}}

    def _post(url, *, json, headers, timeout):
        sent.append(json)
        return _Response()

    monkeypatch.setattr(firecrawl_provider.httpx, "post", _post)
    keyless_mcp.firecrawl_extract_keyless([LIVE, STORED])
    assert sent[0]["maxAge"] == 0
    assert "maxAge" not in sent[1]


def _keenable_response(url: str) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = {"url": url, "title": "P", "content": "body"}
    return response


def test_keenable_fetch_asks_for_live_only_for_live_urls(monkeypatch, web_config):
    from plugins.web.keenable import provider as keenable

    sent = []

    def _get(url, *, params, headers, timeout, verify):
        sent.append(params)
        return _keenable_response(params["url"])

    monkeypatch.setattr(keenable, "provider_env", lambda _name: "k-test")
    monkeypatch.setattr(keenable, "use_keyless", lambda *_: False)
    monkeypatch.setattr("requests.get", _get)
    results = keenable.KeenableWebSearchProvider().extract([LIVE, STORED])
    assert [r["content"] for r in results] == ["body", "body"]
    assert sent == [{"url": LIVE, "live": "true"}, {"url": STORED}]


def test_keenable_keyless_fetch_asks_for_live_only_for_live_urls(monkeypatch, web_config):
    sent = []

    def _request(method, path, **kwargs):
        sent.append((path, kwargs["params"]))
        return {"url": kwargs["params"]["url"], "title": "P", "content": "body"}

    monkeypatch.setattr(keyless_mcp, "_keenable_request", _request)
    keyless_mcp.keenable_extract_keyless([LIVE, STORED])
    assert sent == [("/v1/fetch/public", {"url": LIVE, "live": "true"}), ("/v1/fetch/public", {"url": STORED})]


@pytest.mark.parametrize(("urls", "fresh"), [([LIVE, STORED], {"max_age_hours": 0}), ([STORED], {})])
def test_exa_contents_go_live_when_a_live_url_is_in_the_batch(monkeypatch, web_config, urls, fresh):
    from plugins.web.exa import provider as exa

    client = MagicMock()
    client.get_contents.return_value = SimpleNamespace(
        results=[SimpleNamespace(url=u, title="P", text="body") for u in urls], statuses=[])
    monkeypatch.setattr(exa, "use_keyless", lambda *_: False)
    monkeypatch.setattr(exa, "_get_exa_client", lambda: client)
    results = exa.ExaWebSearchProvider().extract(urls)
    assert [r["content"] for r in results] == ["body"] * len(urls)
    assert client.get_contents.call_args.kwargs == {"text": True, **fresh}


@pytest.mark.parametrize(
    ("urls", "fresh"),
    [([LIVE, STORED], {"fetch_policy": {"max_age_seconds": 600, "disable_cache_fallback": True}}), ([STORED], {})],
)
def test_parallel_extract_uses_its_freshest_policy_for_live_urls(monkeypatch, web_config, urls, fresh):
    from plugins.web.parallel import provider as parallel

    seen = {}

    async def _extract(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(results=[SimpleNamespace(url=u, title="P", full_content="body", excerpts=[]) for u in urls], errors=[])

    client = SimpleNamespace(beta=SimpleNamespace(extract=_extract))
    monkeypatch.setattr(parallel, "use_keyless", lambda *_: False)
    monkeypatch.setattr(parallel, "_get_async_client", lambda: client)
    results = asyncio.run(parallel.ParallelWebSearchProvider().extract(urls))
    assert [r["content"] for r in results] == ["body"] * len(urls)
    assert seen == {"urls": urls, "full_content": True, **fresh}


def test_web_extract_tool_fetches_an_exempt_host_live(tmp_path, monkeypatch, public_dns):
    """Real config file and extract dispatch (Firecrawl selected, no key: its free tier through the keyless
    ring): the exempt page goes out with ``maxAge: 0``; the other page keeps Firecrawl's default."""
    from hermes_cli.config import atomic_config_write
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_URL", raising=False)
    atomic_config_write(home / "config.yaml", {"web": {"extract_backend": "firecrawl", "cache_exempt_hosts": ["preview.example"]}})
    register_all_web_providers()
    sent = []

    class _Response:
        def __init__(self, url):
            self.url = url

        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True, "data": {"markdown": f"# {self.url}", "metadata": {"url": self.url}}}

    def _post(url, *, json, headers, timeout):
        sent.append(json)
        return _Response(json["url"])

    monkeypatch.setattr(firecrawl_provider.httpx, "post", _post)
    result = json.loads(asyncio.run(web_tools.web_extract_tool([LIVE, STORED])))

    assert [r["content"] for r in result["results"]] == [f"# {LIVE}", f"# {STORED}"]
    assert {post["url"]: post.get("maxAge") for post in sent} == {LIVE: 0, STORED: None}
