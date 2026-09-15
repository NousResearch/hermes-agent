"""Ollama web backend — search + fetch dispatch through the real tools."""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

from tests.tools.conftest import register_all_web_providers


def _ok(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _register_with_ollama():
    register_all_web_providers()
    from agent.web_search_registry import register_provider
    from plugins.web.ollama.provider import OllamaWebSearchProvider
    register_provider(OllamaWebSearchProvider())


def test_search_dispatch_maps_web_search_shape():
    """web_search on backend=ollama hits /api/web_search with Bearer auth and maps content→description."""
    import tools.web_tools as wt

    _register_with_ollama()
    payload = {
        "results": [
            {"title": "Bloom filter", "url": "https://en.wikipedia.org/wiki/Bloom_filter",
             "content": "space-efficient probabilistic structure"},
        ],
    }
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}), \
         patch.object(wt, "_get_search_backend", return_value="ollama"), \
         patch("plugins.web.ollama.provider.httpx.post", return_value=_ok(payload)) as post:
        out = json.loads(wt.web_search_tool("bloom filter", limit=3))

    assert post.call_args.args[0] == "https://ollama.com/api/web_search"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer ollama-test"
    body = post.call_args.kwargs["json"]
    assert body["query"] == "bloom filter"
    assert 1 <= body["max_results"] <= 10  # ollama caps max_results at 10 server-side
    assert out["success"] is True
    assert out["data"]["web"][0] == {
        "title": "Bloom filter",
        "url": "https://en.wikipedia.org/wiki/Bloom_filter",
        "description": "space-efficient probabilistic structure",
        "position": 1,
    }


def test_search_limit_capped_at_10():
    """The tool's limit buckets (10/20/50/100) never push max_results past Ollama's cap."""
    from plugins.web.ollama.provider import OllamaWebSearchProvider, _MAX_SEARCH_RESULTS

    captured = {}

    def fake_post(endpoint, **kwargs):
        captured.update(kwargs["json"])
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": []}
        return resp

    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}), \
         patch("plugins.web.ollama.provider.httpx.post", side_effect=fake_post):
        p = OllamaWebSearchProvider()
        for requested in (5, 20, 100):
            p.search("q", limit=requested)
            assert captured["max_results"] == min(requested, _MAX_SEARCH_RESULTS)
        assert captured["max_results"] == 10


def test_extract_dispatch_fetches_per_url():
    """web_extract on backend=ollama posts every URL to /api/web_fetch and appends links."""
    import tools.web_tools as wt

    _register_with_ollama()
    urls = ["https://tokio.rs/tokio/tutorial", "https://docs.rs/smol"]
    payloads = [
        {"title": "Tokio Tutorial", "content": "Tokio is an asynchronous runtime … for Rust.",
         "links": ["https://tokio.rs/tokio/tutorial/hello-tokio"]},
        {"title": "smol", "content": "smol runtime docs.", "links": None},
    ]
    posts = []

    def fake_post(endpoint, **kwargs):
        posts.append((endpoint, kwargs))
        return _ok(payloads[len(posts) - 1])

    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}), \
         patch.object(wt, "_get_extract_backend", return_value="ollama"), \
         patch("plugins.web.ollama.provider.httpx.post", side_effect=fake_post) as post:
        out = json.loads(asyncio.run(wt.web_extract_tool(urls)))

    assert post.call_count == 2  # one POST per URL
    by_url = {r["url"]: r for r in out["results"]}
    assert "Tokio is an asynchronous runtime" in by_url[urls[0]]["content"]
    assert "https://tokio.rs/tokio/tutorial/hello-tokio" in by_url[urls[0]]["content"]
    assert by_url[urls[0]]["title"] == "Tokio Tutorial"
    # the tool pipeline trims each entry to url/title/content/error
    assert set(by_url[urls[0]]) == {"url", "title", "content", "error"}
    assert by_url[urls[1]]["content"] == "smol runtime docs."


def test_extract_missing_content_becomes_error():
    """A web_fetch reply without content yields a per-URL error entry, not a crash."""
    from plugins.web.ollama.provider import OllamaWebSearchProvider

    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}), \
         patch("plugins.web.ollama.provider.httpx.post", return_value=_ok({"title": "", "content": "", "links": None})):
        docs = OllamaWebSearchProvider().extract(["https://example.com/empty"])
    assert docs[0]["error"] == "no content returned"


def test_missing_key_errors_without_http():
    """No OLLAMA_API_KEY → is_available False; search/extract return typed errors and never POST."""
    import httpx

    from plugins.web.ollama.provider import OllamaWebSearchProvider
    p = OllamaWebSearchProvider()
    with patch.dict(os.environ, {}, clear=False), \
         patch("plugins.web.ollama.provider.httpx.post") as post:
        os.environ.pop("OLLAMA_API_KEY", None)
        # provider_env falls back to the config-aware layer (~/.hermes/.env); hide it there too.
        import agent.web_search_provider as wsp
        with patch.object(wsp, "get_provider_env", return_value=""):
            assert p.is_available() is False
            res = p.search("x")
            assert res["success"] is False and "OLLAMA_API_KEY" in res["error"]
            docs = p.extract(["https://example.com"])
            assert "OLLAMA_API_KEY" in docs[0]["error"]
            post.assert_not_called()

    with patch.dict(os.environ, {"OLLAMA_API_KEY": "bogus"}), \
         patch("plugins.web.ollama.provider.httpx.post", return_value=_ok({}, status_code=401)) as post:
        post.return_value.text = '{"error":"Unauthorized"}'
        res = OllamaWebSearchProvider().search("x")
        assert res["success"] is False and res["error"] == "Unauthorized"


def test_network_error_returns_failure_shape():
    """httpx errors surface as ``Ollama web search failed: …`` instead of raising."""
    import httpx

    from plugins.web.ollama.provider import OllamaWebSearchProvider
    p = OllamaWebSearchProvider()
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}), \
         patch("plugins.web.ollama.provider.httpx.post", side_effect=httpx.ConnectError("boom")):
        res = p.search("x")
        assert res["success"] is False and "Ollama web search failed" in res["error"]
        docs = p.extract(["https://example.com"])
        assert "Ollama web fetch failed" in docs[0]["error"]


def test_registry_and_availability():
    """``ollama`` registers under the strict-selection name and is available with the env key."""
    from agent.web_search_registry import get_provider

    _register_with_ollama()
    p = get_provider("ollama")
    assert p is not None and p.supports_search() and p.supports_extract()
    with patch.dict(os.environ, {"OLLAMA_API_KEY": "ollama-test"}):
        assert p.is_available() is True