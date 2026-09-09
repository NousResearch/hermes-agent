"""Perplexity web backend — search + snippets dispatch through the real tools."""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

from tests.tools.conftest import register_all_web_providers


def _ok(payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def test_search_dispatch_maps_search_api_shape():
    """web_search on backend=perplexity hits /search with Bearer auth and maps snippet→description."""
    import tools.web_tools as wt

    register_all_web_providers()
    payload = {
        "results": [
            {"title": "Bloom filter", "url": "https://en.wikipedia.org/wiki/Bloom_filter",
             "snippet": "space-efficient probabilistic structure", "date": "2004-04-17"},
        ],
        "id": "abc",
    }
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}), \
         patch.object(wt, "_get_search_backend", return_value="perplexity"), \
         patch("plugins.web.perplexity.provider.httpx.post", return_value=_ok(payload)) as post:
        out = json.loads(wt.web_search_tool("bloom filter", limit=3))

    assert post.call_args.args[0] == "https://api.perplexity.ai/search"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer pplx-test"
    body = post.call_args.kwargs["json"]
    assert body["query"] == "bloom filter"
    assert 1 <= body["max_results"] <= 20  # dispatcher bucket-rounds the fetch limit
    assert body["search_context_size"] == "low"
    assert out["success"] is True
    assert out["data"]["web"][0] == {
        "title": "Bloom filter",
        "url": "https://en.wikipedia.org/wiki/Bloom_filter",
        "description": "space-efficient probabilistic structure",
        "position": 1,
    }


def test_extract_dispatch_snippets_per_url_and_missing_key():
    """web_extract on backend=perplexity posts every URL to /sdk/content/snippets;
    a URL the backend failed carries ``error`` instead of content; no key → error, no HTTP."""
    import tools.web_tools as wt

    register_all_web_providers()
    urls = ["https://tokio.rs/tokio/tutorial", "https://docs.rs/smol"]
    payload = {"results": [
        {"url": urls[0], "text": "Tokio is an asynchronous runtime … for Rust.", "tokens_count": 12},
        {"url": urls[1], "error": "Page not found or unavailable."},
    ]}
    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "pplx-test"}), \
         patch.object(wt, "_get_extract_backend", return_value="perplexity"), \
         patch("plugins.web.perplexity.provider.httpx.post", return_value=_ok(payload)) as post:
        out = json.loads(asyncio.run(wt.web_extract_tool(urls)))

    assert post.call_args.args[0] == "https://api.perplexity.ai/sdk/content/snippets"
    body = post.call_args.kwargs["json"]
    assert body["urls"] == urls
    assert body["query"] == "tokio tutorial smol"
    assert body["max_tokens_per_page"] <= body["max_tokens"]
    by_url = {r["url"]: r for r in out["results"]}
    assert "Tokio is an asynchronous runtime" in by_url[urls[0]]["content"]
    assert by_url[urls[1]]["error"] == "Page not found or unavailable."

    with patch.dict(os.environ, {}, clear=False), \
         patch("plugins.web.perplexity.provider.httpx.post") as post:
        os.environ.pop("PERPLEXITY_API_KEY", None)
        from plugins.web.perplexity.provider import PerplexityWebSearchProvider
        p = PerplexityWebSearchProvider()
        assert p.is_available() is False
        res = p.search("x")
        assert res["success"] is False and "PERPLEXITY_API_KEY" in res["error"]
        docs = p.extract(["https://example.com"])
        assert "PERPLEXITY_API_KEY" in docs[0]["error"]
        post.assert_not_called()
