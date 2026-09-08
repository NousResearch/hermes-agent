"""web_extract pairs backend results to the URLs that were REQUESTED, never by list position (#97378).

Parallel appends failures after successes, Exa omits pages it could not fetch, and the keyless ring
inherits both shapes. Positional pairing cached one page's text under another URL's key and served it
for the whole cache TTL.
"""

import json

import pytest

import tools.web_result_cache as wrc
import tools.web_tools as web_tools
from tools.web_tools_extract import _pair_results

A, B = "https://a.example/page", "https://b.example/page"


def _doc(url, text):
    return {"url": url, "title": url, "content": text, "raw_content": text, "metadata": {"sourceURL": url}}


def _fail(url, error="fetch failed"):
    return {"url": url, "title": "", "content": "", "error": error}


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, results):
        self._results = results

    def extract(self, urls, **kwargs):
        return self._results


@pytest.mark.asyncio
async def test_results_are_paired_by_url_not_position(tmp_path, monkeypatch):
    """A batch returned as [failure(B), document(A)] for the request [A, B] must yield A's document at
    A's position and cache A's text under A's key only — B stays a miss."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", lambda: {})
    provider = _ScriptedProvider([_fail(B), _doc(A, "text of A")])
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: provider.name)
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_resolve_extract_provider", lambda backend: (provider, None))

    async def _allow(url):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow)

    results = json.loads(await web_tools.web_extract_tool([A, B]))["results"]

    assert [r["url"] for r in results] == [A, B]
    assert results[0]["content"] == "text of A" and not results[0].get("error")
    assert results[1]["error"] == "fetch failed"
    assert wrc.extract_cache_get(A, provider=provider.name)["content"] == "text of A"
    assert wrc.extract_cache_get(B, provider=provider.name) is None


@pytest.mark.parametrize(
    "urls, results, expected",
    [
        # A backend echoing a cosmetically different URL still pairs: trailing slash, http->https, apex->www.
        (["https://a.example/x"], [_doc("https://a.example/x/", "slash")], ["slash"]),
        (["http://a.example/x"], [_doc("https://a.example/x", "https")], ["https"]),
        (["https://a.example/x"], [_doc("https://www.a.example/x", "www")], ["www"]),
        # The keyless ring emits a "no content" stub NEXT TO the document when it rewrites a URL.
        (["https://a.example/x"], [_fail("https://a.example/x", "no content returned"), _doc("https://a.example/x/", "doc")], ["doc"]),
        # The query is part of the page: a missing ?page=1 never inherits ?page=2's document.
        (["https://a.example/x?page=1", "https://a.example/x?page=2"], [_doc("https://a.example/x?page=2", "p2")], [None, "p2"]),
        # A lone unmatched result belongs to the lone unmatched URL (redirect to another host).
        ([A, B], [_doc("https://cdn.b.example/home", "moved"), _doc(A, "a")], ["a", "moved"]),
        # Several unmatched results are never guessed at: they are dropped, not attached.
        ([A, B], [_doc("https://x.example/1", "x1"), _doc("https://x.example/2", "x2")], [None, None]),
    ],
    ids=["trailing-slash", "scheme", "www", "document-beats-stub", "query-is-a-page", "lone-leftover", "no-guessing"],
)
def test_pair_results_matches_canonical_variants_only(urls, results, expected):
    paired = _pair_results(urls, results)
    assert [None if r.get("error") else r["content"] for r in paired] == expected
