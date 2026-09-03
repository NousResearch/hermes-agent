"""web_extract pairs backend results with the URL that was REQUESTED.

Extract providers do not preserve the order — or the length — of the URL
list they are handed: Tavily and Parallel append their failures after their
successes, Exa builds its list from the documents it managed to fetch, the
keyless normalizers append "no content returned" stubs for URL strings they
did not see echoed back, and a keyless-rescued batch arrives in ring-vendor
order. Pairing by list position therefore wrote one page's text under
another URL's cache key and served it for the whole TTL.

Each test drives the real dispatcher with a scripted provider double and
asserts the per-URL output, the cache state, and — where it is the point —
whether a follow-up call reached the backend at all.
"""

import hashlib
import json
import time
from unittest.mock import patch

import pytest

import tools.web_result_cache as wrc
import tools.web_tools as web_tools
from plugins.web import keyless_mcp

A = "https://alpha.example.com/page"
B = "https://bravo.example.com/page"
C = "https://charlie.example.com/page"

A_TEXT = "Alpha body text, long enough to be recognisable."
B_TEXT = "Bravo body text, long enough to be recognisable."
C_TEXT = "Charlie body text, long enough to be recognisable."


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the extract cache at a temp dir (as tests/tools/test_web_result_cache.py does)."""
    cache_dir = tmp_path / "cache" / "web"
    cache_dir.mkdir(parents=True)
    monkeypatch.setattr(wrc, "_cache_dir", lambda: cache_dir)
    monkeypatch.setattr(wrc, "_web_config", lambda: {})
    return cache_dir


def _doc(url, body, source_url=None):
    """A successful document in the shape every provider normalizes to."""
    return {
        "url": url,
        "title": f"Title of {url}",
        "content": body,
        "raw_content": body,
        "metadata": {"sourceURL": source_url or url, "title": f"Title of {url}"},
    }


def _failed(url, error="extraction failed"):
    """A Tavily ``failed_results`` entry as _normalize_tavily_documents emits it."""
    return {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "error": error,
        "metadata": {"sourceURL": url},
    }


def _stub(url):
    """The keyless normalizers' stub for a requested URL they never saw echoed."""
    return {"url": url, "title": "", "content": "", "error": "no content returned"}


def _cached(url):
    """Content the extract cache would serve for *url*, or None."""
    hit = wrc.extract_cache_get(url, provider="tavily")
    return hit["content"] if hit else None


class _ScriptedProvider:
    """Keyed provider double replaying scripted extract responses.

    Records the URL list of every call so a test can assert that a follow-up
    request was (or was not) served from cache.
    """

    name = "tavily"
    display_name = "Tavily"

    def __init__(self, *batches):
        self._batches = list(batches)
        self.calls = []

    def supports_search(self):
        return True

    def supports_extract(self):
        return True

    def is_available(self):
        return True

    def extract(self, urls, **kwargs):
        self.calls.append(list(urls))
        return self._batches.pop(0) if self._batches else []


async def _extract(monkeypatch, provider, urls, config=None):
    """Run the real web_extract dispatcher against *provider*; return results."""
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(
        "agent.web_search_registry.get_provider", lambda name: provider
    )
    monkeypatch.setattr(
        web_tools, "_load_web_config", lambda: config or {"backend": "tavily"}
    )

    async def _allow_all(url, **kwargs):
        return True

    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_all)
    return json.loads(await web_tools.web_extract_tool(list(urls)))["results"]


# ── dispatcher: pairing by URL ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_reordered_batch_is_paired_by_url(monkeypatch):
    """Tavily emits documents in API order, which need not be request order."""
    provider = _ScriptedProvider([_doc(B, B_TEXT), _doc(A, A_TEXT)])
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["url"] for r in results] == [A, B]
    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    assert _cached(A) == A_TEXT
    assert _cached(B) == B_TEXT

    # The follow-up is served from cache — the backend is not called again.
    again = await _extract(monkeypatch, provider, [A])
    assert again[0]["content"] == A_TEXT
    assert provider.calls == [[A, B]]


@pytest.mark.asyncio
async def test_failure_appended_after_success_does_not_poison_cache(monkeypatch):
    """Tavily appends failed_results AFTER results, shifting everything later."""
    provider = _ScriptedProvider(
        [_doc(B, B_TEXT), _failed(A)],
        [_doc(A, A_TEXT)],
    )
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["url"] for r in results] == [A, B]
    assert results[0]["error"] and results[0]["content"] == ""
    assert results[1]["content"] == B_TEXT
    assert _cached(A) is None  # a failed URL must cache nothing
    assert _cached(B) == B_TEXT

    # No poisoned hit: the failed URL is fetched again and returns its own page.
    again = await _extract(monkeypatch, provider, [A])
    assert again[0]["content"] == A_TEXT
    assert provider.calls == [[A, B], [A]]


@pytest.mark.asyncio
async def test_omitted_url_gets_its_own_error(monkeypatch):
    """Exa builds its list from response.results, so an unfetchable URL vanishes."""
    provider = _ScriptedProvider([_doc(B, B_TEXT)], [_doc(A, A_TEXT)])
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["url"] for r in results] == [A, B]
    assert results[0]["error"] and results[0]["content"] == ""
    assert results[1]["content"] == B_TEXT
    assert _cached(A) is None
    assert _cached(B) == B_TEXT

    again = await _extract(monkeypatch, provider, [A])
    assert again[0]["content"] == A_TEXT


@pytest.mark.asyncio
async def test_trailing_slash_variants_are_paired(monkeypatch):
    """Firecrawl reports the final URL, which often differs only by a slash —
    here in a reordered batch, so only the key itself can line them up."""
    provider = _ScriptedProvider([_doc(B + "/", B_TEXT), _doc(A + "/", A_TEXT)])
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    assert _cached(A) == A_TEXT
    assert _cached(B) == B_TEXT


@pytest.mark.asyncio
async def test_redirected_url_paired_when_it_is_the_only_leftover(monkeypatch):
    """A backend that follows a cross-host redirect reports the destination;
    with one unmatched request and one unmatched result, it can be nothing else."""
    final = "https://alpha-cdn.example.net/page"
    provider = _ScriptedProvider([_doc(B, B_TEXT), _doc(final, A_TEXT)])
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    # The backend's final URL stays visible on the result...
    assert results[0]["url"] == final
    # ...while the cache is keyed by the string the caller asked for.
    assert _cached(A) == A_TEXT
    assert _cached(final) is None


@pytest.mark.asyncio
async def test_scheme_and_www_canonicalization_is_paired(monkeypatch):
    """Backends canonicalize http:// + apex to the https://www. form they fetched."""
    a_http = "http://example.com/alpha"
    b_http = "http://example.com/bravo"
    provider = _ScriptedProvider([
        _doc("https://www.example.com/bravo", B_TEXT),
        _doc("https://www.example.com/alpha", A_TEXT),
    ])
    results = await _extract(monkeypatch, provider, [a_http, b_http])

    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    assert _cached(a_http) == A_TEXT
    assert _cached(b_http) == B_TEXT


@pytest.mark.asyncio
async def test_duplicate_requested_urls_share_one_document(monkeypatch):
    """One document legitimately answers both copies of a repeated URL."""
    provider = _ScriptedProvider([_doc(A, A_TEXT)])
    results = await _extract(monkeypatch, provider, [A, A])

    assert [r["content"] for r in results] == [A_TEXT, A_TEXT]
    assert provider.calls == [[A, A]]


@pytest.mark.asyncio
async def test_unrequested_document_is_dropped(monkeypatch):
    """A document nobody asked for must not be attached to some other URL."""
    extra = "https://unrelated.example.org/page"
    provider = _ScriptedProvider([
        _doc(A, A_TEXT),
        _doc(extra, "Content of a page nobody requested."),
        _doc(B, B_TEXT),
    ])
    results = await _extract(monkeypatch, provider, [A, B])

    assert [r["url"] for r in results] == [A, B]
    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    assert _cached(extra) is None


@pytest.mark.asyncio
async def test_document_beats_keyless_no_content_stub(monkeypatch):
    """Keyless Tavily appends a stub for any requested URL it did not see
    echoed verbatim — here alongside the real document for the same page."""
    provider = _ScriptedProvider([_doc(B + "/", B_TEXT), _doc(A, A_TEXT), _stub(B)])
    results = await _extract(monkeypatch, provider, [A, B])

    assert results[0]["content"] == A_TEXT
    assert results[1]["content"] == B_TEXT
    assert not results[1]["error"]
    assert _cached(B) == B_TEXT


@pytest.mark.asyncio
async def test_rescued_batch_is_reassociated(monkeypatch):
    """A whole-batch failure rides the keyless ring, which answers in
    ring-vendor order; the rescued batch is paired by URL too."""
    # Keyed Tavily with the rescue tier on, as tests/tools/test_web_keyless_rescue.py does.
    monkeypatch.setattr(
        "agent.web_search_provider.get_provider_env",
        lambda name: "tvly-real" if name == "TAVILY_API_KEY" else "",
    )
    monkeypatch.setattr(
        "agent.web_search_registry._keyless_tier_enabled", lambda: True
    )
    provider = _ScriptedProvider([_failed(A, "HTTP 500"), _failed(B, "HTTP 500")])
    with patch.object(
        keyless_mcp, "extract_with_failover",
        return_value=[_doc(B, B_TEXT), _doc(A, A_TEXT)],
    ) as ring:
        results = await _extract(monkeypatch, provider, [A, B])

    ring.assert_called_once()
    assert [r["url"] for r in results] == [A, B]
    assert [r["content"] for r in results] == [A_TEXT, B_TEXT]
    # Rescued batches are never cached: the next call must retry the backend.
    assert _cached(A) is None
    assert _cached(B) is None


@pytest.mark.asyncio
async def test_cache_hits_and_fetched_results_merge_by_url(monkeypatch):
    """A cached URL plus a reordered fetch of the rest still lines up."""
    await _extract(monkeypatch, _ScriptedProvider([_doc(A, A_TEXT)]), [A])
    assert _cached(A) == A_TEXT

    provider = _ScriptedProvider([_doc(C, C_TEXT), _doc(B, B_TEXT)])
    results = await _extract(monkeypatch, provider, [A, B, C])

    assert [r["url"] for r in results] == [A, B, C]
    assert [r["content"] for r in results] == [A_TEXT, B_TEXT, C_TEXT]
    assert provider.calls == [[B, C]]  # A was served from cache


@pytest.mark.asyncio
async def test_empty_backend_response_errors_per_url(monkeypatch):
    """An empty result list is one explicit error per URL, not one generic
    tool error that loses which URLs were asked for."""
    provider = _ScriptedProvider([])
    results = await _extract(
        monkeypatch, provider, [A, B],
        config={"backend": "tavily", "keyless_rescue": False},
    )

    assert [r["url"] for r in results] == [A, B]
    assert all(r["error"] for r in results)
    assert all(r["content"] == "" for r in results)
    assert _cached(A) is None
    assert _cached(B) is None


# ── cache key version ────────────────────────────────────────────────────

def test_pre_v2_cache_entries_are_never_served(_isolated_cache):
    """Entries written before results were paired by URL may hold another
    page's text, so the key version retires all of them at once."""
    stale_file = _isolated_cache / "stale.cache.md"
    stale_file.write_text("Content of a completely different page.", encoding="utf-8")
    legacy_digest = hashlib.sha256(
        f"{A}\nmarkdown\ntavily".encode("utf-8")
    ).hexdigest()[:16]
    (_isolated_cache / wrc._INDEX_FILENAME).write_text(
        json.dumps({
            legacy_digest: {
                "url": A,
                "file": str(stale_file),
                "title": "Stale",
                "fetched_at": time.time(),
            }
        }),
        encoding="utf-8",
    )

    assert wrc.extract_cache_get(A, provider="tavily") is None

    # Freshly written entries still round-trip.
    wrc.extract_cache_put(A, A_TEXT, title="Alpha", provider="tavily")
    hit = wrc.extract_cache_get(A, provider="tavily")
    assert hit is not None
    assert hit["content"] == A_TEXT


# ── association branches the dispatcher tests cannot reach ───────────────

class TestAssociationEdges:
    def test_multiple_leftovers_are_never_guessed(self):
        """Two unmatched requests and two unmatched results: either pairing
        would be a coin flip, so both requests error and both results drop."""
        out = web_tools._associate_extract_results(
            [A, B],
            [_doc("https://one.example.net/p", "ONE"),
             _doc("https://two.example.net/p", "TWO")],
        )
        assert [r["url"] for r in out] == [A, B]
        assert all(r["error"] for r in out)
        assert all(r["content"] == "" for r in out)

    def test_non_dict_entries_do_not_shift_pairing(self):
        """A backend that slips a non-dict into its list must not misalign it."""
        out = web_tools._associate_extract_results(
            [A, B], ["oops", _doc(B, B_TEXT), None, _doc(A, A_TEXT)]
        )
        assert [r["content"] for r in out] == [A_TEXT, B_TEXT]

    def test_superseded_stub_is_not_paired_with_another_url(self):
        """A stub that lost to the real document for its own URL belongs to
        that URL; it must not be handed to a different unmatched request."""
        out = web_tools._associate_extract_results(
            [A, B], [_doc(A + "/", A_TEXT), _stub(A)]
        )
        assert out[0]["content"] == A_TEXT
        assert out[1]["url"] == B and out[1]["error"]

    def test_superseded_stub_is_retired_for_later_tiers_too(self):
        """Once the real document has answered a URL, its stub must not be
        served to a scheme/www variant of that URL in the loose pass either."""
        page, variant = "https://example.com/p", "http://www.example.com/p"
        out = web_tools._associate_extract_results(
            [page, variant], [_doc(page, A_TEXT), _stub(page)]
        )
        assert out[0]["content"] == A_TEXT
        assert out[1]["url"] == variant and out[1]["error"]

    def test_metadata_source_url_pairs_when_url_differs(self):
        """Keenable puts the fetched URL in ``url`` and the requested one in
        ``metadata.sourceURL``; either side is enough to pair."""
        out = web_tools._associate_extract_results(
            [A, B],
            [_doc("https://cdn.example.net/p", A_TEXT, source_url=A), _doc(B, B_TEXT)],
        )
        assert [r["content"] for r in out] == [A_TEXT, B_TEXT]

    def test_match_key_ignores_only_non_identifying_differences(self):
        """Host case, a default port, a trailing slash and a fragment all
        address the same document; a different query string does not."""
        base = web_tools._extract_match_key("https://Example.COM/a")
        assert base == web_tools._extract_match_key("https://example.com:443/a/#frag")
        assert base != web_tools._extract_match_key("https://example.com/a?page=2")
        assert base != web_tools._extract_match_key("https://example.com/b")
        # Only the loose pass forgives the scheme and a leading "www.".
        assert base != web_tools._extract_match_key("http://www.example.com/a")
        assert web_tools._extract_match_key(
            "http://www.example.com/a", loose=True
        ) == web_tools._extract_match_key("https://example.com/a", loose=True)

    def test_match_key_is_empty_for_missing_urls(self):
        """A blank or non-string url yields no key, so it pairs with nothing."""
        assert web_tools._extract_match_key(None) == ""
        assert web_tools._extract_match_key("   ") == ""
