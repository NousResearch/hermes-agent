"""Exa web provider — extract() surfaces per-URL failures from response.statuses.

Covers the 1-doc-per-URL extract contract when Exa's SDK reports dead URLs in
``response.statuses`` instead of ``response.results``:

- Every requested URL yields exactly one doc.
- A result URL maps to content (``document`` shape).
- A statused URL maps to an ``error`` doc (``page_error`` shape).
- A URL absent from both results and statuses still yields a generic error doc
  (never a silent drop).

Per the dev skill, these tests import the real provider and stub only the SDK
client, so the ABC / _common glue and the extract body stay exercised together.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from plugins.web.exa.provider import ExaWebSearchProvider


def _result(url: str, title: str = "T", text: str = "body") -> MagicMock:
    r = MagicMock()
    r.url = url
    r.title = title
    r.text = text
    return r


def _status(url: str, status: str = "CRAWL_NOT_FOUND") -> MagicMock:
    s = MagicMock()
    s.id = url
    s.status = status
    return s


def _run_extract(provider: ExaWebSearchProvider, urls, results, statuses):
    """Call provider.extract with a stubbed SDK response, bypassing keyless."""
    client = MagicMock()
    client.get_contents.return_value = MagicMock(results=results, statuses=statuses)
    with patch("plugins.web.exa.provider._get_exa_client", return_value=client), \
         patch("plugins.web.exa.provider.use_keyless", return_value=False):
        return provider.extract(urls)


class TestExaExtractPerUrlContract:
    def test_all_live_urls_return_content_docs(self) -> None:
        urls = ["https://a.example", "https://b.example"]
        docs = _run_extract(ExaWebSearchProvider(), urls, [_result(urls[0]), _result(urls[1])], [])
        assert len(docs) == len(urls)
        assert all("error" not in d for d in docs)
        assert {d["url"] for d in docs} == set(urls)
        assert docs[0]["content"] == "body"

    def test_dead_url_becomes_error_doc_not_empty_list(self) -> None:
        urls = ["https://dead.example"]
        docs = _run_extract(ExaWebSearchProvider(), urls, [], [_status(urls[0])])
        assert len(docs) == 1
        assert docs[0]["url"] == urls[0]
        assert docs[0]["error"] == "Exa fetch failed (CRAWL_NOT_FOUND)"

    def test_mixed_batch_yields_one_doc_per_url(self) -> None:
        live, dead = "https://live.example", "https://dead.example"
        urls = [live, dead]
        docs = _run_extract(ExaWebSearchProvider(), urls, [_result(live, text="ok body")], [_status(dead)])
        assert len(docs) == 2
        by_url = {d["url"]: d for d in docs}
        assert "error" not in by_url[live]
        assert by_url[live]["content"] == "ok body"
        assert by_url[dead]["error"] == "Exa fetch failed (CRAWL_NOT_FOUND)"

    def test_url_absent_from_both_still_gets_generic_error_doc(self) -> None:
        # Regression for review point 1: the backfill iterates *requested URLs*,
        # so a URL missing from results AND statuses is reported, never dropped.
        urls = ["https://missing.example"]
        docs = _run_extract(ExaWebSearchProvider(), urls, [], [])
        assert len(docs) == 1
        assert docs[0]["url"] == urls[0]
        assert docs[0]["error"] == f"Exa could not fetch {urls[0]}"

    def test_status_id_normalized_away_still_reports_generic_error(self) -> None:
        # Redirect-normalized URL on the status id (review point 2): benign —
        # the doc is still an error, never silence.
        urls = ["https://example.com/page"]
        docs = _run_extract(ExaWebSearchProvider(), urls, [], [_status("https://example.com/page/")])
        assert len(docs) == 1
        assert docs[0]["error"] == f"Exa could not fetch {urls[0]}"
