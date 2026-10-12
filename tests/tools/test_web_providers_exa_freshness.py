"""Exa web provider — keyed extract() must request fresh content.

Exa's ``/contents`` endpoint serves its index snapshot of a page unless told
otherwise; for a news front page that snapshot was observed 10 days stale.
``max_age_hours=0`` is Exa's documented "always fetch fresh" setting (the older
``livecrawl`` flag is deprecated in favour of it). These tests pin the contract
that the keyed extract path never lets Exa fall back to an aged snapshot, and
that the returned documents keep the shared ``document()`` shape.

The real provider is imported and only the SDK client is stubbed, so the
``_common`` glue and the extract body run together.
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


def _run_extract(urls, results):
    """Call the keyed extract path with a stubbed SDK client; return (docs, client)."""
    client = MagicMock()
    client.get_contents.return_value = MagicMock(results=results, statuses=[])
    with patch("plugins.web.exa.provider._get_exa_client", return_value=client), \
         patch("plugins.web.exa.provider.use_keyless", return_value=False):
        docs = ExaWebSearchProvider().extract(urls)
    return docs, client


class TestExaExtractRequestsFreshContent:
    def test_get_contents_is_asked_for_fresh_content(self) -> None:
        urls = ["https://news.example/", "https://blog.example/post"]
        _, client = _run_extract(urls, [_result(u) for u in urls])

        client.get_contents.assert_called_once()
        args, kwargs = client.get_contents.call_args
        assert list(args[0]) == urls
        assert kwargs.get("text") is True
        # 0 = always fetch fresh. Omitting it (or any positive value) lets Exa
        # serve a cached snapshot that can be days old.
        assert kwargs.get("max_age_hours") == 0

    def test_documents_keep_shared_shape(self) -> None:
        url = "https://news.example/"
        docs, _ = _run_extract([url], [_result(url, title="Front page", text="today's stories")])

        assert len(docs) == 1
        doc = docs[0]
        assert "error" not in doc
        assert doc["url"] == url
        assert doc["title"] == "Front page"
        assert doc["content"] == "today's stories"
        assert doc["raw_content"] == doc["content"]
        assert doc["metadata"] == {"sourceURL": url, "title": "Front page"}
