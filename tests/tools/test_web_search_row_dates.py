"""Search rows keep the dates the backend reported (``published_date`` / ``updated_date`` /
``crawl_date``).

web_search answers from a vendor's index, so a snippet is a copy taken at crawl time. When the
vendor says how old the page or the copy is, that date has to reach the model; when it says
nothing, the row keeps its four-key shape.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.web import keyless_mcp
from plugins.web._common import title_hit, web_hit


def _http_response(payload):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _rows(result):
    assert result["success"] is True, result
    return result["data"]["web"]


class TestRowShape:
    def test_row_without_dates_keeps_the_four_keys(self):
        assert list(web_hit("https://a.test", "A", "d", 1)) == ["url", "title", "description", "position"]
        assert list(title_hit("A", "https://a.test", "d", 1)) == ["title", "url", "description", "position"]

    def test_reported_dates_follow_position_and_blanks_are_dropped(self):
        row = web_hit("https://a.test", "A", "d", 1, published_date="2024-03-01", updated_date=" ", crawl_date=None)
        assert row == {"url": "https://a.test", "title": "A", "description": "d", "position": 1, "published_date": "2024-03-01"}
        assert list(row)[-1] == "published_date"


class TestKeyedVendors:
    def test_exa_rows_carry_published_and_crawl_dates(self, monkeypatch):
        from plugins.web.exa import provider as exa

        hit = SimpleNamespace(
            url="https://a.test", title="A", highlights=["h"],
            published_date="2023-11-16T01:36:32.547Z", crawl_date="2026-10-02T08:00:00.000Z",
        )
        client = MagicMock()
        client.search.return_value = SimpleNamespace(results=[hit])
        monkeypatch.setattr(exa, "use_keyless", lambda *_: False)
        monkeypatch.setattr(exa, "_get_exa_client", lambda: client)
        row = _rows(exa.ExaWebSearchProvider().search("q", 1))[0]
        assert row["published_date"] == "2023-11-16T01:36:32.547Z"
        assert row["crawl_date"] == "2026-10-02T08:00:00.000Z"

    def test_parallel_rows_carry_publish_date(self, monkeypatch):
        from plugins.web.parallel import provider as parallel

        client = MagicMock()
        client.beta.search.return_value = SimpleNamespace(results=[
            SimpleNamespace(url="https://a.test", title="A", excerpts=["e"], publish_date="2025-06-01"),
            SimpleNamespace(url="https://b.test", title="B", excerpts=["f"], publish_date=None),
        ])
        monkeypatch.setattr(parallel, "use_keyless", lambda *_: False)
        monkeypatch.setattr(parallel, "_get_sync_client", lambda: client)
        rows = _rows(parallel.ParallelWebSearchProvider().search("q", 2))
        assert rows[0]["published_date"] == "2025-06-01"
        assert "published_date" not in rows[1]

    def test_keenable_rows_carry_published_and_acquired_dates(self, monkeypatch):
        from plugins.web.keenable import provider as keenable

        payload = {"results": [{
            "url": "https://a.test", "title": "A", "snippet": "s",
            "published_at": "2026-08-17T19:09:14Z", "acquired_at": "2026-10-19T03:21:23Z",
        }]}
        monkeypatch.setattr(keenable, "provider_env", lambda _name: "k-test")
        monkeypatch.setattr(keenable, "use_keyless", lambda *_: False)
        with patch("requests.post", return_value=_http_response(payload)):
            row = _rows(keenable.KeenableWebSearchProvider().search("q", 1))[0]
        assert row["published_date"] == "2026-08-17T19:09:14Z"
        assert row["crawl_date"] == "2026-10-19T03:21:23Z"

    def test_brave_rows_carry_page_age_and_page_fetched(self, monkeypatch):
        from plugins.web.brave_free.provider import BraveFreeWebSearchProvider

        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "BSA-test")
        payload = {"web": {"results": [{
            "title": "A", "url": "https://a.test", "description": "d",
            "page_age": "2024-08-01T00:00:00", "page_fetched": "2026-11-01T10:00:00Z", "age": "3 days ago",
        }]}}
        with patch("httpx.get", return_value=_http_response(payload)):
            row = _rows(BraveFreeWebSearchProvider().search("q", 1))[0]
        assert row["published_date"] == "2024-08-01T00:00:00"
        assert row["crawl_date"] == "2026-11-01T10:00:00Z"

    def test_searxng_rows_carry_published_date(self, monkeypatch):
        from plugins.web.searxng.provider import SearXNGWebSearchProvider

        monkeypatch.setenv("SEARXNG_URL", "http://searx.test")
        payload = {"results": [
            {"title": "A", "url": "https://a.test", "content": "c", "publishedDate": "2025-02-03T00:00:00", "score": 2},
            {"title": "B", "url": "https://b.test", "content": "c", "publishedDate": None, "score": 1},
        ]}
        with patch("httpx.get", return_value=_http_response(payload)):
            rows = _rows(SearXNGWebSearchProvider().search("q", 2))
        assert rows[0]["published_date"] == "2025-02-03T00:00:00"
        assert "published_date" not in rows[1]

    def test_tavily_asks_for_and_keeps_published_date(self, monkeypatch):
        from plugins.web.tavily import provider as tavily

        sent = {}

        def _request(endpoint, payload, *, api_key=None):
            sent.update(payload)
            return {"results": [{"title": "A", "url": "https://a.test", "content": "c", "published_date": "2025-09-01"}]}

        monkeypatch.setattr(tavily, "_tavily_request", _request)
        monkeypatch.setattr(tavily, "_auth", lambda _action: ("tvly-test", None, ""))
        row = _rows(tavily.TavilyWebSearchProvider().search("q", 1))[0]
        assert sent["include_published_date"] is True
        assert row["published_date"] == "2025-09-01"


class TestKeylessRing:
    def test_exa_text_published_line_becomes_published_date(self):
        text = (
            "Title: Dated\nURL: https://a.test\nPublished: 2025-06-01T00:00:00.000Z\nAuthor: N/A\n"
            "Highlights:\nsome highlight\n"
            "\n---\n"
            "Title: Undated\nURL: https://b.test\nPublished: N/A\nHighlights:\nother\n"
        )
        rows = keyless_mcp._parse_exa_search_text(text, limit=5)
        assert rows[0]["published_date"] == "2025-06-01T00:00:00.000Z"
        assert "published_date" not in rows[1]

    def test_parallel_keyless_rows_carry_publish_date(self):
        payload = json.dumps({"results": [{"url": "https://a.test", "title": "A", "excerpts": ["x"], "publish_date": "2024-12-24"}]})
        with patch.object(keyless_mcp, "mcp_call", return_value=payload):
            row = _rows(keyless_mcp.parallel_search_keyless("q", limit=1))[0]
        assert row["published_date"] == "2024-12-24"

    def test_keenable_keyless_rows_carry_published_and_acquired_dates(self):
        data = {"results": [{"url": "https://a.test", "title": "A", "snippet": "s", "published_at": None, "acquired_at": "2026-10-17T02:39:02Z"}]}
        with patch.object(keyless_mcp, "_keenable_request", return_value=data):
            row = _rows(keyless_mcp.keenable_search_keyless("q", limit=1))[0]
        assert row["crawl_date"] == "2026-10-17T02:39:02Z"
        assert "published_date" not in row


@pytest.fixture
def perplexity_search_server(monkeypatch):
    """Local stand-in for the Perplexity Search API (``POST /search``) with one dated and one undated row."""
    rows = [
        {"title": "Dated", "url": "https://a.test", "snippet": "s", "date": "2024-03-01", "last_updated": "2025-10-30"},
        {"title": "Undated", "url": "https://b.test", "snippet": "s", "date": None, "last_updated": None},
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"results": rows}).encode()
            self.send_response(200 if self.path == "/direct/search" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PERPLEXITY_BASE_URL", f"http://127.0.0.1:{server.server_port}/direct")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dates_reach_the_model_through_web_search_tool(tmp_path, monkeypatch, perplexity_search_server):
    """Real config, real dispatch, memo cache and JSON serialization: the dates the vendor sent are in
    the tool result the model reads, and an undated row carries no date keys."""
    from hermes_cli.config import atomic_config_write
    from tests.tools.conftest import register_all_web_providers
    from tools import web_tools

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    atomic_config_write(home / "config.yaml", {"web": {"search_backend": "perplexity", "keyless_rescue": False}})
    register_all_web_providers()

    rows = _rows(json.loads(web_tools.web_search_tool("dated query", limit=2)))
    assert rows[0]["published_date"] == "2024-03-01"
    assert rows[0]["updated_date"] == "2025-10-30"
    assert not {"published_date", "updated_date", "crawl_date"} & set(rows[1])
