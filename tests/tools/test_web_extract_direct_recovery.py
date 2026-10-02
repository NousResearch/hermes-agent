"""Self-hosted scraper outage recovery through real web_extract dispatch/HTTP."""
import asyncio
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import requests
from requests.exceptions import ConnectionError, SSLError

from plugins.web.firecrawl import provider as firecrawl
from tools import web_tools as wt
from tools import web_tools_extract as extract


@pytest.fixture
def pages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("security:\n  allow_private_urls: true\n")
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            hits.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/blocked")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if self.path == "/challenge":
                self.wfile.write(b"<title>Just a moment</title><main>Verify you are human " + b"Please wait. " * 20 + b"</main>")
                return
            if self.path == "/nested":
                self.wfile.write(b"<div>" * 257 + b"Public content. " * 20 + b"</div>" * 257)
                return
            self.wfile.write(b"<html><head><title>Public article</title></head><body><nav>Menu</nav><main><h1>Onboarding</h1><p>" + b"Public help article details. " * 10 + b"</p></main><script>bad()</script></body></html>")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(firecrawl, "_env", lambda key: origin if key == "FIRECRAWL_API_URL" else "")
    monkeypatch.setattr(firecrawl, "_use_keyless_ring", lambda: False)
    monkeypatch.setattr(wt, "_firecrawl_client_config", ("direct", origin, None), raising=False)
    provider = firecrawl.FirecrawlWebSearchProvider()
    monkeypatch.setattr(wt, "_get_extract_backend", lambda: "firecrawl")
    monkeypatch.setattr(wt, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(wt, "_resolve_extract_provider", lambda backend: (provider, None))
    monkeypatch.setattr(extract, "_rescue_eligible", lambda provider: False)
    yield origin, hits
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_scraper_connection_failure_recovers_article_without_sticky_cache(pages, monkeypatch):
    origin, hits = pages
    attempts = []
    # Hold a bound, non-listening endpoint so the request really reaches the
    # refused TCP connection boundary without racing another test's listener.
    backend = socket.socket()
    backend.bind(("127.0.0.1", 0))
    backend_url = f"http://127.0.0.1:{backend.getsockname()[1]}/v2/scrape"

    class Scraper:
        def scrape(self, **kwargs):
            attempts.append(kwargs["url"])
            requests.post(backend_url, json=kwargs, timeout=1)

    monkeypatch.setattr(firecrawl, "_get_firecrawl_client", Scraper)
    url = origin + "/article"
    try:
        for format in ("markdown", "markdown", "html"):
            result = json.loads(asyncio.run(wt.web_extract_tool([url], format=format)))["results"][0]
            assert "Public help article details" in result["content"]
            if format == "markdown":
                assert "bad()" not in result["content"] and "Menu" not in result["content"]
            else:
                assert "<main>" in result["content"]
            assert result["metadata"]["extraction_method"] == "direct_html"
            assert result["metadata"]["sourceURL"] == url
            assert "/v2/scrape" in result["metadata"]["backend_error"]
    finally:
        backend.close()
    assert attempts == [url] * 3 and hits == ["/article"] * 3


def test_http_refusal_and_redirect_policy_do_not_fetch_blocked_content(pages, monkeypatch):
    origin, hits = pages

    class Scraper:
        def scrape(self, **kwargs):
            if kwargs["url"].endswith("/refused"):
                response = httpx.Response(401, request=httpx.Request("POST", origin))
                raise httpx.HTTPStatusError("Unauthorized", request=response.request, response=response)
            if kwargs["url"].endswith("/tls"):
                raise SSLError("certificate validation failed")
            raise ConnectionError("self-hosted scraper connection refused")

    monkeypatch.setattr(firecrawl, "_get_firecrawl_client", Scraper)
    monkeypatch.setattr("plugins.web.firecrawl.direct_fetch.check_website_access",
                        lambda url: {"message": "Blocked by website policy"} if url.endswith("/blocked") else None)
    results = json.loads(asyncio.run(wt.web_extract_tool([origin + path for path in
                        ("/refused", "/redirect", "/challenge", "/tls")])))["results"]
    assert all(result.get("error") and not result["content"] for result in results)
    for route in (("tool-gateway", origin, "token"), ("direct", "https://api.firecrawl.dev", "key")):
        monkeypatch.setattr(wt, "_firecrawl_client_config", route)
        result = json.loads(asyncio.run(wt.web_extract_tool([origin + "/managed-or-cloud"])))["results"][0]
        assert result.get("error") and not result["content"]
    assert hits == ["/redirect", "/challenge"]


def test_deeply_nested_html_is_refused(pages, monkeypatch):
    origin, hits = pages

    class Scraper:
        def scrape(self, **kwargs):
            raise ConnectionError("self-hosted scraper connection refused")

    monkeypatch.setattr(firecrawl, "_get_firecrawl_client", Scraper)
    result = json.loads(asyncio.run(wt.web_extract_tool([origin + "/nested"])))["results"][0]
    assert result.get("error") and not result["content"]
    assert hits == ["/nested"]
