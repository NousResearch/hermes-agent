from __future__ import annotations

import asyncio

from plugins.web.firecrawl import provider as firecrawl_provider


def test_firecrawl_extract_400_uses_direct_http_fallback(monkeypatch):
    class ScrapeClient:
        def scrape(self, **kwargs):
            raise RuntimeError("400 Bad Request: upstream returned non-JSON body")

    class Response:
        url = "https://www.aviant.no/"
        headers = {"content-type": "text/html; charset=utf-8"}
        text = """
        <html>
          <head><title>Aviant</title><script>ignored()</script></head>
          <body><h1>Drone delivery in Norway</h1><p>Healthcare logistics.</p></body>
        </html>
        """

        def raise_for_status(self):
            return None

    class HTTPClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            assert url == "https://www.aviant.no/"
            return Response()

    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: ScrapeClient())
    monkeypatch.setattr(firecrawl_provider, "is_safe_url", lambda url: True)
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)

    import httpx

    monkeypatch.setattr(httpx, "Client", HTTPClient)

    result = asyncio.run(
        firecrawl_provider.FirecrawlWebSearchProvider().extract(
            ["https://www.aviant.no/"], format="markdown"
        )
    )

    assert result == [
        {
            "url": "https://www.aviant.no/",
            "title": "Aviant",
            "content": "Drone delivery in Norway\n\nHealthcare logistics.",
            "raw_content": "Drone delivery in Norway\n\nHealthcare logistics.",
            "metadata": {
                "sourceURL": "https://www.aviant.no/",
                "title": "Aviant",
                "fallback": "direct-http",
            },
        }
    ]


def test_firecrawl_extract_non_400_does_not_direct_fetch(monkeypatch):
    class ScrapeClient:
        def scrape(self, **kwargs):
            raise RuntimeError("503 Service Unavailable")

    def fail_client(*args, **kwargs):
        raise AssertionError("direct fallback should not run for non-400 errors")

    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: ScrapeClient())
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)

    import httpx

    monkeypatch.setattr(httpx, "Client", fail_client)

    result = asyncio.run(
        firecrawl_provider.FirecrawlWebSearchProvider().extract(
            ["https://www.aviant.no/"], format="markdown"
        )
    )

    assert result[0]["url"] == "https://www.aviant.no/"
    assert result[0]["error"] == "503 Service Unavailable"
