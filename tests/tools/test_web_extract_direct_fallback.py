import json

import pytest

from agent.web_search_provider import WebSearchProvider


class FailingExtractProvider(WebSearchProvider):
    @property
    def name(self):
        return "failing-extract"

    @property
    def display_name(self):
        return "Failing Extract"

    def is_available(self):
        return True

    def supports_search(self):
        return False

    def supports_extract(self):
        return True

    async def extract(self, urls, **kwargs):
        return [
            {"url": url, "title": "", "content": "", "raw_content": "", "error": "upstream 400"}
            for url in urls
        ]


def test_web_extract_uses_direct_http_fallback_when_provider_returns_only_failures(monkeypatch):
    import asyncio
    from agent.web_search_registry import register_provider, _reset_for_tests
    from tools import web_tools

    async def _allow_url(_url: str) -> bool:
        return True

    async def _fallback(urls, *, format=None, reason=""):
        assert urls == ["https://example.com/article"]
        assert "failing-extract" in reason
        return [
            {
                "url": urls[0],
                "title": "Fallback title",
                "content": "Fallback readable body",
                "raw_content": "Fallback readable body",
                "metadata": {"fallback": "direct-http"},
            }
        ]

    _reset_for_tests()
    register_provider(FailingExtractProvider())
    monkeypatch.setattr(web_tools, "_ensure_web_plugins_loaded", lambda: None)
    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "failing-extract")
    monkeypatch.setattr(web_tools, "async_is_safe_url", _allow_url)
    monkeypatch.setattr(web_tools, "_direct_http_extract_fallback", _fallback)
    monkeypatch.setattr(web_tools, "check_auxiliary_model", lambda: False)

    try:
        result = asyncio.run(
            web_tools.web_extract_tool(
                ["https://example.com/article"],
                use_llm_processing=False,
            )
        )
    finally:
        _reset_for_tests()

    payload = json.loads(result)
    assert payload["results"][0]["title"] == "Fallback title"
    assert payload["results"][0]["content"] == "Fallback readable body"
    assert payload["results"][0]["error"] is None


def test_html_text_payload_is_dependency_free_and_removes_scripts():
    from tools.web_tools import _html_text_payload

    payload = _html_text_payload(
        """
        <html><head><title>Example &amp; Title</title><script>bad()</script></head>
        <body><h1>Hello</h1><p>First paragraph.</p><style>.x{}</style><p>Second paragraph.</p></body></html>
        """
    )

    assert payload["title"] == "Example & Title"
    assert "bad()" not in payload["content"]
    assert "First paragraph." in payload["content"]
    assert "Second paragraph." in payload["content"]
