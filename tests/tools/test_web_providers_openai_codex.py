"""Tests for the OpenAI Codex OAuth web provider.

Covers:
- provider identity/capabilities and cheap availability checks
- streaming Responses API payload shape for hosted web_search
- parsing streamed output into web_search result rows
- URL extraction via the same hosted web_search tool's open_page action
- integration with tools.web_tools backend availability/dispatch helpers
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _sse_event(event_type: str, payload: dict) -> str:
    data = dict(payload)
    data.setdefault("type", event_type)
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class _MockStreamResponse:
    def __init__(self, *, status_code: int = 200, events: list[str] | None = None, text: str = ""):
        self.status_code = status_code
        self._events = events or []
        self._text = text
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        for event in self._events:
            for line in event.splitlines():
                yield line

    def read(self):
        return self._text.encode("utf-8")


def _completed_message(text: str) -> str:
    return _sse_event(
        "response.output_item.done",
        {
            "item": {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        },
    )


def _delta_text(text: str) -> str:
    return _sse_event("response.output_text.delta", {"delta": text})


class TestOpenAICodexProviderIdentity:
    def test_provider_name_and_capabilities(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.openai_codex.provider import OpenAICodexWebSearchProvider

        provider = OpenAICodexWebSearchProvider()

        assert isinstance(provider, WebSearchProvider)
        assert provider.name == "openai-codex"
        assert "OpenAI" in provider.display_name
        assert provider.supports_search() is True
        assert provider.supports_extract() is True

    def test_available_when_codex_oauth_token_is_present(self):
        from plugins.web.openai_codex import provider as codex_provider

        with patch.object(codex_provider, "_has_codex_oauth_token", return_value=True):
            assert codex_provider.OpenAICodexWebSearchProvider().is_available() is True

    def test_unavailable_without_codex_oauth_token(self):
        from plugins.web.openai_codex import provider as codex_provider

        with patch.object(codex_provider, "_has_codex_oauth_token", return_value=False):
            assert codex_provider.OpenAICodexWebSearchProvider().is_available() is False

    def test_is_available_does_not_read_or_refresh_access_token(self):
        from plugins.web.openai_codex import provider as codex_provider

        with patch.object(codex_provider, "_has_codex_oauth_token", return_value=True), \
             patch.object(
                 codex_provider,
                 "_read_codex_access_token",
                 side_effect=AssertionError("is_available must stay cheap"),
             ):
            assert codex_provider.OpenAICodexWebSearchProvider().is_available() is True


class TestOpenAICodexProviderSearch:
    def test_streaming_search_payload_and_result_parsing(self):
        from plugins.web.openai_codex import provider as codex_provider

        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Example Domain",
                        "url": "https://example.com/",
                        "description": "Documentation example domain.",
                    },
                    {"title": "skip missing URL", "description": "bad"},
                    {
                        "title": "IANA",
                        "url": "https://www.iana.org/domains/example",
                        "description": "IANA reserved domains.",
                    },
                ]
            }
        )
        stream = _MockStreamResponse(events=[_completed_message(body)])

        captured: dict = {}

        def fake_stream(method, url, *, headers, json, timeout):
            captured.update(
                {
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return stream

        with patch.object(codex_provider, "_read_codex_access_token", return_value="codex-token"), \
             patch.object(codex_provider, "_codex_cloudflare_headers", return_value={"CF-Access-Client-Id": "id"}), \
             patch.object(codex_provider, "_load_codex_web_config", return_value={"model": "gpt-5.5"}), \
             patch("httpx.stream", side_effect=fake_stream):
            result = codex_provider.OpenAICodexWebSearchProvider().search("example domain", limit=2)

        assert result == {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Example Domain",
                        "url": "https://example.com/",
                        "description": "Documentation example domain.",
                        "position": 1,
                    },
                    {
                        "title": "IANA",
                        "url": "https://www.iana.org/domains/example",
                        "description": "IANA reserved domains.",
                        "position": 2,
                    },
                ]
            },
        }
        assert captured["method"] == "POST"
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
        assert captured["headers"]["Authorization"] == "Bearer codex-token"
        assert captured["headers"]["Accept"] == "text/event-stream"
        payload = captured["json"]
        assert payload["model"] == "gpt-5.5"
        assert payload["store"] is False
        assert payload["stream"] is True
        assert payload["tools"] == [{"type": "web_search"}]
        assert "max_output_tokens" not in payload
        assert "results" in payload["instructions"]

    def test_search_reports_http_error_without_throwing(self):
        from plugins.web.openai_codex import provider as codex_provider

        stream = _MockStreamResponse(status_code=400, text='{"detail":"Unsupported parameter"}')

        with patch.object(codex_provider, "_read_codex_access_token", return_value="codex-token"), \
             patch.object(codex_provider, "_codex_cloudflare_headers", return_value={}), \
             patch.object(codex_provider, "_load_codex_web_config", return_value={}), \
             patch("httpx.stream", return_value=stream):
            result = codex_provider.OpenAICodexWebSearchProvider().search("x", limit=5)

        assert result["success"] is False
        assert "HTTP 400" in result["error"]
        assert "Unsupported parameter" in result["error"]


class TestOpenAICodexProviderExtract:
    def test_extract_opens_each_url_with_hosted_web_tool(self):
        from plugins.web.openai_codex import provider as codex_provider

        stream = _MockStreamResponse(
            events=[
                _sse_event(
                    "response.output_item.done",
                    {
                        "item": {
                            "type": "web_search_call",
                            "status": "completed",
                            "action": {"type": "open_page", "url": "https://example.com/"},
                        }
                    },
                ),
                _completed_message(
                    json.dumps(
                        {
                            "title": "Example Domain",
                            "content": "# Example Domain\n\nThis domain is used in examples.",
                        }
                    )
                ),
            ]
        )
        captured_prompts: list[str] = []

        def fake_stream(method, url, *, headers, json, timeout):
            captured_prompts.append(json["input"][0]["content"])
            return stream

        with patch.object(codex_provider, "_read_codex_access_token", return_value="codex-token"), \
             patch.object(codex_provider, "_codex_cloudflare_headers", return_value={}), \
             patch.object(codex_provider, "_load_codex_web_config", return_value={"model": "gpt-5.5"}), \
             patch("httpx.stream", side_effect=fake_stream):
            result = codex_provider.OpenAICodexWebSearchProvider().extract(["https://example.com/"], format="markdown")

        assert result == [
            {
                "url": "https://example.com/",
                "title": "Example Domain",
                "content": "# Example Domain\n\nThis domain is used in examples.",
                "raw_content": "# Example Domain\n\nThis domain is used in examples.",
                "metadata": {
                    "provider": "openai-codex",
                    "action": "open_page",
                    "format": "markdown",
                },
            }
        ]
        assert "Open https://example.com/" in captured_prompts[0]

    def test_extract_returns_per_url_error_when_no_content_is_returned(self):
        from plugins.web.openai_codex import provider as codex_provider

        stream = _MockStreamResponse(events=[_delta_text('{"title":"x","content":""}')])

        with patch.object(codex_provider, "_read_codex_access_token", return_value="codex-token"), \
             patch.object(codex_provider, "_codex_cloudflare_headers", return_value={}), \
             patch.object(codex_provider, "_load_codex_web_config", return_value={}), \
             patch("httpx.stream", return_value=stream):
            result = codex_provider.OpenAICodexWebSearchProvider().extract(["https://blocked.example/"])

        assert len(result) == 1
        assert result[0]["url"] == "https://blocked.example/"
        assert result[0]["content"] == ""
        assert "No content" in result[0]["error"]


class TestOpenAICodexWebToolsIntegration:
    def test_backend_availability_helper_recognizes_openai_codex(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_has_codex_oauth_token", lambda: True, raising=False)

        assert web_tools._is_backend_available("openai-codex") is True

    def test_backend_selection_accepts_openai_codex_config(self, monkeypatch):
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "firecrawl", "search_backend": "openai-codex"})
        monkeypatch.setattr(web_tools, "_is_backend_available", lambda backend: backend == "openai-codex")

        assert web_tools._get_search_backend() == "openai-codex"
