"""Unit tests for tools.xai_web_search_tool."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
import requests

from tools import xai_web_search_tool
from tools.xai_web_search_tool import (
    WEB_SEARCH_SCHEMA,
    _extract_inline_citations,
    _extract_response_text,
    _normalize_websites,
    check_web_search_requirements,
    web_search_tool,
)


# ---------------------------------------------------------------------------
# Fake requests.post
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self  # type: ignore[attr-defined]
            raise err


class _FakePost:
    def __init__(self, *responses: _FakeResponse):
        self.queue = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, url: str, *, headers, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self.queue:
            raise AssertionError("ran out of mocked responses")
        return self.queue.pop(0)


@pytest.fixture
def fake_post(monkeypatch):
    holder: Dict[str, _FakePost] = {}

    def install(*responses: _FakeResponse) -> _FakePost:
        fp = _FakePost(*responses)
        holder["fp"] = fp
        monkeypatch.setattr(xai_web_search_tool.requests, "post", fp)
        monkeypatch.setattr(xai_web_search_tool.time, "sleep", lambda _s: None)
        return fp

    return install


@pytest.fixture(autouse=True)
def _no_disk_config(monkeypatch):
    monkeypatch.setattr(xai_web_search_tool, "_load_web_search_config", lambda: {})


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "sk-test-web")


def _ok(answer: str = "Some answer", citations=None, inline=None) -> _FakeResponse:
    payload: Dict[str, Any] = {
        "id": "resp-1",
        "object": "response",
        "status": "completed",
        "output": [
            {"type": "message", "role": "assistant",
             "content": [
                 {"type": "output_text", "text": answer, "annotations": list(inline or [])},
             ]},
        ],
        "citations": list(citations or []),
    }
    return _FakeResponse(200, payload)


# ---------------------------------------------------------------------------
# Requirements / schema
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert check_web_search_requirements() is False

    def test_blank_key_unavailable(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "   ")
        assert check_web_search_requirements() is False

    def test_available_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk")
        assert check_web_search_requirements() is True


class TestSchema:
    def test_required_query(self):
        assert WEB_SEARCH_SCHEMA["parameters"]["required"] == ["query"]

    def test_advertised_optional_params(self):
        props = WEB_SEARCH_SCHEMA["parameters"]["properties"]
        for key in ("query", "allowed_websites", "excluded_websites",
                    "from_date", "to_date", "country"):
            assert key in props


# ---------------------------------------------------------------------------
# _normalize_websites
# ---------------------------------------------------------------------------

class TestNormalizeWebsites:
    def test_strips_protocol(self):
        assert _normalize_websites(["https://nytimes.com"], "x") == ["nytimes.com"]
        assert _normalize_websites(["http://example.com"], "x") == ["example.com"]

    def test_strips_www_prefix(self):
        assert _normalize_websites(["www.nytimes.com"], "x") == ["nytimes.com"]

    def test_strips_path(self):
        assert _normalize_websites(["nytimes.com/section/foo"], "x") == ["nytimes.com"]

    def test_strips_protocol_and_path(self):
        assert _normalize_websites(["https://www.nytimes.com/foo/bar"], "x") == ["nytimes.com"]

    def test_drops_empty(self):
        assert _normalize_websites(["", "  "], "x") == []

    def test_too_many_raises(self):
        with pytest.raises(ValueError):
            _normalize_websites([f"site{i}.com" for i in range(11)], "allowed_websites")

    def test_none_yields_empty(self):
        assert _normalize_websites(None, "x") == []


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

class TestArgValidation:
    def test_empty_query_returns_error(self, api_key):
        out = json.loads(web_search_tool(""))
        assert out["success"] is False
        assert "query" in out["error"].lower()

    def test_missing_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        out = json.loads(web_search_tool("hello"))
        assert out["success"] is False
        assert "XAI_API_KEY" in out["error"]

    def test_allowed_and_excluded_are_mutually_exclusive(self, api_key):
        out = json.loads(web_search_tool(
            "hello",
            allowed_websites=["nytimes.com"],
            excluded_websites=["bbc.co.uk"],
        ))
        assert out["success"] is False
        assert "cannot be used together" in out["error"]


# ---------------------------------------------------------------------------
# Body construction
# ---------------------------------------------------------------------------

class TestBodyConstruction:
    def test_basic_query_uses_default_model(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("latest AI news")
        body = fp.calls[0]["json"]
        assert body["model"] == "grok-4.3"
        assert body["input"][0]["content"] == "latest AI news"
        assert body["tools"] == [{"type": "web_search"}]
        assert body["store"] is False

    def test_endpoint_is_responses(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hello")
        assert fp.calls[0]["url"].endswith("/responses")

    def test_allowed_websites_threaded_into_tool(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi", allowed_websites=["https://www.nytimes.com", "bbc.co.uk"])
        tool = fp.calls[0]["json"]["tools"][0]
        assert tool["allowed_websites"] == ["nytimes.com", "bbc.co.uk"]

    def test_excluded_websites_threaded_into_tool(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi", excluded_websites=["pinterest.com"])
        tool = fp.calls[0]["json"]["tools"][0]
        assert tool["excluded_websites"] == ["pinterest.com"]

    def test_date_range_threaded(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi", from_date="2026-01-01", to_date="2026-05-01")
        tool = fp.calls[0]["json"]["tools"][0]
        assert tool["from_date"] == "2026-01-01"
        assert tool["to_date"] == "2026-05-01"

    def test_country_uppercased(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi", country="fr")
        tool = fp.calls[0]["json"]["tools"][0]
        assert tool["country"] == "FR"

    def test_minimal_tool_def_when_no_options(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi")
        tool = fp.calls[0]["json"]["tools"][0]
        assert tool == {"type": "web_search"}

    def test_headers(self, api_key, fake_post):
        fp = fake_post(_ok())
        web_search_tool("hi")
        h = fp.calls[0]["headers"]
        assert h["Authorization"] == "Bearer sk-test-web"
        assert h["Content-Type"] == "application/json"
        assert h["User-Agent"].startswith("Hermes-Agent/")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_returns_answer(self, api_key, fake_post):
        fake_post(_ok(answer="The answer is 42."))
        out = json.loads(web_search_tool("question"))
        assert out["success"] is True
        assert out["answer"] == "The answer is 42."

    def test_returns_top_level_citations(self, api_key, fake_post):
        cites = [{"url": "https://nytimes.com/x", "title": "X"}]
        fake_post(_ok(citations=cites))
        out = json.loads(web_search_tool("q"))
        assert out["citations"] == cites

    def test_returns_inline_citations(self, api_key, fake_post):
        inline = [
            {"type": "url_citation", "url": "https://a.com", "title": "A",
             "start_index": 0, "end_index": 5},
        ]
        fake_post(_ok(inline=inline))
        out = json.loads(web_search_tool("q"))
        assert out["inline_citations"] == [
            {"url": "https://a.com", "title": "A", "start_index": 0, "end_index": 5},
        ]

    def test_extract_text_uses_output_text_legacy(self):
        assert _extract_response_text({"output_text": "legacy"}) == "legacy"

    def test_extract_text_walks_message_output(self):
        payload = {
            "output": [
                {"type": "message",
                 "content": [
                     {"type": "output_text", "text": "Part1"},
                     {"type": "output_text", "text": "Part2"},
                 ]},
            ],
        }
        assert _extract_response_text(payload) == "Part1\n\nPart2"

    def test_extract_inline_citations_skips_non_url_types(self):
        payload = {
            "output": [
                {"type": "message",
                 "content": [
                     {"type": "output_text", "text": "x", "annotations": [
                         {"type": "url_citation", "url": "https://a"},
                         {"type": "footnote", "url": "https://b"},
                     ]},
                 ]},
            ],
        }
        out = _extract_inline_citations(payload)
        assert len(out) == 1
        assert out[0]["url"] == "https://a"


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------

class TestHttpErrors:
    def test_401_returns_error_json(self, api_key, fake_post):
        fake_post(_FakeResponse(401, text='{"error":"bad key"}'))
        out = json.loads(web_search_tool("hi"))
        assert out["success"] is False
        assert "401" in out["error"]
        assert out["error_type"] == "HTTPError"

    def test_500_retries_then_succeeds(self, api_key, monkeypatch, fake_post):
        # First a 500, then a 200.
        fp = fake_post(_FakeResponse(500, text="oops"), _ok(answer="OK"))
        # Bump retries so the loop tries twice.
        monkeypatch.setattr(xai_web_search_tool, "_get_web_search_retries", lambda: 1)
        out = json.loads(web_search_tool("hi"))
        assert out["success"] is True
        assert out["answer"] == "OK"
        assert len(fp.calls) == 2

    def test_500_exhausts_retries(self, api_key, monkeypatch, fake_post):
        fake_post(_FakeResponse(500, text="boom"), _FakeResponse(500, text="boom"))
        monkeypatch.setattr(xai_web_search_tool, "_get_web_search_retries", lambda: 1)
        out = json.loads(web_search_tool("hi"))
        assert out["success"] is False
        assert "500" in out["error"]

    def test_4xx_does_not_retry(self, api_key, monkeypatch, fake_post):
        fp = fake_post(_FakeResponse(403, text="forbidden"))
        monkeypatch.setattr(xai_web_search_tool, "_get_web_search_retries", lambda: 5)
        out = json.loads(web_search_tool("hi"))
        assert out["success"] is False
        assert len(fp.calls) == 1  # no retry on 4xx
