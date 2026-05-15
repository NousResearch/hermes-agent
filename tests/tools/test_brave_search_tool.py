"""Tests for the advanced brave_search tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def _mock_resp(json_data):
    response = MagicMock()
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    return response


def test_schema_default_mode_is_both():
    from tools.brave_search_tool import BRAVE_SEARCH_SCHEMA

    assert BRAVE_SEARCH_SCHEMA["parameters"]["properties"]["mode"]["default"] == "both"


def test_missing_query_returns_error():
    from tools.brave_search_tool import brave_search_tool

    result = json.loads(brave_search_tool({"mode": "web"}))

    assert result["success"] is False
    assert "query" in result["error"]


def test_rejects_oversized_query():
    from tools.brave_search_tool import brave_search_tool

    result = json.loads(brave_search_tool({"query": "x" * 501}))

    assert result["success"] is False
    assert "500 characters" in result["error"]


def test_rejects_invalid_safesearch():
    from tools.brave_search_tool import brave_search_tool

    result = json.loads(brave_search_tool({"query": "Hermes", "safesearch": "wide-open"}))

    assert result["success"] is False
    assert "safesearch" in result["error"]


def test_string_false_does_not_enable_raw(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"web": {"results": []}})):
        result = json.loads(brave_search_tool({"query": "Hermes", "mode": "web", "raw": "false"}))

    assert result["success"] is True
    assert "raw_web" not in result["data"]


def test_missing_key_returns_error_without_secret(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)

    from tools.brave_search_tool import brave_search_tool

    result = json.loads(brave_search_tool({"query": "Hermes", "mode": "web"}))

    assert result["success"] is False
    assert "BRAVE_SEARCH_API_KEY" in result["error"]


def test_both_mode_combines_web_and_llm(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    responses = [
        _mock_resp({"web": {"results": [{"title": "Web", "url": "https://web", "description": "desc"}]}}),
        _mock_resp({"grounding": {"generic": [{"title": "Ground", "url": "https://ground", "snippets": ["snippet"]}]}}),
    ]

    with patch("plugins.web.brave_search.client.httpx.get", side_effect=responses):
        result = json.loads(brave_search_tool({"query": "Hermes", "limit": 1}))

    assert result["success"] is True
    assert result["mode"] == "both"
    assert result["data"]["web"][0]["title"] == "Web"
    assert result["data"]["llm_context"][0]["title"] == "Ground"
    assert "secret-value" not in json.dumps(result)


def test_both_mode_keeps_web_when_llm_context_fails(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    response = MagicMock(status_code=403)
    responses = [
        _mock_resp({"web": {"results": [{"title": "Web", "url": "https://web", "description": "desc"}]}}),
        __import__("httpx").HTTPStatusError("403", request=MagicMock(), response=response),
    ]

    with patch("plugins.web.brave_search.client.httpx.get", side_effect=responses):
        result = json.loads(brave_search_tool({"query": "Hermes", "limit": 1}))

    assert result["success"] is True
    assert result["data"]["web"][0]["title"] == "Web"
    assert result["data"]["errors"][0]["endpoint"] == "llm_context"


def test_raw_both_namespaces_raw_payloads(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    responses = [
        _mock_resp({"web": {"results": []}}),
        _mock_resp({"grounding": {"generic": []}}),
    ]

    with patch("plugins.web.brave_search.client.httpx.get", side_effect=responses):
        result = json.loads(brave_search_tool({"query": "Hermes", "mode": "raw"}))

    assert result["success"] is True
    assert "raw_web" in result["data"]
    assert "raw_llm_context" in result["data"]


def test_llm_mode_passes_locale_options(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return _mock_resp({"grounding": {"generic": []}})

    with patch("plugins.web.brave_search.client.httpx.get", side_effect=fake_get):
        result = json.loads(
            brave_search_tool(
                {
                    "query": "Hermes",
                    "mode": "llm",
                    "country": "US",
                    "search_lang": "en",
                    "safesearch": "strict",
                    "freshness": "pw",
                }
            )
        )

    assert result["success"] is True
    assert captured["url"].endswith("/llm/context")
    assert captured["params"]["country"] == "US"
    assert captured["params"]["freshness"] == "pw"


def test_images_mode_returns_images(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"title": "Image", "url": "https://img"}]})):
        result = json.loads(brave_search_tool({"query": "Hermes", "mode": "images"}))

    assert result["success"] is True
    assert result["data"]["images"][0]["title"] == "Image"


def test_suggest_mode_returns_suggestions(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp({"results": [{"query": "Hermes Agent"}]})):
        result = json.loads(brave_search_tool({"query": "Hermes", "mode": "suggest"}))

    assert result["success"] is True
    assert result["data"]["suggestions"] == ["Hermes Agent"]


def test_news_videos_and_discussions_modes(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.brave_search_tool import brave_search_tool

    cases = [
        ("news", {"results": [{"title": "News", "url": "https://news"}]}, "news"),
        ("videos", {"results": [{"title": "Video", "url": "https://video"}]}, "videos"),
        ("discussions", {"discussions": {"results": [{"title": "Discussion", "url": "https://discussion"}]}}, "discussions"),
    ]
    for mode, payload, key in cases:
        with patch("plugins.web.brave_search.client.httpx.get", return_value=_mock_resp(payload)):
            result = json.loads(brave_search_tool({"query": "Hermes", "mode": mode}))
        assert result["success"] is True
        assert result["data"][key][0]["title"]


def test_unsupported_mode_returns_error():
    from tools.brave_search_tool import brave_search_tool

    result = json.loads(brave_search_tool({"query": "Hermes", "mode": "shopping"}))

    assert result["success"] is False
    assert "Unsupported" in result["error"]


def test_tool_is_registered_when_discovered(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "secret-value")

    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()

    entry = registry.get_entry("brave_search")
    assert entry is not None
    assert entry.schema["parameters"]["properties"]["mode"]["default"] == "both"
