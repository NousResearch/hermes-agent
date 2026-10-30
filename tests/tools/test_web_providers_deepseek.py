"""Tests for the DeepSeek Search + Fetch web provider.

Covers:
- ``is_available()`` key gating
- ``search()`` — request payload, row normalization (``summary`` beats ``snippet``), HTTP errors
- ``extract()`` — per-URL success/failure entries, caller order preserved
- registration under the ``deepseek`` backend name
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(payload: dict | None = None, status_code: int = 200, text: str | None = None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.text = text if text is not None else json.dumps(payload or {})
    return response


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_SEARCH_BASE_URL", raising=False)


def _provider():
    from plugins.web.deepseek.provider import DeepSeekWebSearchProvider

    return DeepSeekWebSearchProvider()


class TestAvailability:
    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _provider()
        assert provider.is_available() is False
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        assert provider.is_available() is True

    def test_declares_both_capabilities(self) -> None:
        provider = _provider()
        assert provider.name == "deepseek"
        assert provider.supports_search() is True
        assert provider.supports_extract() is True


class TestSearch:
    _PAYLOAD = {
        "results": [
            {"title": "A", "url": "https://a.example", "snippet": "short A", "summary": "long A"},
            {"title": "B", "url": "https://b.example", "snippet": "short B", "summary": ""},
        ]
    }

    def test_normalizes_rows_with_summary_as_description(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with patch("plugins.web.deepseek.provider.httpx.post", return_value=_mock_response(self._PAYLOAD)):
            result = _provider().search("query", limit=5)

        assert result["success"] is True
        assert result["data"]["web"] == [
            {"title": "A", "url": "https://a.example", "description": "long A", "position": 1},
            {"title": "B", "url": "https://b.example", "description": "short B", "position": 2},
        ]

    def test_sends_count_and_bearer_auth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        captured = {}

        def fake_post(url, **kwargs):
            captured.update(url=url, **kwargs)
            return _mock_response({"results": []})

        with patch("plugins.web.deepseek.provider.httpx.post", side_effect=fake_post):
            _provider().search("q", limit=7)

        assert captured["url"].endswith("/v1/tools/search")
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        assert captured["json"] == {"query": "q", "count": 7, "need_content": False}

    def test_count_is_clamped_to_api_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        captured = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return _mock_response({"results": []})

        with patch("plugins.web.deepseek.provider.httpx.post", side_effect=fake_post):
            _provider().search("q", limit=500)

        assert captured["json"]["count"] == 50

    def test_http_error_surfaces_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        failure = _mock_response(status_code=402, text="Insufficient Balance")
        with patch("plugins.web.deepseek.provider.httpx.post", return_value=failure):
            result = _provider().search("q")

        assert result["success"] is False
        assert "Insufficient Balance" in result["error"]

    def test_missing_key_reports_credential_error(self) -> None:
        result = _provider().search("q")
        assert result["success"] is False
        assert "DEEPSEEK_API_KEY" in result["error"]


class TestExtract:
    def test_fetches_each_url_and_keeps_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        def fake_post(url, **kwargs):
            target = kwargs["json"]["url"]
            return _mock_response({"title": target, "content": f"body of {target}"})

        with patch("plugins.web.deepseek.provider.httpx.post", side_effect=fake_post):
            docs = _provider().extract(["https://one.example", "https://two.example"])

        assert [d["url"] for d in docs] == ["https://one.example", "https://two.example"]
        assert docs[0]["content"] == "body of https://one.example"
        assert docs[1]["raw_content"] == "body of https://two.example"
        assert all(d["metadata"]["sourceURL"] == d["url"] for d in docs)

    def test_one_failed_url_does_not_fail_the_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

        def fake_post(url, **kwargs):
            if kwargs["json"]["url"].endswith("bad"):
                return _mock_response(status_code=422, text="unsupported url")
            return _mock_response({"title": "ok", "content": "fine"})

        with patch("plugins.web.deepseek.provider.httpx.post", side_effect=fake_post):
            docs = _provider().extract(["https://good.example", "https://bad"])

        assert docs[0]["content"] == "fine"
        assert docs[1]["url"] == "https://bad"
        assert "unsupported url" in docs[1]["error"]

    def test_empty_content_becomes_error_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        with patch(
            "plugins.web.deepseek.provider.httpx.post",
            return_value=_mock_response({"title": "empty", "content": "   "}),
        ):
            docs = _provider().extract(["https://empty.example"])

        assert docs[0]["error"] == "DeepSeek fetch returned empty content"


class TestRegistration:
    def test_registered_as_deepseek_backend(self) -> None:
        from hermes_cli.plugins import _ensure_plugins_discovered
        from agent.web_search_registry import get_provider

        _ensure_plugins_discovered()
        provider = get_provider("deepseek")
        assert provider is not None
        assert provider.supports_search() and provider.supports_extract()
