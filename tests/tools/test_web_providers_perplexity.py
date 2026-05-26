"""Tests for the Perplexity Search web provider."""

from __future__ import annotations

from unittest.mock import MagicMock

from plugins.web.perplexity.provider import (
    PerplexityWebSearchProvider,
    _normalize_perplexity_search_results,
)


def test_perplexity_normalizes_search_results() -> None:
    result = _normalize_perplexity_search_results(
        {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "snippet": "Example snippet",
                    "date": "2026-01-01",
                }
            ]
        }
    )

    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "description": "Example snippet",
                    "position": 1,
                }
            ]
        },
    }


def test_perplexity_search_posts_to_search_api(monkeypatch) -> None:
    response = MagicMock()
    response.json.return_value = {"results": []}
    response.raise_for_status.return_value = None
    post = MagicMock(return_value=response)

    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test")
    monkeypatch.setattr("plugins.web.perplexity.provider.httpx.post", post)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    provider = PerplexityWebSearchProvider()
    result = provider.search("docs", limit=500)

    assert result == {"success": True, "data": {"web": []}}
    post.assert_called_once()
    kwargs = post.call_args.kwargs
    assert kwargs["json"] == {"query": "docs", "max_results": 20}
    assert kwargs["headers"]["Authorization"] == "Bearer pplx-test"


def test_perplexity_search_returns_error_without_key(monkeypatch) -> None:
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    provider = PerplexityWebSearchProvider()
    result = provider.search("docs", limit=5)

    assert result["success"] is False
    assert "PERPLEXITY_API_KEY" in result["error"]
