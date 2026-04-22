"""Tests for Brave Search native tools and Brave web dispatch."""

import json
import os
from unittest.mock import MagicMock, patch


class TestBraveSearch:
    def test_search_sends_documented_headers_and_params(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "description": "Snippet 1",
                    },
                    {
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "snippet": "Snippet 2",
                    },
                ]
            }
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_search

            result = json.loads(
                brave_search(
                    "python testing",
                    count=2,
                    country="us",
                    freshness="pw",
                    extra_snippets=True,
                    summary=True,
                )
            )

        assert result["success"] is True
        assert len(result["data"]["web"]) == 2
        assert result["data"]["web"][0]["title"] == "Result 1"
        assert result["data"]["web"][0]["position"] == 1

        call = mock_get.call_args
        assert call.args[0].endswith("/web/search")
        assert call.kwargs["params"] == {
            "q": "python testing",
            "count": 2,
            "country": "us",
            "freshness": "pw",
            "extra_snippets": True,
            "summary": True,
        }
        assert call.kwargs["headers"]["X-Subscription-Token"] == "brave-test"
        assert call.kwargs["headers"]["Accept"] == "application/json"
        assert call.kwargs["headers"]["Accept-Encoding"] == "gzip"

    def test_search_forwards_documented_brave_query_params(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "description": "Snippet 1",
                    }
                ]
            }
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.registry import registry

            result = json.loads(
                registry.dispatch(
                    "brave_search",
                    {
                        "query": "python testing",
                        "count": 2,
                        "country": "us",
                        "freshness": "pw",
                        "extra_snippets": True,
                        "summary": True,
                        "search_lang": "fr",
                        "ui_lang": "en-US",
                        "safesearch": "strict",
                        "offset": 20,
                        "text_decorations": False,
                        "spellcheck": True,
                        "result_filter": "web",
                        "goggles": "news",
                        "units": "metric",
                    },
                )
            )

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Result 1"

        call = mock_get.call_args
        assert call.args[0].endswith("/web/search")
        params = call.kwargs["params"]
        assert {
            "q": params["q"],
            "count": params["count"],
            "country": params["country"],
            "freshness": params["freshness"],
            "extra_snippets": params["extra_snippets"],
            "summary": params["summary"],
        } == {
            "q": "python testing",
            "count": 2,
            "country": "us",
            "freshness": "pw",
            "extra_snippets": True,
            "summary": True,
        }
        missing = [
            key for key in (
                "search_lang",
                "ui_lang",
                "safesearch",
                "offset",
                "text_decorations",
                "spellcheck",
                "result_filter",
                "goggles",
                "units",
            )
            if key not in params
        ]
        assert not missing, f"Brave search request is missing documented params: {missing}"

    def test_search_preserves_brave_rich_response_sections(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "description": "Snippet 1",
                    }
                ]
            },
            "query": {"original": "python testing", "display": "python testing"},
            "news": [
                {"title": "News result", "url": "https://example.com/news"}
            ],
            "videos": [
                {"title": "Video result", "url": "https://example.com/video"}
            ],
            "summarizer": {"summary": "Summary text"},
            "rich": {"title": "Rich result", "url": "https://example.com/rich"},
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response):
            from tools.brave_search_tool import brave_search

            result = json.loads(brave_search("python testing", count=1))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Result 1"
        assert result["data"]["query"] == {"original": "python testing", "display": "python testing"}
        assert result["data"]["news"] == [{"title": "News result", "url": "https://example.com/news"}]
        assert result["data"]["videos"] == [{"title": "Video result", "url": "https://example.com/video"}]
        assert result["data"]["summarizer"] == {"summary": "Summary text"}
        assert result["data"]["rich"] == {"title": "Rich result", "url": "https://example.com/rich"}

    def test_suggest_forwards_country_lang_and_rich(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "results": [
                {"query": "python testing"},
                {"query": "python testing tutorial"},
            ]
        }

        with patch.dict(os.environ, {"BRAVE_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_suggest

            result = json.loads(
                brave_suggest(
                    "python te",
                    count=2,
                    country="US",
                    lang="en",
                    rich=True,
                )
            )

        assert result["success"] is True
        assert result["data"]["query"] == "python te"
        assert result["data"]["suggestions"] == ["python testing", "python testing tutorial"]

        call = mock_get.call_args
        assert call.args[0].endswith("/suggest/search")
        assert call.kwargs["params"] == {
            "q": "python te",
            "count": 2,
            "country": "US",
            "lang": "en",
            "rich": True,
        }
        assert call.kwargs["headers"]["X-Subscription-Token"] == "brave-test"
        assert call.kwargs["headers"]["Accept-Encoding"] == "gzip"

    def test_suggest_preserves_structured_metadata_from_brave(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "query": "python testing",
                    "metadata": {"source": "brave", "score": 0.93},
                },
                {
                    "query": "python testing tutorial",
                    "metadata": {"source": "brave", "score": 0.88},
                },
            ]
        }

        with patch.dict(os.environ, {"BRAVE_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response):
            from tools.brave_search_tool import brave_suggest

            result = json.loads(brave_suggest("python te", count=2, country="US", lang="en", rich=True))

        assert result["success"] is True
        assert result["data"]["suggestions"] == [
            {"query": "python testing", "metadata": {"source": "brave", "score": 0.93}},
            {"query": "python testing tutorial", "metadata": {"source": "brave", "score": 0.88}},
        ]

    def test_answers_accepts_chat_completions_surface(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [
                {"message": {"content": "Brave answer text."}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.post", return_value=response) as mock_post:
            from tools.registry import registry

            result = json.loads(
                registry.dispatch(
                    "brave_answers",
                    {
                        "query": "What is Brave Search?",
                        "messages": [
                            {"role": "system", "content": "Answer directly."},
                            {"role": "user", "content": "What is Brave Search?"},
                        ],
                        "model": "brave-pro",
                        "stream": False,
                        "country": "US",
                        "language": "en",
                        "enable_entities": True,
                        "enable_citations": True,
                    },
                )
            )

        assert result["success"] is True
        assert result["data"]["query"] == "What is Brave Search?"
        assert result["data"]["model"] == "brave-pro"
        assert result["data"]["answer"] == "Brave answer text."
        assert result["data"]["usage"]["total_tokens"] == 30

        call = mock_post.call_args
        assert call.args[0].endswith("/chat/completions")
        assert call.kwargs["json"] == {
            "model": "brave-pro",
            "messages": [
                {"role": "system", "content": "Answer directly."},
                {"role": "user", "content": "What is Brave Search?"},
            ],
            "stream": False,
            "country": "US",
            "language": "en",
            "enable_entities": True,
            "enable_citations": True,
        }
        assert call.kwargs["headers"]["X-Subscription-Token"] == "brave-test"
        assert call.kwargs["headers"]["Content-Type"] == "application/json"
        assert call.kwargs["headers"]["Accept-Encoding"] == "gzip"


class TestBraveWebDispatch:
    def test_web_search_tool_dispatches_to_brave(self):
        result_payload = {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": "Brave result",
                        "url": "https://example.com",
                        "description": "desc",
                        "position": 1,
                    }
                ]
            },
        }

        with patch("tools.web_tools._get_backend", return_value="brave"), \
             patch("tools.brave_search_tool.brave_search", return_value=json.dumps(result_payload)) as mock_brave_search, \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("brave query", limit=1))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Brave result"
        mock_brave_search.assert_called_once_with("brave query", count=1)
