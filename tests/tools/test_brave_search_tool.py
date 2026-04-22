"""Tests for Brave Search native tools and Brave web dispatch."""

import json
import os
from unittest.mock import MagicMock, patch

import httpx


def _make_brave_http_error_response(payload, *, url, status_code=422):
    request = httpx.Request("GET", url)
    return httpx.Response(status_code, json=payload, request=request)


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

    def test_search_accepts_free_key_alias(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "web": {"results": [{"title": "Result 1", "url": "https://example.com/1", "description": "Snippet 1"}]}
        }

        with patch.dict(os.environ, {"BRAVE_FREE_API_KEY": "brave-free-test"}, clear=True), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_search

            result = json.loads(brave_search("python testing", count=1))

        assert result["success"] is True
        assert mock_get.call_args.kwargs["headers"]["X-Subscription-Token"] == "brave-free-test"

    def test_search_surfaces_structured_brave_http_errors(self):
        error_response = _make_brave_http_error_response(
            {
                "type": "ErrorResponse",
                "error": {
                    "code": "OPTION_NOT_IN_PLAN",
                    "detail": "This endpoint is not available on your current plan.",
                    "meta": {"component": "web-search"},
                },
            },
            url="https://api.search.brave.com/res/v1/web/search?q=python",
        )

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=error_response):
            from tools.brave_search_tool import brave_search

            result = json.loads(brave_search("python testing"))

        assert "OPTION_NOT_IN_PLAN" in result["error"]
        assert "current plan" in result["error"]
        assert result["brave_error"]["code"] == "OPTION_NOT_IN_PLAN"
        assert result["brave_error"]["status"] == 422
        assert result["brave_error"]["component"] == "web-search"

    def test_suggest_forwards_country_lang_and_rich(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "results": [
                {"query": "python testing"},
                {"query": "python testing tutorial"},
            ]
        }

        with patch.dict(os.environ, {"BRAVE_AUTOSUGGEST_API_KEY": "autosuggest-test", "BRAVE_SEARCH_API_KEY": "search-test"}), \
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
        assert call.kwargs["headers"]["X-Subscription-Token"] == "autosuggest-test"
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

        with patch.dict(os.environ, {"BRAVE_AUTOSUGGEST_API_KEY": "autosuggest-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response):
            from tools.brave_search_tool import brave_suggest

            result = json.loads(brave_suggest("python te", count=2, country="US", lang="en", rich=True))

        assert result["success"] is True
        assert result["data"]["suggestions"] == [
            {"query": "python testing", "metadata": {"source": "brave", "score": 0.93}},
            {"query": "python testing tutorial", "metadata": {"source": "brave", "score": 0.88}},
        ]

    def test_suggest_surfaces_structured_brave_http_errors(self):
        error_response = _make_brave_http_error_response(
            {
                "type": "ErrorResponse",
                "error": {
                    "code": "OPTION_NOT_IN_PLAN",
                    "detail": "Suggest is not enabled for this subscription.",
                    "meta": {"component": "suggest"},
                },
            },
            url="https://api.search.brave.com/res/v1/suggest/search?q=python",
        )

        with patch.dict(os.environ, {"BRAVE_AUTOSUGGEST_API_KEY": "autosuggest-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=error_response):
            from tools.brave_search_tool import brave_suggest

            result = json.loads(brave_suggest("python te"))

        assert "OPTION_NOT_IN_PLAN" in result["error"]
        assert "subscription" in result["error"]
        assert result["brave_error"]["component"] == "suggest"

    def test_answers_accepts_chat_completions_surface(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [
                {"message": {"content": "Brave answer text."}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        with patch.dict(os.environ, {"BRAVE_ANSWERS_API_KEY": "answers-test", "BRAVE_SEARCH_API_KEY": "search-test"}), \
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
        assert call.kwargs["headers"]["X-Subscription-Token"] == "answers-test"
        assert call.kwargs["headers"]["Content-Type"] == "application/json"
        assert call.kwargs["headers"]["Accept-Encoding"] == "gzip"

    def test_answers_surfaces_structured_brave_http_errors(self):
        error_response = _make_brave_http_error_response(
            {
                "type": "ErrorResponse",
                "error": {
                    "code": "OPTION_NOT_IN_PLAN",
                    "detail": "Answers is not available on your current plan.",
                    "meta": {"component": "answers"},
                },
            },
            url="https://api.search.brave.com/res/v1/chat/completions",
        )

        with patch.dict(os.environ, {"BRAVE_ANSWERS_API_KEY": "answers-test"}), \
             patch("tools.brave_search_tool.httpx.post", return_value=error_response):
            from tools.brave_search_tool import brave_answers

            result = json.loads(brave_answers("What is Brave Search?"))

        assert "OPTION_NOT_IN_PLAN" in result["error"]
        assert "current plan" in result["error"]
        assert result["brave_error"]["component"] == "answers"

    def test_answers_falls_back_to_search_key_for_back_compat(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "choices": [{"message": {"content": "Fallback answer."}}],
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "search-test"}, clear=True), \
             patch("tools.brave_search_tool.httpx.post", return_value=response) as mock_post:
            from tools.brave_search_tool import brave_answers

            result = json.loads(brave_answers("Fallback?"))

        assert result["success"] is True
        assert mock_post.call_args.kwargs["headers"]["X-Subscription-Token"] == "search-test"

    def test_answers_defaults_to_non_streaming_requests(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {
            "choices": [{"message": {"content": "Default stream false."}}],
        }

        with patch.dict(os.environ, {"BRAVE_ANSWERS_API_KEY": "answers-test"}, clear=True), \
             patch("tools.brave_search_tool.httpx.post", return_value=response) as mock_post:
            from tools.brave_search_tool import brave_answers

            result = json.loads(brave_answers("Default?"))

        assert result["success"] is True
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_answers_streaming_response_is_aggregated(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.headers = {"content-type": "text/event-stream; charset=utf-8"}
        response.text = (
            'data: {"choices":[{"delta":{"role":"assistant","content":"Brave "}}],"usage":null}\n\n'
            'data: {"choices":[{"delta":{"content":"Search"}}],"usage":null}\n\n'
            'data: {"choices":[{"delta":{"content":" works."}}],"usage":{"total_tokens":42}}\n\n'
            'data: [DONE]\n'
        )

        with patch.dict(os.environ, {"BRAVE_ANSWERS_API_KEY": "answers-test"}, clear=True), \
             patch("tools.brave_search_tool.httpx.post", return_value=response):
            from tools.brave_search_tool import brave_answers

            result = json.loads(brave_answers("What is Brave Search?", stream=True))

        assert result["success"] is True
        assert result["data"]["answer"] == "Brave Search works."
        assert result["data"]["usage"] == {"total_tokens": 42}

    def test_news_dispatch_forwards_documented_params(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "type": "news",
            "query": {"original": "python news"},
            "results": [
                {"title": "News result", "url": "https://example.com/news"},
            ],
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.registry import registry

            result = json.loads(
                registry.dispatch(
                    "brave_news",
                    {
                        "query": "python news",
                        "count": 2,
                        "country": "US",
                        "search_lang": "en",
                        "ui_lang": "en-US",
                        "safesearch": "strict",
                        "offset": 1,
                        "spellcheck": False,
                        "freshness": "pw",
                        "extra_snippets": True,
                        "goggles": "news",
                        "include_fetch_metadata": True,
                        "operators": False,
                    },
                )
            )

        assert result["success"] is True
        assert result["data"]["query"] == {"original": "python news"}
        assert result["data"]["news"][0]["title"] == "News result"
        assert result["data"]["news"][0]["position"] == 2

        call = mock_get.call_args
        assert call.args[0].endswith("/news/search")
        assert call.kwargs["params"] == {
            "q": "python news",
            "count": 2,
            "country": "US",
            "search_lang": "en",
            "ui_lang": "en-US",
            "safesearch": "strict",
            "offset": 1,
            "spellcheck": False,
            "freshness": "pw",
            "extra_snippets": True,
            "goggles": "news",
            "include_fetch_metadata": True,
            "operators": False,
        }

    def test_images_preserve_extra_metadata(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "type": "images",
            "query": {"original": "mountain landscape"},
            "results": [
                {"title": "Image result", "url": "https://example.com/image", "thumbnail": {"src": "https://example.com/thumb.jpg"}},
            ],
            "extra": {"family_friendly": True},
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_images

            result = json.loads(
                brave_images("mountain landscape", count=3, country="US", search_lang="en", safesearch="strict", spellcheck=False)
            )

        assert result["success"] is True
        assert result["data"]["images"][0]["title"] == "Image result"
        assert result["data"]["images"][0]["thumbnail"] == {"src": "https://example.com/thumb.jpg"}
        assert result["data"]["extra"] == {"family_friendly": True}
        assert mock_get.call_args.kwargs["params"] == {
            "q": "mountain landscape",
            "count": 3,
            "country": "US",
            "search_lang": "en",
            "safesearch": "strict",
            "spellcheck": False,
        }

    def test_videos_forward_documented_params(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "type": "videos",
            "query": {"original": "python tutorial"},
            "results": [
                {"title": "Video result", "url": "https://example.com/video"},
            ],
            "extra": {"family_friendly": True},
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_videos

            result = json.loads(
                brave_videos(
                    "python tutorial",
                    count=2,
                    country="US",
                    search_lang="en",
                    ui_lang="en-US",
                    safesearch="moderate",
                    offset=1,
                    spellcheck=True,
                    freshness="pm",
                    include_fetch_metadata=True,
                    operators=False,
                )
            )

        assert result["success"] is True
        assert result["data"]["videos"][0]["position"] == 2
        assert result["data"]["extra"] == {"family_friendly": True}
        assert mock_get.call_args.kwargs["params"] == {
            "q": "python tutorial",
            "count": 2,
            "country": "US",
            "search_lang": "en",
            "ui_lang": "en-US",
            "safesearch": "moderate",
            "offset": 1,
            "spellcheck": True,
            "freshness": "pm",
            "include_fetch_metadata": True,
            "operators": False,
        }

    def test_local_pois_forwards_ids_and_optional_params(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "type": "local_pois",
            "results": [
                {"id": "loc-1", "name": "Cafe 1"},
            ],
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_local_pois

            result = json.loads(brave_local_pois(["loc-1", "loc-2"], search_lang="en", ui_lang="en-US", units="metric"))

        assert result["success"] is True
        assert result["data"]["pois"][0]["id"] == "loc-1"
        assert mock_get.call_args.kwargs["params"] == {
            "ids": ["loc-1", "loc-2"],
            "search_lang": "en",
            "ui_lang": "en-US",
            "units": "metric",
        }

    def test_local_descriptions_accepts_ids_and_preserves_results(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "type": "local_descriptions",
            "results": [
                {"id": "loc-1", "description": "A cozy cafe."},
            ],
        }

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "brave-test"}), \
             patch("tools.brave_search_tool.httpx.get", return_value=response) as mock_get:
            from tools.brave_search_tool import brave_local_descriptions

            result = json.loads(brave_local_descriptions(["loc-1"]))

        assert result["success"] is True
        assert result["data"]["descriptions"][0]["description"] == "A cozy cafe."
        assert mock_get.call_args.args[0].endswith("/local/descriptions")
        assert mock_get.call_args.kwargs["params"] == {"ids": ["loc-1"]}


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
