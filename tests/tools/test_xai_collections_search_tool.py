"""Tests for xAI Collections Search via the Responses API file_search tool."""

import json

import requests


class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Error")
            err.response = self
            raise err

    def json(self):
        return self._payload


def _fake_credentials(monkeypatch, *, provider="xai", api_key="xai-test-key"):
    monkeypatch.setattr(
        "tools.xai_collections_search_tool.resolve_xai_http_credentials",
        lambda: {
            "provider": provider,
            "api_key": api_key,
            "base_url": "https://api.x.ai/v1",
        },
    )


def _no_xai_env(monkeypatch):
    for var in ("XAI_API_KEY", "XAI_BASE_URL", "HERMES_XAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_xai_collections_search_posts_responses_request(monkeypatch):
    from hermes_cli import __version__
    from tools.xai_collections_search_tool import xai_collections_search_tool

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output_text": "The collection says Hermes supports tools.",
                "citations": [
                    {
                        "url": "collections://collection_123/files/file_123",
                    }
                ],
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "collections_search",
                            "arguments": '{"query":"Hermes tools","limit":7}',
                        },
                    }
                ],
                "server_side_tool_usage": {
                    "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": 1,
                },
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }
        )

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_collections_search_tool(
            query="What does the collection say about Hermes tools?",
            collection_ids=["collection_123", "collection_456"],
            max_num_results=7,
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer xai-test-key"
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{__version__}"
    assert captured["json"]["model"] == "grok-4.3"
    assert captured["json"]["store"] is False
    assert captured["json"]["input"][0]["content"] == (
        "What does the collection say about Hermes tools?"
    )
    assert captured["timeout"] == 180
    assert tool_def == {
        "type": "file_search",
        "vector_store_ids": ["collection_123", "collection_456"],
        "max_num_results": 7,
    }
    assert result["success"] is True
    assert result["tool"] == "xai_collections_search"
    assert result["xai_tool"] == "file_search"
    assert result["xai_sdk_tool"] == "collections_search"
    assert result["answer"] == "The collection says Hermes supports tools."
    assert result["citations"] == [{"url": "collections://collection_123/files/file_123"}]
    assert result["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "collections_search",
                "arguments": '{"query":"Hermes tools","limit":7}',
            },
        }
    ]
    assert result["server_side_tool_usage"] == {
        "SERVER_SIDE_TOOL_COLLECTIONS_SEARCH": 1,
    }
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 8}


def test_xai_collections_search_uses_config_collection_ids(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    monkeypatch.setattr(
        "tools.xai_collections_search_tool._load_xai_collections_search_config",
        lambda: {
            "collection_ids": ["collection_config"],
            "max_num_results": 3,
        },
    )
    _fake_credentials(monkeypatch)

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "Configured collection OK."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_collections_search_tool(query="Search configured docs"))

    assert result["success"] is True
    assert captured["tool_def"] == {
        "type": "file_search",
        "vector_store_ids": ["collection_config"],
        "max_num_results": 3,
    }


def test_xai_collections_search_accepts_single_collection_id_string(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    _fake_credentials(monkeypatch)

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["tool_def"] = json["tools"][0]
        return _FakeResponse({"output_text": "Single ID OK."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_collections_search_tool(
            query="Search docs",
            collection_ids="collection_single",
        )
    )

    assert result["success"] is True
    assert captured["tool_def"]["vector_store_ids"] == ["collection_single"]


def test_xai_collections_search_empty_query_returns_tool_error(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    _fake_credentials(monkeypatch)

    result = json.loads(xai_collections_search_tool(query="  ", collection_ids=["collection_1"]))

    assert result["error"] == "query is required for xai_collections_search"


def test_xai_collections_search_requires_collection_ids(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    _fake_credentials(monkeypatch)
    monkeypatch.setattr(
        "tools.xai_collections_search_tool._load_xai_collections_search_config",
        lambda: {},
    )

    result = json.loads(xai_collections_search_tool(query="Search docs"))

    assert "collection_ids is required for xai_collections_search" in result["error"]


def test_xai_collections_search_rejects_invalid_max_num_results(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    _fake_credentials(monkeypatch)

    result = json.loads(
        xai_collections_search_tool(
            query="Search docs",
            collection_ids=["collection_1"],
            max_num_results=0,
        )
    )

    assert result["success"] is False
    assert result["error"] == "max_num_results must be a positive integer"


def test_xai_collections_search_extracts_output_citations_and_calls(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "file_search_call",
                        "id": "fs_1",
                        "queries": ["Hermes tools"],
                        "results": [{"file_id": "file_1", "score": 0.91}],
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hermes has toolsets.",
                                "annotations": [
                                    {
                                        "type": "file_citation",
                                        "file_id": "file_1",
                                        "filename": "hermes.md",
                                        "start_index": 0,
                                        "end_index": 6,
                                    }
                                ],
                            }
                        ],
                    },
                ]
            }
        )

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_collections_search_tool(
            query="What does Hermes have?",
            collection_ids=["collection_1"],
        )
    )

    assert result["success"] is True
    assert result["answer"] == "Hermes has toolsets."
    assert result["inline_citations"] == [
        {
            "type": "file_citation",
            "url": "",
            "title": "",
            "file_id": "file_1",
            "filename": "hermes.md",
            "start_index": 0,
            "end_index": 6,
        }
    ]
    assert result["file_search_calls"] == [
        {
            "type": "file_search_call",
            "id": "fs_1",
            "queries": ["Hermes tools"],
            "results": [{"file_id": "file_1", "score": 0.91}],
        }
    ]


def test_xai_collections_search_returns_structured_http_error(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    class _FailingResponse:
        status_code = 403
        text = '{"code":"forbidden","error":"file_search is not enabled"}'

        def json(self):
            return {
                "code": "forbidden",
                "error": "file_search is not enabled",
            }

        def raise_for_status(self):
            err = requests.HTTPError("403 Client Error: Forbidden")
            err.response = self
            raise err

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(
        xai_collections_search_tool(query="Search docs", collection_ids=["collection_1"])
    )

    assert result["success"] is False
    assert result["provider"] == "xai"
    assert result["tool"] == "xai_collections_search"
    assert result["xai_tool"] == "file_search"
    assert result["error_type"] == "HTTPError"
    assert result["error"] == "forbidden: file_search is not enabled"


def test_xai_collections_search_http_error_supports_nested_error(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    class _FailingResponse:
        status_code = 400
        text = '{"error":{"message":"bad collection id"}}'

        def json(self):
            return {"error": {"message": "bad collection id"}}

        def raise_for_status(self):
            err = requests.HTTPError("400 Client Error: Bad Request")
            err.response = self
            raise err

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(
        xai_collections_search_tool(query="Search docs", collection_ids=["bad"])
    )

    assert result["success"] is False
    assert result["error"] == "bad collection id"


def test_xai_collections_search_retries_timeout_then_succeeds(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return _FakeResponse({"output_text": "Recovered after retry."})

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.xai_collections_search_tool.time.sleep", lambda *_: None)

    result = json.loads(
        xai_collections_search_tool(query="Search docs", collection_ids=["collection_1"])
    )

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after retry."


def test_xai_collections_search_retries_5xx_then_succeeds(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(
                {"code": "server_error", "error": "temporary failure"},
                status_code=500,
            )
        return _FakeResponse({"output_text": "Recovered after 5xx retry."})

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.xai_collections_search_tool.time.sleep", lambda *_: None)

    result = json.loads(
        xai_collections_search_tool(query="Search docs", collection_ids=["collection_1"])
    )

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after 5xx retry."


def test_xai_collections_search_uses_xai_oauth_when_available(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_collections_search_tool import (
        check_xai_collections_search_requirements,
        xai_collections_search_tool,
    )

    _no_xai_env(monkeypatch)
    _fake_credentials(monkeypatch, provider="xai-oauth", api_key="oauth-token")
    invalidate_check_fn_cache()

    assert check_xai_collections_search_requirements() is True

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "OAuth worked."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_collections_search_tool(query="Search docs", collection_ids=["collection_1"])
    )

    assert result["success"] is True
    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-token"


def test_xai_collections_search_returns_tool_error_when_no_credentials(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_collections_search_tool import (
        check_xai_collections_search_requirements,
        xai_collections_search_tool,
    )

    _no_xai_env(monkeypatch)
    _fake_credentials(monkeypatch, api_key="")
    invalidate_check_fn_cache()

    assert check_xai_collections_search_requirements() is False

    result = xai_collections_search_tool(query="Search docs", collection_ids=["collection_1"])

    assert "No xAI credentials available" in result
    assert "hermes auth add xai-oauth" in result


def test_xai_collections_search_honors_config_model_timeout_and_retries(monkeypatch):
    from tools.xai_collections_search_tool import xai_collections_search_tool

    monkeypatch.setattr(
        "tools.xai_collections_search_tool._load_xai_collections_search_config",
        lambda: {
            "model": "grok-custom-test",
            "timeout_seconds": 45,
            "retries": 0,
        },
    )
    _fake_credentials(monkeypatch)

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        return _FakeResponse({"output_text": "Custom config OK."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_collections_search_tool(
            query="Search docs",
            collection_ids=["collection_1"],
        )
    )

    assert result["success"] is True
    assert captured["model"] == "grok-custom-test"
    assert captured["timeout"] == 45


def test_xai_collections_search_registered_in_registry_with_check_fn():
    import tools.xai_collections_search_tool  # noqa: F401 - ensures registration runs
    from tools.registry import registry

    entry = registry.get_entry("xai_collections_search")
    assert entry is not None
    assert entry.toolset == "xai_collections_search"
    assert entry.check_fn is not None
    assert entry.check_fn.__name__ == "check_xai_collections_search_requirements"
    assert "XAI_API_KEY" in entry.requires_env
    assert entry.emoji == "🗂️"
