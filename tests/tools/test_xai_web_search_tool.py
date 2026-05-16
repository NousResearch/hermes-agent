"""Tests for the xAI Web Search tool backed by Responses API."""

import json

import requests


class _FakeResponse:
    def __init__(self, payload, *, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code} Client Error")
            err.response = self
            raise err

    def json(self):
        return self._payload


def _no_xai_env(monkeypatch):
    for var in ("XAI_API_KEY", "XAI_BASE_URL", "HERMES_XAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_xai_web_search_posts_responses_request(monkeypatch):
    from hermes_cli import __version__
    from tools.xai_web_search_tool import xai_web_search_tool

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output_text": "xAI Web Search found current documentation.",
                "citations": [{"url": "https://docs.x.ai/", "title": "xAI Docs"}],
                "server_side_tool_usage": {"SERVER_SIDE_TOOL_WEB_SEARCH": 1},
            }
        )

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_web_search_tool(
            query="What does xAI Web Search support?",
            allowed_domains=["https://docs.x.ai/developers/tools/web-search"],
            enable_image_understanding=True,
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{__version__}"
    assert captured["json"]["model"] == "grok-4.3"
    assert captured["json"]["store"] is False
    assert tool_def["type"] == "web_search"
    assert tool_def["filters"]["allowed_domains"] == ["docs.x.ai"]
    assert tool_def["enable_image_understanding"] is True
    assert result["success"] is True
    assert result["tool"] == "xai_web_search"
    assert result["xai_tool"] == "web_search"
    assert result["answer"] == "xAI Web Search found current documentation."
    assert result["server_side_tool_usage"] == {"SERVER_SIDE_TOOL_WEB_SEARCH": 1}


def test_xai_web_search_rejects_conflicting_domain_filters(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    result = json.loads(
        xai_web_search_tool(
            query="latest xAI docs",
            allowed_domains=["docs.x.ai"],
            excluded_domains=["x.com"],
        )
    )

    assert result["error"] == "allowed_domains and excluded_domains cannot be used together"


def test_xai_web_search_empty_query_returns_tool_error(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    result = json.loads(xai_web_search_tool(query="  "))

    assert result["error"] == "query is required for xai_web_search"


def test_xai_web_search_rejects_too_many_domains(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")

    result = json.loads(
        xai_web_search_tool(
            query="latest docs",
            allowed_domains=[
                "a.example",
                "b.example",
                "c.example",
                "d.example",
                "e.example",
                "f.example",
            ],
        )
    )

    assert result["success"] is False
    assert result["error"] == "allowed_domains supports at most 5 domains"
    assert result["error_type"] == "ValueError"


def test_xai_web_search_excluded_domains_payload_and_default_image_flag(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse({"output_text": "Excluded domain search OK."})

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_web_search_tool(
            query="latest xAI docs",
            excluded_domains=["https://docs.x.ai:443/path", ".example.com"],
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert result["success"] is True
    assert tool_def["filters"]["excluded_domains"] == ["docs.x.ai", "example.com"]
    assert "enable_image_understanding" not in tool_def


def test_xai_web_search_extracts_inline_url_citations(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The docs describe Web Search.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://docs.x.ai/developers/tools/web-search",
                                        "title": "Web Search",
                                        "start_index": 4,
                                        "end_index": 8,
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_web_search_tool(query="xAI web search docs"))

    assert result["success"] is True
    assert result["answer"] == "The docs describe Web Search."
    assert result["inline_citations"] == [
        {
            "url": "https://docs.x.ai/developers/tools/web-search",
            "title": "Web Search",
            "start_index": 4,
            "end_index": 8,
        }
    ]


def test_xai_web_search_returns_structured_http_error(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    class _FailingResponse:
        status_code = 403
        text = '{"code":"forbidden","error":"web_search is not enabled for this model"}'

        def json(self):
            return {
                "code": "forbidden",
                "error": "web_search is not enabled for this model",
            }

        def raise_for_status(self):
            err = requests.HTTPError("403 Client Error: Forbidden")
            err.response = self
            raise err

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(xai_web_search_tool(query="latest xAI docs"))

    assert result["success"] is False
    assert result["provider"] == "xai"
    assert result["tool"] == "xai_web_search"
    assert result["xai_tool"] == "web_search"
    assert result["error_type"] == "HTTPError"
    assert result["error"] == "forbidden: web_search is not enabled for this model"


def test_xai_web_search_retries_read_timeout_then_succeeds(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return _FakeResponse({"output_text": "Recovered after retry."})

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.xai_web_search_tool.time.sleep", lambda *_: None)

    result = json.loads(xai_web_search_tool(query="grok web search"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after retry."


def test_xai_web_search_retries_5xx_then_succeeds(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(
                {"code": "Internal error", "error": "Service temporarily unavailable."},
                status_code=500,
            )
        return _FakeResponse({"output_text": "Recovered after 5xx retry."})

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.xai_web_search_tool.time.sleep", lambda *_: None)

    result = json.loads(xai_web_search_tool(query="grok web search"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after 5xx retry."


def test_xai_web_search_uses_xai_oauth_when_available(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_web_search_tool import check_xai_web_search_requirements, xai_web_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve():
        return {
            "provider": "xai-oauth",
            "api_key": "oauth-bearer-token",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.xai_web_search_tool.resolve_xai_http_credentials", _fake_resolve
    )
    invalidate_check_fn_cache()

    assert check_xai_web_search_requirements() is True

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "Found via OAuth."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_web_search_tool(query="anything about xai"))

    assert result["success"] is True
    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-bearer-token"


def test_xai_web_search_returns_tool_error_when_no_credentials(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_web_search_tool import check_xai_web_search_requirements, xai_web_search_tool

    _no_xai_env(monkeypatch)

    def _fake_resolve():
        return {
            "provider": "xai",
            "api_key": "",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(
        "tools.xai_web_search_tool.resolve_xai_http_credentials", _fake_resolve
    )
    invalidate_check_fn_cache()

    assert check_xai_web_search_requirements() is False

    result = xai_web_search_tool(query="anything")
    assert "No xAI credentials available" in result
    assert "hermes auth add xai-oauth" in result


def test_xai_web_search_honors_config_model_and_timeout(monkeypatch):
    from tools.xai_web_search_tool import xai_web_search_tool

    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.setattr(
        "tools.xai_web_search_tool._load_xai_web_search_config",
        lambda: {"model": "grok-custom-test", "timeout_seconds": 45, "retries": 0},
    )

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["model"] = json["model"]
        captured["timeout"] = timeout
        return _FakeResponse({"output_text": "Custom model OK."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_web_search_tool(query="anything"))

    assert result["success"] is True
    assert captured["model"] == "grok-custom-test"
    assert captured["timeout"] == 45


def test_xai_web_search_registered_in_registry_with_check_fn():
    import tools.xai_web_search_tool  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("xai_web_search")
    assert entry is not None
    assert entry.toolset == "xai_web_search"
    assert entry.check_fn is not None
    assert entry.check_fn.__name__ == "check_xai_web_search_requirements"
    assert "XAI_API_KEY" in entry.requires_env
