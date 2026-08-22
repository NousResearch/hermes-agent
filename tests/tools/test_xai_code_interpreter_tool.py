"""Tests for xAI Code Interpreter via the Responses API code_interpreter tool."""

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
        "tools.xai_code_interpreter_tool.resolve_xai_http_credentials",
        lambda: {
            "provider": provider,
            "api_key": api_key,
            "base_url": "https://api.x.ai/v1",
        },
    )


def _no_xai_env(monkeypatch):
    for var in ("XAI_API_KEY", "XAI_BASE_URL", "HERMES_XAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def test_xai_code_interpreter_posts_responses_request(monkeypatch):
    from hermes_cli import __version__
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "output_text": "Compound interest is $16288.95.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "code_execution",
                            "arguments": '{"code":"print(10000*(1.05)**10)"}',
                        },
                    }
                ],
                "server_side_tool_usage": {
                    "SERVER_SIDE_TOOL_CODE_EXECUTION": 1,
                },
                "usage": {"input_tokens": 20, "output_tokens": 15},
            }
        )

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(
        xai_code_interpreter_tool(
            task="Calculate compound interest for $10000 at 5% for 10 years"
        )
    )

    tool_def = captured["json"]["tools"][0]
    assert captured["url"] == "https://api.x.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer xai-test-key"
    assert captured["headers"]["User-Agent"] == f"Hermes-Agent/{__version__}"
    assert captured["json"]["model"] == "grok-4.5"
    assert captured["json"]["store"] is False
    assert captured["json"]["input"][0]["content"] == (
        "Calculate compound interest for $10000 at 5% for 10 years"
    )
    assert captured["timeout"] == 180
    assert tool_def == {"type": "code_interpreter"}
    assert result["success"] is True
    assert result["tool"] == "xai_code_interpreter"
    assert result["xai_tool"] == "code_interpreter"
    assert result["xai_sdk_tool"] == "code_execution"
    assert result["answer"] == "Compound interest is $16288.95."
    assert result["tool_calls"] == [
        {
            "type": "function",
            "function": {
                "name": "code_execution",
                "arguments": '{"code":"print(10000*(1.05)**10)"}',
            },
        }
    ]
    assert result["server_side_tool_usage"] == {
        "SERVER_SIDE_TOOL_CODE_EXECUTION": 1,
    }
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 15}


def test_xai_code_interpreter_extracts_output_and_calls(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    def _fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse(
            {
                "output": [
                    {
                        "type": "code_interpreter_call",
                        "id": "ci_1",
                        "status": "completed",
                        "code": "print(2 + 2)",
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The result is 4.",
                            }
                        ],
                    },
                ]
            }
        )

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_code_interpreter_tool(task="What is 2+2?"))

    assert result["success"] is True
    assert result["answer"] == "The result is 4."
    assert result["code_interpreter_calls"] == [
        {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "completed",
            "code": "print(2 + 2)",
        }
    ]


def test_xai_code_interpreter_empty_task_returns_tool_error(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    _fake_credentials(monkeypatch)

    result = json.loads(xai_code_interpreter_tool(task="  "))

    assert result["error"] == "task is required for xai_code_interpreter"


def test_xai_code_interpreter_returns_structured_http_error(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    class _FailingResponse:
        status_code = 403
        text = '{"code":"forbidden","error":"code_interpreter is not enabled"}'

        def json(self):
            return {
                "code": "forbidden",
                "error": "code_interpreter is not enabled",
            }

        def raise_for_status(self):
            err = requests.HTTPError("403 Client Error: Forbidden")
            err.response = self
            raise err

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert result["success"] is False
    assert result["provider"] == "xai"
    assert result["tool"] == "xai_code_interpreter"
    assert result["xai_tool"] == "code_interpreter"
    assert result["error_type"] == "HTTPError"
    assert result["error"] == "forbidden: code_interpreter is not enabled"


def test_xai_code_interpreter_http_error_supports_nested_error(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    class _FailingResponse:
        status_code = 400
        text = '{"error":{"message":"bad request"}}'

        def json(self):
            return {"error": {"message": "bad request"}}

        def raise_for_status(self):
            err = requests.HTTPError("400 Client Error: Bad Request")
            err.response = self
            raise err

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", lambda *a, **k: _FailingResponse())

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert result["success"] is False
    assert result["error"] == "bad request"


def test_xai_code_interpreter_retries_timeout_then_succeeds(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    calls = {"count": 0}

    def _fake_post(url, headers=None, json=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ReadTimeout("timed out")
        return _FakeResponse({"output_text": "Recovered after retry."})

    _fake_credentials(monkeypatch)
    monkeypatch.setattr("requests.post", _fake_post)
    monkeypatch.setattr("tools.xai_code_interpreter_tool.time.sleep", lambda *_: None)

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after retry."


def test_xai_code_interpreter_retries_5xx_then_succeeds(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

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
    monkeypatch.setattr("tools.xai_code_interpreter_tool.time.sleep", lambda *_: None)

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert calls["count"] == 2
    assert result["success"] is True
    assert result["answer"] == "Recovered after 5xx retry."


def test_xai_code_interpreter_uses_xai_oauth_when_available(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_code_interpreter_tool import (
        check_xai_code_interpreter_requirements,
        xai_code_interpreter_tool,
    )

    _no_xai_env(monkeypatch)
    _fake_credentials(monkeypatch, provider="xai-oauth", api_key="oauth-token")
    invalidate_check_fn_cache()

    assert check_xai_code_interpreter_requirements() is True

    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["headers"] = headers
        return _FakeResponse({"output_text": "OAuth worked."})

    monkeypatch.setattr("requests.post", _fake_post)

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert result["success"] is True
    assert result["credential_source"] == "xai-oauth"
    assert captured["headers"]["Authorization"] == "Bearer oauth-token"


def test_xai_code_interpreter_returns_tool_error_when_no_credentials(monkeypatch):
    from tools.registry import invalidate_check_fn_cache
    from tools.xai_code_interpreter_tool import (
        check_xai_code_interpreter_requirements,
        xai_code_interpreter_tool,
    )

    _no_xai_env(monkeypatch)
    _fake_credentials(monkeypatch, api_key="")
    invalidate_check_fn_cache()

    assert check_xai_code_interpreter_requirements() is False

    result = xai_code_interpreter_tool(task="Compute something")

    assert "No xAI credentials available" in result
    assert "hermes auth add xai-oauth" in result


def test_xai_code_interpreter_honors_config_model_timeout_and_retries(monkeypatch):
    from tools.xai_code_interpreter_tool import xai_code_interpreter_tool

    monkeypatch.setattr(
        "tools.xai_code_interpreter_tool._load_xai_code_interpreter_config",
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

    result = json.loads(xai_code_interpreter_tool(task="Compute something"))

    assert result["success"] is True
    assert captured["model"] == "grok-custom-test"
    assert captured["timeout"] == 45


def test_xai_code_interpreter_registered_in_registry_with_check_fn():
    import tools.xai_code_interpreter_tool  # noqa: F401 - ensures registration runs
    from tools.registry import registry

    entry = registry.get_entry("xai_code_interpreter")
    assert entry is not None
    assert entry.toolset == "xai_code_interpreter"
    assert entry.check_fn is not None
    assert entry.check_fn.__name__ == "check_xai_code_interpreter_requirements"
    assert "XAI_API_KEY" in entry.requires_env
    assert entry.emoji == "🧮"
