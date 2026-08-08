"""Tests for the http tool — structured HTTP without going through bash.

The whole point of this tool is to skip the bash-quoting bug class. Tests
mock httpx.Client.request directly so they don't make real network calls,
and verify that the JSON-encoded result envelope matches the convention.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import tools.http_tool as ht
from tools.http_tool import (
    HTTP_SCHEMA,
    _MAX_RESPONSE_BYTES,
    http_tool,
)


def _mock_response(
    status: int = 200,
    body: bytes = b"ok",
    headers: dict | None = None,
    encoding: str = "utf-8",
) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.content = body
    resp.headers = headers or {}
    resp.encoding = encoding
    return resp


@pytest.fixture(autouse=True)
def reset_module_client():
    """Reset the module-level httpx.Client between tests so monkeypatching is clean."""
    ht._http_client = None
    yield
    ht._http_client = None


class TestHttpToolBasics:
    @patch("tools.http_tool._get_http_client")
    def test_get_happy_path_returns_status_headers_body(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response(
            status=200, body=b"hello", headers={"Content-Type": "text/plain"}
        )
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://localhost/foo"))

        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "text/plain"
        assert result["body"] == "hello"
        assert "elapsed_ms" in result and isinstance(result["elapsed_ms"], int)

        client.request.assert_called_once()
        kwargs = client.request.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "http://localhost/foo"

    @patch("tools.http_tool._get_http_client")
    def test_method_lowercase_is_normalised(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response()
        mock_get_client.return_value = client

        http_tool(method="post", url="http://x/y")
        assert client.request.call_args.kwargs["method"] == "POST"

    def test_invalid_method_returns_error(self):
        result = json.loads(http_tool(method="INVALID", url="http://x"))
        assert "error" in result
        assert "INVALID" in result["error"]

    def test_missing_url_returns_error(self):
        result = json.loads(http_tool(method="GET", url=""))
        assert "error" in result
        assert "url" in result["error"].lower()


class TestHttpToolBodyResolution:
    @patch("tools.http_tool._get_http_client")
    def test_post_json_routes_to_httpx_json_kwarg(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response(status=201, body=b'{"id":1}')
        mock_get_client.return_value = client

        result = json.loads(
            http_tool(
                method="POST",
                url="http://localhost:3100/api/issues/AOS-8",
                json_body={"status": "done", "comment": "It's done"},
            )
        )

        assert result["status"] == 201
        assert result["body"] == '{"id":1}'

        kwargs = client.request.call_args.kwargs
        # Crucial: the JSON body goes through httpx's `json` kwarg — never
        # through any string-quoting layer. The apostrophe in "It's done"
        # would have killed the bash path.
        assert kwargs["json"] == {"status": "done", "comment": "It's done"}
        assert "content" not in kwargs
        assert "body" not in kwargs

    @patch("tools.http_tool._get_http_client")
    def test_post_body_routes_to_httpx_content_kwarg(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response()
        mock_get_client.return_value = client

        http_tool(
            method="POST",
            url="http://x/y",
            body="raw text body with 'apostrophe'",
        )
        kwargs = client.request.call_args.kwargs
        assert kwargs["content"] == "raw text body with 'apostrophe'"
        assert "json" not in kwargs

    def test_json_and_body_together_returns_error(self):
        result = json.loads(
            http_tool(
                method="POST",
                url="http://x",
                json_body={"a": 1},
                body="raw",
            )
        )
        assert "error" in result
        assert "either" in result["error"].lower() or "both" in result["error"].lower()


class TestHttpToolErrorHandling:
    @patch("tools.http_tool._get_http_client")
    def test_4xx_response_returned_as_success_envelope(self, mock_get_client):
        # 4xx is a successful HTTP exchange — only transport errors become tool_error.
        client = MagicMock()
        client.request.return_value = _mock_response(
            status=404, body=b'{"error":"not found"}'
        )
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://x/missing"))
        assert result["status"] == 404
        assert "error" not in result  # the tool envelope itself succeeded
        assert result["body"] == '{"error":"not found"}'

    @patch("tools.http_tool._get_http_client")
    def test_timeout_returns_tool_error_with_elapsed(self, mock_get_client):
        client = MagicMock()
        client.request.side_effect = httpx.TimeoutException("read timeout")
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://slow", timeout=1))
        assert "error" in result
        assert "timeout" in result["error"].lower()
        assert result.get("timeout") is True
        assert "elapsed_ms" in result

    @patch("tools.http_tool._get_http_client")
    def test_connection_refused_returns_tool_error(self, mock_get_client):
        client = MagicMock()
        client.request.side_effect = httpx.ConnectError("Connection refused")
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://localhost:9"))
        assert "error" in result
        assert "Transport error" in result["error"]
        assert "ConnectError" in result["error"]

    def test_invalid_timeout_type_returns_error(self):
        result = json.loads(
            http_tool(method="GET", url="http://x", timeout="thirty")
        )
        assert "error" in result
        assert "timeout" in result["error"].lower()

    def test_timeout_above_max_returns_error(self):
        result = json.loads(http_tool(method="GET", url="http://x", timeout=999))
        assert "error" in result
        assert "maximum" in result["error"].lower()


class TestHttpToolBodyCap:
    @patch("tools.http_tool._get_http_client")
    def test_oversized_body_is_truncated_with_hint(self, mock_get_client):
        big = b"a" * (_MAX_RESPONSE_BYTES + 100)
        client = MagicMock()
        client.request.return_value = _mock_response(body=big)
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://x/big"))
        assert result["status"] == 200
        assert result.get("truncated") is True
        assert "_hint" in result
        # body string length matches the byte cap (single-byte chars).
        assert len(result["body"]) == _MAX_RESPONSE_BYTES


class TestHttpToolHeaderRedaction:
    @patch("tools.http_tool._get_http_client")
    def test_authorization_header_redacted_in_response(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response(
            headers={
                "Authorization": "Bearer super-secret",
                "Set-Cookie": "session=abc123",
                "X-Custom-Token": "leaked-token",
                "Content-Type": "application/json",
            }
        )
        mock_get_client.return_value = client

        result = json.loads(http_tool(method="GET", url="http://x"))
        assert result["headers"]["Authorization"] == "[REDACTED]"
        assert result["headers"]["Set-Cookie"] == "[REDACTED]"
        assert result["headers"]["X-Custom-Token"] == "[REDACTED]"
        # Non-sensitive headers pass through untouched.
        assert result["headers"]["Content-Type"] == "application/json"


class TestHttpToolSchemaShape:
    """Schema must advertise both `json` and `body` so the model can pick."""

    def test_schema_has_method_and_url_required(self):
        params = HTTP_SCHEMA["parameters"]
        assert params["required"] == ["method", "url"]

    def test_schema_advertises_json_and_body_separately(self):
        props = HTTP_SCHEMA["parameters"]["properties"]
        assert "json" in props
        assert "body" in props
        # Description should make the auto-serialisation contract clear.
        assert "Content-Type" in props["json"]["description"]
        assert "json" in props["body"]["description"].lower()

    def test_schema_has_method_enum_in_description(self):
        props = HTTP_SCHEMA["parameters"]["properties"]
        # Methods enumerated in description (provider-safe vs JSON Schema enum
        # rejection by some sanitisers — same workaround as PATCH_SCHEMA).
        desc = props["method"]["description"]
        for m in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            assert m in desc

    def test_tool_registered_under_http_toolset(self):
        from tools.registry import registry
        entry = registry._tools.get("http")
        assert entry is not None, "http tool not registered"
        assert entry.toolset == "http"
        assert entry.handler is not None


class TestHttpToolRequestParams:
    @patch("tools.http_tool._get_http_client")
    def test_query_params_passed_through(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response()
        mock_get_client.return_value = client

        http_tool(method="GET", url="http://x", params={"q": "abc", "page": "2"})
        kwargs = client.request.call_args.kwargs
        assert kwargs["params"] == {"q": "abc", "page": "2"}

    @patch("tools.http_tool._get_http_client")
    def test_request_headers_passed_through(self, mock_get_client):
        client = MagicMock()
        client.request.return_value = _mock_response()
        mock_get_client.return_value = client

        http_tool(
            method="POST",
            url="http://x",
            headers={"Authorization": "Bearer xyz", "X-Run-Id": "r1"},
            json_body={"a": 1},
        )
        kwargs = client.request.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer xyz"
        assert kwargs["headers"]["X-Run-Id"] == "r1"

    def test_invalid_headers_type_returns_error(self):
        result = json.loads(
            http_tool(method="GET", url="http://x", headers="not-a-dict")  # type: ignore[arg-type]
        )
        assert "error" in result
        assert "headers" in result["error"].lower()
