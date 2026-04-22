"""Tests for the http_request tool."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tools.http_tool import (
    HTTP_REQUEST_SCHEMA,
    _ALLOWED_METHODS,
    _handle_http_request,
    MAX_RESPONSE_SIZE,
)
from tools.registry import registry


# ── Schema & Registration ────────────────────────────────────────────────

class TestSchema:
    def test_schema_name(self):
        assert HTTP_REQUEST_SCHEMA["name"] == "http_request"

    def test_schema_has_url_required(self):
        assert "url" in HTTP_REQUEST_SCHEMA["parameters"]["required"]

    def test_schema_has_all_methods(self):
        method_enum = HTTP_REQUEST_SCHEMA["parameters"]["properties"]["method"]["enum"]
        assert set(method_enum) == _ALLOWED_METHODS

    def test_schema_has_expected_properties(self):
        props = HTTP_REQUEST_SCHEMA["parameters"]["properties"]
        for key in ("url", "method", "headers", "params", "json", "body",
                     "form_data", "timeout", "follow_redirects"):
            assert key in props, f"Missing property: {key}"


class TestRegistration:
    def test_http_request_registered(self):
        assert "http_request" in registry._tools

    def test_http_request_has_handler(self):
        tool = registry._tools["http_request"]
        assert tool.handler is _handle_http_request

    def test_http_request_toolset(self):
        tool = registry._tools["http_request"]
        assert tool.toolset == "http"


# ── Input Validation ──────────────────────────────────────────────────────

class TestInputValidation:
    def test_missing_url_returns_error(self):
        result = _handle_http_request({})
        assert "Error" in result
        assert "url" in result.lower()

    def test_empty_url_returns_error(self):
        result = _handle_http_request({"url": ""})
        assert "Error" in result

    def test_invalid_method_returns_error(self):
        result = _handle_http_request({"url": "https://example.com", "method": "INVALID"})
        assert "Error" in result
        assert "method" in result.lower()

    def test_invalid_headers_json_returns_error(self):
        result = _handle_http_request({
            "url": "https://example.com",
            "headers": "not-valid-json",
        })
        assert "Error" in result
        assert "headers" in result.lower()

    def test_invalid_params_json_returns_error(self):
        result = _handle_http_request({
            "url": "https://example.com",
            "params": "not-valid-json",
        })
        assert "Error" in result
        assert "params" in result.lower()

    def test_invalid_form_data_json_returns_error(self):
        result = _handle_http_request({
            "url": "https://example.com",
            "form_data": "not-valid-json",
        })
        assert "Error" in result
        assert "form_data" in result.lower()


# ── SSRF Protection ──────────────────────────────────────────────────────

class TestSSRFProtection:
    def test_localhost_blocked(self):
        result = _handle_http_request({"url": "http://127.0.0.1:8080/admin"})
        assert "Error" in result
        assert "private" in result.lower() or "internal" in result.lower()

    def test_private_ip_10_blocked(self):
        result = _handle_http_request({"url": "http://10.0.0.1/api"})
        assert "Error" in result

    def test_private_ip_192_168_blocked(self):
        result = _handle_http_request({"url": "http://192.168.1.1:3000/"})
        assert "Error" in result

    def test_metadata_endpoint_blocked(self):
        result = _handle_http_request({"url": "http://169.254.169.254/latest/meta-data/"})
        assert "Error" in result

    def test_google_metadata_hostname_blocked(self):
        result = _handle_http_request({"url": "http://metadata.google.internal/computeMetadata/v1/"})
        assert "Error" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_public_url_allowed(self, mock_safe):
        """When is_safe_url passes, the handler proceeds (mocked HTTP to avoid real calls)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.reason_phrase = "OK"
        mock_response.url = "https://example.com/"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "OK"
        mock_response.is_success = True

        with patch("httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.request.return_value = mock_response
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_ctx

            result = _handle_http_request({"url": "https://example.com"})
            assert "200" in result
            assert "OK" in result


# ── Successful Request Handling ──────────────────────────────────────────

def _make_mock_response(status_code=200, content_type="text/plain", body="hello",
                         reason="OK", url="https://example.com/test"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.reason_phrase = reason
    resp.url = url
    resp.headers = {"content-type": content_type}
    resp.text = body
    resp.is_success = 200 <= status_code < 400
    resp.json.return_value = json.loads(body) if "json" in content_type else None
    return resp


def _patch_httpx_client(mock_response):
    """Return a context manager patch that makes httpx.Client return mock_response."""
    mock_ctx = MagicMock()
    mock_ctx.request.return_value = mock_response
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return patch("httpx.Client", return_value=mock_ctx)


class TestSuccessfulRequests:
    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_get_request(self, mock_safe):
        resp = _make_mock_response(body="hi there")
        with _patch_httpx_client(resp):
            result = _handle_http_request({"url": "https://example.com"})
            assert "200" in result
            assert "hi there" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_json_response_parsed(self, mock_safe):
        payload = {"key": "value", "count": 42}
        resp = _make_mock_response(
            content_type="application/json",
            body=json.dumps(payload),
        )
        resp.json.return_value = payload
        with _patch_httpx_client(resp):
            result = _handle_http_request({"url": "https://api.example.com/data"})
            assert "200" in result
            assert "value" in result
            assert "42" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_error_status_code(self, mock_safe):
        resp = _make_mock_response(
            status_code=404,
            reason="Not Found",
            body="not found",
        )
        resp.is_success = False
        with _patch_httpx_client(resp):
            result = _handle_http_request({"url": "https://example.com/missing"})
            assert "404" in result
            assert "✗" in result  # error icon

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_success_icon_for_2xx(self, mock_safe):
        resp = _make_mock_response(status_code=201, reason="Created")
        with _patch_httpx_client(resp):
            result = _handle_http_request({"url": "https://example.com"})
            assert "✓" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_headers_string_parsed(self, mock_safe):
        resp = _make_mock_response()
        with _patch_httpx_client(resp):
            result = _handle_http_request({
                "url": "https://example.com",
                "headers": '{"Authorization": "Bearer tok"}',
            })
            assert "200" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_params_string_parsed(self, mock_safe):
        resp = _make_mock_response()
        with _patch_httpx_client(resp):
            result = _handle_http_request({
                "url": "https://example.com",
                "params": '{"page": "1"}',
            })
            assert "200" in result


# ── Timeout Handling ──────────────────────────────────────────────────────

class TestTimeout:
    def test_timeout_capped_at_120(self):
        """Verify timeout is capped at 120s even if user specifies more."""
        with patch("tools.url_safety.is_safe_url", return_value=True):
            with patch("httpx.Client") as mock_client_cls:
                mock_ctx = MagicMock()
                mock_response = _make_mock_response()
                mock_ctx.request.return_value = mock_response
                mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
                mock_ctx.__exit__ = MagicMock(return_value=False)
                mock_client_cls.return_value = mock_ctx

                _handle_http_request({"url": "https://example.com", "timeout": 500})
                call_kwargs = mock_client_cls.call_args
                # httpx.Client called with timeout=min(500, 120) = 120
                assert call_kwargs.kwargs.get("timeout") == 120 or \
                       call_kwargs[1].get("timeout") == 120

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_timeout_error_message(self, mock_safe):
        with patch("httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.request.side_effect = httpx.TimeoutException("timed out")
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_ctx

            result = _handle_http_request({"url": "https://example.com"})
            assert "timed out" in result.lower()


# ── Error Handling ────────────────────────────────────────────────────────

class TestErrorHandling:
    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_connect_error(self, mock_safe):
        with patch("httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.request.side_effect = httpx.ConnectError("Connection refused")
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_ctx

            result = _handle_http_request({"url": "https://example.com"})
            assert "Error" in result

    @patch("tools.url_safety.is_safe_url", return_value=True)
    def test_unsupported_protocol(self, mock_safe):
        with patch("httpx.Client") as mock_client_cls:
            mock_ctx = MagicMock()
            mock_ctx.request.side_effect = httpx.UnsupportedProtocol("bad")
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value = mock_ctx

            result = _handle_http_request({"url": "https://example.com"})
            assert "Error" in result
