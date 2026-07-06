"""Tests for the http_request tool."""

import json
import pytest
from unittest.mock import patch, MagicMock
import httpx

from tools.http_request import http_request_tool, redact_url_secrets, redact_sensitive_headers


def test_invalid_method():
    # 1. invalid method rejected
    res_str = http_request_tool(method="INVALID", url="https://example.com")
    res = json.loads(res_str)
    assert res["success"] is False
    assert "Unsupported HTTP method" in res["error"]
    assert res["error_type"] == "ValueError"


def test_private_url_blocked():
    # 2. localhost/private URL blocked
    # is_safe_url will reject 127.0.0.1
    res_str = http_request_tool(method="GET", url="http://127.0.0.1")
    res = json.loads(res_str)
    assert res["success"] is False
    assert "URL is not safe to access" in res["error"]
    assert res["error_type"] == "PermissionError"


def test_metadata_ip_blocked():
    # 3. metadata IP blocked
    # 169.254.169.254 is link-local and blocked by is_safe_url
    res_str = http_request_tool(method="GET", url="http://169.254.169.254/latest/meta-data")
    res = json.loads(res_str)
    assert res["success"] is False
    assert "URL is not safe to access" in res["error"]
    assert res["error_type"] == "PermissionError"


@patch("tools.http_request.is_safe_url", return_value=True)
def test_public_url_allowed_and_get_json(mock_safe):
    # 4. public URL allowed when safety check indicates safe
    # 5. GET JSON response parsed correctly
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.headers = {"content-type": "application/json; charset=utf-8"}
    mock_response.text = '{"status": "ok", "items": [1, 2]}'
    mock_response.json.return_value = {"status": "ok", "items": [1, 2]}
    
    with patch("httpx.Client.request", return_value=mock_response) as mock_request:
        res_str = http_request_tool(method="GET", url="https://api.github.com/users/test")
        res = json.loads(res_str)
        
        assert res["success"] is True
        assert res["status"] == 200
        assert res["ok"] is True
        assert "application/json" in res["content_type"]
        assert res["json"] == {"status": "ok", "items": [1, 2]}
        assert res["truncated"] is False
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == "https://api.github.com/users/test"


@patch("tools.http_request.is_safe_url", return_value=True)
def test_post_json_body(mock_safe):
    # 6. POST JSON body sent correctly
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.is_success = True
    mock_response.headers = {"content-type": "application/json"}
    mock_response.text = '{"id": 123}'
    mock_response.json.return_value = {"id": 123}
    
    with patch("httpx.Client.request", return_value=mock_response) as mock_request:
        payload = {"title": "new issue", "body": "description"}
        res_str = http_request_tool(
            method="POST",
            url="https://api.github.com/repos/test/issues",
            json_body=payload
        )
        res = json.loads(res_str)
        
        assert res["success"] is True
        assert res["status"] == 201
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["json"] == payload
        assert kwargs["data"] is None


@patch("tools.http_request.is_safe_url", return_value=True)
def test_query_params_and_custom_headers(mock_safe):
    # 7. query params handled correctly
    # 8. custom headers handled correctly
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = "<html>Hello</html>"
    
    headers = {"X-Custom-Header": "value1", "Authorization": "Bearer supersecret"}
    query = {"page": "1", "q": "test"}
    
    with patch("httpx.Client.request", return_value=mock_response) as mock_request:
        res_str = http_request_tool(
            method="GET",
            url="https://example.com/search",
            headers=headers,
            query=query
        )
        res = json.loads(res_str)
        
        assert res["success"] is True
        
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert kwargs["params"] == query
        # Request headers should have Authorization
        assert kwargs["headers"]["Authorization"] == "Bearer supersecret"
        
        # Output url and headers must be redacted
        assert res["url"] == "https://example.com/search"
        assert res["headers"].get("Authorization") is None  # response headers didn't have it


@patch("tools.http_request.is_safe_url", return_value=True)
@patch("tools.http_request._get_env_value", return_value="env-secret-token")
def test_bearer_token_auth_from_env(mock_get_env, mock_safe):
    # 9. bearer token auth from env handled correctly
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.headers = {"content-type": "application/json", "Authorization": "Bearer env-secret-token"}
    mock_response.text = '{"data": "auth_ok"}'
    mock_response.json.return_value = {"data": "auth_ok"}
    
    with patch("httpx.Client.request", return_value=mock_response) as mock_request:
        res_str = http_request_tool(
            method="GET",
            url="https://api.github.com/user",
            auth_mode="bearer_env",
            auth_token_env="MY_GITHUB_TOKEN"
        )
        res = json.loads(res_str)
        
        assert res["success"] is True
        mock_get_env.assert_called_once_with("MY_GITHUB_TOKEN")
        
        # Verify Authorization header was sent with the env value
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer env-secret-token"
        
        # Response headers had Authorization (mocked), verify it is redacted in output
        assert res["headers"]["Authorization"] == "[REDACTED]"


@patch("tools.http_request.is_safe_url", return_value=True)
def test_timeout_error(mock_safe):
    # 10. timeout returns structured error
    with patch("httpx.Client.request", side_effect=httpx.TimeoutException("Request timed out")):
        res_str = http_request_tool(method="GET", url="https://example.com", timeout_seconds=5)
        res = json.loads(res_str)
        assert res["success"] is False
        assert "Request timed out" in res["error"]
        assert res["error_type"] == "TimeoutException"


@patch("tools.http_request.is_safe_url", return_value=True)
def test_connection_error(mock_safe):
    # 11. connection error returns structured error
    with patch("httpx.Client.request", side_effect=httpx.ConnectError("Connection refused")):
        res_str = http_request_tool(method="GET", url="https://example.com")
        res = json.loads(res_str)
        assert res["success"] is False
        assert "Connection refused" in res["error"]
        assert res["error_type"] == "ConnectError"


@patch("tools.http_request.is_safe_url", return_value=True)
@patch("tools.http_request.registry.get_max_result_size", return_value=15)
def test_response_truncation(mock_max_size, mock_safe):
    # 12. large text response is truncated or previewed safely
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.is_success = True
    mock_response.headers = {"content-type": "text/plain"}
    # Length of "abcdefghijklmnopqrstuvwxyz" is 26, which is > 15
    mock_response.text = "abcdefghijklmnopqrstuvwxyz"
    
    with patch("httpx.Client.request", return_value=mock_response):
        res_str = http_request_tool(method="GET", url="https://example.com/large")
        res = json.loads(res_str)
        
        assert res["success"] is True
        assert res["truncated"] is True
        # It should slice at 15 and add the truncated note
        assert res["text_preview"].startswith("abcdefghijklmno")
        assert "[TRUNCATED]" in res["text_preview"]
        # Since it is truncated, json should be None
        assert res["json"] is None


@patch("tools.http_request.is_safe_url", return_value=True)
def test_non_200_response(mock_safe):
    # 13. non-200 response still returns structured result cleanly
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.is_success = False
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.text = "Not Found"
    
    with patch("httpx.Client.request", return_value=mock_response):
        res_str = http_request_tool(method="GET", url="https://example.com/missing")
        res = json.loads(res_str)
        
        assert res["success"] is True
        assert res["status"] == 404
        assert res["ok"] is False
        assert res["text_preview"] == "Not Found"


def test_redact_url_secrets():
    # Test that secret query parameters are redacted
    url_with_secret = "https://example.com/api?api_key=secret123&q=query&token=abc"
    redacted = redact_url_secrets(url_with_secret)
    assert "api_key=%5BREDACTED%5D" in redacted
    assert "token=%5BREDACTED%5D" in redacted
    assert "q=query" in redacted


def test_redact_sensitive_headers():
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123",
        "x-api-key": "secret",
        "Accept": "*/*"
    }
    redacted = redact_sensitive_headers(headers)
    assert redacted["Content-Type"] == "application/json"
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["x-api-key"] == "[REDACTED]"
    assert redacted["Accept"] == "*/*"
