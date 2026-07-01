"""Tests that Camofox browser sends Authorization header when CAMOFOX_API_KEY is set.

Regression test for https://github.com/NousResearch/hermes-agent/issues/20476
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_camofox import (
    _auth_headers,
    camofox_back,
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_press,
    camofox_scroll,
    camofox_snapshot,
    camofox_type,
)


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    payload = json.dumps(json_data or {}).encode("utf-8")
    resp.iter_content.return_value = [payload]
    resp.encoding = "utf-8"
    resp.json.side_effect = AssertionError(
        "Camofox JSON must be read through the bounded stream reader"
    )
    resp.content = b"\x89PNG\r\n\x1a\nfake"
    resp.raise_for_status = MagicMock()
    resp.close = MagicMock()
    return resp


class TestAuthHeaders:
    """Unit tests for _auth_headers() helper."""

    def test_empty_when_no_key(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)
        assert _auth_headers() == {}

    def test_bearer_when_key_set(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_API_KEY", "test-secret-123")
        assert _auth_headers() == {"Authorization": "Bearer test-secret-123"}

    def test_empty_when_key_blank(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_API_KEY", "   ")
        assert _auth_headers() == {}


class TestAuthHeadersSent:
    """Verify all HTTP call sites include auth headers when CAMOFOX_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_API_KEY", "my-api-key")

    @patch("tools.browser_camofox.requests.post")
    def test_ensure_tab_sends_auth(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t1"})
        camofox_navigate("https://example.com", task_id="auth_test_1")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    def test_post_sends_auth(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t2"})
        camofox_navigate("https://example.com", task_id="auth_test_2")
        mock_post.return_value = _mock_response(json_data={"ok": True, "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="auth_test_2")
        # The second call is a POST to /tabs/{tabId}/navigate
        last_call = mock_post.call_args_list[-1]
        assert last_call.kwargs.get("headers") == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.get")
    def test_get_sends_auth(self, mock_get, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t3"})
        camofox_navigate("https://example.com", task_id="auth_test_3")
        mock_get.return_value = _mock_response(json_data={
            "snapshot": '- heading "Hello"',
            "refsCount": 1,
        })
        camofox_snapshot(task_id="auth_test_3")
        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.delete")
    def test_delete_sends_auth(self, mock_delete, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t4"})
        camofox_navigate("https://example.com", task_id="auth_test_4")
        mock_delete.return_value = _mock_response(json_data={"ok": True})
        camofox_close(task_id="auth_test_4")
        _, kwargs = mock_delete.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}


class TestNoAuthHeadersWhenKeyUnset:
    """Verify HTTP calls send empty headers when CAMOFOX_API_KEY is not set."""

    @pytest.fixture(autouse=True)
    def _unset_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)

    @patch("tools.browser_camofox.requests.post")
    def test_no_auth_on_tab_creation(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t5"})
        camofox_navigate("https://example.com", task_id="noauth_test_1")
        _, kwargs = mock_post.call_args
        assert kwargs.get("headers") == {}
