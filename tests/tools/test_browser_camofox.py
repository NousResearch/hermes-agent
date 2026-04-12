"""Tests for the Camofox browser backend."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_camofox import (
    camofox_back,
    camofox_click,
    camofox_close,
    camofox_console,
    camofox_get_images,
    camofox_navigate,
    camofox_press,
    camofox_scroll,
    camofox_snapshot,
    camofox_type,
    camofox_vision,
    check_camofox_available,
    is_camofox_mode,
)


# ---------------------------------------------------------------------------
# Configuration detection
# ---------------------------------------------------------------------------


class TestCamofoxMode:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        assert is_camofox_mode() is False

    def test_enabled_when_url_set(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        assert is_camofox_mode() is True

    def test_health_check_unreachable(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:19999")
        assert check_camofox_available() is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = b"\x89PNG\r\n\x1a\nfake"
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Navigate
# ---------------------------------------------------------------------------


class TestCamofoxNavigate:
    @patch("tools.browser_camofox.requests.post")
    def test_creates_tab_on_first_navigate(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab1", "url": "https://example.com"})

        result = json.loads(camofox_navigate("https://example.com", task_id="t1"))
        assert result["success"] is True
        assert result["url"] == "https://example.com"

    @patch("tools.browser_camofox.requests.post")
    def test_navigates_existing_tab(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        # First call creates tab
        mock_post.return_value = _mock_response(json_data={"tabId": "tab2", "url": "https://a.com"})
        camofox_navigate("https://a.com", task_id="t2")

        # Second call navigates
        mock_post.return_value = _mock_response(json_data={"ok": True, "url": "https://b.com"})
        result = json.loads(camofox_navigate("https://b.com", task_id="t2"))
        assert result["success"] is True
        assert result["url"] == "https://b.com"

    def test_connection_error_returns_helpful_message(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:19999")
        result = json.loads(camofox_navigate("https://example.com", task_id="t_err"))
        assert result["success"] is False
        assert "Cannot connect" in result["error"]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestCamofoxSnapshot:
    def test_no_session_returns_error(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(camofox_snapshot(task_id="no_such_task"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.get")
    def test_returns_snapshot(self, mock_get, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        # Create session
        mock_post.return_value = _mock_response(json_data={"tabId": "tab3", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t3")

        # Return snapshot
        mock_get.return_value = _mock_response(json_data={
            "snapshot": "- heading \"Test\" [e1]\n- button \"Submit\" [e2]",
            "refsCount": 2,
        })
        result = json.loads(camofox_snapshot(task_id="t3"))
        assert result["success"] is True
        assert "[e1]" in result["snapshot"]
        assert result["element_count"] == 2


# ---------------------------------------------------------------------------
# Click / Type / Scroll / Back / Press
# ---------------------------------------------------------------------------


class TestCamofoxInteractions:
    @patch("tools.browser_camofox.requests.post")
    def test_click(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab4", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t4")

        mock_post.return_value = _mock_response(json_data={"ok": True, "url": "https://x.com"})
        result = json.loads(camofox_click("@e5", task_id="t4"))
        assert result["success"] is True
        assert result["clicked"] == "e5"

    @patch("tools.browser_camofox.requests.post")
    def test_type(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab5", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t5")

        mock_post.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_type("@e3", "hello world", task_id="t5"))
        assert result["success"] is True
        assert result["typed"] == "hello world"

    @patch("tools.browser_camofox.requests.post")
    def test_scroll(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab6", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t6")

        mock_post.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_scroll("down", task_id="t6"))
        assert result["success"] is True
        assert result["scrolled"] == "down"

    @patch("tools.browser_camofox.requests.post")
    def test_back(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab7", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t7")

        mock_post.return_value = _mock_response(json_data={"ok": True, "url": "https://prev.com"})
        result = json.loads(camofox_back(task_id="t7"))
        assert result["success"] is True

    @patch("tools.browser_camofox.requests.post")
    def test_press(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab8", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t8")

        mock_post.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_press("Enter", task_id="t8"))
        assert result["success"] is True
        assert result["pressed"] == "Enter"


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestCamofoxClose:
    @patch("tools.browser_camofox.requests.delete")
    @patch("tools.browser_camofox.requests.post")
    def test_close_session(self, mock_post, mock_delete, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab9", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t9")

        mock_delete.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_close(task_id="t9"))
        assert result["success"] is True
        assert result["closed"] is True

    def test_close_nonexistent_session(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(camofox_close(task_id="nonexistent"))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Console (limited support)
# ---------------------------------------------------------------------------


class TestCamofoxConsole:
    def test_console_returns_empty_with_note(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(camofox_console(task_id="t_console"))
        assert result["success"] is True
        assert result["total_messages"] == 0
        assert "not available" in result["note"]


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestCamofoxGetImages:
    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.get")
    def test_get_images(self, mock_get, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab10", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t10")

        # camofox_get_images parses images from the accessibility tree snapshot
        snapshot_text = (
            '- img "Logo"\n'
            '  /url: https://x.com/img.png\n'
        )
        mock_get.return_value = _mock_response(json_data={
            "snapshot": snapshot_text,
        })
        result = json.loads(camofox_get_images(task_id="t10"))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["images"][0]["src"] == "https://x.com/img.png"


# ---------------------------------------------------------------------------
# Routing integration — verify browser_tool routes to camofox
# ---------------------------------------------------------------------------


class TestBrowserToolRouting:
    """Verify that browser_tool.py delegates to camofox when CAMOFOX_URL is set."""

    @patch("tools.browser_camofox.requests.post")
    def test_browser_navigate_routes_to_camofox(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab_rt", "url": "https://example.com"})

        from tools.browser_tool import browser_navigate
        # Bypass SSRF check for test URL
        with patch("tools.browser_tool._is_safe_url", return_value=True):
            result = json.loads(browser_navigate("https://example.com", task_id="t_route"))
        assert result["success"] is True

    def test_check_requirements_passes_with_camofox(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_tool import check_browser_requirements
        assert check_browser_requirements() is True


