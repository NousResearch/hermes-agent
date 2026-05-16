"""Tests for the Kimi WebBridge toolset."""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.kimi_webbridge import (
    _check_bridge,
    _get_daemon_url,
    _validate_screenshot_path,
    kimi_webbridge_click,
    kimi_webbridge_close_session,
    kimi_webbridge_close_tab,
    kimi_webbridge_evaluate,
    kimi_webbridge_fill,
    kimi_webbridge_find_tab,
    kimi_webbridge_list_tabs,
    kimi_webbridge_navigate,
    kimi_webbridge_save_pdf,
    kimi_webbridge_save_screenshot,
    kimi_webbridge_screenshot,
    kimi_webbridge_snapshot,
)


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


class TestGetDaemonUrl:
    def test_default_fallback(self):
        with patch("hermes_cli.config.load_config", side_effect=Exception("no config")):
            assert _get_daemon_url() == "http://127.0.0.1:10086"

    def test_from_config(self):
        fake_cfg = {"providers": {"kimi_webbridge": {"base_url": "http://localhost:9999"}}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            assert _get_daemon_url() == "http://localhost:9999"

    def test_trailing_slash_stripped(self):
        fake_cfg = {"providers": {"kimi_webbridge": {"base_url": "http://localhost:9999/"}}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            assert _get_daemon_url() == "http://localhost:9999"


class TestCheckBridge:
    def test_true_when_daemon_responds(self):
        with patch("tools.kimi_webbridge.requests.post", return_value=_mock_response({})):
            assert _check_bridge() is True

    def test_false_when_request_fails(self):
        with patch("tools.kimi_webbridge.requests.post", side_effect=Exception("connection refused")):
            assert _check_bridge() is False

    def test_false_when_non_200(self):
        resp = _mock_response({}, status_code=500)
        with patch("tools.kimi_webbridge.requests.post", return_value=resp):
            assert _check_bridge() is False


class TestValidateScreenshotPath:
    def test_default_path(self):
        path = _validate_screenshot_path(None)
        assert str(path).startswith("/tmp/kimi-webbridge-screenshots/")
        assert path.suffix == ".png"

    def test_valid_tmp_path(self):
        path = _validate_screenshot_path("/tmp/foo.png")
        assert str(path) == "/tmp/foo.png"

    def test_valid_home_path(self):
        path = _validate_screenshot_path("~/foo.png")
        assert "foo.png" in str(path)

    def test_invalid_path_rejected(self):
        with pytest.raises(ValueError, match="must be under /tmp or home directory"):
            _validate_screenshot_path("/etc/passwd")


class TestNavigate:
    def test_navigate_success(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "url": "https://example.com", "tabId": 42}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_navigate("https://example.com"))
        assert result["ok"] is True
        assert result["data"]["url"] == "https://example.com"

    def test_navigate_with_group_title(self):
        mock = _mock_response({"ok": True, "data": {"success": True}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock) as post:
            kimi_webbridge_navigate("https://example.com", group_title="my-group")
            call_args = post.call_args[1]["json"]
            assert call_args["args"]["group_title"] == "my-group"


class TestSnapshot:
    def test_snapshot_returns_tree(self):
        mock = _mock_response({"ok": True, "data": {"url": "https://example.com", "title": "Example", "tree": []}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_snapshot())
        assert result["data"]["title"] == "Example"


class TestClick:
    def test_click_by_selector(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "tag": "button", "text": "Submit"}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_click("@e5"))
        assert result["data"]["tag"] == "button"


class TestFill:
    def test_fill_input(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "tag": "input", "mode": "value"}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_fill("@e3", "hello"))
        assert result["data"]["mode"] == "value"


class TestEvaluate:
    def test_evaluate_js(self):
        mock = _mock_response({"ok": True, "data": {"type": "string", "value": "42"}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_evaluate("document.title"))
        assert result["data"]["value"] == "42"


class TestListTabs:
    def test_list_tabs(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "tabs": [{"tabId": 1, "url": "https://a.com"}]}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_list_tabs())
        assert len(result["data"]["tabs"]) == 1


class TestCloseTab:
    def test_close_tab(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "closed": True}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_close_tab())
        assert result["data"]["closed"] is True


class TestCloseSession:
    def test_close_session(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "closed": 3}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_close_session())
        assert result["data"]["closed"] == 3


class TestFindTab:
    def test_find_tab(self):
        mock = _mock_response({"ok": True, "data": {"success": True, "url": "https://kimi.com", "tabId": 7}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_find_tab("https://kimi.com", active=True))
        assert result["data"]["tabId"] == 7


class TestSavePdf:
    def test_save_pdf(self):
        mock = _mock_response({"ok": True, "data": {"path": "/tmp/test.pdf", "sizeBytes": 1234}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_save_pdf(file_name="test.pdf"))
        assert result["data"]["path"] == "/tmp/test.pdf"


class TestScreenshot:
    def test_screenshot_strips_base64(self):
        mock = _mock_response({"ok": True, "data": {"format": "png", "dataLength": 50000, "data": "iVBORw0KGgo" * 1000}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_screenshot())
        inner = result["data"]
        assert "base64 image data" in inner["data"]


class TestSaveScreenshot:
    def test_save_screenshot_success(self, tmp_path):
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        mock = _mock_response({"ok": True, "data": {"format": "png", "dataLength": len(fake_b64), "data": fake_b64}})
        output = tmp_path / "shot.png"
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_save_screenshot(str(output)))
        assert result["success"] is True
        assert result["path"] == str(output)
        assert output.exists()

    def test_save_screenshot_failure_no_data(self):
        mock = _mock_response({"ok": True, "data": {"format": "png"}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_save_screenshot())
        assert "error" in result

    def test_save_screenshot_invalid_path(self):
        fake_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        mock = _mock_response({"ok": True, "data": {"format": "png", "dataLength": len(fake_b64), "data": fake_b64}})
        with patch("tools.kimi_webbridge.requests.post", return_value=mock):
            result = json.loads(kimi_webbridge_save_screenshot("/etc/evil.png"))
        assert "error" in result
        assert "must be under /tmp or home directory" in result.get("message", "")


class TestErrorHandling:
    def test_request_exception_returns_error_dict(self):
        import requests
        with patch("tools.kimi_webbridge.requests.post", side_effect=requests.ConnectionError("boom")):
            result = json.loads(kimi_webbridge_navigate("https://example.com"))
        assert result["error"] is True
        assert "boom" in result["message"]
