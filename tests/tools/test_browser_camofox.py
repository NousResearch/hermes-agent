"""Tests for the Camofox browser backend."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests


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
    _rewrite_loopback_url_for_camofox,
)


# ---------------------------------------------------------------------------
# Configuration detection
# ---------------------------------------------------------------------------


class TestCamofoxMode:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        assert is_camofox_mode() is False


    def test_health_check_unreachable(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:19999")
        assert check_camofox_available() is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_with_camofox(**camofox_config):
    return {"browser": {"camofox": camofox_config}}


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


class TestCamofoxLoopbackRewrite:
    @patch("tools.browser_camofox.load_config")
    def test_rewrites_localhost_when_enabled(self, mock_config, monkeypatch):
        monkeypatch.delenv("CAMOFOX_REWRITE_LOOPBACK_URLS", raising=False)
        monkeypatch.delenv("CAMOFOX_LOOPBACK_HOST_ALIAS", raising=False)
        mock_config.return_value = _config_with_camofox(rewrite_loopback_urls=True)

        rewritten, metadata = _rewrite_loopback_url_for_camofox("http://127.0.0.1:8766/#settings")

        assert rewritten == "http://host.docker.internal:8766/#settings"
        assert metadata == {
            "from": "127.0.0.1",
            "to": "host.docker.internal",
            "original_url": "http://127.0.0.1:8766/#settings",
            "rewritten_url": "http://host.docker.internal:8766/#settings",
        }


    @patch("tools.browser_camofox.load_config")
    def test_env_alias_takes_precedence(self, mock_config, monkeypatch):
        monkeypatch.setenv("CAMOFOX_REWRITE_LOOPBACK_URLS", "true")
        monkeypatch.setenv("CAMOFOX_LOOPBACK_HOST_ALIAS", "192.168.1.10")
        mock_config.return_value = _config_with_camofox(
            rewrite_loopback_urls=False,
            loopback_host_alias="host.docker.internal",
        )

        rewritten, metadata = _rewrite_loopback_url_for_camofox("http://[::1]:8080/path")

        assert rewritten == "http://192.168.1.10:8080/path"
        assert metadata is not None
        assert metadata["from"] == "::1"
        assert metadata["to"] == "192.168.1.10"


class TestCamofoxNavigate:
    @patch("tools.browser_camofox.requests.post")
    def test_creates_tab_on_first_navigate(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab1", "url": "https://example.com"})

        result = json.loads(camofox_navigate("https://example.com", task_id="t1"))
        assert result["success"] is True
        assert result["url"] == "https://example.com"


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
    def test_type_redacts_api_key(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab5b", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t5b")

        secret = "sk-proj-ABCD1234567890EFGH"
        mock_post.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_type("@apikey", secret, task_id="t5b"))
        assert result["success"] is True
        assert secret not in json.dumps(result)
        assert result["typed"].startswith("sk-pro")

    @patch("tools.browser_camofox.requests.post")
    def test_type_failure_redacts_api_key(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab5c", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t5c")

        secret = "sk-proj-ABCD1234567890EFGH"
        mock_post.side_effect = RuntimeError(f"camofox failed while typing {secret}")
        raw_result = camofox_type("@apikey", secret, task_id="t5c")
        result = json.loads(raw_result)

        assert result["success"] is False
        assert secret not in raw_result
        assert "sk-pro" in raw_result


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


class TestCamofoxVisionConfig:
    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox._get")
    @patch("tools.browser_camofox._get_raw")
    def test_camofox_vision_uses_configured_temperature_and_timeout(self, mock_get_raw, mock_get, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab11", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t11")

        snapshot_text = '- button "Submit"\n'
        raw_resp = MagicMock()
        raw_resp.content = b"fakepng"
        mock_get_raw.return_value = raw_resp
        mock_get.return_value = {"snapshot": snapshot_text}

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Camofox screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("tools.browser_camofox.open", create=True) as mock_open,
            patch("agent.auxiliary_client.call_llm", return_value=mock_response) as mock_llm,
            patch("tools.browser_camofox.load_config", return_value={"auxiliary": {"vision": {"temperature": 1, "timeout": 45}}}),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"fakepng"
            result = json.loads(camofox_vision("what is on the page?", annotate=True, task_id="t11"))

        assert result["success"] is True
        assert result["analysis"] == "Camofox screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 1.0
        assert mock_llm.call_args.kwargs["timeout"] == 45.0

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox._get")
    @patch("tools.browser_camofox._get_raw")
    def test_camofox_vision_defaults_temperature_when_config_omits_it(self, mock_get_raw, mock_get, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab12", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t12")

        snapshot_text = '- button "Submit"\n'
        raw_resp = MagicMock()
        raw_resp.content = b"fakepng"
        mock_get_raw.return_value = raw_resp
        mock_get.return_value = {"snapshot": snapshot_text}

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Default camofox screenshot analysis"
        mock_response.choices = [mock_choice]

        with (
            patch("tools.browser_camofox.open", create=True) as mock_open,
            patch("agent.auxiliary_client.call_llm", return_value=mock_response) as mock_llm,
            patch("tools.browser_camofox.load_config", return_value={"auxiliary": {"vision": {}}}),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"fakepng"
            result = json.loads(camofox_vision("what is on the page?", annotate=True, task_id="t12"))

        assert result["success"] is True
        assert result["analysis"] == "Default camofox screenshot analysis"
        assert mock_llm.call_args.kwargs["temperature"] == 0.1
        assert mock_llm.call_args.kwargs["timeout"] == 120.0


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
        from tools.browser_tool_install import check_browser_requirements
        assert check_browser_requirements() is True




# ---------------------------------------------------------------------------
# HTTP error detail — camofox's own explanation must survive raise_for_status()
# ---------------------------------------------------------------------------


def _failing_response(status=500, json_data=None, text="", url="http://localhost:9377/tabs/t1/evaluate"):
    """A response whose raise_for_status() fails the way requests' does."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    if json_data is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_data
    reason = "Internal Server Error" if status >= 500 else "Bad Request"
    resp.raise_for_status.side_effect = requests.HTTPError(
        f"{status} Server Error: {reason} for url: {url}", response=resp
    )
    return resp


class TestErrorDetail:
    def test_extracts_error_field(self):
        from tools.browser_camofox import _error_detail
        resp = _failing_response(json_data={"error": "page.evaluate: return not in function"})
        assert _error_detail(resp) == "page.evaluate: return not in function"

    def test_appends_structured_code(self):
        from tools.browser_camofox import _error_detail
        resp = _failing_response(
            status=400,
            json_data={"error": "page.evaluate: return not in function",
                       "code": "invalid_expression", "retryable": False},
        )
        detail = _error_detail(resp)
        assert "return not in function" in detail
        assert "code: invalid_expression" in detail

    def test_falls_back_to_text_when_not_json(self):
        from tools.browser_camofox import _error_detail
        resp = _failing_response(text="<html><body>502 Bad Gateway</body></html>")
        assert "502 Bad Gateway" in _error_detail(resp)

    def test_collapses_whitespace(self):
        from tools.browser_camofox import _error_detail
        resp = _failing_response(json_data={"error": "line one\n  line two\t\tline three"})
        assert _error_detail(resp) == "line one line two line three"

    def test_caps_long_bodies(self):
        from tools.browser_camofox import _error_detail, _MAX_ERROR_DETAIL_CHARS
        resp = _failing_response(text="x" * 10_000)
        detail = _error_detail(resp)
        assert len(detail) == _MAX_ERROR_DETAIL_CHARS + 3
        assert detail.endswith("...")

    def test_empty_body_yields_empty_detail(self):
        from tools.browser_camofox import _error_detail
        assert _error_detail(_failing_response(text="")) == ""


class TestRequestSurfacesServerError:
    @patch("tools.browser_camofox.requests.post")
    def test_server_message_reaches_the_caller(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_camofox import _post
        mock_post.return_value = _failing_response(
            json_data={"error": "page.evaluate: return not in function"}
        )

        with pytest.raises(requests.HTTPError) as excinfo:
            _post("/tabs/t1/evaluate", {"userId": "u", "expression": "return document.title"})

        message = str(excinfo.value)
        assert "return not in function" in message
        # the original status line is kept so existing status-code checks still match
        assert "500" in message

    @patch("tools.browser_camofox.requests.post")
    def test_status_line_preserved_when_body_is_empty(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_camofox import _post
        mock_post.return_value = _failing_response(text="")

        with pytest.raises(requests.HTTPError) as excinfo:
            _post("/tabs/t1/evaluate", {"userId": "u", "expression": "document.title"})

        assert "500 Server Error" in str(excinfo.value)

    @patch("tools.browser_camofox.requests.post")
    def test_successful_response_is_untouched(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_camofox import _post
        mock_post.return_value = _mock_response(json_data={"ok": True, "result": "Example Domain"})

        assert _post("/tabs/t1/evaluate", {"userId": "u", "expression": "document.title"}) == {
            "ok": True, "result": "Example Domain"
        }
