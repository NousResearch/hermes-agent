"""Tests for the Camofox browser backend."""

import json
from unittest.mock import MagicMock, patch


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

    def test_enabled_when_url_set(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        assert is_camofox_mode() is True

    def test_cdp_override_takes_priority(self, monkeypatch):
        """When BROWSER_CDP_URL is set (via /browser connect), CDP takes priority over Camofox."""
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
        assert is_camofox_mode() is False

    def test_cdp_override_blank_does_not_disable_camofox(self, monkeypatch):
        """Empty/whitespace BROWSER_CDP_URL should not suppress Camofox."""
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("BROWSER_CDP_URL", "  ")
        assert is_camofox_mode() is True

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
    def test_rewrite_is_opt_in(self, mock_config, monkeypatch):
        monkeypatch.delenv("CAMOFOX_REWRITE_LOOPBACK_URLS", raising=False)
        mock_config.return_value = _config_with_camofox(rewrite_loopback_urls=False)

        rewritten, metadata = _rewrite_loopback_url_for_camofox("http://localhost:3000/app?x=1")

        assert rewritten == "http://localhost:3000/app?x=1"
        assert metadata is None

    @patch("tools.browser_camofox.load_config")
    def test_preserves_public_urls_when_enabled(self, mock_config, monkeypatch):
        monkeypatch.delenv("CAMOFOX_REWRITE_LOOPBACK_URLS", raising=False)
        mock_config.return_value = _config_with_camofox(rewrite_loopback_urls=True)

        rewritten, metadata = _rewrite_loopback_url_for_camofox("https://example.com:8443/path?q=1#top")

        assert rewritten == "https://example.com:8443/path?q=1#top"
        assert metadata is None

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

    @patch("tools.browser_camofox.load_config")
    @patch("tools.browser_camofox.requests.post")
    def test_navigate_uses_rewritten_loopback_url(self, mock_post, mock_config, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("CAMOFOX_REWRITE_LOOPBACK_URLS", raising=False)
        monkeypatch.delenv("CAMOFOX_LOOPBACK_HOST_ALIAS", raising=False)
        mock_config.return_value = _config_with_camofox(rewrite_loopback_urls=True)
        mock_post.return_value = _mock_response(json_data={"tabId": "tab_rewrite"})

        result = json.loads(camofox_navigate("http://127.0.0.1:8766/#settings", task_id="t_rewrite"))

        assert result["success"] is True
        assert result["url"] == "http://host.docker.internal:8766/#settings"
        assert result["requested_url"] == "http://127.0.0.1:8766/#settings"
        assert result["url_rewrite"]["to"] == "host.docker.internal"
        assert "Rewrote loopback URL" in result["warning"]
        assert mock_post.call_args.kwargs["json"]["url"] == "http://host.docker.internal:8766/#settings"

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
        assert result["typed"] == "[redacted typed text]"

    @patch("tools.browser_camofox.requests.post")
    def test_type_never_echoes_raw_secret(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab5b", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t5b")

        typed_text = "my_secret_password_123"
        mock_post.return_value = _mock_response(json_data={"ok": True})
        result = json.loads(camofox_type("@password", typed_text, task_id="t5b"))
        assert result["success"] is True
        assert typed_text not in json.dumps(result)
        assert result["typed"] == "[redacted typed text]"

    @patch("tools.browser_camofox.requests.post")
    def test_type_failure_never_echoes_raw_secret(self, mock_post, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(json_data={"tabId": "tab5c", "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="t5c")

        typed_text = "my_secret_password_123"
        mock_post.side_effect = RuntimeError(f"camofox failed while typing {typed_text}")
        raw_result = camofox_type("@password", typed_text, task_id="t5c")
        result = json.loads(raw_result)

        assert result["success"] is False
        assert typed_text not in raw_result
        assert "[redacted typed text]" in raw_result

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
        from tools.browser_tool import check_browser_requirements
        assert check_browser_requirements() is True


