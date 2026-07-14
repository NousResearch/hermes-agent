"""Tests for the Camofox browser backend."""

import json
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
        # Normal text is left readable.
        assert result["typed"] == "hello world"

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


# ---------------------------------------------------------------------------
# Cookie import — Netscape parser
# ---------------------------------------------------------------------------


class TestNetscapeParser:
    """Unit tests for the Netscape cookies.txt parser."""

    def test_basic_seven_field_line(self):
        from tools.browser_camofox import _netscape_parse
        text = ".example.com\tTRUE\t/\tFALSE\t1735689600\tsession\tabc123"
        cookies = _netscape_parse(text)
        assert cookies == [{
            "name": "session",
            "value": "abc123",
            "domain": ".example.com",
            "path": "/",
            "expires": 1735689600,
            "httpOnly": False,
            "secure": False,
        }]

    def test_httponly_prefix_recognized(self):
        from tools.browser_camofox import _netscape_parse
        text = "#HttpOnly_.example.com\tTRUE\t/\tTRUE\t0\tauth\ttoken"
        cookies = _netscape_parse(text)
        assert len(cookies) == 1
        assert cookies[0]["httpOnly"] is True
        assert cookies[0]["secure"] is True
        assert cookies[0]["domain"] == ".example.com"
        assert cookies[0]["name"] == "auth"

    def test_skips_regular_comments_and_blank_lines(self):
        from tools.browser_camofox import _netscape_parse
        text = (
            "# Netscape HTTP Cookie File\n"
            "\n"
            "# generated by Firefox extension\n"
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n"
            "\n"
        )
        cookies = _netscape_parse(text)
        assert len(cookies) == 1
        assert cookies[0]["name"] == "foo"

    def test_strips_utf8_bom(self):
        from tools.browser_camofox import _netscape_parse
        text = "\ufeff.example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar"
        cookies = _netscape_parse(text)
        assert len(cookies) == 1
        assert cookies[0]["domain"] == ".example.com"

    def test_value_preserves_embedded_tabs(self):
        """Upstream parses value as parts.slice(6).join('\\t')."""
        from tools.browser_camofox import _netscape_parse
        text = ".example.com\tTRUE\t/\tFALSE\t0\ttoken\taaa\tbbb\tccc"
        cookies = _netscape_parse(text)
        assert cookies[0]["value"] == "aaa\tbbb\tccc"

    def test_malformed_lines_skipped(self):
        from tools.browser_camofox import _netscape_parse
        text = (
            "not\tenough\tfields\n"
            ".good.com\tTRUE\t/\tFALSE\t0\tname\tvalue\n"
        )
        cookies = _netscape_parse(text)
        assert len(cookies) == 1
        assert cookies[0]["domain"] == ".good.com"

    def test_handles_crlf_line_endings(self):
        from tools.browser_camofox import _netscape_parse
        text = ".a.com\tTRUE\t/\tFALSE\t0\tn1\tv1\r\n.b.com\tTRUE\t/\tFALSE\t0\tn2\tv2\r\n"
        cookies = _netscape_parse(text)
        assert len(cookies) == 2
        assert {c["domain"] for c in cookies} == {".a.com", ".b.com"}


# ---------------------------------------------------------------------------
# Cookie import — file reader
# ---------------------------------------------------------------------------


class TestReadCookieFile:
    """Integration tests for _read_cookie_file with real tmp filesystem."""

    def _write(self, tmp_path, name, content):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def _configure_dir(self, monkeypatch, path):
        import tools.browser_camofox as camofox
        monkeypatch.setattr(
            camofox,
            "_get_camofox_config",
            lambda: {"cookies_dir": str(path)},
        )

    def test_reads_file_within_cookies_dir(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        self._configure_dir(monkeypatch, tmp_path)
        self._write(tmp_path, "site.txt", ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n")

        cookies = _read_cookie_file("site.txt")
        assert len(cookies) == 1
        assert cookies[0]["name"] == "foo"

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        cookies_dir = tmp_path / "cookies"
        cookies_dir.mkdir()
        (tmp_path / "secret.txt").write_text(".evil\tTRUE\t/\tFALSE\t0\tleaked\tvalue\n")
        self._configure_dir(monkeypatch, cookies_dir)

        with pytest.raises(ValueError, match="relative path"):
            _read_cookie_file("../secret.txt")

    def test_absolute_path_blocked(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        self._configure_dir(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="relative path"):
            _read_cookie_file("/etc/passwd")

    def test_file_too_large_rejected(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        self._configure_dir(monkeypatch, tmp_path)
        big = tmp_path / "big.txt"
        big.write_text("x" * (6 * 1024 * 1024), encoding="utf-8")

        with pytest.raises(ValueError, match="too large"):
            _read_cookie_file("big.txt")

    def test_domain_suffix_filter(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        self._configure_dir(monkeypatch, tmp_path)
        self._write(
            tmp_path, "mixed.txt",
            ".linkedin.com\tTRUE\t/\tFALSE\t0\tli_at\ttokA\n"
            ".google.com\tTRUE\t/\tFALSE\t0\tSID\ttokB\n"
            "www.linkedin.com\tTRUE\t/\tFALSE\t0\tJSESSIONID\ttokC\n"
        )
        cookies = _read_cookie_file("mixed.txt", domain_suffix=".linkedin.com")
        assert {c["name"] for c in cookies} == {"li_at", "JSESSIONID"}

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file
        self._configure_dir(monkeypatch, tmp_path)
        with pytest.raises(FileNotFoundError):
            _read_cookie_file("does_not_exist.txt")

    def test_default_cookies_dir_is_home(self, monkeypatch):
        from tools.browser_camofox import _resolve_cookies_dir
        import tools.browser_camofox as camofox
        monkeypatch.setattr(camofox, "_get_camofox_config", lambda: {})
        resolved = _resolve_cookies_dir()
        assert resolved.endswith("/.camofox/cookies")

    def test_cookies_dir_is_a_config_field_not_an_optional_env_var(self):
        from hermes_cli.config import DEFAULT_CONFIG, OPTIONAL_ENV_VARS

        assert DEFAULT_CONFIG["browser"]["camofox"]["cookies_dir"] == "~/.camofox/cookies"
        assert "CAMOFOX_COOKIES_DIR" not in OPTIONAL_ENV_VARS

    def test_reads_cookies_dir_through_real_config_chain(self, tmp_path, monkeypatch):
        from tools.browser_camofox import _read_cookie_file

        hermes_home = tmp_path / "hermes-home"
        configured_dir = tmp_path / "configured-cookies"
        legacy_env_dir = tmp_path / "legacy-env-cookies"
        hermes_home.mkdir()
        configured_dir.mkdir()
        legacy_env_dir.mkdir()
        (hermes_home / "config.yaml").write_text(
            "browser:\n"
            "  camofox:\n"
            f"    cookies_dir: {configured_dir}\n",
            encoding="utf-8",
        )
        self._write(
            configured_dir,
            "site.txt",
            ".example.com\tTRUE\t/\tFALSE\t0\tsource\tconfig\n",
        )
        self._write(
            legacy_env_dir,
            "site.txt",
            ".example.com\tTRUE\t/\tFALSE\t0\tsource\tlegacy-env\n",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("CAMOFOX_COOKIES_DIR", str(legacy_env_dir))

        cookies = _read_cookie_file("site.txt")

        assert cookies[0]["value"] == "config"


# ---------------------------------------------------------------------------
# Cookie import — camofox_import_cookies HTTP entry point
# ---------------------------------------------------------------------------


class TestCamofoxImportCookies:
    """Tests for the top-level tool entry point that reads the file and POSTs."""

    def _prep(self, tmp_path, monkeypatch, api_key="k-secret-123"):
        import tools.browser_camofox as camofox
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr(
            camofox,
            "_get_camofox_config",
            lambda: {"cookies_dir": str(tmp_path)},
        )
        if api_key is not None:
            monkeypatch.setenv("CAMOFOX_API_KEY", api_key)
        else:
            monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)

    def test_missing_api_key_returns_error(self, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch, api_key=None)
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        result = json.loads(camofox_import_cookies("x.txt", task_id="t1"))
        assert result["success"] is False
        assert "CAMOFOX_API_KEY" in result["error"]

    @patch("tools.browser_camofox.requests.post")
    def test_sends_bearer_token(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch, api_key="k-secret-123")
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        mock_post.return_value = _mock_response(
            json_data={"ok": True, "userId": "u", "count": 1}
        )

        result = json.loads(camofox_import_cookies("x.txt", task_id="t1"))
        assert result["success"] is True
        assert result["imported"] == 1

        call = mock_post.call_args
        headers = call.kwargs.get("headers") or {}
        assert headers.get("Authorization") == "Bearer k-secret-123"

        url = call.args[0] if call.args else call.kwargs.get("url", "")
        assert "/sessions/" in url and url.endswith("/cookies")

        body = call.kwargs.get("json") or {}
        assert "cookies" in body
        assert isinstance(body["cookies"], list)
        assert body["cookies"][0]["name"] == "foo"

    @patch("tools.browser_camofox.requests.post")
    def test_uses_configured_command_timeout(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        mock_post.return_value = _mock_response(json_data={"ok": True, "count": 1})

        with patch("tools.browser_camofox._get_command_timeout", return_value=73):
            camofox_import_cookies("x.txt", task_id="t1")
        assert mock_post.call_args.kwargs["timeout"] == 73

    @patch("tools.browser_camofox.requests.post")
    def test_body_sanitized_to_allowlist(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        mock_post.return_value = _mock_response(json_data={"ok": True, "count": 1})

        camofox_import_cookies("x.txt", task_id="t1")
        allowed = {"name", "value", "domain", "path", "expires", "httpOnly", "secure", "sameSite"}
        sent = mock_post.call_args.kwargs["json"]["cookies"][0]
        assert set(sent.keys()).issubset(allowed)

    @patch("tools.browser_camofox.requests.post")
    def test_server_403_surfaced(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        import requests as _req
        resp = MagicMock()
        resp.status_code = 403
        resp.json.return_value = {"error": "Forbidden"}
        resp.raise_for_status.side_effect = _req.HTTPError("403 Forbidden", response=resp)
        mock_post.return_value = resp

        result = json.loads(camofox_import_cookies("x.txt", task_id="t1"))
        assert result["success"] is False
        assert "403" in result["error"] or "Forbidden" in result["error"]

    @patch("tools.browser_camofox.requests.post")
    def test_server_400_surfaced(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "x.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        import requests as _req
        resp = MagicMock()
        resp.status_code = 400
        resp.json.return_value = {"error": "bad"}
        resp.raise_for_status.side_effect = _req.HTTPError("400 Bad Request", response=resp)
        mock_post.return_value = resp

        result = json.loads(camofox_import_cookies("x.txt", task_id="t1"))
        assert result["success"] is False

    def test_max_500_cookies_enforced(self, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        lines = [
            f".example{i}.com\tTRUE\t/\tFALSE\t0\tn{i}\tv{i}"
            for i in range(501)
        ]
        (tmp_path / "big.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = json.loads(camofox_import_cookies("big.txt", task_id="t1"))
        assert result["success"] is False
        assert "500" in result["error"]

    @patch("tools.browser_camofox.requests.post")
    def test_domain_suffix_forwarded_to_reader(self, mock_post, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "mixed.txt").write_text(
            ".linkedin.com\tTRUE\t/\tFALSE\t0\tli_at\ttokA\n"
            ".google.com\tTRUE\t/\tFALSE\t0\tSID\ttokB\n",
            encoding="utf-8",
        )
        mock_post.return_value = _mock_response(json_data={"ok": True, "count": 1})

        result = json.loads(camofox_import_cookies(
            "mixed.txt", domain_suffix=".linkedin.com", task_id="t1"
        ))
        assert result["success"] is True
        assert result["imported"] == 1
        sent_names = [c["name"] for c in mock_post.call_args.kwargs["json"]["cookies"]]
        assert sent_names == ["li_at"]

    def test_empty_file_returns_helpful_error(self, tmp_path, monkeypatch):
        from tools.browser_camofox import camofox_import_cookies
        self._prep(tmp_path, monkeypatch)
        (tmp_path / "empty.txt").write_text("# only comments\n", encoding="utf-8")
        result = json.loads(camofox_import_cookies("empty.txt", task_id="t1"))
        assert result["success"] is False
        assert "no cookies" in result["error"].lower()


# ---------------------------------------------------------------------------
# browser_import_cookies — delegation + camofox-only gating
# ---------------------------------------------------------------------------


class TestBrowserImportCookiesRouting:
    """Verify browser_tool.browser_import_cookies delegates correctly."""

    @patch("tools.browser_camofox.requests.post")
    def test_delegates_to_camofox_when_enabled(self, mock_post, tmp_path, monkeypatch):
        import tools.browser_camofox as camofox
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setattr(
            camofox,
            "_get_camofox_config",
            lambda: {"cookies_dir": str(tmp_path)},
        )
        monkeypatch.setenv("CAMOFOX_API_KEY", "k")
        (tmp_path / "c.txt").write_text(
            ".example.com\tTRUE\t/\tFALSE\t0\tfoo\tbar\n", encoding="utf-8"
        )
        mock_post.return_value = _mock_response(json_data={"ok": True, "count": 1})

        from tools.browser_tool import browser_import_cookies
        result = json.loads(browser_import_cookies("c.txt", task_id="t_route"))
        assert result["success"] is True
        assert result["imported"] == 1

    def test_errors_clearly_when_not_camofox(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        from tools.browser_tool import browser_import_cookies
        result = json.loads(browser_import_cookies("anything.txt", task_id="t_no"))
        assert result["success"] is False
        assert "camofox" in result["error"].lower()

    def test_hidden_from_registry_without_camofox(self, monkeypatch):
        """check_fn should gate the tool so non-camofox users don't see it."""
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        from tools.browser_tool import _check_import_cookies_requirements
        assert _check_import_cookies_requirements() is False

    @pytest.mark.parametrize("api_key", [None, "", "   "])
    def test_hidden_from_registry_without_api_key(self, monkeypatch, api_key):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        if api_key is None:
            monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)
        else:
            monkeypatch.setenv("CAMOFOX_API_KEY", api_key)
        from tools.browser_tool import _check_import_cookies_requirements
        assert _check_import_cookies_requirements() is False

    def test_visible_when_camofox_enabled(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_API_KEY", "secret")
        from tools.browser_tool import _check_import_cookies_requirements
        assert _check_import_cookies_requirements() is True
