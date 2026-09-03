"""Tests for the Camofox browser backend."""

import json

import requests
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

    def test_stale_tab_410_recreates_tab(self, monkeypatch):
        """HTTP 410 Gone after camofox-browser restart must recover like 404 (#80276)."""
        import tools.browser_camofox as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with mod._sessions_lock:
            mod._sessions["t_410"] = {
                "user_id": "hermes_test",
                "tab_id": "dead-tab",
                "session_key": "task_t_410",
                "managed": False,
                "adopt_existing_tab": False,
            }

        gone = requests.HTTPError("410 Gone")
        gone.response = MagicMock(status_code=410)
        navigate_calls: list[str] = []

        def _post_side_effect(path, body=None, **kwargs):
            navigate_calls.append(path)
            if path == "/tabs/dead-tab/navigate":
                raise gone
            return {"ok": True, "url": (body or {}).get("url", "")}

        def _ensure_tab(task_id, url="about:blank"):
            session = mod._get_session(task_id)
            session["tab_id"] = "fresh-tab"
            return session

        with patch.object(mod, "_post", side_effect=_post_side_effect), \
             patch.object(mod, "_get", return_value={"snapshot": "", "refsCount": 0}), \
             patch.object(mod, "_ensure_tab", side_effect=_ensure_tab) as mock_ensure:
            result = json.loads(camofox_navigate("https://example.com/next", task_id="t_410"))

        assert result["success"] is True
        assert navigate_calls[0] == "/tabs/dead-tab/navigate"
        assert navigate_calls[1] == "/tabs/fresh-tab/navigate"
        assert result["url"] == "https://example.com/next"
        mock_ensure.assert_called()
        with mod._sessions_lock:
            assert mod._sessions["t_410"]["tab_id"] == "fresh-tab"

    def test_navigate_posts_to_adopted_tab(self, monkeypatch):
        """Adopted tabs must receive the requested /navigate, not a synthesized URL."""
        import tools.browser_camofox as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with mod._sessions_lock:
            mod._sessions["t_adopt"] = {
                "user_id": "hermes_test",
                "tab_id": "adopted-tab",
                "session_key": "task_t_adopt",
                "managed": True,
                "adopt_existing_tab": True,
            }

        posts: list[tuple[str, str]] = []

        def _post_side_effect(path, body=None, **kwargs):
            posts.append((path, (body or {}).get("url", "")))
            return {"ok": True, "url": (body or {}).get("url", ""), "title": "live"}

        with patch.object(mod, "_post", side_effect=_post_side_effect), \
             patch.object(mod, "_get", return_value={"snapshot": "", "refsCount": 0}), \
             patch.object(mod, "_ensure_tab") as mock_ensure:
            result = json.loads(
                camofox_navigate("https://example.com/target", task_id="t_adopt")
            )

        mock_ensure.assert_not_called()
        assert posts == [("/tabs/adopted-tab/navigate", "https://example.com/target")]
        assert result["success"] is True
        assert result["url"] == "https://example.com/target"



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
# Stale tab 404/410 cleanup — sibling ops clear tab_id so navigate recovers
# (#54729 follow-up + #80276). No blind action replay.
# ---------------------------------------------------------------------------


def _mock_stale_response(status_code: int):
    """Build a requests.Response that raises HTTPError for stale-tab codes."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {}
    resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


class TestStaleTabCleanup:
    """404 GC and 410 post-restart must clear cached tab_id on sibling ops."""

    def _setup_session(self, mock_post, monkeypatch, task_id):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        mock_post.return_value = _mock_response(
            json_data={"tabId": "stale-tab", "url": "https://x.com"}
        )
        camofox_navigate("https://x.com", task_id=task_id)

    @patch("tools.browser_camofox.requests.get")
    @patch("tools.browser_camofox.requests.post")
    def test_snapshot_clears_tab_on_404(self, mock_post, mock_get, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_snap")
        mock_get.return_value = _mock_stale_response(404)
        result = json.loads(camofox_snapshot(task_id="stale_snap"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_snap")["tab_id"] is None

    @patch("tools.browser_camofox.requests.get")
    @patch("tools.browser_camofox.requests.post")
    def test_snapshot_clears_tab_on_410(self, mock_post, mock_get, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_snap_410")
        mock_get.return_value = _mock_stale_response(410)
        result = json.loads(camofox_snapshot(task_id="stale_snap_410"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_snap_410")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_click_clears_tab_on_410(self, mock_post, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_click")
        mock_post.return_value = _mock_stale_response(410)
        result = json.loads(camofox_click("e1", task_id="stale_click"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_click")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_type_clears_tab_on_404(self, mock_post, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_type")
        mock_post.return_value = _mock_stale_response(404)
        result = json.loads(camofox_type("e1", "hello", task_id="stale_type"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_type")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_scroll_clears_tab_on_404(self, mock_post, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_scroll")
        mock_post.return_value = _mock_stale_response(404)
        result = json.loads(camofox_scroll("down", task_id="stale_scroll"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_scroll")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_back_clears_tab_on_404(self, mock_post, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_back")
        mock_post.return_value = _mock_stale_response(404)
        result = json.loads(camofox_back(task_id="stale_back"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_back")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_press_clears_tab_on_404(self, mock_post, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_press")
        mock_post.return_value = _mock_stale_response(404)
        result = json.loads(camofox_press("Enter", task_id="stale_press"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_press")["tab_id"] is None

    @patch("tools.browser_camofox.requests.get")
    @patch("tools.browser_camofox.requests.post")
    def test_get_images_clears_tab_on_404(self, mock_post, mock_get, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_img")
        mock_get.return_value = _mock_stale_response(404)
        result = json.loads(camofox_get_images(task_id="stale_img"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_img")["tab_id"] is None

    @patch("tools.browser_camofox._get_raw")
    @patch("tools.browser_camofox.requests.post")
    def test_vision_clears_tab_on_410(self, mock_post, mock_get_raw, monkeypatch):
        self._setup_session(mock_post, monkeypatch, "stale_vis")
        mock_get_raw.side_effect = requests.HTTPError(
            response=MagicMock(status_code=410),
        )
        result = json.loads(camofox_vision("what?", task_id="stale_vis"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        from tools.browser_camofox import _get_session
        assert _get_session("stale_vis")["tab_id"] is None

    @patch("tools.browser_camofox.requests.post")
    def test_non_stale_error_keeps_tab_id(self, mock_post, monkeypatch):
        """A 500 must NOT clear tab_id — only 404/410 trigger cleanup."""
        self._setup_session(mock_post, monkeypatch, "stale_500")
        mock_post.return_value = _mock_stale_response(500)
        result = json.loads(camofox_click("e1", task_id="stale_500"))
        assert result["success"] is False
        from tools.browser_camofox import _get_session
        assert _get_session("stale_500")["tab_id"] == "stale-tab"

    @patch("tools.browser_camofox.requests.get")
    @patch("tools.browser_camofox.requests.post")
    def test_navigate_recovers_after_sibling_clears_tab(self, mock_post, mock_get, monkeypatch):
        """After click clears a stale tab, navigate creates a fresh one."""
        self._setup_session(mock_post, monkeypatch, "stale_recover")
        mock_post.return_value = _mock_stale_response(410)
        camofox_click("e1", task_id="stale_recover")

        mock_post.return_value = _mock_response(
            json_data={"tabId": "fresh-tab", "url": "https://y.com"}
        )
        mock_get.return_value = _mock_response(json_data={"snapshot": "", "refsCount": 0})
        result = json.loads(camofox_navigate("https://y.com", task_id="stale_recover"))
        assert result["success"] is True
        assert result["url"] == "https://y.com"
        from tools.browser_camofox import _get_session
        assert _get_session("stale_recover")["tab_id"] == "fresh-tab"

    @patch("tools.browser_camofox.requests.get")
    @patch("tools.browser_camofox.requests.post")
    def test_bonus_snapshot_410_clears_tab_after_successful_navigate(
        self, mock_post, mock_get, monkeypatch
    ):
        """Post-navigation bonus snapshot must invalidate a tab that dies mid-call."""
        mock_get.return_value = _mock_response(json_data={"snapshot": "", "refsCount": 0})
        self._setup_session(mock_post, monkeypatch, "bonus_snap")
        mock_post.return_value = _mock_response(
            json_data={"ok": True, "url": "https://y.com", "title": ""}
        )
        mock_get.return_value = _mock_stale_response(410)
        result = json.loads(camofox_navigate("https://y.com", task_id="bonus_snap"))
        assert result["success"] is True
        from tools.browser_camofox import _get_session
        assert _get_session("bonus_snap")["tab_id"] is None

    @patch("tools.browser_camofox._get")
    @patch("tools.browser_camofox._get_raw")
    @patch("tools.browser_camofox.requests.post")
    def test_vision_annotation_snapshot_410_clears_tab(
        self, mock_post, mock_get_raw, mock_get, monkeypatch, tmp_path
    ):
        mock_get.return_value = {"snapshot": "", "refsCount": 0}
        self._setup_session(mock_post, monkeypatch, "vis_ann")
        mock_get_raw.return_value = MagicMock(content=b"\x89PNG\r\n\x1a\nfake")
        mock_get.side_effect = requests.HTTPError(
            response=MagicMock(status_code=410, json=lambda: {}),
        )
        llm = MagicMock()
        llm.choices = [MagicMock(message=MagicMock(content="ok"))]
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path), \
             patch("agent.auxiliary_client.call_llm", return_value=llm), \
             patch("agent.redact.redact_sensitive_text", side_effect=lambda s: s):
            result = json.loads(
                camofox_vision("what?", annotate=True, task_id="vis_ann")
            )
        assert result["success"] is True
        from tools.browser_camofox import _get_session
        assert _get_session("vis_ann")["tab_id"] is None


class TestCamofoxEvalStaleVsCapability:
    def _session(self, monkeypatch, task_id="eval_t"):
        import tools.browser_camofox as mod

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        session = {
            "user_id": "hermes_test",
            "tab_id": "eval-tab",
            "session_key": f"task_{task_id}",
            "managed": False,
            "adopt_existing_tab": False,
        }
        with mod._sessions_lock:
            mod._sessions[task_id] = session
        return session

    def test_eval_410_clears_tab_and_asks_navigate(self, monkeypatch):
        from tools.browser_tool import _camofox_eval
        import tools.browser_camofox as mod

        session = self._session(monkeypatch)
        gone = requests.HTTPError("410 Gone")
        gone.response = MagicMock(status_code=410, json=lambda: {"recovery": "create_new_tab"})

        with patch.object(mod, "_ensure_tab", return_value=session), \
             patch.object(mod, "_post", side_effect=gone):
            result = json.loads(_camofox_eval("1+1", task_id="eval_t"))

        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        assert session["tab_id"] is None

    def test_eval_404_without_tab_payload_is_capability(self, monkeypatch):
        from tools.browser_tool import _camofox_eval
        import tools.browser_camofox as mod

        session = self._session(monkeypatch, "eval_cap")
        missing = requests.HTTPError("404")
        missing.response = MagicMock(status_code=404, json=lambda: {}, text="Not Found")

        with patch.object(mod, "_ensure_tab", return_value=session), \
             patch.object(mod, "_post", side_effect=missing):
            result = json.loads(_camofox_eval("1+1", task_id="eval_cap"))

        assert result["success"] is False
        assert "not supported" in result["error"]
        assert session["tab_id"] == "eval-tab"

    def test_eval_404_tab_missing_payload_is_stale(self, monkeypatch):
        from tools.browser_tool import _camofox_eval
        import tools.browser_camofox as mod

        session = self._session(monkeypatch, "eval_miss")
        missing = requests.HTTPError("404")
        missing.response = MagicMock(
            status_code=404,
            json=lambda: {"code": "tab_destroyed", "recovery": "create_new_tab"},
            text="tab destroyed",
        )

        with patch.object(mod, "_ensure_tab", return_value=session), \
             patch.object(mod, "_post", side_effect=missing):
            result = json.loads(_camofox_eval("1+1", task_id="eval_miss"))

        assert result["success"] is False
        assert "browser_navigate" in result["error"]
        assert session["tab_id"] is None

    def test_ssrf_probe_410_clears_tab(self, monkeypatch):
        from tools.browser_tool import _camofox_current_page_private_url
        import tools.browser_camofox as mod

        session = self._session(monkeypatch, "probe_t")
        gone = requests.HTTPError("410 Gone")
        gone.response = MagicMock(status_code=410, json=lambda: {})

        with patch.object(mod, "_post", side_effect=gone):
            assert _camofox_current_page_private_url(
                "eval-tab", "hermes_test", session=session
            ) is None
        assert session["tab_id"] is None


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


