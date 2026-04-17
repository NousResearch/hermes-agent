from unittest.mock import Mock, patch


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"
BROWSERLESS_LAUNCH_URL_1 = (
    "wss://browserless.example/chromium?token=abc&"
    "launch=%7B%22stealth%22%3Atrue%2C%22userDataDir%22%3A%22%2Fdata%2Fhermes-profiles%2Fpasclaw%22%7D"
)
BROWSERLESS_LAUNCH_URL_2 = (
    "wss://browserless.example/chromium?token=abc&"
    "launch=%7B%22stealth%22%3Atrue%2C%22userDataDir%22%3A%22%2Fdata%2Fhermes-profiles%2Fother-profile%22%7D"
)


class TestResolveCdpOverride:
    def test_keeps_full_devtools_websocket_url(self):
        from tools.browser_tool import _resolve_cdp_override

        assert _resolve_cdp_override(WS_URL) == WS_URL

    def test_resolves_http_discovery_endpoint_to_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(HTTP_URL)

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_resolves_bare_ws_hostport_to_discovery_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(f"ws://{HOST}:{PORT}")

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_falls_back_to_raw_url_when_discovery_fails(self):
        from tools.browser_tool import _resolve_cdp_override

        with patch("tools.browser_tool.requests.get", side_effect=RuntimeError("boom")):
            assert _resolve_cdp_override(HTTP_URL) == HTTP_URL

    def test_normalizes_provider_returned_http_cdp_url_when_creating_session(self, monkeypatch):
        import tools.browser_tool as browser_tool

        provider = Mock()
        provider.create_session.return_value = {
            "session_name": "cloud-session",
            "bb_session_id": "bu_123",
            "cdp_url": "https://cdp.browser-use.example/session",
            "features": {"browser_use": True},
        }

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
        monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: provider)

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            session_info = browser_tool._get_session_info("task-browser-use")

        assert session_info["cdp_url"] == WS_URL
        provider.create_session.assert_called_once_with("task-browser-use")
        mock_get.assert_called_once_with(
            "https://cdp.browser-use.example/session/json/version",
            timeout=10,
        )

    def test_reuses_browserless_managed_persistence_session_across_task_ids(self, monkeypatch):
        import tools.browser_tool as browser_tool

        created = []

        def fake_create(task_id, cdp_url, shared_cdp_key=None):
            created.append((task_id, cdp_url))
            session = {
                "session_name": "cdp_shared_session",
                "bb_session_id": None,
                "cdp_url": cdp_url,
                "features": {"cdp_override": True},
            }
            if shared_cdp_key:
                session["_shared_cdp_key"] = shared_cdp_key
            return session

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: BROWSERLESS_LAUNCH_URL_1)
        monkeypatch.setattr(browser_tool, "_create_cdp_session", fake_create)

        first = browser_tool._get_session_info("task-a")
        second = browser_tool._get_session_info("task-b")

        assert created == [("task-a", BROWSERLESS_LAUNCH_URL_1)]
        assert first["session_name"] == "cdp_shared_session"
        assert second["session_name"] == "cdp_shared_session"
        assert browser_tool._active_sessions["task-a"] is first
        assert browser_tool._active_sessions["task-b"] is first

    def test_different_browserless_user_data_dirs_do_not_share_session(self, monkeypatch):
        import tools.browser_tool as browser_tool

        current_url = {"value": BROWSERLESS_LAUNCH_URL_1}
        created = []

        def fake_create(task_id, cdp_url, shared_cdp_key=None):
            created.append((task_id, cdp_url))
            session = {
                "session_name": f"cdp_{len(created)}",
                "bb_session_id": None,
                "cdp_url": cdp_url,
                "features": {"cdp_override": True},
            }
            if shared_cdp_key:
                session["_shared_cdp_key"] = shared_cdp_key
            return session

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
        monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: current_url["value"])
        monkeypatch.setattr(browser_tool, "_create_cdp_session", fake_create)

        first = browser_tool._get_session_info("task-a")
        current_url["value"] = BROWSERLESS_LAUNCH_URL_2
        second = browser_tool._get_session_info("task-b")

        assert len(created) == 2
        assert first["session_name"] != second["session_name"]
