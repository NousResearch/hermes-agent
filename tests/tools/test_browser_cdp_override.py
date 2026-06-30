from unittest.mock import Mock, patch


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"


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

    def test_redacts_secret_query_params_in_success_log(self):
        from tools.browser_tool import _resolve_cdp_override

        raw = "https://cdp.example/json/version?access_token=super-secret-token-123456"
        resolved_ws = "wss://cdp.example/devtools/browser/abc?token=super-secret-token-123456"

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": resolved_ws}

        with patch("tools.browser_tool.requests.get", return_value=response), \
                patch("tools.browser_tool.logger.info") as mock_info:
            resolved = _resolve_cdp_override(raw)

        assert resolved == resolved_ws
        mock_info.assert_called_once()
        _, logged_raw, logged_ws = mock_info.call_args.args
        assert "super-secret-token-123456" not in logged_raw
        assert "super-secret-token-123456" not in logged_ws
        assert "access_token=***" in logged_raw
        assert "token=***" in logged_ws

    def test_redacts_secret_query_params_in_failure_log(self):
        from tools.browser_tool import _resolve_cdp_override

        raw = "https://cdp.example?access_token=super-secret-token-123456"
        secret_error = RuntimeError(
            "upstream rejected https://cdp.example/json/version?access_token=super-secret-token-123456"
        )

        with patch("tools.browser_tool.requests.get", side_effect=secret_error), \
                patch("tools.browser_tool.logger.warning") as mock_warning:
            resolved = _resolve_cdp_override(raw)

        assert resolved == raw
        mock_warning.assert_called_once()
        _, logged_raw, logged_version_url, logged_error = mock_warning.call_args.args
        assert "super-secret-token-123456" not in logged_raw
        assert "super-secret-token-123456" not in logged_version_url
        assert "super-secret-token-123456" not in logged_error
        assert "access_token=***" in logged_raw
        assert "access_token=***" in logged_version_url
        assert "access_token=***" in logged_error
        assert logged_version_url.startswith("https://cdp.example")

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


class TestGetCdpOverride:
    def test_prefers_env_var_over_config(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        monkeypatch.setattr(
            browser_tool,
            "read_raw_config",
            lambda: {"browser": {"cdp_url": "http://config-host:9222"}},
            raising=False,
        )

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_uses_config_browser_cdp_url_when_env_missing(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("hermes_cli.config.read_raw_config", return_value={"browser": {"cdp_url": HTTP_URL}}), \
             patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = browser_tool._get_cdp_override()

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)


class TestCreateCdpSession:
    """_create_cdp_session() must sanitize the CDP URL before logging.

    PR #54851 added _sanitize_url_for_logs() and wired it into the three log
    sites inside _resolve_cdp_override(). This test guards the fourth site
    that was missed: the logger.info call inside _create_cdp_session(), which
    receives the already-resolved CDP URL and could contain a query-string
    token (e.g. wss://provider.example/session?token=secret).
    """

    def test_redacts_token_in_session_creation_log(self):
        from tools.browser_tool import _create_cdp_session

        cdp_url_with_token = "wss://cdp.example/devtools/browser/abc?token=super-secret-token-999"

        with patch("tools.browser_tool.logger.info") as mock_info:
            result = _create_cdp_session("task-1", cdp_url_with_token)

        assert result["cdp_url"] == cdp_url_with_token, "raw URL must be stored unmodified"

        mock_info.assert_called_once()
        logged_args = " ".join(str(a) for a in mock_info.call_args.args)
        assert "super-secret-token-999" not in logged_args
        assert "token=***" in logged_args

    def test_plain_url_without_secrets_passes_through(self):
        from tools.browser_tool import _create_cdp_session

        plain_url = "ws://localhost:9222/devtools/browser/abc123"

        with patch("tools.browser_tool.logger.info") as mock_info:
            _create_cdp_session("task-2", plain_url)

        logged_args = " ".join(str(a) for a in mock_info.call_args.args)
        assert "localhost:9222" in logged_args


class TestCDPSupervisorTimeoutRedaction:
    """CDPSupervisor.start() TimeoutError must not expose raw CDP credentials.

    The supervisor raises TimeoutError(f"... (cdp_url={self.cdp_url[:80]}...)")
    when attach times out.  A URL with a query-string token (e.g.
    wss://provider.example/session?token=secret) would embed the raw secret
    in the exception message, which propagates to caller logs and tracebacks.
    """

    def _make_timed_out_supervisor(self, cdp_url: str):
        """Return a CDPSupervisor whose start() will time out immediately."""
        import threading
        from tools.browser_supervisor import CDPSupervisor

        sup = CDPSupervisor.__new__(CDPSupervisor)
        sup.task_id = "test-task"
        sup.cdp_url = cdp_url
        sup._start_error = None
        sup._stop_requested = False
        sup._loop = None
        # _thread = None so the is_alive() early-return guard is skipped.
        sup._thread = None
        # _ready_event that never fires so wait() always returns False.
        never_ready = threading.Event()
        sup._ready_event = never_ready
        return sup

    def test_timeout_error_redacts_query_token(self):
        cdp_url = "wss://cdp.example/devtools/browser/abc?token=super-secret-999"
        sup = self._make_timed_out_supervisor(cdp_url)

        with patch("threading.Thread") as mock_thread_cls, patch.object(sup, "stop"):
            mock_thread_cls.return_value = Mock()
            try:
                sup.start(timeout=0.001)
            except TimeoutError as exc:
                msg = str(exc)
                assert "super-secret-999" not in msg, (
                    "raw token must not appear in TimeoutError message"
                )
                assert "cdp_url=" in msg
            else:
                raise AssertionError("TimeoutError was not raised")

    def test_timeout_error_preserves_plain_url(self):
        plain_url = "ws://127.0.0.1:9222/devtools/browser/abc"
        sup = self._make_timed_out_supervisor(plain_url)

        with patch("threading.Thread") as mock_thread_cls, patch.object(sup, "stop"):
            mock_thread_cls.return_value = Mock()
            try:
                sup.start(timeout=0.001)
            except TimeoutError as exc:
                assert "127.0.0.1:9222" in str(exc)
            else:
                raise AssertionError("TimeoutError was not raised")

