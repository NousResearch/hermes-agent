import logging
from unittest.mock import Mock, patch

import pytest


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"


@pytest.fixture(autouse=True)
def clear_cdp_discovery_failure_cache():
    import tools.browser_tool as browser_tool

    with browser_tool._cdp_discovery_failure_cache_lock:
        browser_tool._cdp_discovery_failure_cache.clear()
    yield
    with browser_tool._cdp_discovery_failure_cache_lock:
        browser_tool._cdp_discovery_failure_cache.clear()


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

    def test_resolves_bare_hostport_config_shorthand_to_discovery_websocket(self):
        from tools.browser_tool import _resolve_cdp_override

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"webSocketDebuggerUrl": WS_URL}

        with patch("tools.browser_tool.requests.get", return_value=response) as mock_get:
            resolved = _resolve_cdp_override(f"{HOST}:{PORT}")

        assert resolved == WS_URL
        mock_get.assert_called_once_with(VERSION_URL, timeout=10)

    def test_treats_discovery_endpoint_as_unavailable_when_discovery_fails(self):
        from tools.browser_tool import _resolve_cdp_override

        with patch("tools.browser_tool.requests.get", side_effect=RuntimeError("boom")):
            assert _resolve_cdp_override(HTTP_URL) == ""

    def test_unreachable_discovery_endpoint_is_debug_logged_and_negative_cached(self, caplog):
        from tools.browser_tool import _resolve_cdp_override

        with caplog.at_level(logging.DEBUG, logger="tools.browser_tool"), \
             patch("tools.browser_tool.requests.get", side_effect=RuntimeError("boom")) as mock_get:
            assert _resolve_cdp_override(f"ws://{HOST}:{PORT}") == ""
            assert _resolve_cdp_override(f"ws://{HOST}:{PORT}") == ""

        mock_get.assert_called_once_with(VERSION_URL, timeout=10)
        assert "not reachable" in caplog.text
        assert "WARNING" not in caplog.text

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
                patch("tools.browser_tool.logger.debug") as mock_debug:
            resolved = _resolve_cdp_override(raw)

        assert resolved == ""
        mock_debug.assert_called_once()
        _, logged_raw, logged_version_url, logged_error = mock_debug.call_args.args
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

    def test_camofox_yields_to_config_cdp_override(self, monkeypatch):
        """CAMOFOX_URL + a persistent browser.cdp_url config override must NOT
        report camofox mode: the CDP browser takes precedence so navigation is
        not routed through Camofox, and the CDP backend stays non-local for SSRF
        checks. Regression for the env-only suppression gap (config CDP was
        ignored, so CAMOFOX_URL + config CDP still dispatched to Camofox)."""
        import tools.browser_camofox as bc

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)

        # No CDP anywhere -> camofox mode is on.
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bc.is_camofox_mode() is True

        # A config-only CDP override suppresses camofox.
        with patch("hermes_cli.config.read_raw_config",
                   return_value={"browser": {"cdp_url": HTTP_URL}}):
            assert bc.is_camofox_mode() is False

        # The env override still suppresses camofox.
        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            assert bc.is_camofox_mode() is False

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


class TestCDPSupervisorStartErrorRedaction:
    """CDPSupervisor.start() must not leak the CDP URL via the connect-error path.

    The more common failure mode than attach-timeout: the first
    websockets.connect(self.cdp_url) raises (bad URI, refused, TLS), the raw
    exception is stashed as self._start_error, and start() re-raises it. Those
    websockets exceptions embed the full raw cdp_url -- token and userinfo --
    in their message. start() must re-raise a REDACTED error and must not leak
    the secret via the exception message or the traceback cause chain.
    """

    def _run_start_hitting_error(self, cdp_url: str, start_error: BaseException):
        """Invoke start() so it takes the _start_error re-raise branch.

        start() clears _ready_event / _start_error and launches a thread, so we
        can't pre-seed them. Instead we stub threading.Thread: the fake thread's
        start() synchronously populates _start_error and sets the ready event,
        exactly as the real supervisor loop does on a first-connect failure.
        """
        import threading
        from tools.browser_supervisor import CDPSupervisor

        sup = CDPSupervisor.__new__(CDPSupervisor)
        sup.task_id = "test-task"
        sup.cdp_url = cdp_url
        sup._start_error = None
        sup._stop_requested = False
        sup._loop = None
        sup._thread = None
        sup._ready_event = threading.Event()

        def _fake_thread(*args, **kwargs):
            fake = Mock()

            def _start():
                sup._start_error = start_error
                sup._ready_event.set()

            fake.start.side_effect = _start
            fake.is_alive.return_value = False
            return fake

        with patch("threading.Thread", side_effect=_fake_thread), patch.object(sup, "stop"):
            sup.start(timeout=5.0)

    def test_start_error_redacts_query_token(self):
        # A realistic websockets-style error embedding the raw URL + token.
        raw = "wss://cdp.example/devtools/browser/abc?token=super-secret-999"
        err = ValueError(f"{raw} isn't a valid URI: hostname isn't provided")
        try:
            self._run_start_hitting_error(raw, err)
        except Exception as exc:  # noqa: BLE001 - asserting on the surface
            msg = str(exc)
            assert "super-secret-999" not in msg, (
                "raw token must not appear in the re-raised error message"
            )
            # The raw cause must be suppressed so it can't leak via traceback.
            assert exc.__cause__ is None
            assert getattr(exc, "__suppress_context__", False) is True
        else:
            raise AssertionError("start() did not re-raise the start error")

    def test_start_error_redacts_userinfo_password(self):
        raw = "wss://user:p4ssw0rd@cdp.example/devtools/browser/x"
        err = ValueError(f"{raw} isn't a valid URI: hostname isn't provided")
        try:
            self._run_start_hitting_error(raw, err)
        except Exception as exc:  # noqa: BLE001
            assert "p4ssw0rd" not in str(exc)
        else:
            raise AssertionError("start() did not re-raise the start error")


class TestRedactCdpErrorText:
    """The supervisor's error-text chokepoint masks credentials, keeps context."""

    def test_masks_query_token_in_exception(self):
        from tools.browser_supervisor import _redact_cdp_error_text

        err = ConnectionError("connect wss://h/x?token=leak-me failed")
        out = _redact_cdp_error_text(err)
        assert "leak-me" not in out

    def test_preserves_non_secret_context(self):
        from tools.browser_supervisor import _redact_cdp_error_text

        err = ConnectionError("connect ws://127.0.0.1:9222/x failed: refused")
        out = _redact_cdp_error_text(err)
        assert "127.0.0.1:9222" in out
        assert "refused" in out
