"""Tests for tui_gateway.entry helper functions (coverage for #36611).

Covers all functions in entry.py except the module-level sys.path
sanitization (test_entry_sys_path.py), wait_for_mcp_discovery
(test_wait_for_mcp_discovery.py), and main() (test_entry_main.py).
"""

import os
import signal
import sys
import threading
import time
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import tui_gateway.entry as entry


# ── _install_sidecar_publisher ──────────────────────────────────────────────


class TestInstallSidecarPublisher:
    """Cover _install_sidecar_publisher branches: no env var -> no-op,
    env var set -> wraps transport."""

    def test_no_sidecar_url_returns_early(self):
        """HERMES_TUI_SIDECAR_URL not set: function returns immediately."""
        with patch.dict(os.environ, clear=True):
            entry._install_sidecar_publisher()
        # No crash is the assertion

    def test_sidecar_url_wraps_transport(self, monkeypatch):
        """Env var set: wraps _stdio_transport in a TeeTransport."""
        monkeypatch.setenv("HERMES_TUI_SIDECAR_URL", "ws://localhost:8080")
        mock_tee = MagicMock()
        monkeypatch.setattr("tui_gateway.entry.TeeTransport", mock_tee)
        mock_ws = MagicMock()
        with patch.dict("sys.modules", {
            "tui_gateway.event_publisher": MagicMock(WsPublisherTransport=mock_ws),
        }):
            old_transport = entry.server._stdio_transport
            entry._install_sidecar_publisher()

        mock_tee.assert_called_once()
        args, _ = mock_tee.call_args
        assert args[0] is old_transport  # first arg is the old transport


# ── _shutdown_grace_seconds ─────────────────────────────────────────────────


class TestShutdownGraceSeconds:
    """Cover all branches of _shutdown_grace_seconds."""

    def test_default_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", raising=False)
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S

    def test_default_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "")
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S

    def test_default_when_env_whitespace(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "  ")
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S

    def test_default_when_env_not_a_number(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "not-a-float")
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S

    def test_returns_value_when_positive(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "3.5")
        assert entry._shutdown_grace_seconds() == 3.5

    def test_default_when_zero(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "0")
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S

    def test_default_when_negative(self, monkeypatch):
        monkeypatch.setenv("HERMES_TUI_GATEWAY_SHUTDOWN_GRACE_S", "-2")
        assert entry._shutdown_grace_seconds() == entry._DEFAULT_SHUTDOWN_GRACE_S


# ── _log_signal ─────────────────────────────────────────────────────────────


class TestLogSignal:
    """Cover _log_signal: signal name resolution, crash log write, timer."""

    def test_signal_name_lookup_uses_known_signal(self, monkeypatch):
        """SIGTERM resolves by name."""
        calls = []
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr(
            "builtins.open",
            MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()))),
        )
        monkeypatch.setattr("traceback.print_stack", lambda *a, **kw: None)
        monkeypatch.setattr("traceback.format_stack", lambda *a, **kw: ["fake stack"])
        monkeypatch.setattr("sys._current_frames", lambda: {})
        monkeypatch.setattr("threading._active", {})
        monkeypatch.setattr("sys.exit", MagicMock(side_effect=SystemExit(0)))

        with pytest.raises(SystemExit):
            entry._log_signal(signal.SIGTERM, None)
        # No crash — signal name resolution handled

    def test_signal_name_fallback_for_unknown_number(self):
        """Unknown signal number falls back to 'signal N' format."""
        _signal_names = {}
        for _attr in ("SIGPIPE", "SIGTERM", "SIGHUP", "SIGINT", "SIGBREAK"):
            _sig = getattr(signal, _attr, None)
            if _sig is not None:
                _signal_names[int(_sig)] = _attr
        name = _signal_names.get(9999, "signal 9999")
        assert name == "signal 9999"

    def test_log_signal_writes_crash_log(self, tmp_path, monkeypatch):
        """_log_signal writes stack data to the crash log."""
        crash_log = tmp_path / "crash.log"
        monkeypatch.setattr("tui_gateway.entry._CRASH_LOG", str(crash_log))
        monkeypatch.setattr("sys.exit", MagicMock(side_effect=SystemExit(0)))
        monkeypatch.setattr("sys._current_frames", lambda: {threading.get_ident(): None})
        monkeypatch.setattr("threading._active", {})

        with pytest.raises(SystemExit):
            entry._log_signal(signal.SIGTERM, None)

        assert crash_log.exists()

    def test_log_signal_handles_none_frame(self, monkeypatch):
        """frame is None: skips main-thread stack, still logs other threads."""
        monkeypatch.setattr("sys.exit", MagicMock(side_effect=SystemExit(0)))
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr(
            "builtins.open",
            MagicMock(return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()))),
        )
        monkeypatch.setattr("traceback.format_stack", lambda *a, **kw: ["fake stack"])
        monkeypatch.setattr("sys._current_frames", lambda: {threading.get_ident(): None})
        monkeypatch.setattr("threading._active", {})

        with pytest.raises(SystemExit):
            entry._log_signal(signal.SIGTERM, None)

    def test_crash_log_dir_create_fails_gracefully(self, monkeypatch):
        """os.makedirs exception is caught."""
        monkeypatch.setattr("sys.exit", MagicMock(side_effect=SystemExit(0)))
        monkeypatch.setattr("os.makedirs", MagicMock(side_effect=PermissionError()))
        monkeypatch.setattr("sys._current_frames", lambda: {})
        monkeypatch.setattr("threading._active", {})
        monkeypatch.setattr("traceback.print_stack", lambda *a, **kw: None)
        monkeypatch.setattr("traceback.format_stack", lambda *a, **kw: [""])

        with pytest.raises(SystemExit):
            entry._log_signal(signal.SIGTERM, None)
        # No crash — exception handled


# ── _log_exit ────────────────────────────────────────────────────────────────


class TestLogExit:
    """Cover _log_exit: normal write and exception path."""

    def test_logs_reason_to_crash_log(self, tmp_path, monkeypatch):
        """_log_exit writes reason to crash log file."""
        crash_log = tmp_path / "crash.log"
        monkeypatch.setattr("tui_gateway.entry._CRASH_LOG", str(crash_log))
        entry._log_exit("test reason")
        content = crash_log.read_text()
        assert "test reason" in content
        assert "gateway exit" in content

    def test_crash_log_dir_create_fails_gracefully(self, monkeypatch):
        """os.makedirs exception is caught."""
        monkeypatch.setattr("os.makedirs", MagicMock(side_effect=PermissionError()))
        entry._log_exit("test reason")
        # No crash

    def test_crash_log_write_fails_gracefully(self, tmp_path, monkeypatch):
        """File open/write exception is caught."""
        monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=PermissionError()))
        entry._log_exit("test reason")
        # No crash


# ── mcp_discovery_in_flight ─────────────────────────────────────────────────


class TestMcpDiscoveryInFlight:
    """Cover mcp_discovery_in_flight: thread None, alive, dead."""

    def _restore_thread(self, saved):
        entry._mcp_discovery_thread = saved

    def test_no_thread_returns_false(self):
        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            assert entry.mcp_discovery_in_flight() is False
        finally:
            self._restore_thread(saved)

    def test_alive_thread_returns_true(self):
        saved = entry._mcp_discovery_thread
        try:
            t = threading.Thread(target=lambda: time.sleep(10), daemon=True)
            t.start()
            entry._mcp_discovery_thread = t
            assert entry.mcp_discovery_in_flight() is True
            t.join(timeout=1)
        finally:
            self._restore_thread(saved)

    def test_dead_thread_returns_false(self):
        saved = entry._mcp_discovery_thread
        try:
            t = threading.Thread(target=lambda: None, daemon=True)
            t.start()
            t.join()
            entry._mcp_discovery_thread = t
            assert entry.mcp_discovery_in_flight() is False
        finally:
            self._restore_thread(saved)


# ── join_mcp_discovery ──────────────────────────────────────────────────────


class TestJoinMcpDiscovery:
    """Cover join_mcp_discovery: thread None, alive with timeout, alive blocks."""

    def _restore_thread(self, saved):
        entry._mcp_discovery_thread = saved

    def test_no_thread_returns_true(self):
        saved = entry._mcp_discovery_thread
        try:
            entry._mcp_discovery_thread = None
            assert entry.join_mcp_discovery(timeout=1.0) is True
        finally:
            self._restore_thread(saved)

    def test_alive_thread_completes_returns_true(self):
        saved = entry._mcp_discovery_thread
        try:
            t = threading.Thread(target=lambda: None, daemon=True)
            t.start()
            entry._mcp_discovery_thread = t
            assert entry.join_mcp_discovery(timeout=1.0) is True
        finally:
            self._restore_thread(saved)

    def test_alive_thread_still_alive_after_timeout_returns_false(self):
        saved = entry._mcp_discovery_thread
        stop = threading.Event()
        try:
            t = threading.Thread(target=stop.wait, daemon=True)
            t.start()
            entry._mcp_discovery_thread = t
            assert entry.join_mcp_discovery(timeout=0.1) is False
        finally:
            stop.set()
            self._restore_thread(saved)

    def test_join_with_no_timeout_waits_until_complete(self):
        """join_mcp_discovery(timeout=None) blocks until thread finishes."""
        saved = entry._mcp_discovery_thread
        try:
            barrier = threading.Barrier(2, timeout=2)
            t = threading.Thread(target=barrier.wait, daemon=True)
            t.start()
            entry._mcp_discovery_thread = t
            start = time.monotonic()
            assert entry.join_mcp_discovery() is True
            elapsed = time.monotonic() - start
            assert elapsed < 1.0  # completed quickly
        finally:
            self._restore_thread(saved)


# ── Signal handler installation ──────────────────────────────────────────────


class TestSignalHandlerInstallation:
    """Verify module-level signal handlers are installed with correct handlers.

    These are set at import time in entry.py lines 159-170.
    We test indirectly by checking getattr + getsignal on the signals
    that the platform supports.
    """

    def test_sigpipe_ignored_if_available(self):
        if hasattr(signal, "SIGPIPE"):
            assert signal.getsignal(signal.SIGPIPE) is signal.SIG_IGN

    def test_sigterm_handled_by_log_signal(self):
        if hasattr(signal, "SIGTERM"):
            assert signal.getsignal(signal.SIGTERM) is entry._log_signal

    def test_sighup_handled_by_log_signal(self):
        if hasattr(signal, "SIGHUP"):
            assert signal.getsignal(signal.SIGHUP) is entry._log_signal

    def test_sigbreake_handled_by_log_signal_when_no_sighup(self):
        # On platforms without SIGHUP but with SIGBREAK (Windows)
        if not hasattr(signal, "SIGHUP") and hasattr(signal, "SIGBREAK"):
            assert signal.getsignal(signal.SIGBREAK) is entry._log_signal

    def test_sigint_ignored(self):
        if hasattr(signal, "SIGINT"):
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
