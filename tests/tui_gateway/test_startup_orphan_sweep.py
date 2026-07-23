"""Tests for #65194: the gateway's startup-time orphaned-session sweep.

A gateway restart destroys the in-process ws-orphan grace timers
(``_schedule_ws_orphan_reap``), so rows for sessions that died with the
previous process stay ``ended_at IS NULL`` forever.  ``entry.main()`` must
trigger a DB-level sweep on every boot — gated by ``sessions.orphan_reaper``
(default on) and the gateway's session TTL — without ever blocking or
crashing startup.
"""

import io
import threading
from unittest.mock import MagicMock, patch

from tui_gateway import entry, server


def _sessions_cfg(**kw):
    return {"sessions": kw}


class TestReapOrphanedSessions:
    @patch("hermes_cli.config.load_config")
    @patch("tui_gateway.server._get_db")
    def test_sweeps_with_session_ttl_threshold(self, mock_get_db, mock_load_config):
        mock_load_config.return_value = _sessions_cfg()
        db = MagicMock()
        db.sweep_orphaned_sessions.return_value = 3
        mock_get_db.return_value = db

        with patch.object(server, "_SESSION_TTL_S", 6 * 3600.0):
            entry._reap_orphaned_sessions()

        db.sweep_orphaned_sessions.assert_called_once_with(
            max_idle_seconds=6 * 3600.0
        )

    @patch("hermes_cli.config.load_config")
    @patch("tui_gateway.server._get_db")
    def test_flag_off_skips_sweep(self, mock_get_db, mock_load_config):
        mock_load_config.return_value = _sessions_cfg(orphan_reaper=False)

        entry._reap_orphaned_sessions()

        mock_get_db.assert_not_called()

    @patch("hermes_cli.config.load_config")
    @patch("tui_gateway.server._get_db")
    def test_zero_ttl_disables_sweep(self, mock_get_db, mock_load_config):
        mock_load_config.return_value = _sessions_cfg()

        with patch.object(server, "_SESSION_TTL_S", 0.0):
            entry._reap_orphaned_sessions()

        mock_get_db.assert_not_called()

    @patch("hermes_cli.config.load_config")
    @patch("tui_gateway.server._get_db")
    def test_missing_db_is_noop(self, mock_get_db, mock_load_config):
        mock_load_config.return_value = _sessions_cfg()
        mock_get_db.return_value = None

        with patch.object(server, "_SESSION_TTL_S", 6 * 3600.0):
            entry._reap_orphaned_sessions()  # must not raise

    @patch("hermes_cli.config.load_config")
    @patch("tui_gateway.server._get_db")
    def test_db_errors_never_propagate(self, mock_get_db, mock_load_config):
        mock_load_config.return_value = _sessions_cfg()
        db = MagicMock()
        db.sweep_orphaned_sessions.side_effect = RuntimeError("boom")
        mock_get_db.return_value = db

        with patch.object(server, "_SESSION_TTL_S", 6 * 3600.0):
            entry._reap_orphaned_sessions()  # must not raise

    @patch("hermes_cli.config.load_config", side_effect=RuntimeError("no config"))
    @patch("tui_gateway.server._get_db")
    def test_config_failure_defaults_to_enabled(self, mock_get_db, mock_load_config):
        db = MagicMock()
        db.sweep_orphaned_sessions.return_value = 0
        mock_get_db.return_value = db

        with patch.object(server, "_SESSION_TTL_S", 6 * 3600.0):
            entry._reap_orphaned_sessions()

        db.sweep_orphaned_sessions.assert_called_once()


class TestMainWiring:
    def test_main_triggers_sweep(self, monkeypatch):
        """main() spawns the sweep on every boot (daemon thread, off the
        gateway.ready path) and still serves the stdin loop."""
        swept = threading.Event()
        monkeypatch.setattr(entry, "_reap_orphaned_sessions", swept.set)
        monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
        monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
        monkeypatch.setattr(entry, "write_json", lambda _payload: True)
        monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
        with patch("hermes_cli.config.read_raw_config", return_value={}):
            entry.main()

        assert swept.wait(timeout=5), "startup sweep was never triggered"
