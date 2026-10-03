"""The TUI/connector re-auth path must force OAuth for servers that never challenge.

``probe_with_rollback`` (tools/connectors/mcp_oauth.py) drives the OAuth probe behind the TUI
``oauth.start`` RPC and card installs. A server that answers ``initialize`` with 200 and no
``WWW-Authenticate`` (gmailmcp.googleapis.com, Blynk) never triggers the SDK's reactive 401
branch, so without the one-shot force flag the flow ends with "no OAuth token was obtained"
(#53870, #89412).
"""

import contextlib
import io

import pytest


def test_probe_with_rollback_marks_server_for_forced_oauth(monkeypatch, tmp_path):
    """``probe_with_rollback`` sets the one-shot force flag after ``manager.remove`` and before
    the probe, mirroring ``hermes mcp login`` and the dashboard re-auth worker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.mcp_config as mc
    import tools.mcp_oauth_manager as mgr_mod
    from tools.connectors.mcp_oauth import probe_with_rollback

    forced = []
    real_manager = mgr_mod.get_manager()

    class _RecordingManager:
        def remove(self, name, hermes_home=None):
            return real_manager.remove(name, hermes_home=hermes_home)

        def set_force_oauth(self, name):
            forced.append(name)

        def restore_entry(self, *args, **kwargs):
            pass

    monkeypatch.setattr(mgr_mod, "get_manager", lambda: _RecordingManager())
    monkeypatch.setattr(mc, "_probe_single_server", lambda name, cfg, connect_timeout=None, details=None: [("t", "d")])
    monkeypatch.setattr(mc, "_oauth_tokens_present", lambda name: True)
    monkeypatch.setattr(mc, "_save_mcp_server", lambda name, cfg: True)

    with contextlib.redirect_stdout(io.StringIO()):
        probe_with_rollback(
            "gmail",
            {"url": "https://gmailmcp.example.test/mcp/v1", "auth": "oauth"},
            str(tmp_path),
            None,
            False,
        )

    assert forced == ["gmail"]


def test_probe_with_rollback_failure_rolls_back_without_forcing(monkeypatch, tmp_path):
    """A failed probe still records the force attempt, and the rollback path restores the
    previous manager entry — the flag is one-shot either way."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.mcp_config as mc
    import tools.mcp_oauth_manager as mgr_mod
    from tools.connectors.mcp_oauth import probe_with_rollback

    forced = []
    real_manager = mgr_mod.get_manager()

    class _RecordingManager:
        def remove(self, name, hermes_home=None):
            return real_manager.remove(name, hermes_home=hermes_home)

        def set_force_oauth(self, name):
            forced.append(name)

        def restore_entry(self, *args, **kwargs):
            pass

    monkeypatch.setattr(mgr_mod, "get_manager", lambda: _RecordingManager())

    def _boom(name, cfg, connect_timeout=None, details=None):
        raise RuntimeError("probe body replaced by recorder")

    monkeypatch.setattr(mc, "_probe_single_server", _boom)

    with pytest.raises(RuntimeError, match="probe body replaced by recorder"):
        with contextlib.redirect_stdout(io.StringIO()):
            probe_with_rollback(
                "gmail",
                {"url": "https://gmailmcp.example.test/mcp/v1", "auth": "oauth"},
                str(tmp_path),
                None,
                False,
            )

    assert forced == ["gmail"]
