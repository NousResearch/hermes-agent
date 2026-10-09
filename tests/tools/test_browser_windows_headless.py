"""Regression for #64867: local Chromium stays headless on Windows Desktop.

Windows Desktop opened a blank top-level window during local browser automation:
the agent-browser daemon inherits the launch environment, so anything that
resolves its ``--headed`` setting to a window turns local mode's documented
"zero-cost headless Chromium" into a visible browser over the Desktop chat.

The option builder is pure in ``(options, is_windows)`` so the invariant is
testable on any host (root AGENTS.md); the wiring tests run only where the
behaviour lives, on the Windows lane.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("AGENT_BROWSER_HEADED", raising=False)
    yield


# ---------------------------------------------------------------------------
# windows_headless_browser_options (pure, runs on every host)
# ---------------------------------------------------------------------------

from tools.browser_tool_session import windows_headless_browser_options


def test_windows_pins_headless_for_local_launch():
    env = {"PATH": "/usr/bin", "AGENT_BROWSER_SOCKET_DIR": "/tmp/s"}
    pinned = windows_headless_browser_options(env, is_windows=True)
    assert pinned["AGENT_BROWSER_HEADED"] == "false"
    assert pinned["PATH"] == "/usr/bin"
    assert pinned["AGENT_BROWSER_SOCKET_DIR"] == "/tmp/s"
    # Pure: the caller's dict (shared _build_browser_env output) is never mutated.
    assert "AGENT_BROWSER_HEADED" not in env


def test_non_windows_options_unchanged():
    env = {"PATH": "/usr/bin"}
    assert windows_headless_browser_options(env, is_windows=False) == env
    assert "AGENT_BROWSER_HEADED" not in env


def test_explicit_headed_setting_is_never_overridden():
    for value in ("true", "1", "yes"):
        env = {"AGENT_BROWSER_HEADED": value}
        assert windows_headless_browser_options(env, is_windows=True) == env


# ---------------------------------------------------------------------------
# Wiring: local (non-CDP) session commands carry the pin into the daemon env.
# Windows-only by nature — the guard is a no-op on other hosts.
# ---------------------------------------------------------------------------

HEADLESS_SNAPSHOT_STDOUT = (
    '{"success": true, "data": {"snapshot": '
    '"- heading \\\\"Hi\\\\" [ref=e1]", "refs": {"e1": {}}}}'
)


@pytest.mark.platforms("windows")
@patch("tools.browser_tool_session._get_session_info")
@patch("tools.browser_tool_install._find_agent_browser", return_value="/usr/bin/agent-browser")
@patch("tools.browser_tool_cloud._is_local_mode", return_value=True)
@patch("tools.browser_tool_install._chromium_installed", return_value=True)
@patch("tools.browser_tool_cloud._get_cloud_provider", return_value=None)
@patch("tools.browser_tool_cdp._get_cdp_override", return_value="")
@patch("tools.browser_tool._is_camofox_mode", return_value=False)
def test_local_command_env_pins_headed_false(
    _camofox, _cdp, _cloud, _chromium, _local, _find, _session
):
    """A local session's daemon env pins AGENT_BROWSER_HEADED=false (#64867)."""
    import tools.browser_tool as bt
    import tools.browser_tool_session as bt_session

    bt._cached_headed_mode = False
    bt._headed_mode_resolved = True
    _session.return_value = {"session_name": "test-sess"}

    captured = []
    mock_proc = MagicMock()
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    def capture_popen(cmd, **kwargs):
        captured.append((cmd, kwargs.get("env")))
        return mock_proc

    with (
        patch("subprocess.Popen", side_effect=capture_popen),
        patch("os.open", return_value=99),
        patch("os.close"),
        patch("os.unlink"),
        patch("os.makedirs"),
        patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=HEADLESS_SNAPSHOT_STDOUT))),
            __exit__=MagicMock(return_value=False),
        ))),
        patch("tools.interrupt.is_interrupted", return_value=False),
        patch("tools.browser_tool_lifecycle._write_owner_pid"),
    ):
        bt_session._run_browser_command("task1", "snapshot", [], _engine_override="auto")

    assert len(captured) == 1
    cmd, env = captured[0]
    assert env.get("AGENT_BROWSER_HEADED") == "false"
    # The pin must not contradict the argv: headless local commands carry no --headed.
    assert "--headed" not in cmd
