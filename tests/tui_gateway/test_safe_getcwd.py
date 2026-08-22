"""Regression tests: tui_gateway must tolerate a deleted working directory.

``os.getcwd()`` raises ``FileNotFoundError`` once the process's working
directory is removed out from under it (the folder Hermes was launched in gets
deleted, rebuilt, or ``git worktree remove``'d mid-session). ``tui_gateway``
called it unguarded on seven fallback paths.

Severity is process death, not degradation: ``session.create`` and
``session.resume`` are NOT in ``server._LONG_HANDLERS``, so they run inline on
the reader thread, and ``tui_gateway/entry.py`` calls ``dispatch(req)`` with no
``try``/``except`` — an escaping FileNotFoundError exits the stdio gateway.

``server._safe_getcwd`` mirrors the already-merged ``tools/terminal_tool.py``
helper (#39491). These tests pin every substituted site so the crash class
cannot silently regress, and pin the ``_SlashWorker`` subprocess contract that
the substitution must not disturb.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as server


def _getcwd_raises() -> MagicMock:
    """A stand-in for os.getcwd() under a deleted CWD."""
    return MagicMock(side_effect=FileNotFoundError(2, "No such file or directory"))


@pytest.fixture
def deleted_dir(tmp_path):
    """A real path that existed and no longer does."""
    d = tmp_path / "workspace"
    d.mkdir()
    path = str(d)
    d.rmdir()
    assert not os.path.isdir(path)
    return path


# ── the helper itself ────────────────────────────────────────────────────

def test_safe_getcwd_returns_real_cwd_when_available():
    assert server._safe_getcwd() == os.getcwd()


def test_safe_getcwd_prefers_terminal_cwd_when_getcwd_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    with patch("os.getcwd", _getcwd_raises()):
        assert server._safe_getcwd() == str(tmp_path)


def test_safe_getcwd_falls_back_to_home_when_getcwd_raises(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    expected = os.path.expanduser("~")
    with patch("os.getcwd", _getcwd_raises()):
        assert server._safe_getcwd() == expected


def test_safe_getcwd_does_not_swallow_unrelated_oserrors(monkeypatch):
    """Parity with tools/terminal_tool.py: only FileNotFoundError is tolerated,
    so a genuine PermissionError still surfaces instead of being masked."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    with patch("os.getcwd", MagicMock(side_effect=PermissionError(13, "denied"))):
        with pytest.raises(PermissionError):
            server._safe_getcwd()


# ── _default_session_cwd — the gap the sweeper named on #40153 ───────────

def test_default_session_cwd_survives_deleted_cwd(monkeypatch):
    """session.create / session.resume resolve through here and run INLINE, so
    an unguarded getcwd on this path kills the gateway process."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    expected = os.path.expanduser("~")
    with patch.object(server, "_launch_configured_cwd", return_value=None):
        with patch("os.getcwd", _getcwd_raises()):
            assert server._default_session_cwd() == expected


# ── _completion_cwd — both substituted sites ─────────────────────────────

def test_completion_cwd_or_chain_survives_deleted_cwd(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    expected = os.path.expanduser("~")
    with patch.object(server, "_profile_configured_cwd", return_value=None):
        with patch.object(server, "_launch_configured_cwd", return_value=None):
            with patch("os.getcwd", _getcwd_raises()):
                assert server._completion_cwd({}) == expected


def test_completion_cwd_survives_deleted_cwd_supplied_by_client(deleted_dir, monkeypatch):
    """The isdir-failed tail. In the real scenario the client's last-known cwd
    IS the deleted directory, so ``os.path.isdir`` returns False and the tail
    fires even though a cwd was supplied — the ``except Exception: pass`` above
    it does not cover that ``return``."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    expected = os.path.expanduser("~")
    with patch("os.getcwd", _getcwd_raises()):
        assert server._completion_cwd({"cwd": deleted_dir}) == expected


# ── _SlashWorker: guarded cwd + untouched subprocess contract ────────────

def test_slash_worker_spawns_with_fallback_cwd_and_preserves_contract(monkeypatch, tmp_path):
    """The cwd substitution must not disturb the profile-home env scoping
    (#40677), UTF-8 lossy decode (#53137), windows_hide_flags() or
    start_new_session=True that this block has accumulated."""
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    # Deliberately NOT tmp_path: that is the expected fallback cwd, so reusing
    # it would let the HERMES_HOME assertion pass without the override firing.
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    with patch.dict("sys.modules", {
        "hermes_constants": MagicMock(
            get_hermes_home=MagicMock(return_value=str(tmp_path))
        ),
    }):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value.stdout = MagicMock()
            mock_popen.return_value.stderr = MagicMock()
            with patch("os.getcwd", _getcwd_raises()):
                server._SlashWorker(
                    session_key="k", model="m", profile_home=str(profile_home)
                )

    assert mock_popen.called, "Popen was not invoked"
    kwargs = mock_popen.call_args[1]
    assert kwargs["cwd"] == str(tmp_path)
    # preservation guarantee, asserted rather than promised
    assert kwargs["env"]["HERMES_HOME"] == str(profile_home)
    assert kwargs["start_new_session"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert "creationflags" in kwargs


# ── methods_tools.py handlers ────────────────────────────────────────────

@pytest.mark.parametrize("method_name", ["cli.exec", "config.show", "shell.exec"])
def test_methods_tools_handlers_resolve_safe_getcwd(method_name):
    """HandlerRegistry.install() rebuilds each handler with server.py's
    globals, so methods_tools.py calls the bare name with no import. Pin that
    the name really is resolvable, or these sites ship a latent NameError."""
    handler = server._methods[method_name]
    assert "_safe_getcwd" in handler.__globals__


def test_cli_exec_runs_with_fallback_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    completed = MagicMock(stdout="", stderr="", returncode=0)
    with patch("subprocess.run", return_value=completed) as mock_run:
        with patch("os.getcwd", _getcwd_raises()):
            resp = server._methods["cli.exec"]("1", {"argv": ["--version"]})

    assert "error" not in resp, resp
    assert mock_run.called
    assert mock_run.call_args[1]["cwd"] == str(tmp_path)


def test_shell_exec_runs_with_fallback_cwd(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    completed = MagicMock(stdout="", stderr="", returncode=0)
    with patch("subprocess.run", return_value=completed) as mock_run:
        with patch("os.getcwd", _getcwd_raises()):
            resp = server._methods["shell.exec"]("2", {"command": "echo hi"})

    assert "error" not in resp, resp
    assert mock_run.called
    assert mock_run.call_args[1]["cwd"] == str(tmp_path)


def test_config_show_reports_fallback_working_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    with patch("os.getcwd", _getcwd_raises()):
        resp = server._methods["config.show"]("3", {})

    assert "error" not in resp, resp
    rows = [
        row
        for section in resp["result"]["sections"]
        for row in section["rows"]
        if row[0] == "Working Dir"
    ]
    assert rows == [["Working Dir", str(tmp_path)]]
