"""Browser cleanup fallback contracts, retained after retiring the npx warmer."""

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_tool_lifecycle import _legacy_kill_process_tree


@pytest.mark.platforms("linux", "darwin")
def test_posix_kills_process_group_term_then_kill(monkeypatch):
    proc = MagicMock(pid=999)
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    _legacy_kill_process_tree(proc)
    assert calls == [(999, signal.SIGTERM), (999, signal.SIGKILL)]


@pytest.mark.platforms("linux", "darwin")
def test_posix_missing_process_returns_silently(monkeypatch):
    def missing(pid):
        raise ProcessLookupError()

    monkeypatch.setattr(os, "getpgid", missing)
    _legacy_kill_process_tree(MagicMock(pid=999))


@pytest.mark.platforms("linux", "darwin")
@pytest.mark.parametrize("kill_error", [None, OSError("already reaped")])
def test_missing_killpg_falls_back_to_proc_kill(monkeypatch, kill_error):
    proc = MagicMock(pid=999)
    proc.kill.side_effect = kill_error
    monkeypatch.delattr(os, "killpg", raising=False)
    _legacy_kill_process_tree(proc)
    proc.kill.assert_called_once()


@pytest.mark.platforms("linux", "darwin")
def test_permission_denied_does_not_attempt_sigkill(monkeypatch):
    monkeypatch.setattr(os, "getpgid", lambda pid: 999)
    calls = []

    def denied(pgid, sig):
        calls.append((pgid, sig))
        raise PermissionError()

    monkeypatch.setattr(os, "killpg", denied)
    _legacy_kill_process_tree(MagicMock(pid=999))
    assert calls == [(999, signal.SIGTERM)]


@pytest.mark.platforms("win32")
@pytest.mark.parametrize("error", [None, OSError("taskkill missing")])
def test_windows_taskkill_targets_tree_and_is_best_effort(error):
    with patch("subprocess.run", side_effect=error) as run:
        _legacy_kill_process_tree(MagicMock(pid=4321))
    assert run.call_args.args[0] == ["taskkill", "/PID", "4321", "/T", "/F"]