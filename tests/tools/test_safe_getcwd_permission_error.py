"""Regression tests for _safe_getcwd() PermissionError handling (macOS TCC).

Background: on macOS, when the process CWD is under a TCC-protected location
(``~/Documents``, ``~/Desktop``, ``~/Downloads``) and the calling process
lacks Full Disk Access, ``os.getcwd()`` raises ``PermissionError: [Errno 1]
Operation not permitted`` — not ``FileNotFoundError``.

Before the fix, ``_safe_getcwd`` only caught ``FileNotFoundError``, so the
terminal-tool cleanup thread (which calls ``_get_env_config()`` →
``_safe_getcwd()`` every 60 s) logged a full stack trace on every tick,
accumulating hundreds of MB of noise in ``mcp-stderr.log`` without breaking
functionality. After the fix, ``PermissionError`` falls back to
``TERMINAL_CWD`` or ``$HOME`` just like a deleted CWD already does.
"""

import os

import pytest

from tools import terminal_tool


class _GetcwdPatcher:
    """Context manager that temporarily replaces ``os.getcwd`` with *fn*."""

    def __init__(self, fn):
        self.fn = fn
        self._original = None

    def __enter__(self):
        self._original = os.getcwd
        os.getcwd = self.fn
        return self

    def __exit__(self, *exc):
        os.getcwd = self._original


def _raise(exc):
    def _fn():
        raise exc

    return _fn


def test_permission_error_falls_back_to_home(monkeypatch):
    """macOS TCC EPERM on os.getcwd() must fall back to $HOME, not propagate."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    with _GetcwdPatcher(_raise(PermissionError(1, "Operation not permitted"))):
        result = terminal_tool._safe_getcwd()

    assert result == os.path.expanduser("~")


def test_permission_error_prefers_existing_terminal_cwd(monkeypatch, tmp_path):
    """TERMINAL_CWD takes priority over $HOME when the live CWD is TCC-blocked.

    The fallback must itself exist: a TERMINAL_CWD pointing at a deleted
    directory (the mid-session stale-cwd failure — the session's recorded cwd
    died after resume pinned it) is NOT a valid fallback, because spawning a
    subprocess into it fails exactly like the original getcwd error. Home is
    the guaranteed-valid last resort.
    """
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    with _GetcwdPatcher(_raise(PermissionError(1, "Operation not permitted"))):
        result = terminal_tool._safe_getcwd()

    assert result == str(tmp_path)


def test_permission_error_skips_dead_terminal_cwd(monkeypatch, tmp_path):
    """A TERMINAL_CWD whose directory is gone must fall through to $HOME.

    Regression guard for the stale-cwd-on-resume fix: the fallback chain can
    no longer return the dead path it is falling back from.
    """
    dead = tmp_path / "gone"
    monkeypatch.setenv("TERMINAL_CWD", str(dead))

    with _GetcwdPatcher(_raise(FileNotFoundError(2, "No such file or directory"))):
        result = terminal_tool._safe_getcwd()

    assert result == os.path.expanduser("~")


def test_file_not_found_still_handled(monkeypatch):
    """Regression guard: the existing FileNotFoundError path must keep working."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    with _GetcwdPatcher(_raise(FileNotFoundError(2, "No such file or directory"))):
        result = terminal_tool._safe_getcwd()

    assert result == os.path.expanduser("~")


def test_happy_path_unchanged():
    """Normal os.getcwd() must pass through untouched."""
    # Don't patch getcwd — use the real one
    result = terminal_tool._safe_getcwd()
    assert result == os.getcwd()


def test_os_error_not_swallowed(monkeypatch):
    """Unrelated OSError subclasses must still propagate (don't over-catch)."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    # NotADirectoryError is an OSError but neither FileNotFoundError nor
    # PermissionError — it should escape so callers see the real problem.
    with pytest.raises(OSError):
        with _GetcwdPatcher(_raise(NotADirectoryError(20, "Not a directory"))):
            terminal_tool._safe_getcwd()
