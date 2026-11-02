"""submit_stdin presses Enter the way the child's input expects it: CR on a POSIX PTY, LF on a pipe."""

import time
from unittest.mock import MagicMock

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture()
def registry():
    return ProcessRegistry()


def _make_session() -> ProcessSession:
    return ProcessSession(id="proc_submit", command="cat", task_id="t1", started_at=time.time())


@pytest.mark.platforms("posix")
def test_submit_stdin_pty_mode_sends_cr(registry):
    """PTY Enter is \\r — what a real keyboard sends. Raw-mode TUIs bind
    Enter to \\r and ignore \\n; canonical-mode children still see \\n via
    the tty driver's ICRNL mapping.
    """
    pty = MagicMock()
    s = _make_session()
    s._pty = pty
    registry._running[s.id] = s

    result = registry.submit_stdin(s.id, "hello")

    pty.write.assert_called_once_with(b"hello\r")
    assert result["status"] == "ok"


def test_submit_stdin_pipe_mode_sends_newline(registry):
    proc = MagicMock()
    proc.stdin = MagicMock()
    s = _make_session()
    s.process = proc
    registry._running[s.id] = s

    result = registry.submit_stdin(s.id, "hello")

    proc.stdin.write.assert_called_once_with("hello\n")
    assert result["status"] == "ok"


def _dispatch_submit(monkeypatch) -> MagicMock:
    """The model-facing `process_manage submit` is one PTY write of text + Enter, the shape of
    a one-call `tmux send-keys 'text' Enter` that the CLI submits once input goes quiet
    (hermes_cli/cli_enter_burst.py). Two writes would put a gap between text and Enter."""
    import json

    import tools.process_registry as pr
    from tools.registry import registry as tool_registry

    pty = MagicMock()
    s = _make_session()
    s._pty = pty
    monkeypatch.setitem(pr.process_registry._running, s.id, s)

    result = tool_registry.dispatch("process_manage", {"action": "submit", "session_id": s.id, "data": "/help"})

    assert json.loads(result)["status"] == "ok"
    return pty


@pytest.mark.platforms("posix")
def test_process_manage_submit_is_one_pty_write_posix(monkeypatch):
    """ptyprocess takes bytes; Enter is the CR a keyboard sends."""
    _dispatch_submit(monkeypatch).write.assert_called_once_with(b"/help\r")


@pytest.mark.platforms("windows")
def test_process_manage_submit_is_one_pty_write_windows(monkeypatch):
    """pywinpty takes str; Enter is CRLF so ConPTY ends the line read."""
    _dispatch_submit(monkeypatch).write.assert_called_once_with("/help\r\n")
