"""A viewer renders PTY and pipe output with different line discipline, so the
registry must say which one a process is, on every surface that feeds a viewer."""

import time
from types import SimpleNamespace

from tools.process_registry import ProcessRegistry, ProcessSession


def _session(sid: str, *, pty) -> ProcessSession:
    session = ProcessSession(id=sid, command="cmd", task_id="t", session_key="k", started_at=time.time())
    session.process = SimpleNamespace(poll=lambda: None, stdout=None, stderr=None, stdin=None)
    session._pty = pty
    return session


def test_is_pty_follows_the_pty_handle():
    assert _session("proc_pty", pty=object()).is_pty is True
    # A use_pty spawn that fell back to pipes has no handle: it is a pipe process.
    assert _session("proc_pipe", pty=None).is_pty is False


def test_list_sessions_flags_only_pty_processes():
    registry = ProcessRegistry()
    registry._running["proc_pty"] = _session("proc_pty", pty=SimpleNamespace(isalive=lambda: True))
    registry._running["proc_pipe"] = _session("proc_pipe", pty=None)

    rows = {row["session_id"]: row for row in registry.list_sessions(session_key="k")}

    assert rows["proc_pty"].get("pty") is True
    assert "pty" not in rows["proc_pipe"]
