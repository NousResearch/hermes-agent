"""process_manage write/submit on a background process without stdin names the fix that works.

A local ``terminal(background=true)`` spawn without ``pty=true`` connects stdin to /dev/null by
design; the old error blamed a "non-local backend or stdin closed", so agents retried blind.
"""
import pytest

from tests.waiting import wait_until
from tools.process_registry import ProcessRegistry


@pytest.fixture()
def registry():
    reg = ProcessRegistry()
    yield reg
    for sid in list(reg._running):
        reg.kill_process(sid)


def _spawn_reader(registry, cwd):
    # Alive and waiting; a bare ``read`` would hit EOF on /dev/null and exit before the write.
    return registry.spawn_local("sleep 30", cwd=str(cwd), task_id="t-stdin")


@pytest.mark.platforms("posix")
def test_pipe_mode_write_says_to_relaunch_with_pty(registry, tmp_path):
    session = _spawn_reader(registry, tmp_path)
    result = registry.write_stdin(session.id, "hello\n")
    assert result["status"] == "error"
    assert "pty=true" in result["error"] and "non-local" not in result["error"]


@pytest.mark.platforms("posix")
def test_no_pty_advice_where_a_pty_cannot_be_had(registry, monkeypatch, tmp_path):
    import tools.process_registry_stdin as stdin_mod

    session = _spawn_reader(registry, tmp_path)
    monkeypatch.setattr(stdin_mod, "_pty_available", lambda: False)
    error = stdin_mod.stdin_unavailable_error(session)
    assert "pty=true would not help" in error
    session.output_log = "worker.log"  # an isolated worker's log-backed spawn
    monkeypatch.setattr(stdin_mod, "_pty_available", lambda: True)
    assert "pty=true would not help" in stdin_mod.stdin_unavailable_error(session)


def test_non_local_session_keeps_the_backend_explanation():
    from types import SimpleNamespace

    from tools.process_registry_stdin import stdin_unavailable_error

    error = stdin_unavailable_error(SimpleNamespace(process=None, output_log=""))
    assert "non-local terminal backend" in error and "pty=true" not in error


@pytest.mark.platforms("posix")
def test_a_reader_that_hit_eof_says_why_it_already_exited(registry, tmp_path):
    """``read x`` on /dev/null returns at once, so the write lands on an exited process."""
    session = registry.spawn_local("read x; echo got:$x", cwd=str(tmp_path), task_id="t-stdin-eof")
    wait_until(lambda: session.exited, what="the EOF reader to exit")
    result = registry.write_stdin(session.id, "hello\n")
    assert result["status"] == "already_exited"
    assert "without stdin" in result["error"] and "pty=true" in result["error"]


@pytest.mark.platforms("posix")
def test_a_failed_pty_request_is_not_told_to_retry_pty(registry, monkeypatch, tmp_path):
    """PTY library installed but the spawn fails: spawn_local falls back to a pipe, and the error
    names that failure instead of telling the agent to repeat the pty=true that just failed."""
    import tools.process_registry_stdin as stdin_mod

    monkeypatch.setattr(stdin_mod, "_pty_available", lambda: True)

    def _pty_fails(*args, **kwargs):
        raise OSError("openpty: out of pseudo-terminals")

    monkeypatch.setattr(registry, "_spawn_local_pty", _pty_fails)
    session = registry.spawn_local("sleep 30", cwd=str(tmp_path), task_id="t-stdin-ptyfail", use_pty=True)
    error = registry.write_stdin(session.id, "hello\n")["error"]
    assert "out of pseudo-terminals" in error and "pty=true would not help" in error
    assert "Relaunch it with terminal(background=true, pty=true)" not in error
