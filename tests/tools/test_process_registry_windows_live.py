"""LIVE Windows E2E for background-executor spawn parity (#70716 / PR salvage).

Runs ONLY on a real Windows host (the on-demand ``windows-venv-e2e.yml``
lane). The systemd cgroup-isolation feature for local background executors
must be a strict no-op on Windows: jobs spawn exactly as before, output is
captured, exit codes are correct, and no systemd code path is ever reached
— even when the process claims gateway identity.

These tests drive the REAL ``ProcessRegistry.spawn_local`` pipe path on the
live Windows process table (real Popen, real Git Bash shell, real reader
thread) — no mocked spawn.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time

import pytest

pytestmark = pytest.mark.windows_only


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    import tools.process_registry as pr

    reg = pr.ProcessRegistry()
    yield reg
    for sid in list(reg._running):
        try:
            reg.kill_process(sid)
        except Exception:
            pass


def _wait_exit(reg, sid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sess = reg._finished.get(sid) or reg._running.get(sid)
        if sess is not None and sess.exited:
            return sess
        time.sleep(0.2)
    raise AssertionError(f"session {sid} did not exit within {timeout}s")


class TestWindowsSpawnParity:
    def test_background_job_runs_output_and_exit_code_unchanged(self, registry):
        """Plain background job: spawned, output captured, exit code correct."""
        session = registry.spawn_local("echo win-live-parity; exit 7")
        done = _wait_exit(registry, session.id)

        assert done.exit_code == 7
        assert "win-live-parity" in done.output_buffer
        # The systemd scope identity must never be recorded on Windows.
        assert done.systemd_unit == ""

    def test_gateway_identity_never_reaches_systemd_path_on_windows(
        self, registry, monkeypatch
    ):
        """Even with full (faked) gateway identity, the Windows spawn takes
        the legacy path: no scope argv is built, no probe runs, and the job
        behaves exactly as without the identity."""
        import tools.process_registry as pr

        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(
            "gateway.status.get_running_pid",
            lambda *, cleanup_stale=False: os.getpid(),
        )
        monkeypatch.setattr(
            "gateway.restart.is_gateway_supervisor_process", lambda: True
        )
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)

        scope_builds = []
        monkeypatch.setattr(
            pr,
            "_build_systemd_scope_argv",
            lambda *a, **k: scope_builds.append(a) or a[0],
        )

        session = registry.spawn_local("echo win-live-gateway; exit 3")
        done = _wait_exit(registry, session.id)

        assert done.exit_code == 3
        assert "win-live-gateway" in done.output_buffer
        assert done.systemd_unit == ""
        assert scope_builds == [], "Windows must never build a systemd scope argv"
        # The availability probe must not have flipped to True on Windows.
        assert pr._SYSTEMD_SCOPE_AVAILABLE is not True

    def test_kill_process_windows_plain_path(self, registry):
        """kill_process on Windows works without any systemd unit cleanup."""
        session = registry.spawn_local("sleep 60")
        time.sleep(1.0)
        result = registry.kill_process(session.id)
        assert result.get("status") in {"killed", "already_exited"}
        assert session.systemd_unit == ""


@pytest.mark.windows_only
def test_pty_close_reports_unsupported_without_injecting_input(registry, tmp_path):
    """Native Windows input stays writable, but unsupported EOF must not write Ctrl-D."""
    code = (
        "import sys\n"
        "print('native-pty-ready',flush=True)\n"
        "for line in sys.stdin:\n"
        "    text=line.rstrip('\\r\\n')\n"
        "    print('RECEIVED:'+text.encode().hex(),flush=True)\n"
        "    if text.endswith('exit'): break\n"
    )
    python = shlex.quote(sys.executable.replace("\\", "/"))
    session = registry.spawn_local(
        f"{python} -u -c {shlex.quote(code)}", cwd=str(tmp_path), use_pty=True,
    )
    try:
        assert session._pty is not None, "native WinPTY backend must be exercised"
        deadline = time.monotonic() + 10
        while "native-pty-ready" not in session.output_buffer:
            assert time.monotonic() < deadline, "native Python child did not become ready"
            time.sleep(0.02)
        for _ in range(2):
            result = registry.close_stdin(session.id)
            assert result["status"] == "error"
            assert "EOF_UNSUPPORTED_FOR_PTY_BACKEND" in result["error"]
        assert registry.submit_stdin(session.id, "probe")["status"] == "ok"
        deadline = time.monotonic() + 5
        while "RECEIVED:70726f6265" not in session.output_buffer:
            assert time.monotonic() < deadline, session.output_buffer
            time.sleep(0.02)
        assert "RECEIVED:04" not in session.output_buffer
        assert registry.poll(session.id)["status"] == "running"
    finally:
        registry.submit_stdin(session.id, "exit")
        deadline = time.monotonic() + 5
        while not session.exited and time.monotonic() < deadline:
            time.sleep(0.02)
        if not session.exited:
            registry.kill_process(session.id)


@pytest.mark.windows_only
@pytest.mark.parametrize("operation", ["poll", "wait", "list_sessions"])
def test_exited_child_status_does_not_wait_for_inherited_pipe(
    registry, tmp_path, operation
):
    """A live descendant may retain stdout after the tracked child exits.

    Exercise the actual Windows buffered reader, not a fake pipe or platform.
    Release both fixture processes in finally even when the status call blocks.
    """
    exit_gate = tmp_path / "parent-exit"
    release_gate = tmp_path / "writer-exit"
    writer_code = (
        "import pathlib,sys,time; "
        "gate=pathlib.Path(sys.argv[1]); deadline=time.monotonic()+20; "
        "print('inherited-pipe-ready',flush=True)\n"
        "while not gate.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[3]], "
        "stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)\n"
        "gate=pathlib.Path(sys.argv[2]); deadline=time.monotonic()+20\n"
        "while not gate.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "sys.exit(7)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code, writer_code, str(exit_gate), str(release_gate)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW,
    )
    session = registry.adopt_local(
        proc, command="windows-inherited-pipe-test", cwd=str(tmp_path),
        task_id="pipe-task", session_key="pipe-session", owner_task_id="pipe-owner",
    )
    outcome = {}
    query_thread = None

    def query_status():
        try:
            if operation == "wait":
                outcome["result"] = registry.wait(session.id, timeout=2)
            elif operation == "list_sessions":
                outcome["result"] = registry.list_sessions(session_key="pipe-session")[0]
            else:
                outcome["result"] = registry.poll(session.id)
        except Exception as error:
            outcome["error"] = error

    try:
        deadline = time.monotonic() + 5
        while "inherited-pipe-ready" not in session.output_buffer:
            assert time.monotonic() < deadline, "descendant did not open the inherited pipe"
            time.sleep(0.01)
        exit_gate.touch()
        assert proc.wait(timeout=5) == 7
        assert session._reader_thread.is_alive()
        assert not session.exited

        query_thread = threading.Thread(target=query_status, daemon=True)
        query_thread.start()
        query_thread.join(timeout=2)
        assert not query_thread.is_alive(), "status waited for the descendant's stdout pipe"
        assert "error" not in outcome, repr(outcome.get("error"))
        assert outcome["result"]["status"] == "exited"
        assert outcome["result"]["exit_code"] == 7
        assert session._completion_event.is_set()
        event = registry.completion_queue.get(timeout=2)
        assert event["session_id"] == session.id
        assert event["owner_task_id"] == "pipe-owner"

        release_gate.touch()
        session._reader_thread.join(timeout=5)
        assert not session._reader_thread.is_alive()
        assert proc.stdout.closed, "reader must release the pipe after EOF"
        assert registry.completion_queue.empty(), "reader emitted a duplicate completion"
        assert "inherited-pipe-ready" in registry.read_log(session.id)["output"]
    finally:
        exit_gate.touch()
        release_gate.touch()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        session._reader_thread.join(timeout=5)
        if query_thread is not None:
            query_thread.join(timeout=5)
