"""Tests for the Windows kill-on-exit Job Object that stops terminal-tool
shells (bash.exe and everything they spawn) from being orphaned when Hermes
exits ungracefully (issue #69033).

Mock-level tests assert the spawn wrapper requests job assignment on Windows
and leaves the POSIX path (``start_new_session=True``) untouched. A real
subprocess integration test (Windows-only, skipped elsewhere) proves a
grandchild process actually dies when the job handle is dropped.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest


def test_spawn_bash_with_kill_on_exit_assigns_job_on_windows(monkeypatch):
    """On Windows, the wrapper must call AssignProcessToJobObject with the
    spawned process's handle after Popen returns."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

    assigned = []
    monkeypatch.setattr(
        _subprocess_compat,
        "assign_to_kill_job",
        lambda proc: assigned.append(proc),
    )

    sentinel = object()
    result = _subprocess_compat.spawn_bash_with_kill_on_exit(lambda: sentinel)

    assert result is sentinel
    assert assigned == [sentinel]


def test_spawn_bash_with_kill_on_exit_noop_on_posix(monkeypatch):
    """On POSIX, the wrapper must NOT touch job assignment at all — the
    existing start_new_session=True (setsid) + pgid-kill machinery is
    already correct there."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", False)

    def boom(proc):  # pragma: no cover - must not be called
        raise AssertionError("assign_to_kill_job must not run on POSIX")

    monkeypatch.setattr(_subprocess_compat, "assign_to_kill_job", boom)

    sentinel = object()
    result = _subprocess_compat.spawn_bash_with_kill_on_exit(lambda: sentinel)
    assert result is sentinel


def test_assign_to_kill_job_calls_win32_api(monkeypatch):
    """assign_to_kill_job wires AssignProcessToJobObject(job, proc._handle)
    when the job object and pywin32 are available."""
    from hermes_cli import _subprocess_compat

    fake_job = object()
    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: fake_job)

    calls = []

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            calls.append((job, handle))

    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)

    class _FakeProc:
        _handle = 4242

    _subprocess_compat.assign_to_kill_job(_FakeProc())
    assert calls == [(fake_job, 4242)]


def test_assign_to_kill_job_fails_open_when_job_unavailable(monkeypatch):
    """No job object available (non-Windows, missing pywin32, creation
    failure) -> no-op, no raise."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: None)

    class _FakeProc:
        _handle = 4242

    # Must not raise.
    _subprocess_compat.assign_to_kill_job(_FakeProc())


def test_assign_to_kill_job_swallows_win32_errors(monkeypatch):
    """A Win32 assignment failure (pre-Win8 nested-job restriction, access
    denied, process already exited) must fail open, never raise into the
    spawn path."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: object())

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            raise OSError("access denied")

    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)

    class _FakeProc:
        _handle = 4242

    # Must not raise.
    _subprocess_compat.assign_to_kill_job(_FakeProc())


def test_get_kill_on_exit_job_is_singleton(monkeypatch):
    """The job object is created once and reused across calls."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_kill_on_exit_job", None)
    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", True)

    created = []

    class _FakeWin32Job:
        JobObjectExtendedLimitInformation = "info-class"
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        @staticmethod
        def CreateJobObject(security, name):
            job = object()
            created.append(job)
            return job

        @staticmethod
        def QueryInformationJobObject(job, info_class):
            return {"BasicLimitInformation": {"LimitFlags": 0}}

        @staticmethod
        def SetInformationJobObject(job, info_class, info):
            pass

    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)

    job1 = _subprocess_compat._get_kill_on_exit_job()
    job2 = _subprocess_compat._get_kill_on_exit_job()
    assert job1 is job2
    assert len(created) == 1


def test_get_kill_on_exit_job_returns_none_when_unavailable(monkeypatch):
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_kill_on_exit_job", None)
    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", False)

    assert _subprocess_compat._get_kill_on_exit_job() is None


def test_popen_bash_uses_kill_on_exit_wrapper(monkeypatch):
    """tools.environments.base._popen_bash must route its Popen call through
    spawn_bash_with_kill_on_exit rather than calling subprocess.Popen
    directly, so docker/ssh/singularity all get covered via the shared
    helper (mirrors the local-backend wiring)."""
    from tools.environments import base as env_base
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

    assigned = []
    monkeypatch.setattr(_subprocess_compat, "assign_to_kill_job", lambda proc: assigned.append(proc))

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs

    monkeypatch.setattr(env_base.subprocess, "Popen", _FakeProc)

    proc = env_base._popen_bash(["bash", "-c", "echo hi"])
    assert isinstance(proc, _FakeProc)
    assert assigned == [proc]


def test_local_backend_run_bash_uses_kill_on_exit_wrapper(monkeypatch):
    """LocalEnvironment's bash spawn also routes through the same wrapper."""
    from tools.environments import local as env_local
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

    assigned = []
    monkeypatch.setattr(_subprocess_compat, "assign_to_kill_job", lambda proc: assigned.append(proc))

    import inspect

    src = inspect.getsource(env_local)
    assert "spawn_bash_with_kill_on_exit" in src


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: Job Object semantics")
def test_real_grandchild_dies_when_job_handle_closes():
    """End-to-end proof: a child spawned via spawn_bash_with_kill_on_exit,
    which itself spawns a grandchild, loses BOTH processes when the shared
    kill-on-exit job's last handle is closed — even though neither process
    was asked to exit and neither is a direct descendant relationship the
    OS process tree would show as "parent".

    This is the deterministic contract test justified in the module
    docstring: Job Objects notify+kill synchronously on
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so there is no timing flakiness to
    poll around beyond OS scheduling latency (bounded wait below).
    """
    import psutil
    import win32api

    from hermes_cli import _subprocess_compat

    # Reset the module-global singleton so this test creates (and owns) its
    # own job, independent of any earlier import-time state.
    _subprocess_compat._kill_on_exit_job = None

    child_script = (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(gc.pid, flush=True)\n"
        "time.sleep(30)\n"
    )

    proc = _subprocess_compat.spawn_bash_with_kill_on_exit(
        lambda: subprocess.Popen(
            [sys.executable, "-c", child_script],
            stdout=subprocess.PIPE,
            text=True,
        )
    )

    try:
        grandchild_pid = int(proc.stdout.readline().strip())

        assert psutil.pid_exists(proc.pid)
        assert psutil.pid_exists(grandchild_pid)

        # Drop the one handle keeping the job alive -> OS kills every
        # process still assigned to it, synchronously.
        job = _subprocess_compat._kill_on_exit_job
        assert job is not None, "job assignment did not happen"
        win32api.CloseHandle(job)
        _subprocess_compat._kill_on_exit_job = None

        deadline = time.time() + 5
        while time.time() < deadline and (
            psutil.pid_exists(proc.pid) or psutil.pid_exists(grandchild_pid)
        ):
            time.sleep(0.1)

        assert not psutil.pid_exists(proc.pid), "child survived job close"
        assert not psutil.pid_exists(grandchild_pid), "grandchild survived job close"
    finally:
        for pid in (proc.pid,):
            try:
                psutil.Process(pid).kill()
            except Exception:
                pass
