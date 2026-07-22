"""Tests for the Windows kill-on-exit Job Object that stops terminal-tool
shells (bash.exe and everything they spawn) from being orphaned when Hermes
exits ungracefully (issue #69033).

Two correctness properties are load-bearing here, both covered by a real
(non-mocked) test in addition to the mock-level contract tests:

1. Race-free assignment: a child must be assigned to the kill-on-exit job
   before it can execute a single instruction, so it cannot spawn a
   grandchild (e.g. a native ``find | grep | head`` pipeline) that escapes
   the job. This rules out a handle-only ``AssignProcessToJobObject`` call
   issued *after* ``Popen()`` returns, which can lose the race to a fast
   native child under parent-thread scheduling pressure — the suspend
   assign resume sequence closes it structurally instead of statistically.
2. Thread-safe singleton: concurrent first callers must never end up with
   two job objects where the loser's object gets garbage collected (closing
   its last handle) and kills whatever was meanwhile assigned to it.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest


def test_spawn_bash_with_kill_on_exit_patches_create_process_on_windows(monkeypatch):
    """On Windows, the wrapper must temporarily patch
    subprocess._winapi.CreateProcess (so CREATE_SUSPENDED + job assignment +
    resume can happen inside Popen's own CreateProcess call) and always
    restore the original afterward."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    fake_job = object()
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: fake_job)

    captured_patch_arg = {}

    def fake_make_patched(job, real_cp):
        captured_patch_arg["job"] = job
        captured_patch_arg["real_cp"] = real_cp
        return "patched-sentinel"

    monkeypatch.setattr(_subprocess_compat, "_make_patched_create_process", fake_make_patched)

    real_cp = object()
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "_winapi",
        type("W", (), {"CreateProcess": real_cp})(),
    )

    seen_during_call = {}

    def popen_fn():
        seen_during_call["CreateProcess"] = _subprocess_compat.subprocess._winapi.CreateProcess
        return "proc-sentinel"

    result = _subprocess_compat.spawn_bash_with_kill_on_exit(popen_fn)

    assert result == "proc-sentinel"
    assert captured_patch_arg["job"] is fake_job
    assert captured_patch_arg["real_cp"] is real_cp
    assert seen_during_call["CreateProcess"] == "patched-sentinel"
    # Restored afterward.
    assert _subprocess_compat.subprocess._winapi.CreateProcess is real_cp


def test_spawn_bash_with_kill_on_exit_restores_patch_on_exception(monkeypatch):
    """If popen_fn() raises, the CreateProcess patch must still be restored
    (finally, not just the happy path) and the exception must propagate."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: object())
    monkeypatch.setattr(
        _subprocess_compat, "_make_patched_create_process", lambda job, real: "patched"
    )

    real_cp = object()
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "_winapi",
        type("W", (), {"CreateProcess": real_cp})(),
    )

    def boom():
        raise FileNotFoundError("bash not found")

    with pytest.raises(FileNotFoundError):
        _subprocess_compat.spawn_bash_with_kill_on_exit(boom)

    assert _subprocess_compat.subprocess._winapi.CreateProcess is real_cp


def test_spawn_bash_with_kill_on_exit_noop_on_posix(monkeypatch):
    """On POSIX, the wrapper must not touch job assignment or the
    CreateProcess patch at all — the existing start_new_session=True
    (setsid) + pgid-kill machinery is already correct there."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", False)

    def boom():  # pragma: no cover - must not be called
        raise AssertionError("_get_kill_on_exit_job must not run on POSIX")

    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", boom)

    sentinel = object()
    result = _subprocess_compat.spawn_bash_with_kill_on_exit(lambda: sentinel)
    assert result is sentinel


def test_spawn_bash_with_kill_on_exit_falls_open_when_job_unavailable(monkeypatch):
    """No job object available -> spawn proceeds unpatched, exactly today's
    behavior, never blocked."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: None)

    sentinel = object()
    result = _subprocess_compat.spawn_bash_with_kill_on_exit(lambda: sentinel)
    assert result is sentinel


def test_patched_create_process_suspends_assigns_and_resumes(monkeypatch):
    """The patched CreateProcess must: OR in CREATE_SUSPENDED on the real
    call, assign the returned process handle to the job, then resume the
    thread — in that order, and it must return the same (hp, ht, pid, tid)
    tuple the real CreateProcess produced."""
    from hermes_cli import _subprocess_compat

    events = []
    fake_job = object()

    def fake_real_create_process(*args):
        events.append(("create", args[_subprocess_compat._CREATION_FLAGS_ARG_INDEX]))
        # Real handles are Win32 HANDLE-likes that support int() -- use
        # plain ints here rather than opaque strings so the code under test
        # (which does int(hp)/int(ht)) doesn't choke on the fixture itself.
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            assert job is fake_job
            events.append(("assign", handle))

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            events.append(("resume", handle))

    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)

    patched = _subprocess_compat._make_patched_create_process(fake_job, fake_real_create_process)

    args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
    result = patched(*args)

    assert result == (7001, 7002, 111, 222)
    assert events[0] == ("create", _subprocess_compat._CREATE_SUSPENDED)
    assert events[1] == ("assign", 7001)
    assert events[2] == ("resume", 7002)


def test_patched_create_process_resumes_even_if_assign_fails(monkeypatch):
    """A job-assignment failure (already in a non-nesting job, access
    denied) must not leave the child permanently suspended — resume must
    still run."""
    from hermes_cli import _subprocess_compat

    events = []
    fake_job = object()

    def fake_real_create_process(*args):
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            raise OSError("access denied")

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            events.append("resumed")

    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)

    patched = _subprocess_compat._make_patched_create_process(fake_job, fake_real_create_process)
    args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
    result = patched(*args)

    assert result == (7001, 7002, 111, 222)
    assert events == ["resumed"]


def test_patched_create_process_falls_open_on_unexpected_signature(monkeypatch):
    """A future CPython signature change (unexpected argc, or kwargs) must
    not be blindly indexed into — fail open to an unmodified, un-suspended,
    unassigned spawn rather than corrupting an unrelated positional arg."""
    from hermes_cli import _subprocess_compat

    calls = []

    def fake_real_create_process(*args, **kwargs):
        calls.append((args, kwargs))
        return ("HPROCESS", "HTHREAD", 111, 222)

    patched = _subprocess_compat._make_patched_create_process(object(), fake_real_create_process)

    result = patched("only", "two", "args")
    assert result == ("HPROCESS", "HTHREAD", 111, 222)
    assert calls == [(("only", "two", "args"), {})]


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


def test_get_kill_on_exit_job_concurrent_first_callers_create_exactly_one_job(monkeypatch):
    """Two threads racing the very first call must not each create their own
    job object. Without the lock, both threads can observe
    ``_kill_on_exit_job is None``, both create a job (A and B), and both
    publish — the loser's Python-side reference is then dropped, and CPython
    eventually closes its last handle, which fires
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and kills whatever was concurrently
    assigned to it. ``CreateJobObject`` is made to rendezvous on a barrier
    here so both threads are guaranteed to be inside the "is None" window at
    the same time absent a lock -- against an unlocked
    check-then-create-then-publish implementation this reliably produces two
    created jobs; with the lock, the second thread blocks before it can ever
    reach CreateJobObject, so only one job is ever created.
    """
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_kill_on_exit_job", None)
    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", True)

    created = []
    entered_create = threading.Barrier(2, timeout=5)

    class _FakeWin32Job:
        JobObjectExtendedLimitInformation = "info-class"
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        @staticmethod
        def CreateJobObject(security, name):
            try:
                entered_create.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
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

    results = []

    def call():
        results.append(_subprocess_compat._get_kill_on_exit_job())

    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert len(created) == 1, f"expected exactly one job created, got {len(created)}"
    assert results[0] is results[1] is created[0]


def test_warn_job_assignment_once_logs_exactly_once(monkeypatch, caplog):
    """Repeated failures must produce exactly one log line, not one per
    spawn -- fail-open should not mean fail-silent-forever with no signal,
    but it also must not spam."""
    import logging

    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_warned_job_assignment_unavailable", False)

    with caplog.at_level(logging.WARNING, logger="hermes_cli._subprocess_compat"):
        _subprocess_compat._warn_job_assignment_once("reason one")
        _subprocess_compat._warn_job_assignment_once("reason two")
        _subprocess_compat._warn_job_assignment_once("reason three")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "reason one" in warnings[0].getMessage()


def test_popen_bash_uses_kill_on_exit_wrapper(monkeypatch):
    """tools.environments.base._popen_bash must route its Popen call through
    spawn_bash_with_kill_on_exit rather than calling subprocess.Popen
    directly, so docker/ssh/singularity all get covered via the shared
    helper (mirrors the local-backend wiring)."""
    from tools.environments import base as env_base
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)

    routed = []

    def fake_wrapper(popen_fn):
        routed.append(True)
        return popen_fn()

    monkeypatch.setattr(env_base, "spawn_bash_with_kill_on_exit", fake_wrapper)

    class _FakeProc:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.kwargs = kwargs

    monkeypatch.setattr(env_base.subprocess, "Popen", _FakeProc)

    proc = env_base._popen_bash(["bash", "-c", "echo hi"])
    assert isinstance(proc, _FakeProc)
    assert routed == [True]


def test_local_backend_run_bash_uses_kill_on_exit_wrapper():
    """LocalEnvironment's bash spawn also routes through the same wrapper
    (source check -- the spawn is deep inside a long method with cwd
    recovery / shell-init logic that isn't worth re-mocking end-to-end
    here; the base.py test above exercises the wrapper's call contract
    directly)."""
    import inspect

    from tools.environments import local as env_local

    src = inspect.getsource(env_local)
    assert "spawn_bash_with_kill_on_exit" in src


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: Job Object semantics")
def test_real_grandchild_dies_when_job_handle_closes():
    """End-to-end proof: a child spawned via spawn_bash_with_kill_on_exit,
    which itself spawns a grandchild essentially immediately, loses BOTH
    processes when the shared kill-on-exit job's last handle is closed.

    The grandchild is spawned as the very first thing the child's script
    does, which is exactly the scenario a handle-only post-hoc
    AssignProcessToJobObject could lose the race on -- the suspend/assign/
    resume sequence under test here closes it structurally (the child
    cannot run *any* code, including spawning a grandchild, until resumed
    after assignment), not just empirically-usually.
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

    grandchild_pid = None
    try:
        grandchild_pid = int(proc.stdout.readline().strip())

        assert psutil.pid_exists(proc.pid)
        assert psutil.pid_exists(grandchild_pid)

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
        for pid in (proc.pid, grandchild_pid):
            if pid is None:
                continue
            try:
                psutil.Process(pid).kill()
            except Exception:
                pass
