"""Tests for the Windows kill-on-exit Job Object that stops terminal-tool
shells (bash.exe and everything they spawn) from being orphaned when Hermes
exits ungracefully (issue #69033).

Three correctness properties are load-bearing here, each covered by a real
(non-mocked) test in addition to the mock-level contract tests:

1. Race-free assignment: a child must be assigned to the kill-on-exit job
   before it can execute a single instruction, so it cannot spawn a
   grandchild (e.g. a native ``find | grep | head`` pipeline) that escapes
   the job.
2. Thread isolation: the ``subprocess._winapi.CreateProcess`` patch is
   installed once, permanently, process-wide -- but it must be a no-op for
   every thread except the one currently inside
   ``spawn_bash_with_kill_on_exit``. An earlier per-call capture/restore
   design could both (a) leak a stale patch permanently under a
   capture-before-lock race, and (b) sweep totally unrelated subprocess
   spawns from other threads into our job even when correctly serialized.
   The thread-local owner gate replaces that design.
3. Thread-safe singletons: concurrent first callers of the job object and
   of the CreateProcess-patch installer must never end up with two
   competing objects where the loser gets garbage collected and takes
   something down with it (job) or silently un-does the winner's install
   (patch).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# spawn_bash_with_kill_on_exit: top-level wrapper contract
# ---------------------------------------------------------------------------


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
    """No job object available -> spawn proceeds without ever installing the
    patch, exactly today's behavior, never blocked."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: None)

    installed = []
    monkeypatch.setattr(
        _subprocess_compat,
        "_install_job_owned_create_process_once",
        lambda: installed.append(True),
    )

    sentinel = object()
    result = _subprocess_compat.spawn_bash_with_kill_on_exit(lambda: sentinel)
    assert result is sentinel
    assert installed == []


def test_spawn_bash_with_kill_on_exit_sets_and_clears_owner(monkeypatch):
    """The wrapper must set _spawn_owner.job for the duration of exactly the
    popen_fn() call on the calling thread, then clear it -- clearing is what
    keeps a later unrelated Popen on the SAME thread (e.g. the caller doing
    something else afterward) from being swept into the job too."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    fake_job = object()
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: fake_job)
    monkeypatch.setattr(_subprocess_compat, "_install_job_owned_create_process_once", lambda: None)

    assert getattr(_subprocess_compat._spawn_owner, "job", None) is None

    seen_during_call = {}

    def popen_fn():
        seen_during_call["job"] = getattr(_subprocess_compat._spawn_owner, "job", None)
        return "proc-sentinel"

    result = _subprocess_compat.spawn_bash_with_kill_on_exit(popen_fn)

    assert result == "proc-sentinel"
    assert seen_during_call["job"] is fake_job
    assert getattr(_subprocess_compat._spawn_owner, "job", None) is None


def test_spawn_bash_with_kill_on_exit_clears_owner_on_exception(monkeypatch):
    """If popen_fn() raises, _spawn_owner.job must still be cleared
    (finally, not just the happy path), and the exception must propagate."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
    monkeypatch.setattr(_subprocess_compat, "_get_kill_on_exit_job", lambda: object())
    monkeypatch.setattr(_subprocess_compat, "_install_job_owned_create_process_once", lambda: None)

    def boom():
        raise FileNotFoundError("bash not found")

    with pytest.raises(FileNotFoundError):
        _subprocess_compat.spawn_bash_with_kill_on_exit(boom)

    assert getattr(_subprocess_compat._spawn_owner, "job", None) is None


# ---------------------------------------------------------------------------
# _job_owned_create_process: the permanently-installed, thread-gated patch
# ---------------------------------------------------------------------------


def test_job_owned_create_process_passes_through_when_no_owner(monkeypatch):
    """A thread with no _spawn_owner.job set (i.e. any thread not currently
    inside spawn_bash_with_kill_on_exit) must reach the real CreateProcess
    completely unmodified -- no CREATE_SUSPENDED, no job assignment. This is
    the core of the fix for "process-wide patch bleeds into unrelated
    Popens": the patch is always installed, but inert by default."""
    from hermes_cli import _subprocess_compat

    assert getattr(_subprocess_compat._spawn_owner, "job", None) is None

    calls = []

    def fake_original(*args, **kwargs):
        calls.append((args, kwargs))
        return ("HP", "HT", 1, 2)

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)

    result = _subprocess_compat._job_owned_create_process("a", "b", flags=0)
    assert result == ("HP", "HT", 1, 2)
    assert calls == [(("a", "b"), {"flags": 0})]


def test_job_owned_create_process_suspends_assigns_and_resumes_when_owned(monkeypatch):
    """When the calling thread owns a job (_spawn_owner.job is set), the
    patch must: OR in CREATE_SUSPENDED on the real call, assign the
    returned process handle to the job, then resume the thread -- in that
    order -- and return the same (hp, ht, pid, tid) tuple."""
    from hermes_cli import _subprocess_compat

    events = []
    fake_job = object()

    def fake_original(*args):
        events.append(("create", args[_subprocess_compat._CREATION_FLAGS_ARG_INDEX]))
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

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)
    _subprocess_compat._spawn_owner.job = fake_job
    try:
        args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
        result = _subprocess_compat._job_owned_create_process(*args)
    finally:
        _subprocess_compat._spawn_owner.job = None

    assert result == (7001, 7002, 111, 222)
    assert events[0] == ("create", _subprocess_compat._CREATE_SUSPENDED)
    assert events[1] == ("assign", 7001)
    assert events[2] == ("resume", 7002)


def test_job_owned_create_process_resumes_even_if_assign_fails(monkeypatch):
    """A job-assignment failure (already in a non-nesting job, access
    denied) must not leave the child permanently suspended — resume must
    still run."""
    from hermes_cli import _subprocess_compat

    events = []
    fake_job = object()

    def fake_original(*args):
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            raise OSError("access denied")

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            events.append("resumed")

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)
    _subprocess_compat._spawn_owner.job = fake_job
    try:
        args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
        result = _subprocess_compat._job_owned_create_process(*args)
    finally:
        _subprocess_compat._spawn_owner.job = None

    assert result == (7001, 7002, 111, 222)
    assert events == ["resumed"]


def test_job_owned_create_process_terminates_and_raises_when_resume_fails(monkeypatch):
    """A ResumeThread failure leaves a live but permanently-suspended child
    -- any caller reading its stdout would hang forever. The patch must
    terminate the child and raise, not return a stuck process silently."""
    from hermes_cli import _subprocess_compat

    events = []
    fake_job = object()

    def fake_original(*args):
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            events.append(("assign", handle))

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            raise OSError("invalid handle")

        @staticmethod
        def TerminateProcess(handle, code):
            events.append(("terminate", handle, code))

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)
    _subprocess_compat._spawn_owner.job = fake_job
    try:
        args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
        with pytest.raises(OSError):
            _subprocess_compat._job_owned_create_process(*args)
    finally:
        _subprocess_compat._spawn_owner.job = None

    assert ("assign", 7001) in events
    assert ("terminate", 7001, 1) in events


def test_job_owned_create_process_closes_handles_via_close_handle_on_resume_failure(monkeypatch):
    """CPython's raw ``_winapi.CreateProcess`` returns plain integer
    handles, not PyHANDLE wrapper objects -- there is no ``.Close()``
    method on them. Calling ``.Close()`` used to raise AttributeError,
    silently swallowed by the surrounding ``except Exception``, leaking
    both handles on every ResumeThread failure. The correct API for raw
    int handles is ``_winapi.CloseHandle``; this must be called for both
    the process and thread handle, with no AttributeError swallowed along
    the way."""
    from hermes_cli import _subprocess_compat

    fake_job = object()

    def fake_original(*args):
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            pass

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            raise OSError("invalid handle")

        @staticmethod
        def TerminateProcess(handle, code):
            pass

    closed = []

    def fake_close_handle(handle):
        closed.append(handle)

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "_winapi",
        type("W", (), {"CloseHandle": staticmethod(fake_close_handle)})(),
    )
    _subprocess_compat._spawn_owner.job = fake_job
    try:
        args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
        with pytest.raises(OSError):
            _subprocess_compat._job_owned_create_process(*args)
    finally:
        _subprocess_compat._spawn_owner.job = None

    assert sorted(closed) == [7001, 7002]


def test_job_owned_create_process_warns_but_still_raises_when_terminate_also_fails(monkeypatch, caplog):
    """If TerminateProcess ALSO fails after ResumeThread failed, that must
    not crash the cleanup path or swallow the original error -- it should
    be visible (a warning) and the original ResumeThread failure must
    still propagate."""
    import logging

    from hermes_cli import _subprocess_compat

    fake_job = object()

    def fake_original(*args):
        return (7001, 7002, 111, 222)

    class _FakeWin32Job:
        @staticmethod
        def AssignProcessToJobObject(job, handle):
            pass

    class _FakeWin32Process:
        @staticmethod
        def ResumeThread(handle):
            raise OSError("resume failed")

        @staticmethod
        def TerminateProcess(handle, code):
            raise OSError("terminate also failed")

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    monkeypatch.setattr(_subprocess_compat, "win32job", _FakeWin32Job)
    monkeypatch.setattr(_subprocess_compat, "win32process", _FakeWin32Process)
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "_winapi",
        type("W", (), {"CloseHandle": staticmethod(lambda h: None)})(),
    )
    _subprocess_compat._spawn_owner.job = fake_job
    try:
        args = ["app", "cmdline", None, None, 0, 0, {}, None, "startupinfo"]
        with caplog.at_level(logging.WARNING, logger="hermes_cli._subprocess_compat"):
            with pytest.raises(OSError, match="resume failed"):
                _subprocess_compat._job_owned_create_process(*args)
    finally:
        _subprocess_compat._spawn_owner.job = None

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("TerminateProcess also failed" in r.getMessage() for r in warnings)


def test_job_owned_create_process_falls_open_on_unexpected_signature(monkeypatch):
    """A future CPython signature change (unexpected argc, or kwargs) must
    not be blindly indexed into — fail open to an unmodified, un-suspended,
    unassigned spawn rather than corrupting an unrelated positional arg."""
    from hermes_cli import _subprocess_compat

    calls = []

    def fake_original(*args, **kwargs):
        calls.append((args, kwargs))
        return (7001, 7002, 111, 222)

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", fake_original)
    _subprocess_compat._spawn_owner.job = object()
    try:
        result = _subprocess_compat._job_owned_create_process("only", "two", "args")
    finally:
        _subprocess_compat._spawn_owner.job = None

    assert result == (7001, 7002, 111, 222)
    assert calls == [(("only", "two", "args"), {})]


def test_install_job_owned_create_process_once_is_idempotent(monkeypatch):
    """Installing twice must not re-capture (and thereby lose) the true
    original -- the second call is a no-op."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", None)
    real_cp = object()
    monkeypatch.setattr(
        _subprocess_compat.subprocess,
        "_winapi",
        type("W", (), {"CreateProcess": real_cp})(),
    )

    _subprocess_compat._install_job_owned_create_process_once()
    installed_once = _subprocess_compat.subprocess._winapi.CreateProcess
    assert _subprocess_compat._original_create_process is real_cp
    assert installed_once is _subprocess_compat._job_owned_create_process

    _subprocess_compat._install_job_owned_create_process_once()
    assert _subprocess_compat.subprocess._winapi.CreateProcess is installed_once
    assert _subprocess_compat._original_create_process is real_cp


def test_install_job_owned_create_process_once_concurrent_first_callers(monkeypatch):
    """Two threads racing the very first install must not each capture a
    different 'original' -- CreateJobObject-style double-checked locking
    applies here too. Without the lock, thread B could capture thread A's
    already-installed patched function as "the original" (finding 1's
    failure mode), permanently wrapping CreateProcess in a way nothing can
    ever fully undo. This test rendezvous's both threads inside the capture
    step to force the race window open."""
    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_original_create_process", None)
    real_cp = object()

    captured = []
    entered_capture = threading.Barrier(2, timeout=5)

    class _WinApi:
        @property
        def CreateProcess(self):
            captured.append(True)
            try:
                entered_capture.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
            return real_cp

        @CreateProcess.setter
        def CreateProcess(self, value):
            self._cp = value

    monkeypatch.setattr(_subprocess_compat.subprocess, "_winapi", _WinApi())

    def call():
        _subprocess_compat._install_job_owned_create_process_once()

    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert len(captured) == 1, (
        f"expected CreateProcess to be captured exactly once, got {len(captured)} -- "
        "a second capture means the lock did not serialize the install"
    )
    assert _subprocess_compat._original_create_process is real_cp


def test_install_job_owned_create_process_once_is_reload_safe():
    """importlib.reload(hermes_cli._subprocess_compat) re-executes the
    module's top-level code in place, which resets _original_create_process
    back to None -- but subprocess._winapi.CreateProcess still points at
    the PRIOR generation's installed wrapper (reload never uninstalls it).
    A naive re-install would capture that already-installed wrapper as "the
    original" and recurse into itself on every future call. This is a real
    (non-mocked) end-to-end reproduction against the actual subprocess
    module, confirmed to RecursionError before this fix and to pass cleanly
    after it."""
    import importlib
    import subprocess as real_subprocess

    from hermes_cli import _subprocess_compat as m

    original_create_process = real_subprocess._winapi.CreateProcess
    try:
        m._install_job_owned_create_process_once()
        wrapper_before_reload = real_subprocess._winapi.CreateProcess

        importlib.reload(m)
        m._install_job_owned_create_process_once()

        # The installed wrapper must be untouched by the reload (proves we
        # skipped re-patching rather than nesting a second wrapper around
        # the first), and the true original must have been correctly
        # recovered rather than left as None or as our own wrapper.
        assert real_subprocess._winapi.CreateProcess is wrapper_before_reload
        assert m._original_create_process is original_create_process

        # The actual failure mode: calling through the installed wrapper on
        # a thread that owns no job (the common case) must reach the real
        # CreateProcess without recursing.
        m._spawn_owner.job = None
        try:
            real_subprocess._winapi.CreateProcess(
                "definitely-does-not-exist.exe", "", None, None, 0, 0, {}, None, None
            )
        except RecursionError:
            pytest.fail("CreateProcess patch recursed into itself after reload")
        except OSError:
            pass  # expected: the bogus executable path fails normally
    finally:
        real_subprocess._winapi.CreateProcess = original_create_process
        m._original_create_process = None


# ---------------------------------------------------------------------------
# Job-object singleton (kept from the prior review round)
# ---------------------------------------------------------------------------


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


def test_get_kill_on_exit_job_returns_none_when_unavailable_and_warns_once(monkeypatch, caplog):
    """pywin32 unavailable -> returns None AND emits the one-time warning
    (previously this path returned before the warning could fire, so the
    "job assignment unavailable" condition was completely silent)."""
    import logging

    from hermes_cli import _subprocess_compat

    monkeypatch.setattr(_subprocess_compat, "_kill_on_exit_job", None)
    monkeypatch.setattr(_subprocess_compat, "_WIN32_JOB_AVAILABLE", False)
    monkeypatch.setattr(_subprocess_compat, "_warned_job_assignment_unavailable", False)

    with caplog.at_level(logging.WARNING, logger="hermes_cli._subprocess_compat"):
        assert _subprocess_compat._get_kill_on_exit_job() is None

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "pywin32 unavailable" in warnings[0].getMessage()


def test_get_kill_on_exit_job_concurrent_first_callers_create_exactly_one_job(monkeypatch):
    """Two threads racing the very first call must not each create their own
    job object -- see the module-level rationale on _job_singleton_lock."""
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


# ---------------------------------------------------------------------------
# Call-site wiring
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Real (non-mocked) integration tests -- Windows only
# ---------------------------------------------------------------------------


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
    after assignment), not just empirically-usually. This exercises the
    Python-Python spawn path; a git-bash-native equivalent was investigated
    but is not included -- see the commit message for why (MSYS PID
    virtualization makes deterministic native-child identification
    impractical). This test still exercises the identical kernel primitive
    (CreateProcess + Job Object inheritance) that a bash pipeline would.
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only: Job Object semantics")
def test_unrelated_thread_spawn_is_not_swept_into_job():
    """The critical thread-isolation proof: while thread A is inside
    spawn_bash_with_kill_on_exit (owns _spawn_owner.job on its own thread),
    thread B calls plain subprocess.Popen directly, concurrently, on the
    SAME process (so it goes through the same, permanently-patched
    subprocess._winapi.CreateProcess). Thread B's child must NOT be
    assigned to the job -- it must survive closing the job handle, proving
    the patch stayed inert for a thread that never opted in.

    This is the load-bearing regression test for "process-wide patch bleeds
    into unrelated Popens": a naive per-call patch swap (rather than the
    thread-local owner gate) would make thread B's process die here too.
    """
    import psutil
    import win32api

    from hermes_cli import _subprocess_compat

    _subprocess_compat._kill_on_exit_job = None

    owned_ready = threading.Event()
    unrelated_ready = threading.Event()
    release_owned = threading.Event()

    owned_proc = {}
    unrelated_proc = {}

    def owned_thread():
        def popen_fn():
            proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
            )
            owned_proc["proc"] = proc
            owned_ready.set()
            # Hold this thread "inside" spawn_bash_with_kill_on_exit (i.e.
            # _spawn_owner.job still set on this thread) while thread B
            # spawns its own, unrelated process -- this is the window an
            # unsafe patch could leak into.
            release_owned.wait(timeout=5)
            return proc

        _subprocess_compat.spawn_bash_with_kill_on_exit(popen_fn)

    def unrelated_thread():
        owned_ready.wait(timeout=5)
        # Plain, direct subprocess.Popen -- NOT going through
        # spawn_bash_with_kill_on_exit. _spawn_owner.job is unset on this
        # thread regardless of what thread A is doing concurrently.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        unrelated_proc["proc"] = proc
        unrelated_ready.set()

    t_owned = threading.Thread(target=owned_thread)
    t_unrelated = threading.Thread(target=unrelated_thread)
    t_owned.start()
    t_unrelated.start()

    assert unrelated_ready.wait(timeout=5), "unrelated thread never finished spawning"
    release_owned.set()
    t_owned.join(timeout=5)
    t_unrelated.join(timeout=5)

    owned = owned_proc["proc"]
    unrelated = unrelated_proc["proc"]

    try:
        assert psutil.pid_exists(owned.pid)
        assert psutil.pid_exists(unrelated.pid)

        job = _subprocess_compat._kill_on_exit_job
        assert job is not None, "job assignment did not happen"
        win32api.CloseHandle(job)
        _subprocess_compat._kill_on_exit_job = None

        deadline = time.time() + 5
        while time.time() < deadline and psutil.pid_exists(owned.pid):
            time.sleep(0.1)

        assert not psutil.pid_exists(owned.pid), "owned child survived job close"
        # Give the unrelated process a moment too, then assert it is still
        # alive -- it was never assigned to the job.
        time.sleep(0.5)
        assert psutil.pid_exists(unrelated.pid), (
            "unrelated thread's process was swept into the job and killed -- "
            "the CreateProcess patch leaked across threads"
        )
    finally:
        for proc in (owned, unrelated):
            try:
                proc.kill()
            except Exception:
                pass
