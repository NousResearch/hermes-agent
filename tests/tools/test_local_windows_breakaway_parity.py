"""Behavioral tests for LocalEnvironment._run_bash Windows job-breakaway parity.

These drive the REAL patched ``_run_bash`` implementation with
``subprocess.Popen`` monkey-patched to a fake, so we assert the exact
kwargs/argv/env/cwd it constructs and how it reacts to constructor failures.
No real subprocesses and no real Windows APIs are required — flag values come
from the same helpers the production code uses, and both platform branches are
forced explicitly, so every test runs on any host OS.

Contract under test:

* Windows primary call receives the full detach kwargs (no start_new_session).
* A qualifying ERROR_ACCESS_DENIED (winerror 5) while the parent is confirmed
  in a job retries exactly once without CREATE_BREAKAWAY_FROM_JOB.
* winerror 5 with the parent NOT in a job propagates with exactly one attempt.
* winerror != 5 propagates with exactly one attempt.
* A job-membership query failure fails closed: original error propagates,
  no retry.
* POSIX receives exactly one start_new_session=True and no creationflags.
* No preexec_fn is introduced on either platform.
* The retry preserves argv/cwd/env/stdout/stderr/stdin/text/encoding/errors.
* The returned process remains explicitly killable.
* A successful Popen return is never respawned.
"""
import subprocess

import pytest

from hermes_cli._subprocess_compat import (
    windows_detach_flags_without_breakaway,
    windows_detach_popen_kwargs,
)
from tools.environments.local import LocalEnvironment


class _FakeStream:
    def write(self, data):
        return len(data)

    def read(self, n=-1):
        return ""

    def readline(self):
        return ""

    def close(self):
        pass


class _FakeProc:
    """Minimal stand-in for a spawned subprocess.Popen."""

    def __init__(self, args, kw):
        self.args = args
        self._kw = kw
        self.pid = 4242
        self.returncode = None
        self.stdout = _FakeStream()
        self.stderr = None
        self.stdin = _FakeStream()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def terminate(self):
        self.returncode = -15


@pytest.fixture
def spawn_calls(monkeypatch):
    """Record every subprocess.Popen invocation _run_bash makes.

    Patches the heavy environment helpers so _run_bash exercises only the
    spawn path, and suppresses the init_session bootstrap so the command
    under test is the sole Popen caller.
    """
    import tools.environments.local as local_mod

    monkeypatch.setattr(local_mod, "_find_bash", lambda: "bash")
    monkeypatch.setattr(local_mod, "_make_run_env", lambda env: dict(env or {}))
    monkeypatch.setattr(local_mod, "_resolve_safe_cwd", lambda cwd: cwd)
    monkeypatch.setattr(local_mod.BaseEnvironment, "init_session", lambda self: None)

    calls = []

    def _fake_popen(args, *a, **kw):
        calls.append((args, dict(kw)))
        return _FakeProc(args, kw)

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    return calls


def _force_windows(monkeypatch, *, in_job=True):
    import hermes_cli._subprocess_compat as compat_mod
    import tools.environments.local as local_mod

    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(compat_mod, "IS_WINDOWS", True)
    monkeypatch.setattr(local_mod, "_process_in_job", lambda: in_job)


def _force_posix(monkeypatch):
    import os

    import hermes_cli._subprocess_compat as compat_mod
    import tools.environments.local as local_mod

    monkeypatch.setattr(local_mod, "_IS_WINDOWS", False)
    monkeypatch.setattr(compat_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(os, "getpgid", lambda pid: 1000, raising=False)


def _fail_first_popen_with(monkeypatch, calls, exc):
    """Install a Popen fake whose first call raises *exc* and later calls
    succeed, recording every attempt into *calls*."""
    state = {"n": 0}

    def _popen(args, *a, **kw):
        calls.append((args, dict(kw)))
        state["n"] += 1
        if state["n"] == 1:
            raise exc
        return _FakeProc(args, kw)

    monkeypatch.setattr(subprocess, "Popen", _popen)


def _winerror(code, msg="spawn failed"):
    exc = OSError(msg)
    exc.winerror = code
    return exc


def _make_env():
    env = LocalEnvironment(cwd="/tmp")
    env.env = {"PATH": "/usr/bin"}
    return env


def test_windows_primary_full_detach_kwargs(monkeypatch, spawn_calls):
    _force_windows(monkeypatch)
    _make_env()._run_bash("echo hi")

    assert len(spawn_calls) == 1
    kw = spawn_calls[0][1]
    assert kw.get("creationflags") == windows_detach_popen_kwargs()["creationflags"]
    assert "start_new_session" not in kw
    assert "preexec_fn" not in kw
    assert kw["stdout"] is subprocess.PIPE
    assert kw["stderr"] is subprocess.STDOUT
    assert kw["stdin"] is subprocess.DEVNULL
    assert kw["text"] is True
    assert kw["encoding"] == "utf-8"
    assert kw["errors"] == "replace"


def test_windows_breakaway_refusal_retries_once(monkeypatch, spawn_calls):
    _force_windows(monkeypatch, in_job=True)
    _fail_first_popen_with(monkeypatch, spawn_calls, _winerror(5))

    proc = _make_env()._run_bash("echo hi")

    assert len(spawn_calls) == 2
    first, second = spawn_calls[0][1], spawn_calls[1][1]
    assert second.get("creationflags") == windows_detach_flags_without_breakaway()
    assert spawn_calls[0][0] == spawn_calls[1][0]
    for key in ("cwd", "env", "stdout", "stderr", "stdin", "text", "encoding", "errors"):
        assert second.get(key) == first.get(key), key
    assert isinstance(proc, _FakeProc)


def test_winerror_5_not_in_job_propagates(monkeypatch, spawn_calls):
    _force_windows(monkeypatch, in_job=False)

    def _always_fail(args, *a, **kw):
        spawn_calls.append((args, dict(kw)))
        raise _winerror(5)

    monkeypatch.setattr(subprocess, "Popen", _always_fail)

    with pytest.raises(OSError) as excinfo:
        _make_env()._run_bash("echo hi")

    assert excinfo.value.winerror == 5
    assert len(spawn_calls) == 1


def test_winerror_not_5_propagates(monkeypatch, spawn_calls):
    _force_windows(monkeypatch, in_job=True)

    def _always_fail(args, *a, **kw):
        spawn_calls.append((args, dict(kw)))
        raise _winerror(2, "file not found")

    monkeypatch.setattr(subprocess, "Popen", _always_fail)

    with pytest.raises(OSError) as excinfo:
        _make_env()._run_bash("echo hi")

    assert excinfo.value.winerror == 2
    assert len(spawn_calls) == 1


def test_job_membership_query_failure_fails_closed(monkeypatch, spawn_calls):
    """Force the real _process_in_job's ctypes boundary to fail: the helper
    must return False (never raise), the original winerror-5 error must
    propagate unchanged, and no fallback spawn may occur."""
    import ctypes

    import hermes_cli._subprocess_compat as compat_mod
    import tools.environments.local as local_mod

    monkeypatch.setattr(local_mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(compat_mod, "IS_WINDOWS", True)
    # No _process_in_job stub here — exercise the real helper with a broken
    # Win32 boundary underneath it.
    monkeypatch.setattr(
        ctypes, "WinDLL",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("kernel32 unavailable")),
        raising=False,
    )

    def _always_fail(args, *a, **kw):
        spawn_calls.append((args, dict(kw)))
        raise _winerror(5)

    monkeypatch.setattr(subprocess, "Popen", _always_fail)

    with pytest.raises(OSError) as excinfo:
        _make_env()._run_bash("echo hi")

    assert excinfo.value.winerror == 5
    assert len(spawn_calls) == 1


def test_second_constructor_failure_propagates(monkeypatch, spawn_calls):
    _force_windows(monkeypatch, in_job=True)

    def _always_fail(args, *a, **kw):
        spawn_calls.append((args, dict(kw)))
        raise _winerror(5)

    monkeypatch.setattr(subprocess, "Popen", _always_fail)

    with pytest.raises(OSError):
        _make_env()._run_bash("echo hi")

    assert len(spawn_calls) == 2  # primary + exactly one fallback, no third


def test_posix_single_start_new_session(monkeypatch, spawn_calls):
    _force_posix(monkeypatch)
    _make_env()._run_bash("echo hi")

    assert len(spawn_calls) == 1
    kw = spawn_calls[0][1]
    assert kw.get("start_new_session") is True
    assert "creationflags" not in kw
    assert "preexec_fn" not in kw


def test_posix_oserror_propagates_without_retry(monkeypatch, spawn_calls):
    _force_posix(monkeypatch)

    def _always_fail(args, *a, **kw):
        spawn_calls.append((args, dict(kw)))
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", _always_fail)

    with pytest.raises(OSError):
        _make_env()._run_bash("echo hi")

    assert len(spawn_calls) == 1


def test_returned_process_killable(monkeypatch, spawn_calls):
    _force_windows(monkeypatch)
    proc = _make_env()._run_bash("sleep 30")

    assert proc.returncode is None
    proc.kill()
    assert proc.returncode == -9


def test_no_respawn_after_success(monkeypatch, spawn_calls):
    _force_windows(monkeypatch)
    env = _make_env()
    proc = env._run_bash("echo hi")

    assert len(spawn_calls) == 1
    assert isinstance(proc, _FakeProc)
