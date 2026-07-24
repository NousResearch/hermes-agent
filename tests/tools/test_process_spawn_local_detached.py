"""Behavioral tests for ProcessRegistry.spawn_local job-object detachment.

The bug: a background process spawned via
:meth:`ProcessRegistry.spawn_local` (the agent's ``terminal(background=true)``
local path) was launched with only ``windows_hide_flags()``
(``CREATE_NO_WINDOW``). That leaves the child inside the parent's Windows job
object, so it is reaped if that job is torn down.

The fix routes the spawn through ``windows_detach_popen_kwargs()`` (which adds
``CREATE_BREAKAWAY_FROM_JOB`` on Windows and ``start_new_session=True`` on
POSIX) with a breakaway-denied ``OSError`` retry that drops only the breakaway
bit, mirroring ``hermes_cli/gateway_windows.py::_spawn_detached``.

These tests drive the real ``spawn_local`` coroutine-free method with a mocked
``subprocess.Popen`` and assert on the kwargs actually passed — not on source
text. They run on any host: both ``process_registry._IS_WINDOWS`` and
``hermes_cli._subprocess_compat.IS_WINDOWS`` are monkeypatched so each platform
branch is exercised deterministically (the Win32 flag constants are plain ints
defined unconditionally).

Scope: this covers the standard pipe-backed (non-PTY) local background path.
The ``use_pty=True`` branch spawns through pywinpty's own implementation and is
out of scope here; it is a separate spawn path and is not touched by this fix.
The native-Windows job-membership counterpart lives in
``test_process_spawn_local_job_breakaway_win.py``.
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli._subprocess_compat as sc
import tools.process_registry as pr
from hermes_cli._subprocess_compat import (
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
)

_BREAKAWAY = 0x01000000


class _FakeProc:
    """Minimal Popen stand-in. ``stdout=None`` makes the reader thread return
    immediately, so no real I/O or process is involved."""

    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.stdout = None

    def poll(self):
        return None

    def kill(self):
        pass

    def wait(self, timeout=None):
        return 0


def _force_platform(monkeypatch, *, windows: bool):
    monkeypatch.setattr(pr, "_IS_WINDOWS", windows)
    monkeypatch.setattr(sc, "IS_WINDOWS", windows)


def _quiet_registry(monkeypatch):
    """Neutralize disk/OS/thread side effects so spawn_local exercises only the
    spawn: no checkpoint write, no host-start-time probe, and a no-op reader
    loop (the reader thread is not under test and would otherwise touch
    attributes the fake proc doesn't have)."""
    monkeypatch.setattr(pr.ProcessRegistry, "_write_checkpoint", lambda self: None)
    monkeypatch.setattr(
        pr.ProcessRegistry, "_safe_host_start_time", staticmethod(lambda pid: None)
    )
    monkeypatch.setattr(pr.ProcessRegistry, "_reader_loop", lambda self, session: None)
    # _find_shell() probes candidate shells with its own subprocess.Popen calls;
    # pin it so the recorder below captures only spawn_local's own spawn.
    monkeypatch.setattr(pr, "_find_shell", lambda: "/bin/bash")
    return pr.ProcessRegistry()


def _install_popen(monkeypatch, side_effects):
    """Patch subprocess.Popen with a recorder driven by ``side_effects`` (each
    item is either an exception to raise or a proc to return)."""
    calls = []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        effect = side_effects[min(len(calls) - 1, len(side_effects) - 1)]
        if isinstance(effect, BaseException):
            raise effect
        return effect

    monkeypatch.setattr(pr.subprocess, "Popen", fake_popen)
    return calls


def test_spawn_local_windows_retries_without_breakaway_on_oserror(monkeypatch, tmp_path):
    _force_platform(monkeypatch, windows=True)
    reg = _quiet_registry(monkeypatch)
    calls = _install_popen(
        monkeypatch, [OSError(5, "Access is denied"), _FakeProc()]
    )

    session = reg.spawn_local("echo hi", cwd=str(tmp_path))

    assert len(calls) == 2, "restrictive-job OSError must trigger exactly one retry"
    (argv1, kw1), (argv2, kw2) = calls

    # argv is identical across primary and fallback. Do NOT pin the exact
    # shell-flag layout ("-lic", "set +m; ...") — that is _find_shell's concern,
    # not this fix's; assert only that the command reaches the shell.
    assert argv1 == argv2
    assert any("echo hi" in part for part in argv1), "command must reach the shell"

    # Every non-creationflags kwarg is identical across the two attempts (only
    # the breakaway bit in creationflags is allowed to differ).
    assert set(kw1) - {"creationflags"} == set(kw2) - {"creationflags"}
    for key in (set(kw1) | set(kw2)) - {"creationflags"}:
        assert kw1[key] == kw2[key], f"{key} differs across retry"

    # Stdio config is preserved on both attempts — process(action="poll")
    # depends on the captured pipe.
    for kw in (kw1, kw2):
        assert kw["stdout"] is subprocess.PIPE
        assert kw["stderr"] is subprocess.STDOUT
        assert kw["stdin"] is subprocess.DEVNULL
        assert kw["close_fds"] is True

    # Primary carries breakaway; fallback drops only that bit. No session kwarg
    # on the Windows branch, and never a preexec_fn.
    assert kw1["creationflags"] == windows_detach_flags()
    assert kw1["creationflags"] & _BREAKAWAY
    assert kw2["creationflags"] == windows_detach_flags_without_breakaway()
    assert not (kw2["creationflags"] & _BREAKAWAY)
    # The two flag sets differ by *exactly* the breakaway bit — nothing else
    # changes between the primary and the fallback.
    assert kw1["creationflags"] ^ kw2["creationflags"] == _BREAKAWAY
    for kw in (kw1, kw2):
        assert "start_new_session" not in kw
        assert "preexec_fn" not in kw

    # The retry succeeded, so a real session is registered and returned.
    assert session.pid == 4242
    assert session.id in reg._running


def test_spawn_local_windows_dual_failure_surfaces_to_caller(monkeypatch, tmp_path):
    _force_platform(monkeypatch, windows=True)
    reg = _quiet_registry(monkeypatch)
    # Distinct exceptions so we can prove it is the *fallback's* failure that
    # propagates — not the primary's, silently re-raised.
    primary_exc = OSError(5, "primary breakaway denied")
    fallback_exc = OSError(1, "fallback also failed")
    calls = _install_popen(monkeypatch, [primary_exc, fallback_exc])

    # Both attempts fail -> the failure is surfaced, not swallowed into a
    # session that falsely claims the process started.
    with pytest.raises(OSError) as excinfo:
        reg.spawn_local("echo hi", cwd=str(tmp_path))

    assert excinfo.value is fallback_exc, "the fallback's exception must propagate"
    assert len(calls) == 2
    assert reg._running == {}, "no session may be registered when the spawn failed"


def test_spawn_local_windows_setup_failure_tree_kills_and_registers_no_session(
    monkeypatch, tmp_path
):
    """A post-Popen setup failure on Windows must tree-kill the child, not just
    proc.kill() it. The child has broken away from our job and owns descendants
    (the shell's grandchildren), so a bare kill would leak them untracked."""
    _force_platform(monkeypatch, windows=True)
    reg = _quiet_registry(monkeypatch)
    _install_popen(monkeypatch, [_FakeProc(pid=9999)])
    # Sentinel host start time so we can prove the tree terminator receives the
    # captured start time (its recycled-PID guard), not just the PID.
    monkeypatch.setattr(
        pr.ProcessRegistry, "_safe_host_start_time", staticmethod(lambda pid: 123456)
    )

    # Fail during post-Popen setup, BEFORE the session is registered.
    def boom(*args, **kwargs):
        raise RuntimeError("thread creation failed")

    monkeypatch.setattr(pr.threading, "Thread", boom)

    # Capture the start-time-validated tree terminator.
    killed = {}
    monkeypatch.setattr(
        pr.ProcessRegistry,
        "_terminate_host_pid",
        classmethod(lambda cls, pid, expected=None: killed.update(pid=pid, expected=expected)),
    )

    with pytest.raises(RuntimeError, match="thread creation failed"):
        reg.spawn_local("echo hi", cwd=str(tmp_path))

    # Tree terminator invoked with BOTH the child pid and the captured start time.
    assert killed == {"pid": 9999, "expected": 123456}
    assert reg._running == {}, "no session may be registered after a setup failure"


def test_spawn_local_windows_happy_path_single_spawn(monkeypatch, tmp_path):
    _force_platform(monkeypatch, windows=True)
    reg = _quiet_registry(monkeypatch)
    calls = _install_popen(monkeypatch, [_FakeProc()])

    reg.spawn_local("echo hi", cwd=str(tmp_path))

    assert len(calls) == 1, "no retry when the primary spawn succeeds"
    _, kw = calls[0]
    assert kw["creationflags"] == windows_detach_flags()
    assert kw["creationflags"] & _BREAKAWAY


def test_spawn_local_posix_single_session_no_duplicate_no_preexec(monkeypatch, tmp_path):
    _force_platform(monkeypatch, windows=False)
    reg = _quiet_registry(monkeypatch)
    calls = _install_popen(monkeypatch, [_FakeProc()])

    # Would raise TypeError("multiple values for 'start_new_session'") if the
    # helper's start_new_session collided with a hard-coded one.
    reg.spawn_local("echo hi", cwd=str(tmp_path))

    assert len(calls) == 1
    _, kw = calls[0]
    assert kw.get("start_new_session") is True
    assert "creationflags" not in kw
    assert "preexec_fn" not in kw


def test_spawn_local_posix_oserror_is_not_retried(monkeypatch, tmp_path):
    _force_platform(monkeypatch, windows=False)
    reg = _quiet_registry(monkeypatch)
    calls = _install_popen(monkeypatch, [OSError(2, "No such file")])

    # On POSIX there is no breakaway bit to drop, so a genuine spawn OSError
    # propagates immediately without a second attempt.
    with pytest.raises(OSError):
        reg.spawn_local("echo hi", cwd=str(tmp_path))

    assert len(calls) == 1
    assert reg._running == {}
