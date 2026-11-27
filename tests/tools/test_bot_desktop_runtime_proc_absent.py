"""Bot Desktop runtime: no /proc, no crash (#134452).

A spawned launcher can land inside a mount/user namespace that mounts procfs
``hidepid=invisible,subset=pid`` — no ``/proc/stat``, no ``/proc/meminfo``. psutil
reads ``/proc/stat`` for ``boot_time()``, and the failure is NOT a ``psutil.Error``:
it surfaces as ``FileNotFoundError`` (file absent) or ``RuntimeError`` (line
missing). The runtime's identity read caught only ``psutil.Error``, so ``start()``
aborted after spawning the launcher — before ``launcher.pid`` was written — and
left the launcher's Xvnc + Xfce running as an unregistered orphan every time.

These tests pin the fix: an unmeasurable process must degrade to ``None`` /
not-alive, never raise.
"""

from __future__ import annotations

import os
import sys

import pytest

from tools.bot_desktop import runtime


class _NotPSUtilError(Exception):
    """Stand-in for ``psutil.Error``: the field failures are NOT subclasses of it."""


def _fake_psutil(proc_obj, *, status_zombie: str = "zombie"):
    class _Psutil:
        Error = _NotPSUtilError  # noqa: N801 -- psutil API name
        STATUS_ZOMBIE = status_zombie

        @staticmethod
        def Process(pid):
            return proc_obj

    return _Psutil


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("/proc/stat"),          # subset=pid / hidepid: file absent
        RuntimeError("line 'btime' not found in /proc/stat"),  # file present, unreadable content
        ProcessLookupError(),                      # pid vanished between read and use
        PermissionError("/proc/stat"),             # hardened procfs
    ],
    ids=["FileNotFoundError", "RuntimeError", "ProcessLookupError", "PermissionError"],
)
def test_create_time_degrades_to_none_when_proc_is_unavailable(monkeypatch, exc):
    """Any failure reading process identity is unmeasurable, not fatal."""
    class _Proc:
        def create_time(self):
            raise exc

    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(_Proc()))

    assert runtime._create_time(12345) is None


@pytest.mark.parametrize(
    "exc",
    [FileNotFoundError("/proc/stat"), RuntimeError("line 'btime' not found in /proc/stat")],
    ids=["FileNotFoundError", "RuntimeError"],
)
def test_pid_alive_degrades_to_false_when_proc_is_unavailable(monkeypatch, exc):
    """Same contract as _create_time: an unmeasurable pid is simply not alive."""
    class _Proc:
        def status(self):
            raise exc

    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(_Proc()))

    assert runtime._pid_alive(12345) is False


def test_launcher_pid_reads_as_not_running_when_identity_is_unmeasurable(tmp_path, monkeypatch):
    """``launcher.pid`` must not report running when its birth time cannot be checked:
    the recycled-pid guard compares against a measured create_time."""
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    (tmp_path / "launcher.pid").write_text(f"{os.getpid()} 12345.0", encoding="utf-8")

    monkeypatch.setattr(runtime, "_create_time", lambda pid: None)

    assert runtime._launcher_pid() is None


def test_recorded_pid_stays_the_orphan_sweeps_handle_when_identity_is_unmeasurable(tmp_path, monkeypatch):
    """#134452, the case @liuhao1024 asked to pin in review: with the procfs unreadable ``_create_time()``
    degrades to ``None`` and ``start()`` records ``"<pid> 0"`` — a birth time of 0. ``_launcher_pid()`` must
    read that as not-running (nothing can confirm the pid is ours), but ``_recorded_launcher_pid()`` — what
    ``stop()``'s orphan sweep matches process groups against — must still yield the pid: on a launcher that
    died under a procfs it could not read, this degraded record is the only handle left on its orphaned
    Xvnc group, and losing it would leak the group and its display.

    Credit: the case and its assertions come from @liuhao1024's review of #134452 (PR #134477)."""
    class _Proc:
        def create_time(self):
            raise FileNotFoundError(2, "No such file or directory", "/proc/stat")

    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(_Proc()))
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path)
    (tmp_path / "launcher.pid").write_text(f"{os.getpid()} 0", encoding="utf-8")  # what start() records then

    assert runtime._create_time(os.getpid()) is None
    assert runtime._launcher_pid() is None, "an unconfirmable identity is not running"
    assert runtime._recorded_launcher_pid() == os.getpid(), "the sweep must keep the only handle it has"
