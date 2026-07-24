"""Native-Windows integration test: spawn_local children break away from the
parent job object.

This is the real-OS counterpart to the mocked kwargs tests in
``test_process_spawn_local_detached.py``. It stands up a controlled job object
with ``JOB_OBJECT_LIMIT_BREAKAWAY_OK``, assigns the *current* (pytest) process
to it, calls the real ``ProcessRegistry.spawn_local``, and asserts via
``IsProcessInJob`` that the spawned child is NOT a member of that job — i.e. it
used ``CREATE_BREAKAWAY_FROM_JOB`` and escaped.

Adversarial contract:
- Against pristine ``main`` (``windows_hide_flags()`` only), the child stays in
  the controlled job and ``IsProcessInJob(child, job)`` is True -> this test
  FAILS.
- Against the fix, the child breaks away and ``IsProcessInJob(child, job)`` is
  False -> this test PASSES.

Isolation: assigning a process to a job object is irreversible for that
process's lifetime, so this file must run in its OWN pytest process (it assigns
the interpreter running it). Do not merge it into a shared test module.

Scope: covers the standard pipe-backed (non-PTY) local background path only.
``use_pty=True`` uses pywinpty's separate spawn implementation and is out of
scope.

Uses only the stdlib ``ctypes`` — no pywin32 / pywinpty dependency.
"""

from __future__ import annotations

import ctypes
import sys
import time

import pytest

import tools.process_registry as pr

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="native Windows job-object semantics"
)


def test_spawn_local_child_breaks_away_from_controlled_job(tmp_path):
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ULONG_PTR),
            ("MaximumWorkingSetSize", ULONG_PTR),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ULONG_PTR),
            ("JobMemoryLimit", ULONG_PTR),
            ("PeakProcessMemoryUsed", ULONG_PTR),
            ("PeakJobMemoryUsed", ULONG_PTR),
        ]

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_TERMINATE = 0x0001

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.IsProcessInJob.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)
    ]
    k32.IsProcessInJob.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetCurrentProcess.argtypes = []
    k32.GetCurrentProcess.restype = wintypes.HANDLE  # pseudo-handle, do not close
    k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k32.TerminateProcess.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    def _is_in_job(proc_handle, job_handle) -> bool:
        result = wintypes.BOOL()
        if not k32.IsProcessInJob(proc_handle, job_handle, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(result.value)

    marker = "HERMES_BREAKAWAY_MARKER_42868"
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    child_handle = None
    session = None
    reg = None
    try:
        # BREAKAWAY_OK is required: it is what lets a child that requests
        # CREATE_BREAKAWAY_FROM_JOB actually escape (without it CreateProcess
        # returns ERROR_ACCESS_DENIED, which is the fallback path — a different
        # test). Build the info struct properly and size it via sizeof().
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_BREAKAWAY_OK
        if not k32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        # Assign THIS process to the controlled job. If the environment forbids
        # nesting (the interpreter is already in a job that disallows it), the
        # test cannot be set up — skip rather than error.
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            err = ctypes.get_last_error()
            pytest.skip(
                f"cannot assign the test process to a controlled job "
                f"(WinError {err}); nested job objects not permitted here"
            )

        # Guard against a vacuous test: if we are not actually in the job, a
        # child "not in the job" proves nothing.
        assert _is_in_job(k32.GetCurrentProcess(), job), (
            "setup failed: pytest process is not in the controlled job"
        )

        reg = pr.ProcessRegistry()
        # Long enough to stay alive while we query; the marker proves capture.
        session = reg.spawn_local(f"echo {marker}; sleep 20", cwd=str(tmp_path))
        assert session.pid, "spawn_local did not return a child pid"

        # Captured output must still work over the pipe (the point of the
        # non-PTY path) — poll the rolling buffer for the marker.
        captured = ""
        deadline = time.time() + 15.0
        while time.time() < deadline:
            captured = reg.read_log(session.id).get("output", "")
            if marker in captured:
                break
            time.sleep(0.2)
        assert marker in captured, (
            f"marker not captured within timeout; buffer={captured!r}"
        )

        # The child must have broken away from the controlled job.
        child_handle = k32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE,
            False,
            int(session.pid),
        )
        if not child_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        assert _is_in_job(child_handle, job) is False, (
            "child is still a member of the parent's job object — breakaway did "
            "not take effect (this is the failure pristine main reproduces)"
        )
    finally:
        # Cleanup order matters: run Hermes' /T tree termination FIRST so it can
        # still find the shell's descendants; killing the shell directly first
        # would orphan them. Only then fall back to a direct TerminateProcess on
        # the child handle. Each handle is closed exactly once.
        if reg is not None and session is not None:
            try:
                reg.kill_process(session.id)
            except Exception:
                pass
        if child_handle:
            try:
                time.sleep(0.3)
                # Only if kill_process did not already reap the child: a genuine
                # last-resort fallback, not a second executioner.
                if session is None or session.process is None or session.process.poll() is None:
                    k32.TerminateProcess(child_handle, 1)
            except Exception:
                pass
            finally:
                k32.CloseHandle(child_handle)
        k32.CloseHandle(job)
