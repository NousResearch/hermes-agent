"""Lifecycle helpers for ``tui_gateway.slash_worker`` subprocesses.

Orphaned slash workers (gateway crash, hard kill mid-compaction, session
reaped without worker.close) block Desktop reconnects until manually killed.
This module reaps them on gateway startup and ties new workers to the gateway
process lifetime on Windows via a kill-on-close job object.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Iterable

import psutil

logger = logging.getLogger(__name__)

_SLASH_WORKER_MARKER = "tui_gateway.slash_worker"
_GATEWAY_CMD_MARKERS = (
    "tui_gateway.entry",
    "tui_gateway.server",
    "tui_gateway",
    "hermes_cli.main gateway",
    "hermes gateway",
)

_reaper_ran = False


def _process_ppid(proc: psutil.Process) -> int:
    """Return parent PID across psutil versions (property on 5.x, method on 7.x)."""
    ppid = proc.ppid
    if callable(ppid):
        return int(ppid())
    return int(ppid)


def _cmdline_text(cmdline: Iterable[str] | None) -> str:
    if not cmdline:
        return ""
    return " ".join(str(part) for part in cmdline).lower()


def is_slash_worker_cmdline(cmdline: Iterable[str] | None) -> bool:
    text = _cmdline_text(cmdline)
    return _SLASH_WORKER_MARKER in text


def is_gateway_cmdline(cmdline: Iterable[str] | None) -> bool:
    text = _cmdline_text(cmdline)
    return any(marker in text for marker in _GATEWAY_CMD_MARKERS)


def has_live_gateway_owner(worker_pid: int, *, my_pid: int) -> bool:
    """True when ``worker_pid`` descends from a live gateway (this or another)."""
    try:
        current = _process_ppid(psutil.Process(worker_pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        if current == my_pid:
            return True
        try:
            parent = psutil.Process(current)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        status = parent.status()
        if status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return False
        if is_gateway_cmdline(parent.cmdline()):
            return True
        current = _process_ppid(parent)
    return False


def _terminate_pid(pid: int) -> None:
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return
    try:
        proc.terminate()
        proc.wait(timeout=1)
        return
    except (psutil.TimeoutExpired, psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    try:
        proc.kill()
        proc.wait(timeout=1)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
        pass


def reap_orphan_slash_workers(*, my_pid: int | None = None) -> int:
    """Terminate slash_worker processes with no live gateway owner.

    Safe to call at gateway startup and idempotent across concurrent gateway
    instances: workers owned by another live ``tui_gateway`` process are kept.
    """
    owner_pid = my_pid or os.getpid()
    reaped = 0
    try:
        processes = psutil.process_iter(["pid", "cmdline"])
    except Exception as exc:
        logger.debug("slash_worker orphan reaper: process_iter failed: %s", exc)
        return 0

    for info in processes:
        try:
            pid = int(info.info["pid"])
        except (TypeError, ValueError, KeyError):
            continue
        if pid == owner_pid:
            continue
        cmdline = info.info.get("cmdline") or []
        if not is_slash_worker_cmdline(cmdline):
            continue
        if has_live_gateway_owner(pid, my_pid=owner_pid):
            continue
        _terminate_pid(pid)
        reaped += 1
        logger.info(
            "Reaped orphaned slash_worker PID %d (cmd=%r)",
            pid,
            " ".join(cmdline) if cmdline else "",
        )
    return reaped


def maybe_reap_orphan_slash_workers_on_startup() -> None:
    """Run the orphan reaper once per gateway process."""
    global _reaper_ran
    if _reaper_ran:
        return
    _reaper_ran = True
    try:
        count = reap_orphan_slash_workers()
    except Exception:
        logger.debug("slash_worker orphan reaper failed", exc_info=True)
        return
    if count:
        logger.info("Reaped %d orphaned slash_worker process(es) on startup", count)


def attach_slash_worker_kill_job(proc: subprocess.Popen) -> object | None:
    """On Windows, bind the worker to a job that dies with this process."""
    if sys.platform != "win32":
        return None
    handle = getattr(proc, "_handle", None)
    if handle is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(job)
            return None

        if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(handle))):
            kernel32.CloseHandle(job)
            return None
        return job
    except Exception:
        logger.debug("Failed to attach slash_worker to kill-on-close job", exc_info=True)
        return None
