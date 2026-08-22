#!/usr/bin/env python3
"""Parent-death watchdog supervisor for stdio MCP subprocesses.

Problem this fixes (#TBD): a stdio MCP server (e.g. ``npx -y mcp-remote
<url>``) is spawned as a direct child of the Hermes process. Hermes's own
teardown path (``MCPServerTask.shutdown()`` / ``_kill_orphaned_mcp_children``
at final exit) reaps it cleanly on a *graceful* exit. But if the spawning
Hermes process dies hard — ``kill -9``, an OS-level crash, a force-quit of
the TUI/desktop app — that teardown code never runs, and the child (plus any
of its own descendants, e.g. mcp-remote's spawned ``node`` process) is
orphaned. macOS has no direct equivalent of Linux's
``prctl(PR_SET_PDEATHSIG)`` to make the kernel auto-kill a child when its
parent dies, so nothing reaps these until the next Hermes startup's opt-in
``_kill_orphaned_mcp_children()`` sweep — which only runs if something calls
it. Repeated ungraceful session restarts can pile up N orphaned processes,
all racing to hold the same upstream SSE session, producing errors like
"Invalid request parameters" / "Received request before initialization was
complete" on the *legitimate* new connection.

Fix: don't spawn the MCP server command directly. Spawn this supervisor
instead, which:
  1. execs the real command as its own child (own process group via
     ``start_new_session``, so it doesn't inherit the supervisor's
     controlling terminal weirdly and so we can killpg it cleanly);
  2. transparently passes stdin/stdout/stderr through — the MCP stdio
     protocol talks directly over those pipes, so the supervisor must be a
     no-op relay, not a bytes-in-the-middle proxy;
  3. runs a background thread that polls the direct POSIX parent identity:
     compare current ``getppid()`` against the parent PID recorded when the
     wrapper was created, AND — when a startup identity snapshot of that PID
     is available — verify the process now occupying the PID is the same
     incarnation. The second check closes the PID-recycling race: after a
     crash the OS can hand the original PID to an unrelated new process
     before this watchdog notices, which would otherwise defeat the whole
     point of the supervisor;
  4. the instant the original parent is gone, terminates the real child's
     process group (SIGTERM, grace period, then SIGKILL) and exits.

This is intentionally a thin, standard-library-only script so it starts fast
and can't itself become a resource leak.

Usage (see ``tools/mcp_tool.py::_run_stdio``)::

    python3 -m tools.mcp_stdio_watchdog \\
        --ppid <original_parent_pid> -- <real_command> <arg1> <arg2> ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 3.0

# Snapshot of the original parent's process identity, taken once at startup.
# ``getppid() == original_ppid`` is necessary but not sufficient: after an
# ungraceful Hermes crash the kernel may recycle the PID onto a brand-new
# process before this supervisor notices, and the orphaned MCP child would
# then keep holding its upstream SSE session forever. The identity snapshot
# lets the loop tell "same process, still alive" apart from "PID recycled".
_original_parent_identity: Optional[str] = None


def _macos_process_identity(pid: int) -> Optional[str]:
    """macOS start-time identity via ``proc_pidinfo`` (libproc, in libSystem).

    Pure in-process call — no subprocess, so it works even where the ``ps``
    binary is unavailable (sandboxes, minimal PATH), and it returns
    second+microsecond start times, far finer than ``ps -o lstart``'s seconds.
    Returns None on any failure (caller falls back).
    """
    try:
        import ctypes
    except ImportError:  # pragma: no cover — ctypes is stdlib
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)  # current process's symbols
        proc_pidinfo = libc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int

        class _ProcBsdInfo(ctypes.Structure):
            # struct proc_bsdinfo from <sys/proc_info.h>, up to the two
            # start-time fields we need (ctypes handles alignment itself).
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32),
                ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32),
                ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32),
                ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32),
                ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32),
                ("pbi_rfu", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32),
                ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32),
                ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]

        info = _ProcBsdInfo()
        # PROC_PIDTBSDINFO == 3
        n = proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if n <= 0:
            return None  # ESRCH (process gone) or permission error
        return f"{info.pbi_start_tvsec}.{info.pbi_start_tvusec:06d}"
    except (AttributeError, OSError, ctypes.ArgumentError):  # pragma: no cover
        return None


def _parse_starttime_from_proc_stat(data: str) -> Optional[str]:
    """Extract the ``starttime`` field (22nd) from a ``/proc/<pid>/stat`` line.

    ``comm`` (field 2) may itself contain ')' and spaces, so split on the
    LAST ')'.  Fields after it begin at field #3 (state); ``starttime`` is
    field #22, i.e. index 19 of that tail. Returns None on malformed data.
    """
    try:
        return data.rsplit(")", 1)[1].split()[22 - 3]
    except (IndexError, ValueError):
        return None


def _read_process_identity(pid: int) -> Optional[str]:
    """Return a stable, comparable identity string for *pid*, or None.

    Linux: read the kernel ``starttime`` field of ``/proc/<pid>/stat`` — a
    monotonic jiffies counter since boot. Cheap (no subprocess), immune to
    wall-clock skew, and effectively unique per process incarnation, so a
    recycled PID can never collide with a live parent's value.

    macOS (no procfs): ``proc_pidinfo`` via libproc — an in-process ctypes
    call, no subprocess, second+microsecond resolution.

    Last-resort fallback for other POSIX platforms: spawn ``ps -o lstart=``
    with a forced ``LC_ALL=C`` so the emitted date string is
    locale-independent and therefore comparable across calls.

    Returns None when the process is already gone or the platform can't be
    queried. Callers MUST treat None as "cannot verify", never as "recycled",
    so a transient read failure can't cause a false kill.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as fh:
            data = fh.read()
    except OSError:
        data = None
    if data:
        parsed = _parse_starttime_from_proc_stat(data)
        if parsed is not None:
            return parsed
    if sys.platform == "darwin":
        macos_ident = _macos_process_identity(pid)
        if macos_ident is not None:
            return macos_ident
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            env={"LC_ALL": "C", "PATH": os.environ.get("PATH") or "/bin:/usr/bin"},
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip()
    return line or None


def _snapshot_parent_identity(original_ppid: int) -> None:
    """Record the original parent's identity for later recycling checks."""
    global _original_parent_identity
    _original_parent_identity = _read_process_identity(original_ppid)


def _is_orphaned(original_ppid: int, getppid=os.getppid) -> bool:
    """Return whether this process no longer has its original POSIX parent.

    Two-tier check: the direct parent PID must still match AND — when a
    startup identity snapshot exists — the process now occupying that PID
    must be the same incarnation (guards against PID recycling after a
    crash). Identity read failures are conservative: never kill on
    uncertainty, fall back to the legacy PID-only behavior instead.
    """
    if getppid() != original_ppid:
        return True
    if _original_parent_identity is None:
        return False  # no snapshot → legacy PID-only behavior
    current = _read_process_identity(original_ppid)
    if current is None:
        return False  # transient read failure — cannot verify, do not kill
    return current != _original_parent_identity


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child's process group.

    This module only ever runs on POSIX (the wrap site in tools/mcp_tool.py
    gates on ``os.name == "posix"``), but guard the POSIX-only primitives
    anyway so an accidental Windows import/execute degrades to a plain
    child kill instead of AttributeError.
    """
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok — non-POSIX fallback
        try:
            proc.terminate()
            proc.wait(timeout=_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=_TERM_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _watchdog_loop(proc: subprocess.Popen, original_ppid: int) -> None:
    while proc.poll() is None:
        if _is_orphaned(original_ppid):
            _terminate_process_group(proc)
            return
        time.sleep(_POLL_INTERVAL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parent-death watchdog for a stdio MCP subprocess.",
    )
    parser.add_argument("--ppid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    real_argv = list(args.command)
    if real_argv and real_argv[0] == "--":
        real_argv = real_argv[1:]
    if not real_argv:
        print("mcp_stdio_watchdog: no command given after '--'", file=sys.stderr)
        return 2

    # New process group so we can killpg() the whole tree the real command
    # may spawn (e.g. mcp-remote's own child `node` process), without
    # touching our own group or the (already-gone) original parent's.
    proc = subprocess.Popen(
        real_argv,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    # Snapshot the parent's process identity NOW, while the original parent
    # is (almost certainly) still alive, so the loop can later distinguish a
    # live original parent from a recycled PID after a crash.
    _snapshot_parent_identity(args.ppid)

    # Because the real server lives in its OWN process group (above), the
    # parent's graceful-shutdown killpg of *our* group no longer reaches it.
    # Forward SIGTERM/SIGINT to the child's group so graceful teardown
    # (`_kill_orphaned_mcp_children`, shutdown sweeps) still kills a wedged
    # server that ignores stdin EOF — otherwise the watchdog wrap would
    # invert the bug it fixes.
    def _forward_shutdown(signum, frame):  # noqa: ARG001
        _terminate_process_group(proc)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_shutdown)
    signal.signal(signal.SIGINT, _forward_shutdown)

    watchdog = threading.Thread(
        target=_watchdog_loop,
        args=(proc, args.ppid),
        daemon=True,
    )
    watchdog.start()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        return 130


if __name__ == "__main__":
    sys.exit(main())
